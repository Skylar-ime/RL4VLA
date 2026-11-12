import os
import json
import pprint
import random
import gc
import signal
from collections import defaultdict
import time
from pathlib import Path
from typing import Annotated
import torch
import numpy as np
import tyro
import wandb
from dataclasses import dataclass
import yaml
from tqdm import tqdm
from mani_skill.utils import visualization
from mani_skill.utils.visualization.misc import images_to_video

from simpler_env.env.simpler_wrapper import SimlerWrapper
from simpler_env.utils.replay_buffer import SeparatedReplayBuffer

signal.signal(signal.SIGINT, signal.SIG_DFL)  # allow ctrl+c
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "PutCarrotOnPlateInScene-v1"
    """The environment ID of the task you want to simulate. Can be one of
    PutCarrotOnPlateInScene-v1, PutSpoonOnTableClothInScene-v1, StackGreenCubeOnYellowCubeBakedTexInScene-v1, PutEggplantInBasketScene-v1"""

    """Number of environments to run. With more than 1 environment the environment will use the GPU backend 
    which runs faster enabling faster large-scale evaluations. Note that the overall behavior of the simulation
    will be slightly different between CPU and GPU backends."""

    seed: Annotated[int, tyro.conf.arg(aliases=["-s"])] = 0
    """Seed the model and environment. Default seed is 0"""

    name: str = "PPO-test"

    # env
    num_envs: int = 64
    episode_len: int = 80
    use_same_init: bool = False
    sim_backend: str = "gpu"

    steps_max: int = 2000000
    steps_vh: int = 0  # episodes
    interval_eval: int = 10
    interval_save: int = 40
    eval_only: bool = False
    eval_episodes: int = 6
    eval_obj_set: str = "test"

    # buffer
    buffer_inferbatch: int = 32
    buffer_minibatch: int = 8
    buffer_gamma: float = 0.99
    buffer_lambda: float = 0.95

    # vla
    vla_path: str = "openvla/openvla-7b"
    vla_unnorm_key: str = "bridge_orig"
    vla_load_path: str = ""
    vla_lora_rank: int = 32

    vla_lr: float = 1e-4
    vla_vhlr: float = 3e-3
    vla_optim_beta1: float = 0.9
    vla_optim_beta2: float = 0.999
    vla_temperature: float = 1.0
    vla_temperature_eval: float = 0.6

    # ppo & grpo
    alg_name: str = "ppo"  # ppo, grpo, sdpo
    alg_grpo_fix: bool = True
    alg_gradient_accum: int = 20
    alg_ppo_epoch: int = 1
    alg_entropy_coef: float = 0.0
    sdpo_ema_rate: float = 0.01
    sdpo_feedback_mode: str = "correct"  # correct, shuffled, none

    # other
    wandb: bool = True
    only_render: bool = False
    render_info: bool = False
    save_render: bool = True



class Runner:
    def __init__(self, all_args: Args):
        self.args = all_args

        # alg_name
        assert self.args.alg_name in ["ppo", "grpo", "sdpo"]

        # set seed
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

        # set wandb
        wandb.init(
            config=all_args.__dict__,
            project="RLVLA",
            name=self.args.name,
            mode=os.environ.get("WANDB_MODE", "online") if self.args.wandb else "offline",
        )
        self.save_dir = Path(wandb.run.dir)
        self.glob_dir = Path(wandb.run.dir) / ".." / "glob"
        self.glob_dir.mkdir(parents=True, exist_ok=True)

        yaml.dump(all_args.__dict__, open(self.glob_dir / "config.yaml", "w"))

        # policy
        from simpler_env.policies.openvla.openvla_train import OpenVLAPolicy, OpenVLAPPO
        device_id = 0
        device_id_other = 1 if torch.cuda.device_count() > 1 else 0
        self.device = torch.device("cuda:" + str(device_id))
        self.policy = OpenVLAPolicy(all_args, device_id_other)

        self.alg = OpenVLAPPO(all_args, self.policy)

        # env
        unnorm_state = self.policy.vla.get_action_stats(self.args.vla_unnorm_key)
        self.env = SimlerWrapper(self.args, unnorm_state)

        # buffer
        self.buffer = SeparatedReplayBuffer(
            all_args,
            obs_dim=(480, 640, 3),
            act_dim=7,
        )
        minibatch_count = self.buffer.get_minibatch_count()
        print(f"Buffer minibatch count: {minibatch_count}")

    @torch.no_grad()
    def _get_action(self, obs, deterministic=False):
        total_batch = obs["image"].shape[0]

        values = []
        actions = []
        logprobs = []

        for i in range(0, total_batch, self.args.buffer_inferbatch):
            obs_batch = {k: v[i:i + self.args.buffer_inferbatch] for k, v in obs.items()}
            value, action, logprob = self.policy.get_action(obs_batch, deterministic)
            values.append(value)
            actions.append(action)
            logprobs.append(logprob)

        values = torch.cat(values, dim=0).to(device=self.device)
        actions = torch.cat(actions, dim=0).to(device=self.device)
        logprobs = torch.cat(logprobs, dim=0).to(device=self.device)

        return values, actions, logprobs

    def collect(self):
        self.policy.prep_rollout()

        obs_image = self.buffer.obs[self.buffer.step]
        obs_image = torch.tensor(obs_image).to(self.device)
        obs = dict(image=obs_image, task_description=self.buffer.instruction)
        value, action, logprob = self._get_action(obs)

        return value, action, logprob

    def insert(self, data):
        obs_img, actions, logprob, value_preds, rewards, done, feedback = data
        masks = 1.0 - done.to(torch.float32)

        obs_img = obs_img.cpu().numpy()
        actions = actions.to(torch.int32).cpu().numpy()
        logprob = logprob.to(torch.float32).cpu().numpy()
        value_preds = value_preds.to(torch.float32).cpu().numpy()
        rewards = rewards.cpu().numpy()
        masks = masks.cpu().numpy()

        self.buffer.insert(obs_img, actions, logprob, value_preds, rewards, masks, feedback)

    def compute_endup(self):
        self.policy.prep_rollout()

        obs_image = torch.tensor(self.buffer.obs[-1]).to(self.device)
        obs = dict(image=obs_image, task_description=self.buffer.instruction)
        with torch.no_grad():
            next_value, _, _ = self._get_action(obs)
        next_value = next_value.to(torch.float32).cpu().numpy()

        self.buffer.endup(next_value)

    def train(self):
        self.policy.prep_training()

        if self.args.alg_name == "ppo":
            train_info = self.alg.train_ppo(self.buffer)
        elif self.args.alg_name == "grpo":
            train_info = self.alg.train_grpo(self.buffer)
        elif self.args.alg_name == "sdpo":
            train_info = self.alg.train_sdpo(self.buffer)
        else:
            raise ValueError(f"Unknown alg_name: {self.args.alg_name}")

        info = {f"train/{k}": v for k, v in train_info.items()}
        info["buffer/reward_mean"] = np.mean(self.buffer.rewards)
        info["buffer/mask_mean"] = np.mean(1.0 - self.buffer.masks)

        return info

    def save_feedback_archive(self, episode: int):
        """Persist decision-time privileged records for audit; never used as student input."""
        feedback_dir = self.glob_dir / "feedback"
        feedback_dir.mkdir(parents=True, exist_ok=True)
        path = feedback_dir / f"episode_{episode:06d}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for decision_index in range(self.args.episode_len):
                for env_index in range(self.args.num_envs):
                    record = self.buffer.feedback[decision_index][env_index]
                    if record is not None:
                        handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
        return path

    @torch.no_grad()
    def eval(self, obj_set: str, episode_offset: int = 0) -> dict:
        self.policy.prep_rollout()
        env_infos = defaultdict(lambda: [])
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_states = torch.cuda.get_rng_state_all()
        torch.manual_seed(self.args.seed + 100000 + episode_offset)

        episode_ids = torch.arange(
            episode_offset, episode_offset + self.args.num_envs, device=self.env.env.unwrapped.device
        )
        obs_img, instruction, info = self.env.reset(obj_set=obj_set, episode_ids=episode_ids)
        completed = torch.zeros(self.args.num_envs, dtype=torch.bool, device=self.device)

        for _ in range(self.args.episode_len):
            obs = dict(image=obs_img, task_description=instruction)
            value, action, logprob = self._get_action(obs, deterministic=True)

            obs_img, reward, done, env_info = self.env.step(action)

            # info
            print({k: round(v.to(torch.float32).mean().tolist(), 4) for k, v in env_info.items()
                   if k not in {"episode", "episode_env_ids"}})
            if "episode" in env_info.keys():
                for value_index, env_index in enumerate(env_info["episode_env_ids"]):
                    if completed[env_index]:
                        continue
                    completed[env_index] = True
                    for key, values in env_info["episode"].items():
                        env_infos[key].append(values[value_index])
                if completed.all():
                    break

        if not completed.all():
            raise RuntimeError("evaluation ended before every requested episode completed")

        # infos
        env_stats = {k: np.mean(v) for k, v in env_infos.items()}
        env_stats = env_stats.copy()
        torch.set_rng_state(cpu_rng_state)
        torch.cuda.set_rng_state_all(cuda_rng_states)

        print(pprint.pformat({k: round(v, 4) for k, v in env_stats.items()}))
        print(f"")

        return env_stats

    @torch.no_grad()
    def eval_over_episodes(self, obj_set: str) -> dict:
        batch_count = max(
            1,
            (self.args.eval_episodes + self.args.num_envs - 1) // self.args.num_envs,
        )
        batch_stats = defaultdict(list)
        for batch_index in range(batch_count):
            stats = self.eval(
                obj_set=obj_set,
                episode_offset=batch_index * self.args.num_envs,
            )
            for key, value in stats.items():
                batch_stats[key].append(value)
        return {key: float(np.mean(values)) for key, values in batch_stats.items()}

    def run_eval_only(self):
        if self.args.eval_episodes <= 0 or self.args.eval_episodes % self.args.num_envs != 0:
            raise ValueError("eval_episodes must be a positive multiple of num_envs")
        if self.args.eval_obj_set not in {"train", "test"}:
            raise ValueError("eval_obj_set must be 'train' or 'test'")

        results_path = self.glob_dir / "eval_only.jsonl"
        self.policy.prep_rollout()
        batch_stats = defaultdict(list)
        with results_path.open("w", encoding="utf-8") as handle:
            for batch_index in range(self.args.eval_episodes // self.args.num_envs):
                episode_offset = batch_index * self.args.num_envs
                stats = self.eval(self.args.eval_obj_set, episode_offset=episode_offset)
                for key, value in stats.items():
                    batch_stats[key].append(value)
                record = {
                    "obj_set": self.args.eval_obj_set,
                    "episode_ids": list(range(episode_offset, episode_offset + self.args.num_envs)),
                    **stats,
                }
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                wandb.log({f"baseline/{key}": value for key, value in stats.items()}, step=batch_index)
        overall_stats = {
            f"baseline/overall_{key}": float(np.mean(values))
            for key, values in batch_stats.items()
        }
        wandb.log(overall_stats, step=len(batch_stats["success"]))
        print(f"Saved evaluation records: {results_path}")

    @torch.no_grad()
    def render(self, epoch: int, obj_set: str) -> dict:
        self.policy.prep_rollout()

        # init logger
        env_infos = defaultdict(lambda: [])
        datas = [{
            "image": [],  # obs_t: [0, T-1]
            "instruction": "",
            "action": [],  # a_t: [0, T-1]
            "info": [],  # info after executing a_t: [1, T]
        } for idx in range(self.args.num_envs)]

        obs_img, instruction, info = self.env.reset(obj_set)
        print("instruction[:3]:", instruction[:3])

        # data dump: instruction
        for idx in range(self.args.num_envs):
            datas[idx]["instruction"] = instruction[idx]

        for _ in range(self.args.episode_len):
            obs = dict(image=obs_img, task_description=instruction)
            value, action, logprob = self._get_action(obs, deterministic=True)

            obs_img_new, reward, done, env_info = self.env.step(action)

            # info
            print({k: round(v.to(torch.float32).mean().tolist(), 4) for k, v in env_info.items()
                   if k not in {"episode", "episode_env_ids"}})
            if "episode" in env_info.keys():
                for k, v in env_info["episode"].items():
                    env_infos[f"{k}"] += v

            for i in range(self.args.num_envs):
                post_action = self.env._process_action(action)
                log_image = obs_img[i].cpu().numpy()
                log_action = post_action[i].cpu().numpy().tolist()
                log_info = {k: v[i].tolist() for k, v in env_info.items()
                            if k not in {"episode", "episode_env_ids"}}
                datas[i]["image"].append(log_image)
                datas[i]["action"].append(log_action)
                datas[i]["info"].append(log_info)

            # update obs_img
            obs_img = obs_img_new

        # data dump: last image
        for i in range(self.args.num_envs):
            log_image = obs_img[i].cpu().numpy()
            datas[i]["image"].append(log_image)

        # save video
        exp_dir = Path(self.glob_dir) / f"vis_{epoch}_{obj_set}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        for i in range(self.args.num_envs):
            images = datas[i]["image"]
            infos = datas[i]["info"]
            assert len(images) == len(infos) + 1

            if self.args.render_info:
                for j in range(len(infos)):
                    images[j + 1] = visualization.put_info_on_image(
                        images[j + 1], infos[j],
                        extras=[f"Ins: {instruction[i]}"]
                    )

            success = int(infos[-1]["success"])
            images_to_video(images, str(exp_dir), f"video_{i}-s_{success}",
                            fps=10, verbose=False)

        # infos
        env_stats = {k: np.mean(v) for k, v in env_infos.items()}
        env_stats_ret = env_stats.copy()

        print(pprint.pformat({k: round(v, 4) for k, v in env_stats.items()}))
        print(f"")

        # save stats
        last_info = {
            idx: {k: env_infos[k][idx] for k in env_infos.keys()}
            for idx in range(self.args.num_envs)
        }

        save_stats = {}
        save_stats["env_name"] = self.args.env_id
        save_stats["ep_len"] = self.args.episode_len
        save_stats["epoch"] = epoch
        save_stats["stats"] = {k: v.item() for k, v in env_stats.items()}
        save_stats["instruction"] = {idx: ins for idx, ins in enumerate(instruction)}
        save_stats["last_info"] = last_info

        yaml.dump(save_stats, open(exp_dir / "stats.yaml", "w"))

        return env_stats_ret

    def run(self):
        max_episodes = self.args.steps_max // self.args.episode_len // self.args.num_envs

        for episode in range(max_episodes):
            env_infos = defaultdict(lambda: [])
            ep_time = time.time()
            rollout_time = time.time()
            for device_index in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(device_index)

            obs_img, instruction, info = self.env.reset(obj_set="train", same_init=self.args.use_same_init)
            self.buffer.warmup(obs_img.cpu().numpy(), instruction)

            for _ in tqdm(range(self.args.episode_len), desc="rollout"):
                value, action, logprob = self.collect()
                feedback = self.env.capture_feedback() if self.args.alg_name == "sdpo" else None
                obs_img, reward, done, env_info = self.env.step(action)

                data = (obs_img, action, logprob, value, reward, done, feedback)
                self.insert(data)

                # info
                if "episode" in env_info.keys():
                    for k, v in env_info["episode"].items():
                        env_infos[f"{k}"] += v

            # steps
            steps = (episode + 1) * self.args.episode_len * self.args.num_envs
            print(pprint.pformat({k: round(np.mean(v), 4) for k, v in env_infos.items()}))
            if self.args.alg_name == "sdpo":
                feedback_path = self.save_feedback_archive(episode)
                print(f"Saved feedback archive: {feedback_path}")
            rollout_seconds = time.time() - rollout_time
            feedback_count = sum(
                record is not None
                for step_records in self.buffer.feedback
                for record in step_records
            )
            feedback_total = self.args.episode_len * self.args.num_envs

            # train and process infos
            if self.args.alg_name != "sdpo":
                self.compute_endup()
            del value, action, logprob, obs_img, reward, done
            gc.collect()
            torch.cuda.empty_cache()

            # train
            update_time = time.time()
            infos = self.train()
            update_seconds = time.time() - update_time
            for k, v in env_infos.items():
                infos[f"env/{k}"] = np.mean(v)
            infos["time/rollout_seconds"] = rollout_seconds
            infos["time/update_seconds"] = update_seconds
            infos["time/episode_seconds"] = time.time() - ep_time
            infos["feedback/effective_ratio"] = feedback_count / max(feedback_total, 1)
            for device_index in range(torch.cuda.device_count()):
                infos[f"gpu/cuda_{device_index}_peak_allocated_gib"] = (
                    torch.cuda.max_memory_allocated(device_index) / (1024 ** 3)
                )
                infos[f"gpu/cuda_{device_index}_peak_reserved_gib"] = (
                    torch.cuda.max_memory_reserved(device_index) / (1024 ** 3)
                )

            # log
            wandb.log(infos, step=steps)

            elapsed_time = time.time() - ep_time
            print(f"{self.args.name}: ep {episode:0>4d} | steps {steps} | e {elapsed_time:.2f}s")
            print(pprint.pformat({k: round(v, 4) for k, v in infos.items()}))

            # eval
            if episode % self.args.interval_eval == self.args.interval_eval - 1 or episode == max_episodes - 1:
                print(f"Evaluating at {steps}")
                eval_time = time.time()
                sval_stats = self.eval_over_episodes(obj_set="train")
                sval_stats = {f"eval/{k}": v for k, v in sval_stats.items()}
                sval_stats["time/eval_train_seconds"] = time.time() - eval_time
                wandb.log(sval_stats, step=steps)

                eval_time = time.time()
                sval_stats = self.eval_over_episodes(obj_set="test")
                sval_stats = {f"eval/{k}_ood": v for k, v in sval_stats.items()}
                sval_stats["time/eval_test_seconds"] = time.time() - eval_time
                wandb.log(sval_stats, step=steps)

            # save
            if episode % self.args.interval_save == self.args.interval_save - 1 or episode == max_episodes - 1:
                print(f"Saving model at {steps}")
                save_path = self.glob_dir / f"steps_{episode:0>4d}"
                self.policy.save(save_path)

                if self.args.save_render:
                    self.render(epoch=episode, obj_set="train")
                    self.render(epoch=episode, obj_set="test")


def main():
    args = tyro.cli(Args)
    runner = Runner(args)

    if args.eval_only:
        runner.run_eval_only()
    elif args.only_render:
        ll = [
            "PutOnPlateInScene25VisionImage-v1",
            "PutOnPlateInScene25VisionTexture03-v1",
            "PutOnPlateInScene25VisionTexture05-v1",
            "PutOnPlateInScene25VisionWhole03-v1",
            "PutOnPlateInScene25VisionWhole05-v1",

            "PutOnPlateInScene25Instruct-v1",
            "PutOnPlateInScene25Plate-v1",
            "PutOnPlateInScene25Position-v1",
            "PutOnPlateInScene25EEPose-v1",
            "PutOnPlateInScene25PositionChange-v1",
            "PutOnPlateInScene25PositionChangeTo-v1"
        ]
        if args.env_id not in ll:
            runner.render(epoch=0, obj_set="train")
        runner.render(epoch=0, obj_set="test")
    else:
        runner.run()


if __name__ == "__main__":
    main()
