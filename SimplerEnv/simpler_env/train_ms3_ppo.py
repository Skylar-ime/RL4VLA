import os
import json
import pprint
import random
import gc
import io
import signal
import tarfile
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import time
from pathlib import Path
from typing import Annotated
import torch
import numpy as np
import tyro
import wandb
from dataclasses import dataclass, replace
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
    checkpoint_steps: str = ""
    eval_only: bool = False
    eval_episodes: int = 6
    eval_obj_set: str = "test"
    eval_episode_start: int = 0

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
    sdpo_feedback_version: str = "relspatial-v2"  # relspatial-v2, relspatial-action-v1, relspatial-outcome-v2, multimodal-correction-v2
    external_feedback_url: str = ""
    external_feedback_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    external_feedback_timeout: int = 120

    # other
    wandb: bool = True
    only_render: bool = False
    render_info: bool = False
    save_render: bool = True



def parse_checkpoint_steps(value: str, steps_max: int, num_envs: int) -> tuple[int, ...]:
    if num_envs < 1 or steps_max < 1 or steps_max % num_envs:
        raise ValueError("steps_max must be a positive multiple of num_envs")
    if not value:
        return ()
    try:
        steps = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise ValueError("checkpoint_steps must contain comma-separated integers") from error
    if (not steps or tuple(sorted(set(steps))) != steps or
            any(step < 1 or step > steps_max or step % num_envs for step in steps)):
        raise ValueError("checkpoint_steps must be unique, increasing environment-step multiples within steps_max")
    return steps


def next_rollout_length(completed_steps: int, steps_max: int, num_envs: int,
                        episode_len: int, checkpoint_steps: tuple[int, ...]) -> int:
    if not 0 <= completed_steps < steps_max or episode_len < 1:
        raise ValueError("Rollout position or maximum episode length is invalid")
    next_boundary = min((step for step in checkpoint_steps if step > completed_steps), default=steps_max)
    rollout_length = min(episode_len, (next_boundary - completed_steps) // num_envs)
    if rollout_length < 1:
        raise ValueError("Checkpoint boundary cannot be reached by the allocated environments")
    return rollout_length


class Runner:
    def __init__(self, all_args: Args):
        self.args = all_args

        # alg_name
        assert self.args.alg_name in ["ppo", "grpo", "sdpo"]
        if self.args.sdpo_feedback_version not in {"relspatial-v2", "relspatial-action-v1", "relspatial-outcome-v2", "multimodal-correction-v2"}:
            raise ValueError(f"Unknown sdpo_feedback_version: {self.args.sdpo_feedback_version}")
        self.multimodal_mode = self.args.alg_name == "sdpo" and self.args.sdpo_feedback_version == "multimodal-correction-v2"
        if self.multimodal_mode:
            if not self.args.external_feedback_url:
                raise ValueError("Multimodal correction requires --external-feedback-url")
            if self.args.sdpo_feedback_mode != "correct" or self.args.alg_gradient_accum != 1 or self.args.alg_ppo_epoch != 1:
                raise ValueError("Multimodal correction requires correct feedback, one optimizer step and one EMA update per feedback")
        if self.args.alg_gradient_accum < 1:
            raise ValueError("alg_gradient_accum must be positive")
        if self.args.alg_name == "sdpo":
            rollout_size = self.args.episode_len * self.args.num_envs
            if self.args.buffer_minibatch == 0:
                raise ValueError("buffer_minibatch cannot be zero")
            if self.args.buffer_minibatch > 0 and rollout_size % self.args.buffer_minibatch:
                raise ValueError("SDPO minibatch size must divide the rollout size")
            minibatches = 1 if self.args.buffer_minibatch < 0 else rollout_size // self.args.buffer_minibatch
            if minibatches % self.args.alg_gradient_accum:
                raise ValueError("SDPO accumulation interval must divide the minibatch count")

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
        self.external_feedback = None
        if self.multimodal_mode:
            from simpler_env.external_feedback import ExternalFeedbackClient
            self.external_feedback = ExternalFeedbackClient(
                base_url=self.args.external_feedback_url,
                model=self.args.external_feedback_model,
                api_key=os.environ.get("EXTERNAL_FEEDBACK_API_KEY", ""),
                timeout_seconds=self.args.external_feedback_timeout,
            )

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

    def train_multimodal_feedback(self, image, instruction: str, action, record):
        batch = (image, [instruction], action, None, None, None, None, None, [record])
        self.policy.prep_training()
        metrics = self.alg.train_sdpo_step(0, 1, batch)
        self.policy.prep_rollout()
        if metrics["optimizer_steps"] != 1.0:
            raise RuntimeError("One multimodal feedback must produce one student and EMA update")
        return metrics

    def request_multimodal_feedback(self, instruction, before_image, after_image, outcome,
                                    executed_action, rollout_step):
        started = time.perf_counter()
        record = self.external_feedback.generate(
            instruction=instruction,
            before_image=before_image,
            after_image=after_image,
            outcome=outcome,
            executed_action=executed_action,
            rollout_step=rollout_step,
        )
        return record, time.perf_counter() - started

    def save_feedback_archive(self, episode: int):
        """Persist decision-time privileged records for audit; never used as student input."""
        feedback_dir = self.glob_dir / "feedback"
        feedback_dir.mkdir(parents=True, exist_ok=True)
        path = feedback_dir / f"episode_{episode:06d}.jsonl"
        image_tar = tarfile.open(feedback_dir / f"episode_{episode:06d}_images.tar", "w") if self.multimodal_mode else None
        try:
            with path.open("w", encoding="utf-8") as handle:
                for decision_index in range(self.buffer.ep_len):
                    for env_index in range(self.args.num_envs):
                        record = self.buffer.feedback[decision_index][env_index]
                        if record is not None:
                            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
                            if image_tar is not None:
                                for phase, data in (("before", record.before_jpeg), ("after", record.after_jpeg)):
                                    member = tarfile.TarInfo(record.image_member(phase))
                                    member.size = len(data)
                                    image_tar.addfile(member, io.BytesIO(data))
        finally:
            if image_tar is not None:
                image_tar.close()
        return path

    @torch.no_grad()
    def eval(
        self,
        obj_set: str,
        episode_offset: int = 0,
        return_episode_stats: bool = False,
    ) -> dict | tuple[dict, list[dict]]:
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
        ever_grasped = torch.zeros_like(completed)
        reached_target_after_grasp = torch.zeros_like(completed)
        released_on_target = torch.zeros_like(completed)
        first_grasp_step = torch.full((self.args.num_envs,), -1, dtype=torch.int32, device=self.device)
        grasp_onset_target_distance = torch.full((self.args.num_envs,), float("inf"), device=self.device)
        best_carried_target_distance = torch.full_like(grasp_onset_target_distance, float("inf"))
        episode_stats = {}

        for elapsed_step in range(1, self.args.episode_len + 1):
            obs = dict(image=obs_img, task_description=instruction)
            value, action, logprob = self._get_action(obs, deterministic=True)

            obs_img, reward, done, env_info = self.env.step(action)

            active = ~completed
            grasped_now = self.env.selected_source_grasped().reshape(-1).to(torch.bool)
            on_target_now = env_info["src_on_target"].reshape(-1).to(torch.bool)
            target_distance = env_info["carrot_plate_dist"].reshape(-1).to(torch.float32)
            new_grasp = active & grasped_now & ~ever_grasped
            first_grasp_step[new_grasp] = elapsed_step
            grasp_onset_target_distance[new_grasp] = target_distance[new_grasp]
            ever_grasped |= active & grasped_now
            best_carried_target_distance = torch.where(
                active & grasped_now,
                torch.minimum(best_carried_target_distance, target_distance),
                best_carried_target_distance,
            )
            reached_target_after_grasp |= active & ever_grasped & on_target_now
            released_on_target |= active & ever_grasped & on_target_now & ~grasped_now

            # info
            print({k: round(v.to(torch.float32).mean().tolist(), 4) for k, v in env_info.items()
                   if k not in {"episode", "episode_env_ids"}})
            if "episode" in env_info.keys():
                for value_index, env_index in enumerate(env_info["episode_env_ids"]):
                    if completed[env_index]:
                        continue
                    completed[env_index] = True
                    episode_id = int(episode_ids[env_index].item())
                    episode_record = {"episode_id": episode_id}
                    for key, values in env_info["episode"].items():
                        value = float(values[value_index])
                        env_infos[key].append(value)
                        episode_record[key] = value
                    grasp_step = int(first_grasp_step[env_index].item())
                    episode_record["first_grasp_step"] = grasp_step if grasp_step >= 0 else None
                    episode_record["reached_target_after_grasp"] = bool(reached_target_after_grasp[env_index].item())
                    episode_record["released_on_target"] = bool(released_on_target[env_index].item())
                    episode_record["carried_target_distance_reduction_m"] = (
                        float((grasp_onset_target_distance[env_index] - best_carried_target_distance[env_index]).item())
                        if grasp_step >= 0 else None
                    )
                    env_infos["reached_target_after_grasp"].append(float(episode_record["reached_target_after_grasp"]))
                    env_infos["released_on_target"].append(float(episode_record["released_on_target"]))
                    episode_stats[episode_id] = episode_record
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

        if return_episode_stats:
            return env_stats, [episode_stats[key] for key in sorted(episode_stats)]
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
        if self.args.eval_episode_start < 0:
            raise ValueError("eval_episode_start must be nonnegative")

        results_path = self.save_dir / "eval_only.jsonl"
        self.policy.prep_rollout()
        batch_stats = defaultdict(list)
        with results_path.open("w", encoding="utf-8") as handle:
            for batch_index in range(self.args.eval_episodes // self.args.num_envs):
                episode_offset = self.args.eval_episode_start + batch_index * self.args.num_envs
                stats, episode_stats = self.eval(
                    self.args.eval_obj_set,
                    episode_offset=episode_offset,
                    return_episode_stats=True,
                )
                for key, value in stats.items():
                    batch_stats[key].append(value)
                record = {
                    "obj_set": self.args.eval_obj_set,
                    "episode_ids": list(range(episode_offset, episode_offset + self.args.num_envs)),
                    "episode_stats": episode_stats,
                    **stats,
                }
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                wandb.log({f"baseline/{key}": value for key, value in stats.items()}, step=batch_index)
        overall_stats = {
            f"baseline/overall_{key}": float(np.mean(values))
            for key, values in batch_stats.items()
        }
        wandb.log(overall_stats, step=len(batch_stats["success"]))
        wandb.save(str(results_path), base_path=str(self.save_dir), policy="now")
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
        checkpoint_steps = parse_checkpoint_steps(self.args.checkpoint_steps, self.args.steps_max, self.args.num_envs)
        training_started = time.monotonic()
        evaluation_seconds = 0.0
        optimizer_steps_total = 0.0
        completed_steps = 0
        episode = 0
        feedback_pool = ThreadPoolExecutor(max_workers=min(self.args.num_envs, 4)) if self.multimodal_mode else None
        while completed_steps < self.args.steps_max:
            rollout_length = next_rollout_length(
                completed_steps, self.args.steps_max, self.args.num_envs,
                self.args.episode_len, checkpoint_steps,
            )
            if self.buffer.ep_len != rollout_length:
                self.buffer = SeparatedReplayBuffer(
                    replace(self.args, episode_len=rollout_length), obs_dim=(480, 640, 3), act_dim=7,
                )
            env_infos = defaultdict(lambda: [])
            online_infos = defaultdict(list)
            ep_time = time.time()
            rollout_time = time.time()
            for device_index in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(device_index)

            obs_img, instruction, info = self.env.reset(obj_set="train", same_init=self.args.use_same_init)
            self.buffer.warmup(obs_img.cpu().numpy(), instruction)

            for decision_index in tqdm(range(rollout_length), desc="rollout"):
                value, action, logprob = self.collect()
                outcome_mode = self.args.alg_name == "sdpo" and self.args.sdpo_feedback_version in {
                    "relspatial-outcome-v2", "multimodal-correction-v2",
                }
                reset_after_step = decision_index + 1 < rollout_length
                if outcome_mode:
                    obs_img, reward, done, env_info, feedback = self.env.step_with_outcome_feedback(
                        action, reset_after_step=reset_after_step, include_contact=self.multimodal_mode,
                    )
                    if self.multimodal_mode:
                        executed_actions = self.env._process_action(action).detach().cpu().numpy()
                        after_images = obs_img.detach().cpu().numpy()
                        outcomes = feedback
                        feedback = []
                        requests = [feedback_pool.submit(
                            self.request_multimodal_feedback,
                            self.buffer.instruction[env_index],
                            self.buffer.obs[self.buffer.step, env_index],
                            after_images[env_index],
                            outcome,
                            executed_actions[env_index],
                            decision_index,
                        ) for env_index, outcome in enumerate(outcomes)]
                        for env_index, outcome in enumerate(outcomes):
                            instruction = self.buffer.instruction[env_index]
                            before_image = self.buffer.obs[self.buffer.step, env_index:env_index + 1]
                            record, feedback_seconds = requests[env_index].result()
                            feedback.append(record)
                            online_infos["time/external_feedback_seconds"].append(feedback_seconds)
                            metrics = self.train_multimodal_feedback(
                                before_image, instruction, action[env_index:env_index + 1].detach().cpu().numpy(), record,
                            )
                            for key, metric in metrics.items():
                                online_infos[key].append(metric)
                else:
                    feedback = self.env.capture_feedback(
                        raw_action=action if self.args.sdpo_feedback_version == "relspatial-action-v1" else None,
                        version=self.args.sdpo_feedback_version,
                    ) if self.args.alg_name == "sdpo" else None
                    obs_img, reward, done, env_info = self.env.step(action)

                reset_batch = outcome_mode and reset_after_step and bool(done.any())
                boundary_done = torch.ones_like(done) if reset_batch else done
                data = (obs_img, action, logprob, value, reward, boundary_done, feedback)
                self.insert(data)
                if reset_batch:
                    next_image, next_instruction, _ = self.env.reset(
                        obj_set="train", same_init=self.args.use_same_init,
                    )
                    self.buffer.set_next_observation(next_image.cpu().numpy(), next_instruction)

                # info
                if "episode" in env_info.keys():
                    for k, v in env_info["episode"].items():
                        env_infos[f"{k}"] += v

            # steps
            steps = completed_steps + rollout_length * self.args.num_envs
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
            feedback_total = rollout_length * self.args.num_envs

            # train and process infos
            if self.args.alg_name != "sdpo":
                self.compute_endup()
            del value, action, logprob, obs_img, reward, done
            gc.collect()
            torch.cuda.empty_cache()

            # train
            update_time = time.time()
            if self.multimodal_mode:
                infos = {
                    f"train/{key}": (sum(values) if key == "optimizer_steps" else float(np.mean(values)))
                    for key, values in online_infos.items()
                }
                infos["buffer/reward_mean"] = np.mean(self.buffer.rewards)
                infos["buffer/mask_mean"] = np.mean(1.0 - self.buffer.masks)
            else:
                infos = self.train()
            update_seconds = time.time() - update_time
            for k, v in env_infos.items():
                infos[f"env/{k}"] = np.mean(v)
            optimizer_steps_total += infos.get("train/optimizer_steps", 0.0)
            infos["train/optimizer_steps_cumulative"] = optimizer_steps_total
            infos["time/train_cumulative_seconds"] = time.monotonic() - training_started - evaluation_seconds
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
            if self.args.interval_eval > 0 and (
                episode % self.args.interval_eval == self.args.interval_eval - 1 or steps == self.args.steps_max
            ):
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
                evaluation_seconds += sval_stats["time/eval_train_seconds"] + sval_stats["time/eval_test_seconds"]

            # save
            if (steps in checkpoint_steps or steps == self.args.steps_max or
                    self.args.interval_save > 0 and episode % self.args.interval_save == self.args.interval_save - 1):
                print(f"Saving model at {steps}")
                save_path = self.glob_dir / (
                    f"step_{steps:06d}" if checkpoint_steps else f"steps_{episode:0>4d}"
                )
                self.policy.save(save_path)
                if checkpoint_steps:
                    (save_path / "environment_steps.json").write_text(json.dumps({
                        "environment_steps": steps,
                        "rollout_length": rollout_length,
                        "num_envs": self.args.num_envs,
                        "rollout_index": episode,
                        "optimizer_steps_cumulative": optimizer_steps_total,
                        "train_cumulative_seconds": time.monotonic() - training_started - evaluation_seconds,
                    }, sort_keys=True) + "\n")

                if self.args.save_render:
                    self.render(epoch=episode, obj_set="train")
                    self.render(epoch=episode, obj_set="test")
            completed_steps = steps
            episode += 1


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
