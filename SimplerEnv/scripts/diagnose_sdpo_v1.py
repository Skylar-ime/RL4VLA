"""Collect bounded, read-only v1 calibration rollouts and score frozen policies."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from simpler_env.env.simpler_wrapper import SimlerWrapper
from simpler_env.train_ms3_ppo import Args
from simpler_env.policies.openvla.openvla_train import OpenVLAPolicy


def selected_contact(task):
    if not all(hasattr(task, name) for name in ("select_carrot_ids", "carrot_names", "objs_carrot")):
        return [None] * task.num_envs
    left = torch.zeros(task.num_envs, device=task.device)
    right = torch.zeros_like(left)
    for index, name in enumerate(task.carrot_names):
        obj = task.objs_carrot[name]
        selected = task.select_carrot_ids == index
        left_force = task.scene.get_pairwise_contact_forces(task.agent.finger1_link, obj).norm(dim=1)
        right_force = task.scene.get_pairwise_contact_forces(task.agent.finger2_link, obj).norm(dim=1)
        left = torch.where(selected, left_force, left)
        right = torch.where(selected, right_force, right)
    return np.stack((left.cpu().numpy(), right.cpu().numpy()), axis=1).tolist()


def make_args(checkpoint, seed, num_envs):
    args = Args()
    args.alg_name = "ppo"
    args.env_id = "PutOnPlateInScene25Main-v3"
    args.vla_path = str(Path.home() / ".cache/huggingface/hub/models--gen-robot--openvla-7b-rlvla-warmup/snapshots/735fd5d1f4472c4c9f3f998f0e89aa641af5ad03")
    args.vla_load_path = str(checkpoint) if checkpoint else ""
    args.num_envs = num_envs
    args.episode_len = 80
    args.seed = seed
    return args


def expected_translation(logits, action_stats):
    bins = np.linspace(-1, 1, 256)
    centres = (bins[:-1] + bins[1:]) / 2
    normalized = torch.as_tensor(centres[np.clip(np.arange(255, -1, -1), 0, 254)],
                                 device=logits.device, dtype=logits.dtype)
    expected = (logits[:3].softmax(dim=-1) * normalized).sum(dim=-1)
    low = torch.as_tensor(action_stats["q01"][:3], device=logits.device, dtype=logits.dtype)
    high = torch.as_tensor(action_stats["q99"][:3], device=logits.device, dtype=logits.dtype)
    mask = torch.as_tensor(action_stats.get("mask", [True] * 7)[:3], device=logits.device)
    return torch.where(mask, 0.5 * (expected + 1) * (high - low) + low, expected)


def direction_score(logits, desired_delta_root, action_stats):
    if desired_delta_root is None:
        return None
    desired = torch.as_tensor(desired_delta_root, device=logits.device, dtype=logits.dtype)
    norm = desired.norm()
    if norm < 0.03:
        return None
    expected = expected_translation(logits, action_stats)
    return float(torch.dot(expected, desired / norm).item())


def collect(args):
    if not 0 < args.decisions <= 100 or args.decisions % 2 or args.num_envs != 2:
        raise ValueError("Protocol requires an even number of decisions up to 100 from two environments")
    output = args.output
    if output.exists():
        raise FileExistsError(output)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = make_args(args.checkpoint, args.seed, args.num_envs)
    policy = OpenVLAPolicy(config, 0)
    policy.prep_rollout()
    env = SimlerWrapper(config, policy.vla.get_action_stats(config.vla_unnorm_key))
    episode_ids = torch.arange(args.episode_offset, args.episode_offset + args.num_envs,
                               device=env.env.unwrapped.device)
    image, instructions, _ = env.reset(obj_set="test", episode_ids=episode_ids)
    env.episode_ids = episode_ids.cpu().tolist()
    output.mkdir(parents=True)
    images, records = [], []
    reset_count = 0
    for step in range(args.decisions // args.num_envs):
        with torch.no_grad():
            _, action, logprob = policy.get_action(
                {"image": image, "task_description": instructions}, deterministic=True)
        feedback = env.capture_feedback(raw_action=action, version="relspatial-action-v1")
        task = env.env.unwrapped
        contact = selected_contact(task)
        bbox = task.carrot_bbox_world.detach().cpu().tolist()
        image_cpu = image.cpu().numpy()
        next_image, _, done, _ = env.step(action)
        next_task = env.env.unwrapped
        next_contact = selected_contact(next_task)
        next_source = next_task.extra_stats["extra_pos_carrot"].detach().cpu().tolist()
        next_target = next_task.extra_stats["extra_pos_plate"].detach().cpu().tolist()
        next_gripper = next_task.extra_stats["extra_pos_gripper"].detach().cpu().tolist()
        for index in range(args.num_envs):
            record = feedback[index].to_dict()
            record.update(
                image_index=len(images), instruction=instructions[index],
                action_token_ids=action[index].cpu().tolist(),
                behavior_logprob=float(logprob[index].item()),
                contact_forces_n=contact[index], source_bbox_world_m=bbox[index],
                gripper_command=float(env._process_action(action)[index, 6].item()),
                next_source_grasped=bool(next_task.agent.is_grasping(
                    next_task.objs_carrot[next_task.carrot_names[int(next_task.select_carrot_ids[index])]])[index].item()),
                next_contact_forces_n=next_contact[index],
                next_source_position=next_source[index],
                next_target_position=next_target[index],
                next_gripper_position=next_gripper[index],
                episode_done=bool(done[index].item()),
            )
            records.append(record)
            images.append(image_cpu[index])
        image = next_image
        if done.any():
            reset_count += 1
            episode_ids = torch.arange(args.episode_offset + reset_count * args.num_envs,
                                       args.episode_offset + (reset_count + 1) * args.num_envs,
                                       device=env.env.unwrapped.device)
            image, instructions, _ = env.reset(obj_set="test", episode_ids=episode_ids)
            env.episode_ids = episode_ids.cpu().tolist()
        if step % 10 == 9:
            print(f"collected {len(records)}/{args.decisions}", flush=True)
    np.savez_compressed(output / "images.npz", images=np.stack(images))
    with (output / "records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    (output / "protocol.json").write_text(json.dumps({
        "model": str(args.checkpoint) if args.checkpoint else "warmup",
        "seed": args.seed, "episode_offset": args.episode_offset,
        "episode_resets": reset_count,
        "decisions": len(records), "env_steps": len(records), "obj_set": "test",
        "sampling_temperature": config.vla_temperature_eval,
        "model_parameters_updated": False,
    }, indent=2) + "\n")
    print(f"saved {len(records)} decisions to {output}")


def score(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    config = make_args(args.checkpoint, 0, 2)
    config.alg_name = "sdpo"
    policy = OpenVLAPolicy(config, 0)
    policy.prep_rollout()
    records = []
    images = []
    source_sizes = []
    for source in args.sources:
        with (source / "records.jsonl").open(encoding="utf-8") as handle:
            source_records = [json.loads(line) for line in handle]
        source_images = np.load(source / "images.npz")["images"]
        if len(source_records) != len(source_images):
            raise ValueError(f"Image/record count mismatch: {source}")
        source_sizes.append(len(source_records))
        records.extend(source_records)
        images.extend(source_images)
    if source_sizes[0] != source_sizes[1] or not 0 < source_sizes[0] <= 100 or source_sizes[0] % 2:
        raise ValueError(f"Expected two equal, even calibration sources of at most 100 decisions, got {source_sizes}")
    source_size = source_sizes[0]
    shuffled_feedback = [None] * len(records)
    generator = np.random.default_rng(0)
    for offset in (0, source_size):
        permutation = generator.permutation(source_size)
        for position, source_index in enumerate(permutation):
            replacement_index = permutation[(position + 1) % len(permutation)]
            shuffled_feedback[offset + source_index] = records[offset + replacement_index]["text"]
    rows = []
    action_stats = policy.vla.get_action_stats(config.vla_unnorm_key)
    for start in range(0, len(records), 2):
        batch = records[start:start + 2]
        observation = {"image": torch.from_numpy(np.stack(images[start:start + 2])).cuda(),
                       "task_description": [row["instruction"] for row in batch]}
        actions = torch.tensor([row["action_token_ids"] for row in batch], device="cuda")
        with torch.no_grad():
            temperature = config.vla_temperature_eval
            student = policy.evaluate_action_logits(observation, actions).float() / temperature
            for mode, feedback in (
                ("correct", [row["text"] for row in batch]),
                ("shuffled", shuffled_feedback[start:start + 2]),
                ("none", [None, None]),
            ):
                teacher = policy.evaluate_teacher_action_logits(observation, actions, feedback).float() / temperature
                student_log = student.log_softmax(dim=-1)
                teacher_log = teacher.log_softmax(dim=-1)
                reverse_kl = (student_log.exp() * (student_log - teacher_log)).sum(dim=-1).mean(dim=-1)
                for index, row in enumerate(batch):
                    student_direction = direction_score(student[index], row["desired_delta_root"], action_stats)
                    teacher_direction = direction_score(teacher[index], row["desired_delta_root"], action_stats)
                    rows.append({"image_index": start + index, "episode_id": row["episode_id"],
                                 "source": "warmup" if start + index < source_size else "v1-seed0-2400",
                                 "checkpoint": args.checkpoint_label,
                                 "feedback_mode": mode, "reverse_kl": float(reverse_kl[index].item()),
                                 "student_direction_projection_m": student_direction,
                                 "teacher_direction_projection_m": teacher_direction,
                                 "student_entropy": float(torch.distributions.Categorical(logits=student[index]).entropy().mean().item()),
                                 "teacher_entropy": float(torch.distributions.Categorical(logits=teacher[index]).entropy().mean().item())})
        if start % 20 == 0:
            print(f"scored {start + 2}/{len(records)} decisions", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"saved {len(rows)} scores to {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--checkpoint", type=Path)
    collect_parser.add_argument("--output", type=Path, required=True)
    collect_parser.add_argument("--seed", type=int, default=0)
    collect_parser.add_argument("--episode-offset", type=int, default=0)
    collect_parser.add_argument("--decisions", type=int, default=100)
    collect_parser.add_argument("--num-envs", type=int, default=2)
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--checkpoint", type=Path, required=True)
    score_parser.add_argument("--checkpoint-label", required=True)
    score_parser.add_argument("--sources", type=Path, nargs=2, required=True)
    score_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {"collect": collect, "score": score}[args.command](args)


if __name__ == "__main__":
    main()
