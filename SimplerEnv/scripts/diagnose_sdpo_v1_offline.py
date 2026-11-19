"""Audit frozen v1 teachers on archived images without simulator interaction."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from diagnose_sdpo_v1 import direction_score, make_args
from diagnose_sdpo_v1_counterfactual import phase, root_direction_suggestion
from simpler_env.policies.openvla.openvla_train import OpenVLAPolicy


MODES = (
    "original", "state_only", "state_direction", "state_judgment",
    "direction_only", "judgment_only", "root_direction_only",
    "neutral_length", "none",
)


def prompt(instruction, feedback):
    section = f"Privileged state feedback for the teacher: {feedback}\n" if feedback else ""
    return f"In: What action should the robot take to {instruction.lower()}?\n{section}Out: "


def neutral_feedback(record, tokenizer):
    target_length = len(tokenizer(prompt(record["instruction"], record["text"]))["input_ids"])
    neutral = "Additional note: no task-specific information."
    base_length = len(tokenizer(prompt(record["instruction"], neutral))["input_ids"])
    if target_length < base_length:
        raise ValueError("Neutral feedback is longer than the original prompt")
    neutral += " unknown" * (target_length - base_length)
    if len(tokenizer(prompt(record["instruction"], neutral))["input_ids"]) != target_length:
        raise ValueError("Could not match the original prompt length")
    return neutral


def feedback_variants(record, tokenizer):
    state = record["text"].split(" Suggested next action: ", 1)[0]
    suggestion = f"Suggested next action: {record['suggested_action']}."
    judgment = f"Sampled action is {record['action_judgment']} with the intended target direction."
    root_suggestion = root_direction_suggestion(record)
    return {
        "original": record["text"],
        "state_only": state,
        "state_direction": f"{state} {suggestion}",
        "state_judgment": f"{state} {judgment}",
        "direction_only": suggestion,
        "judgment_only": judgment,
        "root_direction_only": (
            f"Suggested next action: {root_suggestion}." if root_suggestion else None
        ),
        "neutral_length": neutral_feedback(record, tokenizer),
        "none": None,
    }


def decode_action(token_ids, action_stats):
    bins = np.linspace(-1, 1, 256)
    centers = (bins[:-1] + bins[1:]) / 2
    indices = np.clip(32000 - np.asarray(token_ids) - 1, 0, 254)
    normalized = centers[indices]
    low = np.asarray(action_stats["q01"])
    high = np.asarray(action_stats["q99"])
    mask = np.asarray(action_stats.get("mask", [True] * len(normalized)))
    return np.where(mask, 0.5 * (normalized + 1) * (high - low) + low, normalized)


def projection(action, desired):
    if desired is None:
        return None
    desired = np.asarray(desired)
    norm = np.linalg.norm(desired)
    return None if norm < 0.03 else float(np.dot(action[:3], desired / norm))


def reverse_kl_by_dim(student_logits, teacher_logits):
    student_log = student_logits.float().log_softmax(dim=-1)
    teacher_log = teacher_logits.float().log_softmax(dim=-1)
    return (student_log.exp() * (student_log - teacher_log)).sum(dim=-1)


def make_features(policy, images, prompts):
    images = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).to(**policy.tpdv)
    return policy.processor(prompts, images, padding=True).to(**policy.tpdv)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-label", required=True)
    parser.add_argument("--sources", type=Path, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-decisions", type=int, default=160)
    parser.add_argument("--pair-stride", type=int, choices=(1, 2, 4), default=1)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 0 < args.max_decisions <= 160 or args.max_decisions % 2:
        raise ValueError("max-decisions must be an even count from 2 to 160")

    records, images = [], []
    for source in args.sources:
        source_records = [json.loads(line) for line in (source / "records.jsonl").read_text().splitlines()]
        with np.load(source / "images.npz") as archive:
            source_images = archive["images"]
        if len(source_records) != 80 or len(source_images) != 80:
            raise ValueError(f"Expected 80 paired archived decisions from {source}")
        records.extend(source_records)
        images.extend(source_images)
    selected = [decision_index for offset in (0, 80)
                for pair_start in range(offset, offset + 80, 2 * args.pair_stride)
                for decision_index in (pair_start, pair_start + 1)]
    selected = selected[:args.max_decisions]
    records = [records[index] for index in selected]
    images = [images[index] for index in selected]
    source_split = 80 // args.pair_stride

    config = make_args(args.checkpoint, 0, 2)
    config.alg_name = "sdpo"
    policy = OpenVLAPolicy(config, 0)
    policy.prep_rollout()
    action_stats = policy.vla.get_action_stats(config.vla_unnorm_key)
    rows = []
    with torch.inference_mode():
        for start in range(0, len(records), 2):
            batch = records[start:start + 2]
            batch_images = images[start:start + 2]
            observation = {
                "image": torch.from_numpy(np.stack(batch_images)).cuda(),
                "task_description": [record["instruction"] for record in batch],
            }
            actions = torch.tensor([record["action_token_ids"] for record in batch], device="cuda")
            student_logits = policy.evaluate_action_logits(observation, actions)
            variants = [feedback_variants(record, policy.tokenizer) for record in batch]
            for mode in MODES:
                if any(variant[mode] is None for variant in variants) and mode != "none":
                    raise ValueError(f"Missing {mode} feedback in batch {start}")
                feedback = [variant[mode] for variant in variants]
                teacher_logits = policy.evaluate_teacher_action_logits(observation, actions, feedback)
                conditional_kl = reverse_kl_by_dim(student_logits, teacher_logits)
                features = make_features(
                    policy, batch_images,
                    [prompt(record["instruction"], item) for record, item in zip(batch, feedback)],
                )
                _, generated, _ = policy.teacher_vla.predict_action_batch(
                    **features, unnorm_key=config.vla_unnorm_key, do_sample=False,
                )
                for index, record in enumerate(batch):
                    generated_ids = generated[index].cpu().tolist()
                    decoded = decode_action(generated_ids, action_stats)
                    rows.append({
                        "checkpoint": args.checkpoint_label,
                        "source": "warmup" if start < source_split else "v1-seed0-2400",
                        "image_index": selected[start + index],
                        "episode_id": record["episode_id"],
                        "decision_index": record["decision_index"],
                        "phase": phase(record),
                        "mode": mode,
                        "prompt_tokens": int(features["attention_mask"][index].sum().item()),
                        "reverse_kl_by_dim": conditional_kl[index].cpu().tolist(),
                        "conditional_direction_projection_m": direction_score(
                            teacher_logits[index].float() / config.vla_temperature_eval,
                            record["desired_delta_root"], action_stats,
                        ),
                        "free_action_token_ids": generated_ids,
                        "free_action": decoded.tolist(),
                        "free_direction_projection_m": projection(decoded, record["desired_delta_root"]),
                    })
            if start % 20 == 0:
                print(f"scored {start + 2}/{len(records)} decisions", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
