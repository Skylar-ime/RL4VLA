"""Score a frozen v2 teacher against archived feedback without environment interaction."""

import argparse
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from diagnose_sdpo_v1 import direction_score, make_args
from diagnose_sdpo_v1_offline import decode_action, make_features, neutral_feedback, projection, prompt, reverse_kl_by_dim
from simpler_env.external_feedback import world_to_root
from simpler_env.policies.openvla.openvla_train import OpenVLAPolicy


def read_archive(path):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if not records:
        raise ValueError("Feedback archive is empty")
    images = []
    with tarfile.open(path.with_name(path.stem + "_images.tar")) as archive:
        for record in records:
            if record.get("feedback_schema") != "robot-root-direction-v2":
                raise ValueError("Expected robot-root-direction-v2 feedback archive")
            member = archive.extractfile(record["before_image_member"])
            if member is None:
                raise ValueError("Archived before image is missing")
            images.append(np.asarray(Image.open(io.BytesIO(member.read())).convert("RGB")))
    return records, images


def instruction(record):
    source = record["before"]["source_name"]
    target = record["before"]["target_name"]
    if not source or not target:
        raise ValueError("Task instruction cannot be reconstructed from archived state")
    return f"put {source} on {target}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    records, images = read_archive(args.archive)
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
            instructions = [instruction(record) for record in batch]
            observation = {
                "image": torch.from_numpy(np.stack(batch_images)).cuda(),
                "task_description": instructions,
            }
            actions = torch.tensor([record["action_token_ids"] for record in batch], device="cuda")
            student_logits = policy.evaluate_action_logits(observation, actions)
            for mode in ("original", "neutral_length", "none"):
                feedback = [
                    record["text"] if mode == "original" else (
                        neutral_feedback({**record, "instruction": task}, policy.tokenizer)
                        if mode == "neutral_length" else None
                    )
                    for record, task in zip(batch, instructions)
                ]
                teacher_logits = policy.evaluate_teacher_action_logits(observation, actions, feedback)
                conditional_kl = reverse_kl_by_dim(student_logits, teacher_logits)
                features = make_features(
                    policy, batch_images,
                    [prompt(task, item) for task, item in zip(instructions, feedback)],
                )
                _, generated, _ = policy.teacher_vla.predict_action_batch(
                    **features, unnorm_key=config.vla_unnorm_key, do_sample=False,
                )
                for index, record in enumerate(batch):
                    desired = world_to_root(
                        record["before"]["source_to_gripper"], record["root_quaternion_wxyz"],
                    )
                    generated_ids = generated[index].cpu().tolist()
                    decoded = decode_action(generated_ids, action_stats)
                    rows.append({
                        "archive_row": start + index + 1,
                        "episode_id": record["episode_id"],
                        "rollout_step": record["rollout_step"],
                        "env_index": record["env_index"],
                        "source": record["before"]["source_name"],
                        "mode": mode,
                        "prompt_tokens": int(features["attention_mask"][index].sum().item()),
                        "reverse_kl_by_dim": conditional_kl[index].cpu().tolist(),
                        "conditional_direction_projection_m": direction_score(
                            teacher_logits[index].float() / config.vla_temperature_eval,
                            desired, action_stats,
                        ),
                        "free_action_token_ids": generated_ids,
                        "free_action": decoded.tolist(),
                        "free_direction_projection_m": projection(decoded, desired),
                    })
            print(f"scored {min(start + 2, len(records))}/{len(records)} decisions", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
