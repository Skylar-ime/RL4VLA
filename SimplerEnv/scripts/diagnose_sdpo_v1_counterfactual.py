"""Score frozen v1 EMA teachers under controlled feedback edits on archived observations."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from diagnose_sdpo_v1 import direction_score, make_args
from simpler_env.policies.openvla.openvla_train import OpenVLAPolicy


OPPOSITE = {
    "down": "up", "up": "down", "left": "right", "right": "left",
    "forward": "backward", "backward": "forward",
}
OPPOSITE_JUDGMENT = {"aligned": "opposed", "opposed": "aligned", "not aligned": "aligned"}


def feedback_variants(record):
    state = record["text"].split(" Suggested next action: ", 1)[0]
    suggestion = record["suggested_action"]
    words = suggestion.split()
    reverse_direction = None
    if len(words) > 1 and words[0] == "move" and words[1] in OPPOSITE:
        words[1] = OPPOSITE[words[1]]
        reverse_direction = " ".join(words)
    judgment = record["action_judgment"]
    reversed_judgment = OPPOSITE_JUDGMENT.get(judgment)
    return {
        "original": record["text"],
        "state_only": state,
        "reverse_direction": (
            f"{state} Suggested next action: {reverse_direction}. "
            f"Sampled action is {judgment} with this suggestion."
            if reverse_direction else None
        ),
        "reverse_judgment": (
            f"{state} Suggested next action: {suggestion}. "
            f"Sampled action is {reversed_judgment} with this suggestion."
            if reversed_judgment else None
        ),
        "no_feedback": None,
    }


def phase(record):
    if record["source_grasped"]:
        return "held"
    forces = record["contact_forces_n"]
    if forces is None:
        return "contact_unknown"
    contacts = sum(force >= 0.5 for force in forces)
    return ("approach", "one_finger_contact", "two_finger_contact")[contacts]


def close_probability(logits):
    bins = np.linspace(-1, 1, 256)
    centres = (bins[:-1] + bins[1:]) / 2
    decoded = centres[np.clip(np.arange(255, -1, -1), 0, 254)]
    mask = torch.as_tensor(decoded <= 0.5, device=logits.device)
    return logits[6].softmax(dim=-1)[mask].sum().item()


def teacher_kl(original, alternative):
    original_log = original.log_softmax(dim=-1)
    alternative_log = alternative.log_softmax(dim=-1)
    return (original_log.exp() * (original_log - alternative_log)).sum(dim=-1).mean(dim=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-label", required=True)
    parser.add_argument("--sources", type=Path, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    records, images = [], []
    for source in args.sources:
        source_records = [json.loads(line) for line in (source / "records.jsonl").read_text().splitlines()]
        with np.load(source / "images.npz") as archive:
            source_images = archive["images"]
        if len(source_records) != 80 or len(source_images) != 80:
            raise ValueError(f"Expected 80 paired decisions from {source}")
        records.extend(source_records)
        images.extend(source_images)

    config = make_args(args.checkpoint, 0, 2)
    config.alg_name = "sdpo"
    policy = OpenVLAPolicy(config, 0)
    policy.prep_rollout()
    action_stats = policy.vla.get_action_stats(config.vla_unnorm_key)
    rows = []
    with torch.inference_mode():
        for start in range(0, len(records), 2):
            batch = records[start:start + 2]
            observation = {
                "image": torch.from_numpy(np.stack(images[start:start + 2])).cuda(),
                "task_description": [record["instruction"] for record in batch],
            }
            actions = torch.tensor([record["action_token_ids"] for record in batch], device="cuda")
            variants = [feedback_variants(record) for record in batch]
            logits_by_mode = {}
            for mode in ("original", "state_only", "reverse_direction", "reverse_judgment", "no_feedback"):
                feedback = [variant[mode] for variant in variants]
                if any(value is None for value in feedback) and mode not in {"no_feedback"}:
                    if mode == "reverse_direction" and all(value is None for value in feedback):
                        continue
                    if mode == "reverse_judgment" and all(value is None for value in feedback):
                        continue
                    raise ValueError(f"Mixed applicability in batch {start}: {mode}")
                logits_by_mode[mode] = policy.evaluate_teacher_action_logits(
                    observation, actions, feedback).float() / config.vla_temperature_eval
            original = logits_by_mode["original"]
            for mode, logits in logits_by_mode.items():
                kl = teacher_kl(original, logits)
                for index, record in enumerate(batch):
                    rows.append({
                        "checkpoint": args.checkpoint_label,
                        "source": "warmup" if start < 80 else "v1-seed0-2400",
                        "image_index": start + index,
                        "episode_id": record["episode_id"],
                        "decision_index": record["decision_index"],
                        "phase": phase(record),
                        "mode": mode,
                        "teacher_kl_from_original": float(kl[index].item()),
                        "direction_projection_m": direction_score(
                            logits[index], record["desired_delta_root"], action_stats),
                        "close_probability": close_probability(logits[index]),
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
