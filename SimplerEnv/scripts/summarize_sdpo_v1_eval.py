#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import yaml


ARMS = (
    ("ppo", "none", "relspatial-v2"),
    ("sdpo", "correct", "relspatial-v2"),
    ("sdpo", "correct", "relspatial-action-v1"),
    ("sdpo", "shuffled", "relspatial-action-v1"),
)


def expected_labels() -> list[str]:
    labels = ["v1-warmup"]
    for seed in (0, 1):
        for algorithm, mode, version in ARMS:
            name = f"v1-{algorithm}-{mode}-{version}-seed{seed}-2400steps"
            labels.extend(f"{name}-steps{steps}" for steps in (800, 1600, 2400))
    return labels


def summarize_run(run_dir: Path, label: str) -> tuple[int, int] | None:
    candidates = list(run_dir.glob("wandb/*run-*/files/eval_only.jsonl"))
    complete = []
    for path in candidates:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        if len(rows) == 5:
            complete.append((path, rows))
    if not complete:
        return None
    if len(complete) != 1:
        raise ValueError(f"{label}: multiple complete evaluation runs")

    path, rows = complete[0]
    config = yaml.safe_load((path.parent.parent / "glob/config.yaml").read_text())
    expected = {
        "name": "v1-warmup-test30" if label == "v1-warmup" else label,
        "eval_only": True,
        "eval_episodes": 30,
        "eval_obj_set": "test",
        "num_envs": 6,
        "episode_len": 80,
        "seed": 0,
        "env_id": "PutOnPlateInScene25Main-v3",
        "vla_temperature_eval": 0.6,
    }
    mismatches = [key for key, value in expected.items() if config.get(key) != value]
    if mismatches:
        raise ValueError(f"{label}: mismatched config fields: {', '.join(mismatches)}")

    episodes = []
    for batch_index, row in enumerate(rows):
        batch_ids = list(range(batch_index * 6, (batch_index + 1) * 6))
        if row.get("obj_set") != "test" or row.get("episode_ids") != batch_ids:
            raise ValueError(f"{label}: unexpected test episode IDs")
        batch_episodes = row.get("episode_stats", [])
        if [episode.get("episode_id") for episode in batch_episodes] != batch_ids:
            raise ValueError(f"{label}: episode statistics misaligned with IDs")
        episodes.extend(batch_episodes)
    return (
        sum(int(episode["success"]) for episode in episodes),
        sum(int(episode["is_src_obj_grasped"]) for episode in episodes),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, default=Path("experiments/sdpo-v1/evaluation"))
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    missing = []
    print("label\tsuccess\tgrasp\tepisodes")
    for label in expected_labels():
        result = summarize_run(args.eval_root / label, label)
        if result is None:
            missing.append(label)
            continue
        success, grasp = result
        print(f"{label}\t{success}\t{grasp}\t30")
    if args.strict and missing:
        raise ValueError(f"Missing {len(missing)} complete evaluations: {', '.join(missing)}")


if __name__ == "__main__":
    main()
