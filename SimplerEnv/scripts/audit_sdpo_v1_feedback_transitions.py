"""Audit archived v1 feedback against the next decision-time state."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def audit(paths, threshold):
    counts = defaultdict(lambda: [0, 0, 0.0, 0])
    for path in paths:
        episodes = defaultdict(list)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                episodes[record["episode_id"]].append(record)
        for records in episodes.values():
            records.sort(key=lambda record: record["decision_index"])
            for current, following in zip(records, records[1:]):
                if following["decision_index"] != current["decision_index"] + 1:
                    continue
                if current["source_grasped"] is None:
                    continue
                target_key = "target_position" if current["source_grasped"] else "source_position"
                cursor_key = "source_position" if current["source_grasped"] else "gripper_position"
                if any(record[key] is None for record in (current, following)
                       for key in (target_key, cursor_key)):
                    continue
                before = math.dist(current[target_key], current[cursor_key])
                after = math.dist(following[target_key], following[cursor_key])
                delta = before - after
                key = (current["action_judgment"], bool(current["source_grasped"]))
                counts[key][0] += 1
                counts[key][1] += delta > threshold
                counts[key][2] += delta
                if not current["source_grasped"]:
                    counts[key][3] += following["source_grasped"] is True
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("experiments/sdpo-v1/training_offline"))
    parser.add_argument("--threshold", type=float, default=0.001,
                        help="Minimum distance decrease in metres")
    args = parser.parse_args()
    for seed in (0, 1):
        paths = sorted(args.root.glob(
            f"v1-sdpo-correct-relspatial-action-v1-seed{seed}-2400steps/"
            "wandb/offline-run-*/glob/feedback/episode_*.jsonl"))
        if len(paths) != 15:
            raise ValueError(f"seed {seed}: expected 15 feedback archives, got {len(paths)}")
        print(f"seed {seed}")
        for (judgment, grasped), (count, improved, total_delta, grasp_events) in sorted(
                audit(paths, args.threshold).items()):
            print(f"  {judgment}, grasped={grasped}: {improved}/{count} "
                  f"improved ({improved / count:.3f}), mean delta {total_delta / count:.5f} m"
                  + (f", new grasps {grasp_events}/{count}" if not grasped else ""))


if __name__ == "__main__":
    main()
