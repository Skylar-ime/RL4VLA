#!/usr/bin/env python3
import argparse
import ast
import re
import sys
from collections import defaultdict
from pathlib import Path


EPISODE_PATTERN = re.compile(r"^(v1-.+-seed[01]-2400steps): ep (\d{4}) \| steps (\d+) \| e ")


def load_training_metrics(path: Path) -> dict[str, list[dict[str, float]]]:
    lines = path.read_text(errors="replace").splitlines()
    runs = defaultdict(list)
    index = 0
    while index < len(lines):
        match = EPISODE_PATTERN.match(lines[index])
        if match is None:
            index += 1
            continue
        name, episode_text, steps_text = match.groups()
        episode = int(episode_text)
        if int(steps_text) != (episode + 1) * 160:
            raise ValueError(f"{name}: unexpected step count at episode {episode}")
        start = index + 1
        index = start
        while index < len(lines) and not lines[index].endswith("}"):
            index += 1
        if index == len(lines):
            raise ValueError(f"{name}: incomplete metrics at episode {episode}")
        metrics = ast.literal_eval("\n".join(lines[start:index + 1]))
        metrics["episode_index"] = episode
        runs[name].append(metrics)
        index += 1
    return dict(runs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, default=Path("experiments/sdpo-v1/rlvla-v1-16633.out"))
    args = parser.parse_args()
    runs = load_training_metrics(args.log)
    if len(runs) != 8:
        raise ValueError(f"Expected eight training runs, found {len(runs)}")

    print("name\tepisodes\ttrain_s\trollout_s\tupdate_s\tpeak_allocated_gib\tmean_entropy\tmean_loss")
    for name, episodes in sorted(runs.items()):
        if [episode["episode_index"] for episode in episodes] != list(range(15)):
            raise ValueError(f"{name}: expected episodes 0..14")
        expected_feedback = 0.0 if "-ppo-" in name else 1.0
        if any(episode["feedback/effective_ratio"] != expected_feedback for episode in episodes):
            raise ValueError(f"{name}: unexpected feedback availability")
        for episode in episodes:
            gap = (episode["time/episode_seconds"] - episode["time/rollout_seconds"]
                   - episode["time/update_seconds"])
            if gap < -1:
                print(f"Warning: inconsistent timing in {name} episode {episode['episode_index']}: "
                      f"episode time is {-gap:.1f}s shorter than rollout plus update", file=sys.stderr)

        def total(key: str) -> float:
            return sum(float(episode[key]) for episode in episodes)

        print(
            f"{name}\t15\t{total('time/episode_seconds'):.1f}"
            f"\t{total('time/rollout_seconds'):.1f}"
            f"\t{total('time/update_seconds'):.1f}"
            f"\t{max(episode['gpu/cuda_0_peak_allocated_gib'] for episode in episodes):.2f}"
            f"\t{total('train/policy_entropy') / 15:.3f}"
            f"\t{total('train/loss') / 15:.3f}"
        )


if __name__ == "__main__":
    main()
