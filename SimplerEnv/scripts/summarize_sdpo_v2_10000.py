"""Summarize paired 60-episode validation curves for the 10,000-step study."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CHECKPOINT_STEPS = (480, 2400, 5000, 10000)
EPISODE_IDS = tuple(range(60))


def read_episodes(eval_root: Path, label: str) -> dict[int, dict]:
    files = list((eval_root / label / "wandb").glob("offline-run-*/files/eval_only.jsonl"))
    if len(files) != 1:
        raise ValueError(f"Expected one offline evaluation for {label}, found {len(files)}")
    episodes = {}
    with files[0].open(encoding="utf-8") as handle:
        for line in handle:
            batch = json.loads(line)
            if batch["obj_set"] != "test":
                raise ValueError(f"Evaluation object set is not test: {label}")
            for episode in batch["episode_stats"]:
                episode_id = int(episode["episode_id"])
                if episode_id in episodes:
                    raise ValueError(f"Duplicate episode {episode_id} in {label}")
                episodes[episode_id] = episode
    if tuple(sorted(episodes)) != EPISODE_IDS:
        raise ValueError(f"Evaluation episode IDs are incomplete or changed: {label}")
    required = {
        "success", "is_src_obj_grasped", "consecutive_grasp", "first_grasp_step",
        "reached_target_after_grasp", "released_on_target", "carried_target_distance_reduction_m",
    }
    if any(not required <= episode.keys() for episode in episodes.values()):
        raise ValueError(f"Evaluation lacks phase metrics: {label}")
    return episodes


def summarize(arm: str, step: int, seconds: float, optimizer_steps: float,
              episodes: dict[int, dict], baseline: dict[int, dict]) -> dict:
    success = sum(episode["success"] for episode in episodes.values())
    grasped = sum(episode["is_src_obj_grasped"] for episode in episodes.values())
    sustained = sum(episode["consecutive_grasp"] for episode in episodes.values())
    reached = sum(episode["reached_target_after_grasp"] for episode in episodes.values())
    released = sum(episode["released_on_target"] for episode in episodes.values())
    grasp_steps = [episode["first_grasp_step"] for episode in episodes.values()
                   if episode["first_grasp_step"] is not None]
    distance_reductions = [episode["carried_target_distance_reduction_m"] for episode in episodes.values()
                           if episode["carried_target_distance_reduction_m"] is not None]
    gains = sum(episodes[episode_id]["success"] and not baseline[episode_id]["success"]
                for episode_id in EPISODE_IDS)
    losses = sum(not episodes[episode_id]["success"] and baseline[episode_id]["success"]
                 for episode_id in EPISODE_IDS)
    return {
        "arm": arm,
        "environment_steps": step,
        "train_cumulative_seconds": seconds,
        "optimizer_steps_cumulative": optimizer_steps,
        "episodes": len(episodes),
        "success_count": int(success),
        "success_rate": success / len(episodes),
        "ever_grasped_count": int(grasped),
        "sustained_grasp_count": int(sustained),
        "reached_target_after_grasp_count": int(reached),
        "released_on_target_count": int(released),
        "mean_first_grasp_step_among_grasped": sum(grasp_steps) / len(grasp_steps) if grasp_steps else None,
        "mean_carried_target_distance_reduction_m": (
            sum(distance_reductions) / len(distance_reductions) if distance_reductions else None
        ),
        "paired_success_gains": int(gains),
        "paired_success_losses": int(losses),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--ppo-run", type=Path, required=True)
    parser.add_argument("--sdpo-run", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    baseline = read_episodes(args.eval_root, "warmup-validation-60-h800")
    rows = [summarize("warmup", 0, 0.0, 0.0, baseline, baseline)]
    for arm, train_run in (("ppo", args.ppo_run), ("sdpo", args.sdpo_run)):
        if train_run is None:
            continue
        previous_seconds = -1.0
        for step in CHECKPOINT_STEPS:
            label = f"{arm}-step_{step:06d}-validation-60-h800"
            metadata_path = train_run / "glob" / f"step_{step:06d}" / "environment_steps.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata["environment_steps"] != step:
                raise ValueError(f"Checkpoint metadata step mismatch: {metadata_path}")
            seconds = float(metadata["train_cumulative_seconds"])
            if seconds <= previous_seconds:
                raise ValueError(f"Non-increasing training wall time: {metadata_path}")
            previous_seconds = seconds
            episodes = read_episodes(args.eval_root, label)
            rows.append(summarize(
                arm, step, seconds, float(metadata["optimizer_steps_cumulative"]), episodes, baseline,
            ))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "validation_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    baseline_rate = rows[0]["success_rate"] * 100
    for arm, color in (("ppo", "#3366cc"), ("sdpo", "#cc5533")):
        arm_rows = [row for row in rows if row["arm"] == arm]
        if not arm_rows:
            continue
        points = [rows[0], *arm_rows]
        for axis, x_key, scale in (
            (axes[0], "environment_steps", 1.0),
            (axes[1], "train_cumulative_seconds", 60.0),
        ):
            axis.plot([point[x_key] / scale for point in points],
                      [point["success_rate"] * 100 for point in points],
                      marker="o", label=arm.upper(), color=color)
    for axis, label in ((axes[0], "Environment action steps"), (axes[1], "Training wall time (min)")):
        axis.axhline(baseline_rate, color="gray", linestyle="--", linewidth=1)
        axis.set(xlabel=label, ylabel="Success rate (%)", ylim=(0, 100))
        axis.grid(alpha=0.2)
        axis.legend()
    figure.suptitle("Fixed test episodes 0–59; seed 0")
    figure.tight_layout()
    figure_path = args.output_dir / "validation_learning_curves.png"
    figure.savefig(figure_path, dpi=180)
    print(summary_path)
    print(figure_path)


if __name__ == "__main__":
    main()
