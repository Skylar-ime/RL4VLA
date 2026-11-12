#!/usr/bin/env python3
import argparse
from pathlib import Path

import yaml


ARMS = (
    ("ppo", "none", "relspatial-v2"),
    ("sdpo", "correct", "relspatial-v2"),
    ("sdpo", "correct", "relspatial-action-v1"),
    ("sdpo", "shuffled", "relspatial-action-v1"),
)
CHECKPOINT_EPISODES = (4, 9, 14)


def build_manifest(run_root: Path) -> list[str]:
    lines = ["v1-warmup\t-"]
    for seed in (0, 1):
        for algorithm, mode, version in ARMS:
            name = f"v1-{algorithm}-{mode}-{version}-seed{seed}-2400steps"
            run_dir = run_root / name / "wandb"
            configs = [
                path for path in run_dir.glob("*run-*/glob/config.yaml")
                if (path.parent / "steps_0014" / "adapter_model.safetensors").is_file()
            ]
            if len(configs) != 1:
                raise ValueError(f"{name}: expected one complete W&B run, found {len(configs)}")
            config = yaml.safe_load(configs[0].read_text())
            expected = {
                "name": name,
                "alg_name": algorithm,
                "sdpo_feedback_mode": mode,
                "sdpo_feedback_version": version,
                "seed": seed,
                "steps_max": 2400,
                "episode_len": 80,
                "num_envs": 2,
                "env_id": "PutOnPlateInScene25Main-v3",
            }
            mismatches = [key for key, value in expected.items() if config.get(key) != value]
            if mismatches:
                raise ValueError(f"{name}: mismatched config fields: {', '.join(mismatches)}")
            for episode in CHECKPOINT_EPISODES:
                checkpoint = configs[0].parent / f"steps_{episode:04d}"
                if not (checkpoint / "adapter_model.safetensors").is_file():
                    raise ValueError(f"{name}: missing adapter: {checkpoint}")
                if algorithm == "sdpo":
                    protocol = checkpoint / "sdpo_protocol.json"
                    if not protocol.is_file():
                        raise ValueError(f"{name}: missing SDPO protocol: {checkpoint}")
                steps = (episode + 1) * 160
                lines.append(f"{name}-steps{steps}\t{checkpoint.resolve()}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=Path("experiments/sdpo-v1"))
    args = parser.parse_args()
    print("\n".join(build_manifest(args.run_root)))


if __name__ == "__main__":
    main()
