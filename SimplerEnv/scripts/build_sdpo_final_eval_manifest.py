#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import yaml


ARM_NAMES = ("ppo-none", "sdpo-correct", "sdpo-shuffled", "sdpo-none")
EXPECTED_NAMES = {
    f"{arm}-seed{seed}-480steps" for arm in ARM_NAMES for seed in range(3)
}
NAME_PATTERN = re.compile(r"^(ppo-none|sdpo-(?:correct|shuffled|none))-seed([012])-480steps$")


def build_manifest(wandb_dir: Path) -> list[str]:
    checkpoints = {}
    for config_path in sorted(wandb_dir.glob("run-*/glob/config.yaml")):
        config = yaml.safe_load(config_path.read_text())
        name = config.get("name", "")
        match = NAME_PATTERN.fullmatch(name)
        if match is None:
            continue
        arm, seed_text = match.groups()
        seed = int(seed_text)
        expected_alg = "ppo" if arm == "ppo-none" else "sdpo"
        expected_feedback = "none" if arm == "ppo-none" else arm.removeprefix("sdpo-")
        expected = {
            "alg_name": expected_alg,
            "seed": seed,
            "steps_max": 480,
            "episode_len": 80,
            "num_envs": 2,
            "env_id": "PutOnPlateInScene25Main-v3",
        }
        if expected_alg == "sdpo":
            expected["sdpo_feedback_mode"] = expected_feedback
        mismatches = [key for key, value in expected.items() if config.get(key) != value]
        if mismatches:
            raise ValueError(f"{name}: mismatched config fields: {', '.join(mismatches)}")
        checkpoint = config_path.parent / "steps_0002"
        if not (checkpoint / "adapter_model.safetensors").is_file():
            raise ValueError(f"{name}: incomplete checkpoint: {checkpoint}")
        if expected_alg == "sdpo" and not (checkpoint / "sdpo_protocol.json").is_file():
            raise ValueError(f"{name}: missing SDPO protocol: {checkpoint}")
        if name in checkpoints:
            raise ValueError(f"duplicate run name: {name}")
        checkpoints[name] = checkpoint.resolve()

    missing = sorted(EXPECTED_NAMES - checkpoints.keys())
    if missing:
        raise ValueError(f"missing runs: {', '.join(missing)}")
    lines = ["warmup\t-"]
    for arm in ARM_NAMES:
        for seed in range(3):
            name = f"{arm}-seed{seed}-480steps"
            lines.append(f"{name}\t{checkpoints[name]}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-dir", type=Path, default=Path("wandb"))
    args = parser.parse_args()
    print("\n".join(build_manifest(args.wandb_dir)))


if __name__ == "__main__":
    main()
