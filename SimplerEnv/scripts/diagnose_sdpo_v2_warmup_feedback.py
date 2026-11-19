"""Compare controlled feedback on one frozen warm-up teacher and archived decisions."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from diagnose_sdpo_v1 import direction_score, make_args
from diagnose_sdpo_v1_counterfactual import close_probability
from diagnose_sdpo_v1_offline import decode_action, make_features, neutral_feedback, projection, prompt
from simpler_env.policies.openvla.openvla_train import OpenVLAPolicy


SELECTED = {"approach": (0, 1, 2, 3), "grasp": (16, 52, 61, 63), "transport": (18, 20, 54, 56)}


def feedback_pair(record, phase):
    desired = np.asarray(record["desired_delta_root"])
    axes = [index for index in sorted(range(3), key=lambda index: -abs(desired[index])) if abs(desired[index]) >= 0.01]
    assert axes

    def directions(reverse):
        return ", ".join(
            f"{'positive' if (desired[index] > 0) != reverse else 'negative'} {'xyz'[index]}"
            for index in axes
        )

    if phase == "approach":
        correct = (f"High-level plan: approach the {record['source_name']} before grasping or moving to the plate. "
                   f"Low-level guidance: move the end effector toward the {record['source_name']} along robot-root "
                   f"{directions(False)}; keep the gripper open until aligned within grasping reach. "
                   "Original-decision correction: approach source first; do not close prematurely.")
        wrong = (f"High-level plan: move toward the plate before grasping the {record['source_name']}. "
                 f"Low-level guidance: move the end effector away from the {record['source_name']} along robot-root "
                 f"{directions(True)}; close the gripper now. "
                 "Original-decision correction: move away from source first; close prematurely.")
    elif phase == "grasp":
        correct = (f"High-level plan: grasp the {record['source_name']} before transport. "
                   f"Low-level guidance: align with the {record['source_name']} along robot-root {directions(False)} "
                   "and close the gripper only when aligned within grasping reach. "
                   "Original-decision correction: grasp source before moving to plate.")
        wrong = (f"High-level plan: transport the {record['source_name']} without grasping. "
                 f"Low-level guidance: move away from the {record['source_name']} along robot-root {directions(True)} "
                 "and keep the gripper open even when aligned within grasping reach. "
                 "Original-decision correction: skip source grasp and move to plate.")
    else:
        correct = (f"High-level plan: keep holding the {record['source_name']} and transport it toward the plate. "
                   f"Low-level guidance: move toward the plate along robot-root {directions(False)}; "
                   "keep the gripper closed until aligned above the plate. "
                   "Original-decision correction: do not release before target alignment.")
        wrong = (f"High-level plan: stop transporting the {record['source_name']} and move away from the plate. "
                 f"Low-level guidance: move away from the plate along robot-root {directions(True)}; "
                 "open the gripper before reaching the plate. "
                 "Original-decision correction: release before target alignment.")
    return correct, wrong


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    records = [json.loads(line) for line in (args.source / "records.jsonl").read_text().splitlines()]
    with np.load(args.source / "images.npz") as archive:
        images = archive["images"]
    selected = [(phase, index) for phase, indices in SELECTED.items() for index in indices]
    if len(records) != len(images) or max(index for _, index in selected) >= len(records):
        raise ValueError("Archived records/images do not cover selected decisions")
    for phase, index in selected:
        record = records[index]
        if phase == "transport" and not record["source_grasped"]:
            raise ValueError(f"Decision {index} is not transport")
        if phase != "transport" and record["source_grasped"]:
            raise ValueError(f"Decision {index} unexpectedly holds source")
        if phase == "approach" and (np.linalg.norm(record["source_to_gripper"]) <= 0.08
                                     or max(record["contact_forces_n"]) >= 0.5):
            raise ValueError(f"Decision {index} is not a far, contact-free approach")
        if phase == "grasp" and np.linalg.norm(record["source_to_gripper"]) > 0.08:
            raise ValueError(f"Decision {index} is too far for grasp audit")

    config = make_args(None, 0, 2)
    config.alg_name = "sdpo"
    policy = OpenVLAPolicy(config, 0)
    policy.prep_rollout()
    action_stats = policy.vla.get_action_stats(config.vla_unnorm_key)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle, torch.inference_mode():
        for phase, index in selected:
            record = records[index]
            image = images[index]
            observation = {"image": torch.from_numpy(image[None]).cuda(),
                           "task_description": [record["instruction"]]}
            action = torch.tensor([record["action_token_ids"]], device="cuda")
            correct, wrong = feedback_pair(record, phase)
            neutral = neutral_feedback({**record, "text": correct}, policy.tokenizer)
            variants = {"correct": correct, "wrong": wrong, "neutral_length": neutral, "none": None}
            scores = {}
            logits = {}
            for mode, feedback in variants.items():
                conditional = policy.evaluate_teacher_action_logits(observation, action, [feedback]).float()
                logits[mode] = conditional[0].cpu()
                features = make_features(policy, [image], [prompt(record["instruction"], feedback)])
                _, greedy_ids, _ = policy.teacher_vla.predict_action_batch(
                    **features, unnorm_key=config.vla_unnorm_key, do_sample=False,
                )
                torch.manual_seed(2026 + index)
                _, sampled_ids, _ = policy.teacher_vla.predict_action_batch(
                    **features, unnorm_key=config.vla_unnorm_key, do_sample=True,
                    temperature=config.vla_temperature,
                )
                greedy = decode_action(greedy_ids[0].cpu().tolist(), action_stats)
                sampled = decode_action(sampled_ids[0].cpu().tolist(), action_stats)
                scores[mode] = {
                    "feedback": feedback,
                    "prompt_tokens": int(features["attention_mask"][0].sum()),
                    "conditional_projection_m": direction_score(
                        conditional[0] / config.vla_temperature_eval, record["desired_delta_root"], action_stats,
                    ),
                    "conditional_gripper_close_probability": close_probability(conditional[0]),
                    "greedy_action": greedy.tolist(),
                    "greedy_projection_m": projection(greedy, record["desired_delta_root"]),
                    "sampled_action": sampled.tolist(),
                    "sampled_projection_m": projection(sampled, record["desired_delta_root"]),
                    "greedy_action_token_ids": greedy_ids[0].cpu().tolist(),
                    "sampled_action_token_ids": sampled_ids[0].cpu().tolist(),
                }
            for mode in ("wrong", "neutral_length", "none"):
                correct_log = logits["correct"].log_softmax(-1)
                other_log = logits[mode].log_softmax(-1)
                scores[mode]["conditional_kl_correct_to_mode_by_token"] = (
                    correct_log.exp() * (correct_log - other_log)
                ).sum(-1).tolist()
            row = {"archive_index": index, "phase": phase, "instruction": record["instruction"],
                   "action_token_ids": record["action_token_ids"], "desired_delta_root": record["desired_delta_root"],
                   "source_grasped": record["source_grasped"], "contact_forces_n": record["contact_forces_n"],
                   "variants": scores}
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            print(f"scored {phase} decision {index}", flush=True)


if __name__ == "__main__":
    main()
