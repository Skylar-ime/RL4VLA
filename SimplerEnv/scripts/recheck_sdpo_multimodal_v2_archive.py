"""Regenerate v2 feedback from archived decisions without simulator interaction."""

import argparse
import hashlib
import json
import os
import tarfile
from pathlib import Path

from simpler_env.external_feedback import ExternalFeedbackClient
from simpler_env.feedback import ActionOutcomeRecord, RelSpatialRecord


def archived_outcome(row):
    return ActionOutcomeRecord(
        before=RelSpatialRecord(**row["before"]),
        after=RelSpatialRecord(**row["after"]),
        env_index=row["env_index"],
        action_token_ids=tuple(row["action_token_ids"]),
        episode_done=row["episode_done"],
        rollout_interrupted=row["rollout_interrupted"],
        task_success=row["task_success"],
        distance_change_m=row["distance_change_m"],
        source_movement_m=row["source_movement_m"],
        grasp_transition=row["grasp_transition"],
        text="",
        contact_before_n=row["contact_before_n"],
        contact_after_n=row["contact_after_n"],
        root_quaternion_wxyz=row["root_quaternion_wxyz"],
    )


def read_decisions(path, selected_rows):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if not rows or any(number < 1 or number > len(rows) for number in selected_rows):
        raise ValueError("Requested rows must exist in a nonempty feedback archive")
    decisions = []
    with tarfile.open(path.with_name(path.stem + "_images.tar")) as images:
        for number in selected_rows:
            row = rows[number - 1]
            if row.get("feedback_schema") != "robot-root-direction-v2":
                raise ValueError("Expected archived robot-root-direction-v2 feedback")
            if row.get("root_quaternion_wxyz") is None:
                raise ValueError("Archived decision lacks robot-root orientation")
            image_bytes = []
            for phase in ("before", "after"):
                member = images.extractfile(row[f"{phase}_image_member"])
                if member is None:
                    raise ValueError(f"Archived {phase} image is missing")
                content = member.read()
                if hashlib.sha256(content).hexdigest() != row[f"{phase}_image_sha256"]:
                    raise ValueError(f"Archived {phase} image hash mismatch")
                image_bytes.append(content)
            decisions.append((number, row, archived_outcome(row), *image_bytes))
    return decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--endpoint")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if len(set(args.rows)) != len(args.rows):
        raise ValueError("Requested archive rows must be unique")
    if not args.dry_run and (not args.endpoint or args.output is None):
        parser.error("--endpoint and --output are required unless --dry-run is set")
    if args.output is not None and args.output.exists():
        raise FileExistsError(args.output)

    decisions = read_decisions(args.archive, args.rows)
    print(f"verified {len(decisions)} archived decisions and both JPEG hashes per decision", flush=True)
    if args.dry_run:
        return

    client = ExternalFeedbackClient(
        base_url=args.endpoint, model=args.model,
        api_key=os.environ.get("EXTERNAL_FEEDBACK_API_KEY", ""), timeout_seconds=args.timeout,
    )
    results = []
    for number, row, outcome, before_jpeg, after_jpeg in decisions:
        source = outcome.before.source_name
        target = outcome.before.target_name
        if not source or not target:
            raise ValueError("Cannot reconstruct instruction from archived source and target")
        record = client.generate(
            instruction=f"put {source} on {target}",
            before_image=before_jpeg,
            after_image=after_jpeg,
            outcome=outcome,
            executed_action=row["executed_action"],
            rollout_step=row["rollout_step"],
        )
        results.append({
            "archive_row": number,
            "archive_episode_id": row["episode_id"],
            "archive_feedback_schema": row["feedback_schema"],
            "original_model_correction": row["model_correction"],
            "new_feedback": record.to_dict(),
        })
        print(f"regenerated archive row {number}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"saved {len(results)} regenerated decisions to {args.output}", flush=True)


if __name__ == "__main__":
    main()
