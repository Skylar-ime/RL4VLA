"""Retrospective multimodal feedback for an executed student action."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import urllib.request
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from simpler_env.feedback import ActionOutcomeRecord


FIELDS = (
    "observed_outcome",
    "phase",
    "next_subtask",
    "redo_or_reorder",
    "target_adjustment",
    "ee_direction_root",
    "trajectory_correction",
    "gripper_operation",
    "retrospective_correction",
    "retrospective_target_adjustment",
    "retrospective_direction_root",
    "retrospective_gripper_operation",
    "retrospective_order_adjustment",
    "uncertainty",
)
HIGH_LEVEL_FIELDS = ("phase", "next_subtask", "redo_or_reorder")
LOW_LEVEL_FIELDS = ("target_adjustment", "ee_direction_root", "trajectory_correction", "gripper_operation")
RETROSPECTIVE_FIELDS = (
    "retrospective_correction", "retrospective_target_adjustment", "retrospective_direction_root",
    "retrospective_gripper_operation", "retrospective_order_adjustment",
)
RETROSPECTIVE_DETAIL_FIELDS = RETROSPECTIVE_FIELDS[1:]
PHASES = {
    "approach_source", "grasp_source", "transport_to_target", "release_on_target", "verify_completion",
}
NO_CONTACT_APPROACH_DISTANCE_M = 0.08


@dataclass(frozen=True)
class MultimodalCorrectionRecord:
    outcome: ActionOutcomeRecord
    model: str
    rollout_step: int
    executed_action: tuple[float, ...]
    correction: dict[str, str]
    text: str
    before_jpeg: bytes
    after_jpeg: bytes
    generation_attempts: int = 1
    rejection_reasons: tuple[str, ...] = ()
    model_correction: dict[str, str] | None = None
    grounding_adjustments: tuple[str, ...] = ()

    @property
    def timestamp_utc(self) -> str:
        return self.outcome.timestamp_utc

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.outcome.to_dict(),
            "feedback_version": "multimodal-correction-v2",
            "feedback_schema": "robot-root-direction-v2",
            "correction_schema": "high-low-retrospective-v2",
            "ee_direction_frame": "robot_root_at_decision",
            "external_model": self.model,
            "rollout_step": self.rollout_step,
            "executed_action": self.executed_action,
            "correction": self.correction,
            "text": self.text,
            "before_image_member": self.image_member("before"),
            "after_image_member": self.image_member("after"),
            "before_image_sha256": hashlib.sha256(self.before_jpeg).hexdigest(),
            "after_image_sha256": hashlib.sha256(self.after_jpeg).hexdigest(),
            "generation_attempts": self.generation_attempts,
            "rejection_reasons": self.rejection_reasons,
            "model_correction": self.model_correction,
            "grounding_adjustments": self.grounding_adjustments,
        }

    def image_member(self, phase: str) -> str:
        return f"decision_{self.rollout_step:04d}_env_{self.outcome.env_index:02d}_{phase}.jpg"


def _jpeg_bytes(image: Any) -> bytes:
    if isinstance(image, bytes):
        with Image.open(io.BytesIO(image)) as archived:
            if archived.format != "JPEG" or archived.mode != "RGB":
                raise ValueError("External feedback requires an RGB JPEG image")
            archived.verify()
        return image
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
        raise ValueError("External feedback requires an RGB uint8 image")
    output = io.BytesIO()
    Image.fromarray(array).save(output, format="JPEG", quality=85)
    return output.getvalue()


def parse_correction(content: str) -> dict[str, str]:
    try:
        result = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("External model returned invalid JSON") from error
    if not isinstance(result, dict) or any(
        not isinstance(result.get(field), str) or not result[field].strip() for field in FIELDS
    ):
        raise ValueError("External model omitted a required correction field")
    if (result["next_subtask"].strip().lower() == "unknown"
            or result["retrospective_correction"].strip().lower() == "unknown"
            or all(result[field].strip().lower() == "unknown" for field in (
                "target_adjustment", "ee_direction_root", "trajectory_correction", "gripper_operation",
            )) or all(result[field].strip().lower() == "unknown" for field in RETROSPECTIVE_DETAIL_FIELDS)):
        raise ValueError("External model returned no actionable correction")
    if result["phase"] not in PHASES:
        raise ValueError("External model returned an invalid task phase")
    for field in ("ee_direction_root", "retrospective_direction_root"):
        direction = result[field].strip().lower()
        compact_axes = [part.strip() for part in direction.split(",")]
        if all(re.fullmatch(r"[+-]\s*[xyz]", part) for part in compact_axes):
            result[field + "_raw"] = result[field]
            result[field] = ", ".join(
                f"{'positive' if part[0] == '+' else 'negative'} {part[-1]}" for part in compact_axes
            )
            direction = result[field].lower()
        if direction != "unknown" and not re.search(r"\b(positive|negative)\s+[xyz]\b", direction):
            parts = [part.strip() for part in direction.strip("[]() ").split(",")]
            if len(parts) == 3:
                try:
                    vector = [float(part) for part in parts]
                except ValueError:
                    vector = []
                if len(vector) == 3 and np.isfinite(vector).all():
                    normalized = ", ".join(
                        f"{'positive' if vector[index] > 0 else 'negative'} {'xyz'[index]}"
                        for index in sorted(range(3), key=lambda axis: abs(vector[axis]), reverse=True)
                        if abs(vector[index]) > 0.01
                    )
                    if normalized:
                        result[field + "_raw"] = result[field]
                        result[field] = normalized
                        direction = normalized
        if direction != "unknown" and not re.search(r"\b(positive|negative)\s+[xyz]\b", direction):
            raise ValueError(f"External model direction must name signed robot-root axes: {direction[:120]!r}")
        signed_axes = re.findall(r"\b(positive|negative)\s+([xyz])\b", direction)
        if any({sign for sign, named_axis in signed_axes if named_axis == axis} == {"positive", "negative"}
               for axis in "xyz"):
            raise ValueError(f"External model direction contradicts itself: {direction[:120]!r}")
    correction = {field: result[field].strip() for field in FIELDS}
    for field in ("ee_direction_root_raw", "retrospective_direction_root_raw"):
        if field in result:
            correction[field] = result[field]
    return correction


def world_to_root(vector: Any, quaternion_wxyz: Any) -> tuple[float, float, float]:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(-1)
    if quaternion.size != 4 or not np.isfinite(quaternion).all() or np.linalg.norm(quaternion) == 0:
        raise ValueError("Multimodal correction requires a valid robot-root quaternion")
    quaternion /= np.linalg.norm(quaternion)
    scalar, axis = quaternion[0], quaternion[1:]
    world = np.asarray(vector, dtype=np.float64).reshape(3)
    root = world + 2 * np.cross(-axis, np.cross(-axis, world) + scalar * world)
    return tuple(float(value) for value in root)


def validate_stage(correction: dict[str, str], outcome: ActionOutcomeRecord) -> None:
    phase = correction["phase"]
    if outcome.task_success is True:
        allowed = {"verify_completion"}
    elif outcome.after.source_grasped is False:
        allowed = {"approach_source", "grasp_source"}
    elif outcome.after.source_grasped is True:
        allowed = {"transport_to_target", "release_on_target"}
    else:
        return
    if phase not in allowed:
        raise ValueError("External model phase contradicts observed grasp or success state")
    if phase in {"approach_source", "grasp_source"} and outcome.after.target_name:
        if outcome.after.target_name.lower() in correction["next_subtask"].lower():
            raise ValueError("External model next subtask targets the destination before grasp")
    if outcome.after.source_grasped is False and outcome.after.source_to_gripper is not None:
        displacement = world_to_root(outcome.after.source_to_gripper, outcome.root_quaternion_wxyz)
        distance = float(np.linalg.norm(displacement))
        signed_axes = re.findall(r"\b(positive|negative)\s+([xyz])\b", correction["ee_direction_root"].lower())
        if distance > 0.08 and signed_axes and not any(
            displacement["xyz".index(axis)] * (1 if sign == "positive" else -1) > 0.01
            for sign, axis in signed_axes
        ):
            raise ValueError(
                f"External model direction does not approach the ungrasped source: "
                f"{correction['ee_direction_root'][:120]!r}, root-frame source offset {displacement}"
            )
        contact = outcome.contact_after_n
        no_contact = contact is None or max(contact) < 0.5
        if distance > NO_CONTACT_APPROACH_DISTANCE_M and no_contact and phase != "approach_source":
            raise ValueError("External model requests grasp phase while far from source without contact")
        if distance > NO_CONTACT_APPROACH_DISTANCE_M and no_contact and any(immediate_closure(correction[field]) for field in (
                "next_subtask", "target_adjustment", "trajectory_correction",
                "gripper_operation", "retrospective_correction", "retrospective_gripper_operation")):
            raise ValueError("External model requests grasp closure while far from source without contact")
        if outcome.before.source_grasped is False and outcome.before.source_to_gripper is not None:
            original_offset = world_to_root(outcome.before.source_to_gripper, outcome.root_quaternion_wxyz)
            signed_axes = re.findall(
                r"\b(positive|negative)\s+([xyz])\b", correction["retrospective_direction_root"].lower(),
            )
            if np.linalg.norm(original_offset) > NO_CONTACT_APPROACH_DISTANCE_M and signed_axes and not any(
                original_offset["xyz".index(axis)] * (1 if sign == "positive" else -1) > 0.01
                for sign, axis in signed_axes
            ):
                raise ValueError("External model retrospective direction does not approach the original source")


def immediate_closure(operation: str) -> bool:
    operation = operation.lower()
    if re.search(
        r"do not close|avoid closing|without closing|not closing|never close|"
        r"keep (?:the |it |gripper )?open|remain open|"
        r"close only (?:after|when|if)|only close (?:after|when|if)|do not need to close", operation,
    ):
        return False
    if re.fullmatch(r"close(?:\s+(?:now|immediately))?[.!]?", operation.strip()):
        return True
    return bool(re.search(
        r"\bclos(?:e|ing)\s+(?:(?:the|your|its)\s+)?(?:gripper|fingers|jaws|it)\b|"
        r"\b(?:gripper|fingers|jaws)\b.{0,40}\bclosed\b|"
        r"\bclosed\s+(?:(?:the|your|its)\s+)?(?:gripper|fingers|jaws)\b|"
        r"\b(?:gripper|fingers|jaws)\b.{0,40}\b(?:did not|failed to)\s+close\b|"
        r"\bnot yet closed\b|"
        r"\b[+-]?1\s+close\b",
        operation,
    ))


def validate_gripper_attribution(correction: dict[str, str], executed_action: Any) -> None:
    if executed_action[6] <= 0:
        return
    for field in ("observed_outcome", "retrospective_correction", "retrospective_gripper_operation"):
        if re.search(
            r"\b(?:gripper|fingers|jaws)\b.{0,40}\b(?:did not|failed to|could not)\s+close\b|"
            r"\b(?:failed|unsuccessful|improper)\s+(?:gripper\s+)?clos(?:e|ure|ing)\b",
            correction[field].lower(),
        ):
            raise ValueError("External model attributes failed closure to an executed OPEN command")


def validate_retrospective_coverage(correction: dict[str, str], outcome: ActionOutcomeRecord) -> None:
    observed_approach = (
        outcome.before.source_grasped is False and outcome.after.source_grasped is False
        and outcome.before.source_to_gripper is not None and outcome.root_quaternion_wxyz is not None
    )
    if observed_approach:
        missing = [
            field for field in RETROSPECTIVE_DETAIL_FIELDS
            if correction[field].strip().lower() == "unknown"
        ]
        if missing:
            raise ValueError("External model omitted observed-approach retrospective detail: " + ", ".join(missing))


def ground_obvious_contact_errors(correction: dict[str, str], outcome: ActionOutcomeRecord):
    grounded = correction.copy()
    adjustments = []
    if outcome.after.source_grasped is False and outcome.after.source_to_gripper is not None:
        distance = float(np.linalg.norm(outcome.after.source_to_gripper))
        contact = outcome.contact_after_n
        if distance > NO_CONTACT_APPROACH_DISTANCE_M and (contact is None or max(contact) < 0.5):
            source_name = outcome.after.source_name or "source object"
            if grounded["phase"] == "grasp_source":
                grounded["phase"] = "approach_source"
                adjustments.append("phase: grasp_source replaced by approach_source; source far without contact")
            replacements = {
                "next_subtask": f"Approach the {source_name} with the gripper open; align within grasping reach before closing.",
                "target_adjustment": f"Target the {source_name}, not the destination, until grasp is established.",
                "trajectory_correction": f"Align the end effector toward the {source_name}; keep gripper open until within grasping reach.",
                "gripper_operation": "Keep gripper open while approaching source; close only when aligned within grasping reach.",
                "retrospective_correction": (
                    f"At the original decision, approach the {source_name} first and keep gripper open until within grasping reach."
                ),
                "retrospective_gripper_operation": "Keep gripper open at the original decision until aligned within grasping reach.",
            }
            for field, replacement in replacements.items():
                destination_first = (
                    field == "next_subtask" and outcome.after.target_name
                    and outcome.after.target_name.lower() in grounded[field].lower()
                )
                if immediate_closure(grounded[field]) or destination_first:
                    grounded[field] = replacement
                    adjustments.append(f"{field}: premature grasp or destination instruction replaced; source far without contact")
    return grounded, tuple(adjustments)


class ExternalFeedbackClient:
    def __init__(self, *, base_url: str, model: str, api_key: str = "", timeout_seconds: int = 120):
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("External feedback base URL must be HTTP(S)")
        if not model:
            raise ValueError("External feedback model must be specified")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def generate(self, *, instruction: str, before_image: Any, after_image: Any,
                 outcome: ActionOutcomeRecord, executed_action: Any, rollout_step: int) -> MultimodalCorrectionRecord:
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Task instruction is required for external feedback")
        if hasattr(executed_action, "detach"):
            executed_action = executed_action.detach().cpu().numpy()
        action_values = np.asarray(executed_action, dtype=np.float64).reshape(-1)
        if action_values.size != 7 or not np.isfinite(action_values).all():
            raise ValueError("Executed action must have seven finite decoded components")
        if rollout_step < 0:
            raise ValueError("Rollout step must be nonnegative")
        if outcome.root_quaternion_wxyz is None:
            raise ValueError("Multimodal correction requires the robot-root orientation")
        before_jpeg = _jpeg_bytes(before_image)
        after_jpeg = _jpeg_bytes(after_image)
        outcome_data = outcome.to_dict()
        state = {key: outcome_data[key] for key in (
            "before", "after", "distance_change_m", "source_movement_m", "grasp_transition",
            "task_success", "episode_done", "rollout_interrupted", "contact_before_n", "contact_after_n",
            "root_quaternion_wxyz",
        )}
        if outcome.task_success is True:
            stage_hint = "Task succeeded after this action. Use phase verify_completion."
        elif outcome.after.source_grasped is False:
            source_distance = None if outcome.after.source_to_gripper is None else float(
                np.linalg.norm(outcome.after.source_to_gripper)
            )
            contact = outcome.contact_after_n
            far_without_contact = (
                source_distance is not None and source_distance > NO_CONTACT_APPROACH_DISTANCE_M
                and (contact is None or max(contact) < 0.5)
            )
            stage_hint = (
                f"Source {outcome.after.source_name or 'object'} is NOT grasped after this action. "
                "Use phase approach_source or grasp_source; next_subtask must focus on the source, not the destination. "
                "Do not advise transporting to the target or placing/releasing before grasp is established. "
                f"Source center is {source_distance} m from gripper after action; "
                f"finger contact forces are {contact} N. "
                "When beyond observed grasping reach and without contact, keep the gripper open and approach first; "
                "closing the gripper can itself establish contact once aligned and within reach. "
                "The signed robot-root-axis direction should have a meaningful component toward the source."
            )
            if far_without_contact:
                stage_hint += " In this observed state, use phase approach_source and keep the gripper open."
            if outcome.before.source_grasped is False and outcome.before.source_to_gripper is not None:
                stage_hint += (
                    " The original decision was also an observed source-approach state: give concrete "
                    "retrospective target, signed root direction, gripper operation, and subtask order; "
                    "do not use unknown for these four fields."
                )
        elif outcome.after.source_grasped is True:
            stage_hint = (
                "Source is grasped after this action. Use phase transport_to_target or release_on_target. "
                "Do not release until target support and alignment are supported by observation."
            )
        else:
            stage_hint = "Current grasp state is unknown; do not assume the source is held."
        if outcome.after.source_grasped is False:
            target = outcome.after.source_position
            reference = outcome.after.gripper_position
        else:
            target = outcome.after.target_position
            reference = outcome.after.source_position
        relative_world = None if target is None or reference is None else [
            round(target[index] - reference[index], 4) for index in range(3)
        ]
        relative_root = None if relative_world is None else world_to_root(
            relative_world, outcome.root_quaternion_wxyz,
        )
        axis_hint = "unknown" if relative_root is None else ", ".join(
            f"{'positive' if relative_root[index] > 0 else 'negative'} {'xyz'[index]} ({abs(relative_root[index]):.3f} m)"
            for index in sorted(range(3), key=lambda axis: abs(relative_root[axis]), reverse=True)
            if abs(relative_root[index]) > 0.01
        ) or "near the geometric target"
        if outcome.before.source_grasped is False:
            original_target = outcome.before.source_position
            original_reference = outcome.before.gripper_position
        else:
            original_target = outcome.before.target_position
            original_reference = outcome.before.source_position
        original_world = None if original_target is None or original_reference is None else [
            round(original_target[index] - original_reference[index], 4) for index in range(3)
        ]
        original_root = None if original_world is None else world_to_root(
            original_world, outcome.root_quaternion_wxyz,
        )
        prompt = (
            "Task instruction: " + instruction + "\n"
            "These are simulator observations from one action, not an action label. "
            "Image 1 is before the student action; image 2 is after it. "
            "State positions use the simulator world frame in meters; the controller translation uses the robot-root frame. "
            "Do not assume camera, world, and robot-root directions are equal. "
            "The executed student's decoded controller command is shown below; it is not an optimal-action label. "
            "Explain the observed outcome, the next subtask, whether to retry or reorder, the target, "
            "end-effector translation direction using signed ROBOT-ROOT axes (positive x, negative y, etc.), "
            "trajectory/position correction, and gripper operation. "
            "Then look back at the original decision and separately state how its target, ROBOT-ROOT translation "
            "direction, gripper operation, and subtask order should change. For non-direction fields, say 'keep' "
            "when no change is needed; for direction fields, repeat signed axes when unchanged. "
            "Use 'unknown' where evidence is insufficient; never claim distance decrease proves success. "
            "Do not output numeric action tokens, joint commands, or a robot control command. "
            "Return only a JSON object with nonempty string fields: " + ", ".join(FIELDS) + ". "
            "phase must be one of " + ", ".join(sorted(PHASES)) + ". "
            "ee_direction_root and retrospective_direction_root must name signed robot-root axes or be 'unknown'.\n"
            "Observed stage constraint: " + stage_hint + "\n"
            "Current geometric target relative to reference in WORLD x/y/z meters (not a safe trajectory): "
            + json.dumps(relative_world) + "\n"
            "The same target offset transformed to ROBOT-ROOT x/y/z meters: "
            + json.dumps(relative_root) + "\n"
            "Meaningful signed ROBOT-ROOT axes toward that target (geometric evidence, not a collision-free command): "
            + axis_hint + "\n"
            "Original-decision target offset in ROBOT-ROOT x/y/z meters (geometric evidence, not an action label): "
            + json.dumps(original_root) + "\n"
            "Decoded controller command (translation ROBOT-ROOT x/y/z in meters, rotation delta, gripper +1 OPEN/-1 CLOSE): "
            + json.dumps(action_values.tolist()) + "\n"
            "Decoded gripper operation: " + ("OPEN" if action_values[6] > 0 else "CLOSE") + ".\n"
            "If the decoded command kept the gripper OPEN, do not claim a closing attempt failed; "
            "describe it as no closing command instead. Do not infer speed or contact failure from distance alone.\n"
            "Simulator state change: " + json.dumps(state, ensure_ascii=False)
        )
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": 768,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(before_jpeg).decode("ascii")}},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(after_jpeg).decode("ascii")}},
            ]}],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        rejections = []
        for attempt in range(1, 4):
            request = urllib.request.Request(self.url, data=json.dumps(payload).encode("utf-8"), headers=headers)
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                result = json.load(response)
            if result.get("model") != self.model:
                raise ValueError("External model response did not identify the requested model")
            content = result["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("External model response content must be text")
            try:
                model_correction = parse_correction(content)
                validate_gripper_attribution(model_correction, action_values)
                validate_retrospective_coverage(model_correction, outcome)
                correction, adjustments = ground_obvious_contact_errors(model_correction, outcome)
                validate_stage(correction, outcome)
                break
            except ValueError as error:
                rejections.append(str(error))
                if attempt == 3:
                    raise
                payload["messages"][0]["content"][0]["text"] = (
                    prompt + "\nPrevious answer was invalid: " + content[:1500]
                    + "\nValidation error: " + str(error) + "\nReturn corrected JSON only."
                )
        def section(fields):
            return " ".join(f"{field.replace('_', ' ')}: {correction[field]}" for field in fields)

        text = (
            "Retrospective external-model feedback for the teacher only. "
            f"Observed outcome: {correction['observed_outcome']} "
            f"High-level plan: {section(HIGH_LEVEL_FIELDS)} "
            f"Low-level guidance: {section(LOW_LEVEL_FIELDS)} "
            f"Original-decision correction: {section(RETROSPECTIVE_FIELDS)} "
            f"Uncertainty: {correction['uncertainty']}"
        )
        return MultimodalCorrectionRecord(
            outcome, self.model, rollout_step, tuple(float(value) for value in action_values),
            correction, text, before_jpeg, after_jpeg, attempt, tuple(rejections),
            model_correction, adjustments,
        )
