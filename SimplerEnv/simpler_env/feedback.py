"""Deterministic privileged RelSpatial feedback for SimplerEnv rollouts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class RelSpatialRecord:
    episode_id: int | str | None
    decision_index: int
    timestamp_utc: str
    source_name: str | None
    target_name: str | None
    source_position: tuple[float, float, float] | None
    target_position: tuple[float, float, float] | None
    gripper_position: tuple[float, float, float] | None
    source_to_gripper: tuple[float, float, float] | None
    target_to_gripper: tuple[float, float, float] | None
    source_to_target: tuple[float, float, float] | None
    source_grasped: bool | None
    gripper_open: bool | None
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionGuidanceRecord:
    state: RelSpatialRecord
    desired_delta_root: tuple[float, float, float] | None
    action_delta_root: tuple[float, float, float] | None
    suggested_action: str
    action_judgment: str
    text: str

    @property
    def timestamp_utc(self) -> str:
        return self.state.timestamp_utc

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.state.to_dict(),
            "feedback_version": "relspatial-action-v1",
            "desired_delta_root": self.desired_delta_root,
            "action_delta_root": self.action_delta_root,
            "suggested_action": self.suggested_action,
            "action_judgment": self.action_judgment,
            "text": self.text,
        }


def _vec(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != 3 or not np.isfinite(array).all():
        return None
    return tuple(float(x) for x in array)


def _bool(value: Any) -> bool | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value).reshape(-1)
    return None if array.size != 1 else bool(array[0])


def _direction(vector: tuple[float, float, float] | None, deadband_m: float) -> str:
    if vector is None:
        return "unknown"
    values = np.asarray(vector)
    if np.linalg.norm(values) < deadband_m:
        return "near"
    axis = int(np.argmax(np.abs(values)))
    names = (("left", "right"), ("front", "back"), ("below", "above"))
    return names[axis][0 if values[axis] < 0 else 1]


def _state_phrase(name: str, vector: tuple[float, float, float] | None, deadband_m: float) -> str:
    if vector is None:
        return f"{name} position is unknown."
    return f"{name} is {_relation(vector, 'the gripper', deadband_m)} ({np.linalg.norm(vector):.3f} m)."


def _relation(vector: tuple[float, float, float], reference: str, deadband_m: float) -> str:
    direction = _direction(vector, deadband_m)
    prepositions = {
        "left": "left of",
        "right": "right of",
        "front": "in front of",
        "back": "behind",
        "below": "below",
        "above": "above",
        "near": "near",
    }
    return f"{prepositions[direction]} {reference}"


def build_rel_spatial_feedback(*, episode_id: int | str | None, decision_index: int,
                               source_name: str | None, target_name: str | None,
                               source_position: Any, target_position: Any,
                               gripper_position: Any, source_grasped: Any = None,
                               gripper_open: Any = None, timestamp_utc: str | None = None,
                               deadband_m: float = 0.01) -> RelSpatialRecord:
    """Build teacher feedback from current state; missing fields remain unknown."""
    source, target, gripper = _vec(source_position), _vec(target_position), _vec(gripper_position)
    source_to_gripper = None if source is None or gripper is None else tuple(source[i] - gripper[i] for i in range(3))
    target_to_gripper = None if target is None or gripper is None else tuple(target[i] - gripper[i] for i in range(3))
    source_to_target = None if source is None or target is None else tuple(source[i] - target[i] for i in range(3))
    source_label, target_label = source_name or "source object", target_name or "target container"
    grasped, opened = _bool(source_grasped), _bool(gripper_open)
    text = " ".join((
        _state_phrase(source_label, source_to_gripper, deadband_m),
        _state_phrase(target_label, target_to_gripper, deadband_m),
        (f"{source_label} is {_relation(source_to_target, target_label, deadband_m)}."
         if source_to_target is not None else f"Relation between {source_label} and {target_label} is unknown."),
        f"{source_label} is {'being held' if grasped else 'not being held' if grasped is not None else 'in an unknown grasp state'}.",
        f"The gripper is {'open' if opened else 'closed' if opened is not None else 'in an unknown state'}.",
    ))
    return RelSpatialRecord(episode_id, decision_index, timestamp_utc or datetime.now(timezone.utc).isoformat(),
                            source_name, target_name, source, target, gripper, source_to_gripper,
                            target_to_gripper, source_to_target, grasped, opened, text)


def records_to_jsonable(records: Sequence[RelSpatialRecord | None]) -> list[dict[str, Any] | None]:
    return [record.to_dict() if record is not None else None for record in records]


def build_action_guidance_feedback(
    state: RelSpatialRecord,
    *,
    action_delta_root: Any,
    root_quaternion_wxyz: Any,
    gripper_command: Any,
    approach_m: float = 0.03,
    movement_deadband_m: float = 0.005,
) -> ActionGuidanceRecord:
    """Judge a sampled action against a decision-time geometric heuristic."""
    action = _vec(action_delta_root)
    if hasattr(root_quaternion_wxyz, "detach"):
        root_quaternion_wxyz = root_quaternion_wxyz.detach().cpu().numpy()
    quaternion = np.asarray(root_quaternion_wxyz, dtype=np.float64).reshape(-1) if root_quaternion_wxyz is not None else None
    desired_world = state.source_to_gripper if state.source_grasped is False else (
        tuple(-value for value in state.source_to_target) if state.source_to_target is not None else None
    )
    desired = None
    if desired_world is not None and quaternion is not None and quaternion.size == 4 and np.isfinite(quaternion).all():
        norm = np.linalg.norm(quaternion)
        if norm > 0:
            scalar, vector = quaternion[0] / norm, quaternion[1:] / norm
            world = np.asarray(desired_world)
            desired = _vec(world + 2 * np.cross(-vector, np.cross(-vector, world) + scalar * world))

    if state.source_grasped is None:
        suggestion, judgment = "unknown", "unknown"
    elif desired is None:
        suggestion, judgment = "unknown", "unknown"
    elif np.linalg.norm(desired) < approach_m:
        if state.source_grasped:
            suggestion, judgment = "release only when placement is supported", "unknown"
        else:
            suggestion = "close the gripper on the source object"
            command = _bool(gripper_command < 0) if gripper_command is not None else None
            judgment = "aligned" if command else "not aligned" if command is False else "unknown"
    else:
        direction = _direction(desired_world, movement_deadband_m)
        direction = {"below": "down", "above": "up", "front": "forward", "back": "backward"}.get(direction, direction)
        target = "source object" if state.source_grasped is False else "target container"
        suggestion = f"move {direction} in the world frame toward the {target}"
        if action is None or np.linalg.norm(action) < movement_deadband_m:
            judgment = "not aligned" if action is not None else "unknown"
        else:
            cosine = float(np.dot(action, desired) / (np.linalg.norm(action) * np.linalg.norm(desired)))
            judgment = "aligned" if cosine >= 0.5 else "opposed" if cosine <= -0.5 else "not aligned"

    text = f"{state.text} Suggested next action: {suggestion}. Sampled action is {judgment} with this suggestion."
    return ActionGuidanceRecord(state, desired, action, suggestion, judgment, text)
