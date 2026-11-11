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
    return f"{name} is {_direction(vector, deadband_m)} of the gripper ({np.linalg.norm(vector):.3f} m)."


def build_rel_spatial_feedback(*, episode_id: int | str | None, decision_index: int,
                               source_name: str | None, target_name: str | None,
                               source_position: Any, target_position: Any,
                               gripper_position: Any, source_grasped: Any = None,
                               gripper_open: Any = None, timestamp_utc: str | None = None,
                               deadband_m: float = 0.01) -> RelSpatialRecord:
    """Build teacher feedback from current state; missing fields remain unknown."""
    source, target, gripper = _vec(source_position), _vec(target_position), _vec(gripper_position)
    source_to_gripper = None if source is None or gripper is None else tuple(gripper[i] - source[i] for i in range(3))
    target_to_gripper = None if target is None or gripper is None else tuple(gripper[i] - target[i] for i in range(3))
    source_to_target = None if source is None or target is None else tuple(target[i] - source[i] for i in range(3))
    source_label, target_label = source_name or "source object", target_name or "target container"
    grasped, opened = _bool(source_grasped), _bool(gripper_open)
    text = " ".join((
        _state_phrase(source_label, source_to_gripper, deadband_m),
        _state_phrase(target_label, target_to_gripper, deadband_m),
        (f"{source_label} is {_direction(source_to_target, deadband_m)} of {target_label}."
         if source_to_target is not None else f"Relation between {source_label} and {target_label} is unknown."),
        f"{source_label} is {'being held' if grasped else 'not being held' if grasped is not None else 'in an unknown grasp state'}.",
        f"The gripper is {'open' if opened else 'closed' if opened is not None else 'in an unknown state'}.",
    ))
    return RelSpatialRecord(episode_id, decision_index, timestamp_utc or datetime.now(timezone.utc).isoformat(),
                            source_name, target_name, source, target, gripper, source_to_gripper,
                            target_to_gripper, source_to_target, grasped, opened, text)


def records_to_jsonable(records: Sequence[RelSpatialRecord | None]) -> list[dict[str, Any] | None]:
    return [record.to_dict() if record is not None else None for record in records]
