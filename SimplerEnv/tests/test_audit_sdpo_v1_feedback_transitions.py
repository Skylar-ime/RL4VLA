import json
import math

from scripts.audit_sdpo_v1_feedback_transitions import audit, audit_close_suggestions


def test_audit_matches_next_decision_and_separates_grasp_state(tmp_path):
    records = [
        dict(episode_id=0, decision_index=0, action_judgment="aligned", source_grasped=False,
             source_position=[0.0, 0.0, 0.0], target_position=[1.0, 0.0, 0.0],
             gripper_position=[0.2, 0.0, 0.0]),
        dict(episode_id=0, decision_index=1, action_judgment="opposed", source_grasped=True,
             source_position=[0.1, 0.0, 0.0], target_position=[1.0, 0.0, 0.0],
             gripper_position=[0.1, 0.0, 0.0]),
        dict(episode_id=0, decision_index=2, action_judgment="unknown", source_grasped=True,
             source_position=[0.0, 0.0, 0.0], target_position=[1.0, 0.0, 0.0],
             gripper_position=[0.0, 0.0, 0.0]),
        dict(episode_id=1, decision_index=0, action_judgment="aligned", source_grasped=False,
             source_position=[0.0, 0.0, 0.0], target_position=[1.0, 0.0, 0.0],
             gripper_position=[0.0, 0.0, 0.0]),
    ]
    path = tmp_path / "feedback.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    result = audit([path], threshold=0.001)

    assert result[("aligned", False)][:2] == [1, 1]
    assert math.isclose(result[("aligned", False)][2], 0.2)
    assert result[("aligned", False)][3] == 1
    assert result[("opposed", True)][:2] == [1, 0]
    assert math.isclose(result[("opposed", True)][2], -0.1)


def test_audit_close_suggestions_counts_following_grasps(tmp_path):
    records = [
        dict(episode_id=0, decision_index=0, suggested_action="close the gripper on the source object",
             action_judgment="aligned", source_grasped=False),
        dict(episode_id=0, decision_index=1, suggested_action="move right",
             action_judgment="aligned", source_grasped=True),
        dict(episode_id=1, decision_index=0, suggested_action="close the gripper on the source object",
             action_judgment="not aligned", source_grasped=False),
        dict(episode_id=1, decision_index=1, suggested_action="move right",
             action_judgment="aligned", source_grasped=False),
    ]
    path = tmp_path / "feedback.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    assert audit_close_suggestions([path]) == {"aligned": [1, 1], "not aligned": [1, 0]}
