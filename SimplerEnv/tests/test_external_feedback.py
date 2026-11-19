import json
import io
import tarfile
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from simpler_env.external_feedback import (
    ExternalFeedbackClient, FIELDS, MultimodalCorrectionRecord, parse_correction, validate_stage,
    ground_obvious_contact_errors, immediate_closure, validate_gripper_attribution,
    validate_retrospective_coverage,
)
from simpler_env.feedback import build_action_outcome_feedback, build_rel_spatial_feedback
from simpler_env.policies.openvla.openvla_train import OpenVLAPPO
from simpler_env.train_ms3_ppo import Runner


def _outcome():
    before = build_rel_spatial_feedback(
        episode_id="0:reset0:env0", decision_index=0,
        source_name="carrot", target_name="plate",
        source_position=[0.1, 0.0, 0.0], target_position=[0.3, 0.0, 0.0],
        gripper_position=[0.0, 0.0, 0.0], source_grasped=False,
    )
    after = build_rel_spatial_feedback(
        episode_id="0:reset0:env0", decision_index=0,
        source_name="carrot", target_name="plate",
        source_position=[0.1, 0.0, 0.0], target_position=[0.3, 0.0, 0.0],
        gripper_position=[0.01, 0.0, 0.0], source_grasped=False,
    )
    return build_action_outcome_feedback(
        before, after, env_index=0, action_token_ids=[31999] * 7,
        episode_done=False, task_success=False, root_quaternion_wxyz=[1, 0, 0, 0],
    )


def _with_retrospective_details(correction):
    correction.update(
        retrospective_target_adjustment="Keep the carrot as the target.",
        retrospective_direction_root="positive x",
        retrospective_gripper_operation="Keep gripper open until close to carrot.",
        retrospective_order_adjustment="Approach carrot before grasping and transporting.",
    )
    return correction


def test_external_feedback_preserves_archived_rgb_jpeg_bytes():
    from simpler_env.external_feedback import _jpeg_bytes

    output = io.BytesIO()
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(output, format="JPEG")
    original = output.getvalue()
    assert _jpeg_bytes(original) == original
    try:
        _jpeg_bytes(b"not an image")
    except (ValueError, OSError):
        pass
    else:
        raise AssertionError("Invalid archived image accepted")


def test_external_feedback_sends_both_images_and_state(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            correction = {field: "unknown" for field in FIELDS}
            correction["phase"] = "approach_source"
            correction["next_subtask"] = "approach carrot"
            correction["ee_direction_root"] = "positive x"
            correction["retrospective_correction"] = "move toward carrot before closing"
            _with_retrospective_details(correction)
            return json.dumps({"model": "Qwen/Qwen2.5-VL-7B-Instruct",
                               "choices": [{"message": {"content": json.dumps(correction)}}]}).encode()

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = ExternalFeedbackClient(base_url="http://localhost:8000/v1", model="Qwen/Qwen2.5-VL-7B-Instruct")
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    record = client.generate(instruction="put carrot on plate", before_image=image,
                             after_image=image, outcome=_outcome(), executed_action=[0.01, 0, 0, 0, 0, 0, -1], rollout_step=0)
    payload = json.loads(captured["request"].data)
    content = payload["messages"][0]["content"]
    assert len(content) == 3
    assert all(item["image_url"]["url"].startswith("data:image/jpeg;base64,") for item in content[1:])
    assert "put carrot on plate" in content[0]["text"]
    assert "distance_change_m" in content[0]["text"]
    assert "Decoded controller command" in content[0]["text"]
    assert "The same target offset transformed to ROBOT-ROOT" in content[0]["text"]
    assert "Original-decision target offset in ROBOT-ROOT" in content[0]["text"]
    assert "repeat signed axes when unchanged" in content[0]["text"]
    assert "gripper +1 OPEN/-1 CLOSE" in content[0]["text"]
    assert "Decoded gripper operation: CLOSE" in content[0]["text"]
    assert "do not claim a closing attempt failed" in content[0]["text"]
    assert record.to_dict()["feedback_version"] == "multimodal-correction-v2"
    assert record.to_dict()["feedback_schema"] == "robot-root-direction-v2"
    assert record.to_dict()["correction_schema"] == "high-low-retrospective-v2"
    assert record.to_dict()["ee_direction_frame"] == "robot_root_at_decision"
    assert record.correction["next_subtask"] == "approach carrot"
    assert "High-level plan:" in record.text
    assert "Low-level guidance:" in record.text
    assert "Original-decision correction:" in record.text
    assert record.to_dict()["before_image_member"] == "decision_0000_env_00_before.jpg"
    assert len(record.to_dict()["before_image_sha256"]) == 64


def test_external_feedback_rejects_missing_fields():
    try:
        parse_correction('{"next_subtask":"approach"}')
    except ValueError as error:
        assert "required correction field" in str(error)
    else:
        raise AssertionError("Incomplete correction accepted")


def test_external_feedback_rejects_vacuous_guidance():
    try:
        parse_correction(json.dumps({field: "unknown" for field in FIELDS}))
    except ValueError as error:
        assert "no actionable correction" in str(error)
    else:
        raise AssertionError("Vacuous correction accepted")


def test_external_feedback_normalizes_numeric_root_vector():
    correction = {field: "unknown" for field in FIELDS}
    correction.update(phase="approach_source", next_subtask="approach carrot",
                      ee_direction_root="-0.001, -0.020, -0.145",
                      retrospective_correction="approach before closing")
    _with_retrospective_details(correction)
    parsed = parse_correction(json.dumps(correction))
    assert parsed["ee_direction_root"] == "negative z, negative y"
    assert parsed["ee_direction_root_raw"] == "-0.001, -0.020, -0.145"

    correction["ee_direction_root"] = "-z, -y"
    parsed = parse_correction(json.dumps(correction))
    assert parsed["ee_direction_root"] == "negative z, negative y"
    assert parsed["ee_direction_root_raw"] == "-z, -y"

    correction["retrospective_direction_root"] = "-z"
    parsed = parse_correction(json.dumps(correction))
    assert parsed["retrospective_direction_root"] == "negative z"
    assert parsed["retrospective_direction_root_raw"] == "-z"


def test_external_feedback_rejects_conflicting_axis_signs():
    correction = {field: "unknown" for field in FIELDS}
    correction.update(phase="approach_source", next_subtask="approach carrot",
                      ee_direction_root="positive z, negative z",
                      retrospective_correction="approach source")
    _with_retrospective_details(correction)
    try:
        parse_correction(json.dumps(correction))
    except ValueError as error:
        assert "contradicts itself" in str(error)
    else:
        raise AssertionError("Contradictory direction accepted")


def test_world_to_root_matches_widowx_controller_frame():
    from simpler_env.external_feedback import world_to_root

    quarter_turn = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
    assert np.allclose(world_to_root([1, 0, 0], quarter_turn), [0, -1, 0], atol=1e-6)


def test_stage_direction_is_checked_in_robot_root_frame():
    from simpler_env.external_feedback import world_to_root

    outcome = _outcome()
    quarter_turn = (np.sqrt(0.5), 0, 0, np.sqrt(0.5))
    outcome = replace(outcome, root_quaternion_wxyz=quarter_turn)
    correction = {field: "unknown" for field in FIELDS}
    correction.update(phase="approach_source", next_subtask="approach carrot",
                      ee_direction_root="negative y", retrospective_correction="approach source")
    assert world_to_root(outcome.after.source_to_gripper, quarter_turn)[1] < 0
    validate_stage(correction, outcome)
    correction["ee_direction_root"] = "positive y"
    try:
        validate_stage(correction, outcome)
    except ValueError as error:
        assert "direction does not approach" in str(error)
    else:
        raise AssertionError("World-frame direction accepted as robot-root direction")

    correction["ee_direction_root"] = "negative y"
    correction["retrospective_direction_root"] = "positive y"
    try:
        validate_stage(correction, outcome)
    except ValueError as error:
        assert "retrospective direction" in str(error)
    else:
        raise AssertionError("Wrong original-decision direction accepted")


def test_external_feedback_rejects_stage_contradiction():
    correction = {field: "unknown" for field in FIELDS}
    correction.update(phase="release_on_target", next_subtask="place carrot on plate",
                      ee_direction_root="negative z", retrospective_correction="move to plate")
    try:
        validate_stage(correction, _outcome())
    except ValueError as error:
        assert "contradicts observed grasp" in str(error)
    else:
        raise AssertionError("Placement accepted without grasp")


def test_external_feedback_rejects_destination_before_grasp():
    correction = {field: "unknown" for field in FIELDS}
    correction.update(phase="approach_source", next_subtask="move gripper toward plate",
                      ee_direction_root="negative z", retrospective_correction="approach source")
    try:
        validate_stage(correction, _outcome())
    except ValueError as error:
        assert "destination before grasp" in str(error)
    else:
        raise AssertionError("Destination approach accepted before grasp")


def test_external_feedback_rejects_wrong_axis_and_premature_closure():
    state = build_rel_spatial_feedback(
        episode_id="0:reset0:env0", decision_index=0,
        source_name="carrot", target_name="plate",
        source_position=[0.2, 0.0, 0.0], target_position=[0.3, 0.0, 0.0],
        gripper_position=[0.0, 0.0, 0.0], source_grasped=False,
    )
    outcome = build_action_outcome_feedback(
        state, state, env_index=0, action_token_ids=[31999] * 7,
        episode_done=False, task_success=False, contact_after_n=[0.0, 0.0],
        root_quaternion_wxyz=[1, 0, 0, 0],
    )
    correction = {field: "unknown" for field in FIELDS}
    correction.update(phase="approach_source", next_subtask="approach carrot",
                      ee_direction_root="negative x", gripper_operation="keep open",
                      retrospective_correction="approach source")
    try:
        validate_stage(correction, outcome)
    except ValueError as error:
        assert "direction does not approach" in str(error)
    else:
        raise AssertionError("Wrong source direction accepted")
    correction["ee_direction_root"] = "positive x"
    correction["retrospective_correction"] = "The gripper did not come close enough to the source for contact."
    unchanged, adjustments = ground_obvious_contact_errors(correction, outcome)
    assert unchanged == correction
    assert adjustments == ()
    correction["gripper_operation"] = "Close the gripper"
    try:
        validate_stage(correction, outcome)
    except ValueError as error:
        assert "closure while far" in str(error)
    else:
        raise AssertionError("Premature grasp closure accepted")

    correction["phase"] = "grasp_source"
    correction["next_subtask"] = "Approach carrot with the gripper closed"
    correction["trajectory_correction"] = "Ensure the gripper is closed"
    correction["retrospective_correction"] = "The gripper should have been closed earlier"
    grounded, adjustments = ground_obvious_contact_errors(correction, outcome)
    assert grounded["phase"] == "approach_source"
    assert grounded["gripper_operation"].startswith("Keep gripper open")
    assert len(adjustments) == 5
    validate_stage(grounded, outcome)


def test_closure_detection_does_not_confuse_distance_with_gripper_operation():
    assert not immediate_closure("The gripper did not come close enough to the source for contact.")
    assert not immediate_closure("Move closer to the source; keep the gripper open.")
    assert not immediate_closure("Move closer without closing the gripper.")
    assert not immediate_closure("Close only when aligned within grasping reach.")
    assert immediate_closure("Close the gripper now.")
    assert immediate_closure("The gripper should have been closed earlier.")
    assert immediate_closure("+1 close")
    assert immediate_closure("CLOSE")
    assert immediate_closure("The gripper did not close properly, so the object was missed.")
    assert immediate_closure("The approach was too close but not yet closed.")


def test_open_command_cannot_be_described_as_failed_closure():
    correction = {field: "unknown" for field in FIELDS}
    correction["observed_outcome"] = "The bread was not grasped."
    correction["retrospective_correction"] = "The gripper did not close properly, leading to missed contact."
    try:
        validate_gripper_attribution(correction, [0, 0, 0, 0, 0, 0, 1])
    except ValueError as error:
        assert "executed OPEN command" in str(error)
    else:
        raise AssertionError("Failed closure attributed to an OPEN command")
    validate_gripper_attribution(correction, [0, 0, 0, 0, 0, 0, -1])
    correction["retrospective_correction"] = "The gripper remained open; approach before closing."
    validate_gripper_attribution(correction, [0, 0, 0, 0, 0, 0, 1])


def test_approach_feedback_requires_all_retrospective_details():
    correction = _with_retrospective_details({field: "unknown" for field in FIELDS})
    validate_retrospective_coverage(correction, _outcome())
    correction["retrospective_gripper_operation"] = "unknown"
    try:
        validate_retrospective_coverage(correction, _outcome())
    except ValueError as error:
        assert "retrospective_gripper_operation" in str(error)
    else:
        raise AssertionError("Missing retrospective gripper detail accepted")


def test_contact_guard_covers_ungrasped_source_beyond_observed_grasp_reach():
    state = build_rel_spatial_feedback(
        episode_id="0:reset0:env0", decision_index=0,
        source_name="bread", target_name="plate",
        source_position=[0.11, 0.0, 0.0], target_position=[0.3, 0.0, 0.0],
        gripper_position=[0.0, 0.0, 0.0], source_grasped=False,
    )
    outcome = build_action_outcome_feedback(
        state, state, env_index=0, action_token_ids=[31999] * 7,
        episode_done=False, task_success=False, contact_after_n=[0.0, 0.0],
        root_quaternion_wxyz=[1, 0, 0, 0],
    )
    correction = {field: "unknown" for field in FIELDS}
    correction.update(phase="approach_source", next_subtask="Approach bread with gripper closed",
                      ee_direction_root="positive x", gripper_operation="CLOSE",
                      retrospective_correction="The gripper did not close properly.",
                      retrospective_gripper_operation="CLOSE")
    grounded, adjustments = ground_obvious_contact_errors(correction, outcome)
    assert grounded["next_subtask"].startswith("Approach the bread with the gripper open")
    assert grounded["gripper_operation"].startswith("Keep gripper open")
    assert grounded["retrospective_gripper_operation"].startswith("Keep gripper open")
    assert len(adjustments) == 4
    validate_stage(grounded, outcome)


def test_external_feedback_rejects_wrong_model(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return b'{"model":"other-model","choices":[]}'

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: Response())
    client = ExternalFeedbackClient(base_url="http://localhost:8000/v1", model="Qwen/Qwen2.5-VL-7B-Instruct")
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    try:
        client.generate(instruction="put carrot on plate", before_image=image,
                        after_image=image, outcome=_outcome(), executed_action=[0.01, 0, 0, 0, 0, 0, -1], rollout_step=0)
    except ValueError as error:
        assert "requested model" in str(error)
    else:
        raise AssertionError("Mismatched model accepted")


def test_external_feedback_repairs_invalid_first_response(monkeypatch):
    requests = []

    class Response:
        def __init__(self, content):
            self.content = content

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps({"model": "Qwen/Qwen2.5-VL-7B-Instruct",
                               "choices": [{"message": {"content": json.dumps(self.content)}}]}).encode()

    def fake_urlopen(request, timeout):
        requests.append(json.loads(request.data))
        correction = {field: "unknown" for field in FIELDS}
        correction.update(phase="release_on_target" if len(requests) == 1 else "approach_source",
                          next_subtask="approach carrot", ee_direction_root="positive x",
                          retrospective_correction="approach before closing")
        _with_retrospective_details(correction)
        return Response(correction)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = ExternalFeedbackClient(base_url="http://localhost:8000/v1", model="Qwen/Qwen2.5-VL-7B-Instruct")
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    record = client.generate(
        instruction="put carrot on plate", before_image=image, after_image=image,
        outcome=_outcome(), executed_action=[0.01, 0, 0, 0, 0, 0, -1], rollout_step=0,
    )
    assert len(requests) == 2
    assert "Previous answer was invalid" in requests[1]["messages"][0]["content"][0]["text"]
    assert record.generation_attempts == 2
    assert len(record.rejection_reasons) == 1


def test_external_feedback_retries_false_gripper_failure_claim(monkeypatch):
    requests = []

    class Response:
        def __init__(self, correction):
            self.correction = correction

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps({
                "model": "Qwen/Qwen2.5-VL-7B-Instruct",
                "choices": [{"message": {"content": json.dumps(self.correction)}}],
            }).encode()

    def fake_urlopen(request, timeout):
        requests.append(json.loads(request.data))
        correction = {field: "unknown" for field in FIELDS}
        correction.update(
            phase="approach_source", next_subtask="approach carrot",
            ee_direction_root="positive x", gripper_operation="OPEN",
            retrospective_correction=(
                "The gripper did not close properly." if len(requests) == 1
                else "The student kept the gripper open; approach before closing."
            ),
        )
        _with_retrospective_details(correction)
        return Response(correction)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = ExternalFeedbackClient(base_url="http://localhost:8000/v1", model="Qwen/Qwen2.5-VL-7B-Instruct")
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    record = client.generate(
        instruction="put carrot on plate", before_image=image, after_image=image,
        outcome=_outcome(), executed_action=[0.01, 0, 0, 0, 0, 0, 1], rollout_step=0,
    )
    assert record.generation_attempts == 2
    assert len(record.rejection_reasons) == 1
    assert "executed OPEN command" in record.rejection_reasons[0]
    assert "kept the gripper open" in record.correction["retrospective_correction"]


def test_online_feedback_updates_once_per_record():
    calls = []

    class Policy:
        def prep_training(self):
            calls.append("train")

        def prep_rollout(self):
            calls.append("rollout")

    class Algorithm:
        def train_sdpo_step(self, index, total, batch):
            calls.append((index, total, batch[1], batch[2].tolist(), batch[-1]))
            return {"optimizer_steps": 1.0}

    runner = Runner.__new__(Runner)
    runner.policy = Policy()
    runner.alg = Algorithm()
    record = SimpleNamespace(text="retry approach")
    image = np.zeros((1, 8, 8, 3), dtype=np.uint8)
    action = np.full((1, 7), 31999, dtype=np.int32)
    assert runner.train_multimodal_feedback(image, "put carrot on plate", action, record) == {"optimizer_steps": 1.0}
    assert calls == ["train", (0, 1, ["put carrot on plate"], [[31999] * 7], [record]), "rollout"]


def test_online_feedback_updates_student_and_ema_with_original_tokens():
    student_logits = torch.nn.Parameter(torch.tensor([0.5, -0.5]))
    teacher_logits = torch.nn.Parameter(torch.tensor([-0.5, 0.5]), requires_grad=False)
    original_tokens = np.array([[31999, 31998, 31997, 31996, 31995, 31994, 31993]], dtype=np.int32)
    seen = []
    ema_updates = []

    class CountingOptimizer(torch.optim.SGD):
        def __init__(self):
            super().__init__([student_logits], lr=0.1)
            self.steps = 0

        def step(self, closure=None):
            self.steps += 1
            return super().step(closure)

    optimizer = CountingOptimizer()

    def student_forward(observation, actions):
        seen.append((observation["task_description"], actions.tolist()))
        return student_logits.expand(1, 7, 2)

    def teacher_forward(observation, actions, feedback):
        assert feedback == ["retry approach"]
        assert actions.tolist() == seen[-1][1]
        return teacher_logits.expand(1, 7, 2)

    policy = SimpleNamespace(
        tpdv={"device": torch.device("cpu")}, tpdv_vn={"device": torch.device("cpu")},
        params_vla=[student_logits],
        vla_optimizer=optimizer, evaluate_action_logits=student_forward,
        evaluate_teacher_action_logits=teacher_forward,
        update_teacher_ema=lambda rate: ema_updates.append(rate),
        prep_training=lambda: None, prep_rollout=lambda: None,
    )
    runner = Runner.__new__(Runner)
    runner.policy = policy
    runner.alg = OpenVLAPPO(SimpleNamespace(
        alg_entropy_coef=0.0, alg_gradient_accum=1, sdpo_feedback_mode="correct", sdpo_ema_rate=0.01,
    ), policy)
    record = SimpleNamespace(text="retry approach")
    result = runner.train_multimodal_feedback(
        np.zeros((1, 8, 8, 3), dtype=np.uint8), "put carrot on plate", original_tokens, record,
    )
    assert result["optimizer_steps"] == 1.0
    assert optimizer.steps == 1
    assert ema_updates == [0.01]
    assert seen == [(["put carrot on plate"], original_tokens.tolist())]
    assert teacher_logits.grad is None


def test_feedback_archive_retains_both_images_and_hashes():
    correction = {field: "unknown" for field in FIELDS}
    correction.update(next_subtask="approach carrot", ee_direction_root="positive x",
                      retrospective_correction="approach before closing")
    record = MultimodalCorrectionRecord(
        outcome=_outcome(), model="Qwen/Qwen2.5-VL-7B-Instruct", rollout_step=0,
        executed_action=(0.01, 0, 0, 0, 0, 0, -1), correction=correction,
        text="approach before closing", before_jpeg=b"before image", after_jpeg=b"after image",
    )
    with tempfile.TemporaryDirectory() as directory:
        runner = Runner.__new__(Runner)
        runner.args = SimpleNamespace(episode_len=1, num_envs=1)
        runner.buffer = SimpleNamespace(ep_len=1, feedback=[[record]])
        runner.glob_dir = Path(directory)
        runner.multimodal_mode = True
        archive = runner.save_feedback_archive(0)
        row = json.loads(archive.read_text())
        with tarfile.open(archive.with_name("episode_000000_images.tar")) as images:
            assert images.extractfile(row["before_image_member"]).read() == b"before image"
            assert images.extractfile(row["after_image_member"]).read() == b"after image"
        assert "before image" not in archive.read_text()
        assert len(row["before_image_sha256"]) == 64
