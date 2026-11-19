import json
import runpy
import sys
import tempfile
from http.server import HTTPServer
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from simpler_env.external_feedback import ExternalFeedbackClient, FIELDS, MultimodalCorrectionRecord, _jpeg_bytes
from simpler_env.feedback import build_action_outcome_feedback, build_rel_spatial_feedback
from simpler_env.train_ms3_ppo import Runner


def test_chat_server_receives_real_client_request():
    server_code = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/serve_qwen_feedback.py"))

    class Model:
        def generate(self, prompt, images, max_tokens):
            assert "put carrot on plate" in prompt
            assert len(images) == 2 and all(image.size == (8, 8) for image in images)
            assert max_tokens == 768
            response = {field: "unknown" for field in FIELDS}
            response.update(phase="approach_source", next_subtask="approach carrot", ee_direction_root="positive x",
                            retrospective_correction="approach before closing",
                            retrospective_target_adjustment="Keep carrot target",
                            retrospective_direction_root="positive x",
                            retrospective_gripper_operation="Keep gripper open",
                            retrospective_order_adjustment="Approach carrot before grasping")
            return json.dumps(response)

    handler = server_code["make_handler"](Model(), "Qwen/Qwen2.5-VL-7B-Instruct", "")
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        state = build_rel_spatial_feedback(
            episode_id="0:reset0:env0", decision_index=0,
            source_name="carrot", target_name="plate",
            source_position=[0.1, 0.0, 0.0], target_position=[0.3, 0.0, 0.0],
            gripper_position=[0.0, 0.0, 0.0], source_grasped=False,
        )
        outcome = build_action_outcome_feedback(
            state, state, env_index=0, action_token_ids=[31999] * 7,
            episode_done=False, task_success=False, root_quaternion_wxyz=[1, 0, 0, 0],
        )
        client = ExternalFeedbackClient(
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            model="Qwen/Qwen2.5-VL-7B-Instruct",
        )
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        result = client.generate(
            instruction="put carrot on plate", before_image=image,
            after_image=image, outcome=outcome,
            executed_action=[0.01, 0, 0, 0, 0, 0, -1], rollout_step=0,
        )
        assert result.correction["next_subtask"] == "approach carrot"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_archive_recheck_generates_without_environment_interaction():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    server_code = runpy.run_path(str(scripts / "serve_qwen_feedback.py"))

    class Model:
        def generate(self, prompt, images, max_tokens):
            assert len(images) == 2
            assert max_tokens == 768
            correction = {field: "unknown" for field in FIELDS}
            correction.update(
                phase="approach_source", next_subtask="Approach carrot",
                ee_direction_root="positive x", gripper_operation="OPEN",
                retrospective_correction="Approach carrot before grasping.",
                retrospective_target_adjustment="Keep carrot as source target.",
                retrospective_direction_root="positive x",
                retrospective_gripper_operation="Keep gripper open until in reach.",
                retrospective_order_adjustment="Approach before grasp and transport.",
            )
            return json.dumps(correction)

    handler = server_code["make_handler"](Model(), "Qwen/Qwen2.5-VL-7B-Instruct", "")
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        state = build_rel_spatial_feedback(
            episode_id="0:reset0:env0", decision_index=0,
            source_name="carrot", target_name="plate",
            source_position=[0.1, 0.0, 0.0], target_position=[0.3, 0.0, 0.0],
            gripper_position=[0.0, 0.0, 0.0], source_grasped=False,
        )
        outcome = build_action_outcome_feedback(
            state, state, env_index=0, action_token_ids=[31999] * 7,
            episode_done=False, task_success=False, root_quaternion_wxyz=[1, 0, 0, 0],
        )
        image = _jpeg_bytes(np.zeros((8, 8, 3), dtype=np.uint8))
        correction = {field: "unknown" for field in FIELDS}
        archived = MultimodalCorrectionRecord(
            outcome=outcome, model="Qwen/Qwen2.5-VL-7B-Instruct", rollout_step=0,
            executed_action=(0.01, 0, 0, 0, 0, 0, 1), correction=correction,
            text="archived feedback", before_jpeg=image, after_jpeg=image,
            model_correction=correction,
        )
        with tempfile.TemporaryDirectory() as directory:
            runner = Runner.__new__(Runner)
            runner.args = SimpleNamespace(episode_len=1, num_envs=1)
            runner.buffer = SimpleNamespace(ep_len=1, feedback=[[archived]])
            runner.glob_dir = Path(directory)
            runner.multimodal_mode = True
            archive = runner.save_feedback_archive(0)
            output = Path(directory) / "rechecked.jsonl"
            argv = [
                "recheck_sdpo_multimodal_v2_archive.py", "--archive", str(archive),
                "--rows", "1", "--endpoint", f"http://127.0.0.1:{server.server_port}/v1",
                "--output", str(output),
            ]
            with patch.object(sys, "argv", argv):
                runpy.run_path(str(scripts / "recheck_sdpo_multimodal_v2_archive.py"), run_name="__main__")
            row = json.loads(output.read_text())
            assert row["archive_row"] == 1
            assert row["new_feedback"]["correction_schema"] == "high-low-retrospective-v2"
            assert row["new_feedback"]["before_image_sha256"] == archived.to_dict()["before_image_sha256"]
            assert row["new_feedback"]["generation_attempts"] == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
