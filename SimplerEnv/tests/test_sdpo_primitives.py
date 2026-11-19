import torch
import numpy as np
import json
from pathlib import Path
from types import SimpleNamespace
from transformers import BatchFeature

from simpler_env.feedback import build_action_guidance_feedback, build_action_outcome_feedback, build_rel_spatial_feedback
from simpler_env.env.simpler_wrapper import completed_episode_info
from simpler_env.policies.openvla.openvla_train import OpenVLAPolicy, OpenVLAPPO
from simpler_env.sdpo_distillation import reverse_kl_student_to_teacher
from simpler_env.train_ms3_ppo import Runner, load_eval_episode_ids, next_rollout_length, parse_checkpoint_steps
from simpler_env.utils.replay_buffer import SeparatedReplayBuffer


def test_exact_environment_step_boundaries_keep_eighty_step_episode_cap():
    boundaries = parse_checkpoint_steps("480,2400,5000,10000", 10000, 2)
    completed = 0
    checkpoints = []
    rollout_lengths = []
    while completed < 10000:
        length = next_rollout_length(completed, 10000, 2, 80, boundaries)
        completed += length * 2
        rollout_lengths.append(length)
        if completed in boundaries:
            checkpoints.append(completed)
    assert checkpoints == [480, 2400, 5000, 10000]
    assert 80 in rollout_lengths and 20 in rollout_lengths
    assert sum(rollout_lengths) * 2 == 10000


def test_stratified_evaluation_manifest_covers_test_objects_and_backgrounds():
    manifest_path = Path(__file__).resolve().parents[1] / "scripts/sdpo_v2_eval_episodes.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episodes_per_stratum = manifest["position_config_count"] * manifest["orientation_count"]
    validation = load_eval_episode_ids(
        str(manifest_path), "validation", 60, manifest["env_id"], "test", 0,
    )
    final_test = load_eval_episode_ids(
        str(manifest_path), "final_test", 100, manifest["env_id"], "test", 60,
    )
    assert not set(validation) & set(final_test)
    for split in (validation, final_test):
        strata = {episode_id // episodes_per_stratum for episode_id in split}
        assert strata == set(range(9 * 5))
    assert load_eval_episode_ids("", "validation", 3, manifest["env_id"], "test", 4) == (4, 5, 6)


def test_rel_spatial_feedback_uses_current_positions_and_explicit_unknowns():
    record = build_rel_spatial_feedback(
        episode_id=3,
        decision_index=4,
        source_name="carrot",
        target_name="plate",
        source_position=[0.0, 0.0, 0.0],
        target_position=[0.2, 0.0, 0.0],
        gripper_position=[0.0, 0.0, 0.1],
        source_grasped=True,
    )
    assert record.source_to_target == (-0.2, 0.0, 0.0)
    assert "carrot is left of plate" in record.text
    assert "unknown state" in record.text


def test_rel_spatial_feedback_rejects_invalid_vector_as_unknown():
    record = build_rel_spatial_feedback(
        episode_id=0,
        decision_index=0,
        source_name=None,
        target_name=None,
        source_position=[float("nan"), 0.0, 0.0],
        target_position=None,
        gripper_position=[0.0, 0.0, 0.0],
    )
    assert record.source_position is None
    assert "position is unknown" in record.text


def test_rel_spatial_feedback_uses_grammatical_directions():
    cases = [
        ((-0.2, 0.0, 0.0), "left of"),
        ((0.2, 0.0, 0.0), "right of"),
        ((0.0, -0.2, 0.0), "in front of"),
        ((0.0, 0.2, 0.0), "behind"),
        ((0.0, 0.0, -0.2), "below"),
        ((0.0, 0.0, 0.2), "above"),
    ]
    for offset, phrase in cases:
        record = build_rel_spatial_feedback(
            episode_id=0,
            decision_index=0,
            source_name="object",
            target_name="target",
            source_position=offset,
            target_position=[0.0, 0.0, 0.0],
            gripper_position=[0.0, 0.0, 0.0],
        )
        assert f"object is {phrase} target." in record.text
        assert f"object is {phrase} the gripper" in record.text


def test_v1_guidance_transforms_world_direction_to_robot_root():
    state = build_rel_spatial_feedback(
        episode_id=0, decision_index=1, source_name="object", target_name="plate",
        source_position=[0.2, 0.0, 0.0], target_position=[0.4, 0.0, 0.0],
        gripper_position=[0.0, 0.0, 0.0], source_grasped=False,
    )
    record = build_action_guidance_feedback(
        state, action_delta_root=[-0.02, 0.0, 0.0],
        root_quaternion_wxyz=[0.0, 0.0, 0.0, 1.0], gripper_command=1.0,
    )
    assert record.suggested_action == "move right in the world frame toward the source object"
    assert record.action_judgment == "aligned"
    assert record.to_dict()["feedback_version"] == "relspatial-action-v1"
    assert "Sampled action is aligned" in record.text


def test_v1_guidance_uses_current_grasp_and_handles_unknown():
    state = build_rel_spatial_feedback(
        episode_id=0, decision_index=1, source_name="object", target_name="plate",
        source_position=[0.0, 0.0, 0.0], target_position=[0.0, 0.0, -0.2],
        gripper_position=[0.0, 0.0, 0.0], source_grasped=True,
    )
    record = build_action_guidance_feedback(
        state, action_delta_root=[0.0, 0.0, -0.02],
        root_quaternion_wxyz=[0.0, 0.0, 0.0, 1.0], gripper_command=-1.0,
    )
    assert record.suggested_action == "move down in the world frame toward the target container"
    assert record.action_judgment == "aligned"
    unknown = build_action_guidance_feedback(
        state, action_delta_root=[0.0, 0.0, -0.02],
        root_quaternion_wxyz=None, gripper_command=-1.0,
    )
    assert unknown.action_judgment == "unknown"


def test_v1_guidance_near_source_recommends_closing():
    state = build_rel_spatial_feedback(
        episode_id=0, decision_index=1, source_name="object", target_name="plate",
        source_position=[0.0, 0.0, 0.01], target_position=[0.2, 0.0, 0.0],
        gripper_position=[0.0, 0.0, 0.0], source_grasped=False,
    )
    record = build_action_guidance_feedback(
        state, action_delta_root=[0.0, 0.0, 0.0],
        root_quaternion_wxyz=[1.0, 0.0, 0.0, 0.0], gripper_command=-1.0,
    )
    assert record.action_judgment == "aligned"


def test_v1_wrapper_reads_current_grasp_not_episode_history():
    from simpler_env.env.simpler_wrapper import SimlerWrapper

    wrapper = object.__new__(SimlerWrapper)
    wrapper.num_envs = 1
    wrapper.episode_ids = [0]
    wrapper.decision_index = [0]
    robot = SimpleNamespace(
        pose=SimpleNamespace(q=torch.tensor([[1.0, 0.0, 0.0, 0.0]])),
        active_joints=[],
    )
    agent = SimpleNamespace(robot=robot, is_grasping=lambda obj: torch.tensor([False]))
    task = SimpleNamespace(
        agent=agent,
        objs_carrot={"carrot": object()},
        carrot_names=["carrot"],
        model_db_carrot={"carrot": {"name": "carrot"}},
        model_db_plate={"plate": {"name": "plate"}},
        plate_names=["plate"],
        select_carrot_ids=torch.tensor([0]),
        select_plate_ids=torch.tensor([0]),
        episode_stats={"is_src_obj_grasped": torch.tensor([True])},
        extra_stats={
            "extra_pos_carrot": torch.tensor([[0.1, 0.0, 0.0]]),
            "extra_pos_plate": torch.tensor([[0.2, 0.0, 0.0]]),
            "extra_pos_gripper": torch.tensor([[0.0, 0.0, 0.0]]),
        },
    )
    wrapper.env = SimpleNamespace(unwrapped=task)
    wrapper._process_action = lambda action: torch.tensor([[0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    record = wrapper.capture_feedback(torch.ones((1, 7)), version="relspatial-action-v1")[0]
    assert record.state.source_grasped is False
    assert record.suggested_action == "move right in the world frame toward the source object"
    assert wrapper.decision_index == [1]


def test_outcome_feedback_reports_observations_without_action_quality_label():
    before = build_rel_spatial_feedback(
        episode_id=7, decision_index=2, source_name="object", target_name="plate",
        source_position=[0.0, 0.0, 0.0], target_position=[0.2, 0.0, 0.0],
        gripper_position=[0.0, 0.0, 0.1], source_grasped=False,
    )
    after = build_rel_spatial_feedback(
        episode_id=7, decision_index=2, source_name="object", target_name="plate",
        source_position=[0.0, 0.0, 0.002], target_position=[0.2, 0.0, 0.0],
        gripper_position=[0.0, 0.0, 0.06], source_grasped=True,
    )
    record = build_action_outcome_feedback(
        before, after, env_index=0, action_token_ids=[31800] * 7,
        episode_done=True, task_success=False,
    )
    assert record.distance_change_m < 0
    assert record.source_movement_m > 0
    assert record.grasp_transition == "established"
    assert "this alone does not establish action correctness" in record.text
    assert "Task success at this step: false" in record.text
    assert "Suggested next action" not in record.text
    assert "aligned" not in record.text
    assert record.to_dict()["action_token_ids"] == (31800,) * 7


def test_outcome_feedback_marks_missing_measurements_unknown():
    before = build_rel_spatial_feedback(
        episode_id=7, decision_index=2, source_name=None, target_name=None,
        source_position=None, target_position=None, gripper_position=None,
    )
    after = build_rel_spatial_feedback(
        episode_id=7, decision_index=2, source_name=None, target_name=None,
        source_position=None, target_position=None, gripper_position=None,
    )
    record = build_action_outcome_feedback(
        before, after, env_index=0, action_token_ids=[31800] * 7,
        episode_done=False, task_success=None,
    )
    assert record.distance_change_m is None
    assert record.grasp_transition == "unknown"
    assert "distance change is unknown" in record.text
    assert "Task success at this step: unknown" in record.text


def test_outcome_feedback_matches_parallel_actions_and_terminal_states():
    from simpler_env.env.simpler_wrapper import SimlerWrapper

    wrapper = object.__new__(SimlerWrapper)
    wrapper.num_envs = 2
    wrapper.episode_ids = [100, 200]
    wrapper.decision_index = [3, 5]
    positions = {
        "extra_pos_carrot": torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]),
        "extra_pos_plate": torch.tensor([[0.1, 0.0, 0.0], [0.3, 0.0, 0.0]]),
        "extra_pos_gripper": torch.tensor([[0.0, 0.0, 0.1], [0.2, 0.0, 0.1]]),
    }
    grasped = torch.tensor([False, True])
    root_quaternions = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    agent = SimpleNamespace(robot=SimpleNamespace(active_joints=[], pose=SimpleNamespace(q=root_quaternions)),
                            is_grasping=lambda obj: grasped)
    task = SimpleNamespace(
        agent=agent, objs_carrot={"carrot": object()}, carrot_names=["carrot"],
        model_db_carrot={"carrot": {"name": "carrot"}},
        model_db_plate={"plate": {"name": "plate"}}, plate_names=["plate"],
        select_carrot_ids=torch.tensor([0, 0]), select_plate_ids=torch.tensor([0, 0]),
        episode_stats={}, extra_stats=positions,
    )
    wrapper.env = SimpleNamespace(unwrapped=task)

    def execute(action):
        positions["extra_pos_gripper"] = torch.tensor([[0.0, 0.0, 0.05], [0.2, 0.0, 0.1]])
        grasped[:] = torch.tensor([True, False])
        root_quaternions[:] = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])
        return None, None, torch.tensor([[True], [False]]), {"success": torch.tensor([True, False])}

    wrapper.step = execute
    actions = torch.tensor([[31800] * 7, [31900] * 7])
    _, _, done, _, records = wrapper.step_with_outcome_feedback(actions, reset_after_step=True, include_contact=True)
    assert done[:, 0].tolist() == [True, False]
    assert [(record.before.episode_id, record.before.decision_index) for record in records] == [
        ("100:reset0:env0", 3), ("200:reset0:env1", 5),
    ]
    assert [record.grasp_transition for record in records] == ["established", "lost"]
    assert [record.task_success for record in records] == [True, False]
    assert [record.rollout_interrupted for record in records] == [False, True]
    assert [record.env_index for record in records] == [0, 1]
    assert [record.action_token_ids for record in records] == [(31800,) * 7, (31900,) * 7]
    assert [record.root_quaternion_wxyz for record in records] == [(1.0, 0.0, 0.0, 0.0)] * 2
    assert wrapper.decision_index == [4, 6]


def test_outcome_full_reset_updates_all_environments_after_one_finishes():
    from simpler_env.env.simpler_wrapper import SimlerWrapper

    wrapper = object.__new__(SimlerWrapper)
    wrapper.num_envs = 2
    wrapper.episode_ids = [100, 200]
    wrapper.decision_index = [4, 6]
    wrapper.reset_generation = 3
    wrapper.reward_old = torch.tensor([[0.7], [0.9]])
    task = SimpleNamespace(
        device=torch.device("cpu"), _episode_seed=[100, 200],
        get_language_instruction=lambda: ["new first", "new second"],
    )

    def reset(options):
        assert options["obj_set"] == "train"
        assert "env_idx" not in options
        task._episode_seed = [300, 400]
        return {"sensor_data": {"3rd_view_camera": {"rgb": torch.ones((2, 2, 2, 3))}}}, {}

    wrapper.env = SimpleNamespace(unwrapped=task, reset=reset)
    image, instruction, _ = wrapper.reset(obj_set="train")
    assert image.dtype == torch.uint8
    assert instruction == ["new first", "new second"]
    assert wrapper.episode_ids == [300, 400]
    assert wrapper.decision_index == [0, 0]
    assert wrapper.reset_generation == 4
    assert np.allclose(wrapper.reward_old[:, 0].tolist(), [0.0, 0.0])


def test_outcome_sdpo_uses_original_actions_and_one_update_per_two_minibatches():
    student_logits = torch.nn.Parameter(torch.tensor([0.5, -0.5]))
    teacher_logits = torch.nn.Parameter(torch.tensor([-0.5, 0.5]), requires_grad=False)
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
        seen.append((actions.clone(), observation["task_description"]))
        return student_logits.expand(actions.shape[0], actions.shape[1], 2)

    def teacher_forward(observation, actions, feedback):
        assert feedback == ["observed result"] * actions.shape[0]
        assert torch.equal(actions, seen[-1][0])
        return teacher_logits.expand(actions.shape[0], actions.shape[1], 2)

    policy = SimpleNamespace(
        tpdv={"device": torch.device("cpu")}, tpdv_vn={"device": torch.device("cpu")},
        params_vla=[student_logits],
        vla_optimizer=optimizer, evaluate_action_logits=student_forward,
        evaluate_teacher_action_logits=teacher_forward,
        update_teacher_ema=lambda rate: ema_updates.append(rate),
    )
    trainer = OpenVLAPPO(SimpleNamespace(
        alg_entropy_coef=0.0, alg_gradient_accum=2, sdpo_feedback_mode="correct",
        sdpo_ema_rate=0.01,
    ), policy)
    batch = (
        np.zeros((2, 2, 2, 3), dtype=np.uint8), ["task one", "task two"],
        np.array([[31800] * 7, [31900] * 7]), None, None, None, None, None,
        [SimpleNamespace(text="observed result")] * 2,
    )
    first = trainer.train_sdpo_step(0, 2, batch)
    assert first["optimizer_steps"] == 0.0
    assert optimizer.steps == 0
    second = trainer.train_sdpo_step(1, 2, batch)
    assert second["optimizer_steps"] == 1.0
    assert optimizer.steps == 1
    assert ema_updates == [0.01]
    assert teacher_logits.grad is None


def test_reverse_kl_masks_tokens_and_only_student_gets_gradient():
    student = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True)
    teacher = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]], requires_grad=True)
    loss = reverse_kl_student_to_teacher(student, teacher, torch.tensor([[1, 0]], dtype=torch.bool))
    loss.backward()
    assert loss.item() > 0
    assert student.grad is not None
    assert teacher.grad is None


def test_reverse_kl_is_zero_for_identical_distributions():
    logits = torch.tensor([[[2.0, -1.0], [0.5, 0.5]]])
    loss = reverse_kl_student_to_teacher(logits, logits.clone(), torch.ones((1, 2), dtype=torch.bool))
    assert loss.item() < 1e-7


def test_reverse_kl_rejects_misaligned_shapes():
    student = torch.zeros((1, 2, 3))
    teacher = torch.zeros((1, 1, 3))
    try:
        reverse_kl_student_to_teacher(student, teacher, torch.ones((1, 2), dtype=torch.bool))
    except ValueError as error:
        assert "identical aligned shapes" in str(error)
    else:
        raise AssertionError("misaligned student and teacher logits must be rejected")


def test_completed_episode_metrics_include_only_done_environments():
    info = {
        "success": torch.tensor([0.0, 1.0, 0.0]),
        "is_src_obj_grasped": torch.tensor([1.0, 1.0, 0.0]),
        "consecutive_grasp": torch.tensor([0.5, 1.0, 0.0]),
    }
    completed_episode_info(info, torch.tensor([[False], [True], [True]]))
    assert info["episode_env_ids"] == [1, 2]
    assert info["episode"]["success"] == [1.0, 0.0]
    assert info["episode"]["is_src_obj_grasped"] == [1.0, 0.0]


def test_training_evaluation_aggregates_all_requested_episode_batches():
    runner = object.__new__(Runner)
    runner.args = SimpleNamespace(eval_episodes=6, num_envs=2)
    calls = []

    def fake_eval(obj_set, episode_offset=0):
        calls.append((obj_set, episode_offset))
        return {"success": episode_offset / 2}

    runner.eval = fake_eval
    stats = runner.eval_over_episodes("test")

    assert calls == [("test", 0), ("test", 2), ("test", 4)]
    assert stats == {"success": 1.0}


def test_eval_only_preserves_stratified_episode_ids(tmp_path, monkeypatch):
    manifest_path = Path(__file__).resolve().parents[1] / "scripts/sdpo_v2_eval_episodes.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runner = object.__new__(Runner)
    runner.args = SimpleNamespace(
        eval_episodes=60, num_envs=6, eval_obj_set="test", eval_episode_start=0,
        eval_episode_manifest=str(manifest_path), eval_episode_split="validation",
        env_id=manifest["env_id"],
    )
    runner.save_dir = tmp_path
    runner.policy = SimpleNamespace(prep_rollout=lambda: None)
    runner.eval = lambda obj_set, return_episode_stats, selected_episode_ids: (
        {"success": 0.0}, [{"episode_id": episode_id} for episode_id in selected_episode_ids],
    )
    monkeypatch.setattr("simpler_env.train_ms3_ppo.wandb.log", lambda *args, **kwargs: None)
    monkeypatch.setattr("simpler_env.train_ms3_ppo.wandb.save", lambda *args, **kwargs: None)

    runner.run_eval_only()

    batches = [json.loads(line) for line in (tmp_path / "eval_only.jsonl").read_text().splitlines()]
    assert [episode_id for batch in batches for episode_id in batch["episode_ids"]] == manifest["validation"]
    assert all(len(batch["episode_stats"]) == 6 for batch in batches)


def test_eval_transport_metrics_use_live_grasp_not_accumulated_episode_flag():
    runner = object.__new__(Runner)
    runner.args = SimpleNamespace(seed=0, num_envs=1, episode_len=3)
    runner.device = torch.device("cpu")
    runner.policy = SimpleNamespace(prep_rollout=lambda: None)
    runner._get_action = lambda obs, deterministic: (
        torch.zeros((1, 1)), torch.zeros((1, 7)), torch.zeros((1, 1)),
    )

    class Environment:
        def __init__(self):
            self.env = SimpleNamespace(unwrapped=SimpleNamespace(device=torch.device("cpu")))
            self.step_index = 0

        def reset(self, obj_set, episode_ids):
            return torch.zeros((1, 8, 8, 3), dtype=torch.uint8), ["put carrot on plate"], {}

        def step(self, action):
            self.step_index += 1
            info = {
                "is_src_obj_grasped": torch.tensor([True]),
                "src_on_target": torch.tensor([self.step_index == 3]),
                "carrot_plate_dist": torch.tensor([0.2 - self.step_index * 0.05]),
                "success": torch.tensor([self.step_index == 3]),
            }
            if self.step_index == 3:
                info["episode_env_ids"] = [0]
                info["episode"] = {
                    "success": [1], "is_src_obj_grasped": [1], "consecutive_grasp": [0],
                }
            return torch.zeros((1, 8, 8, 3), dtype=torch.uint8), torch.zeros((1, 1)), torch.tensor([[self.step_index == 3]]), info

        def selected_source_grasped(self):
            return torch.tensor([self.step_index == 1])

    runner.env = Environment()
    _, episodes = runner.eval("test", return_episode_stats=True)
    record = episodes[0]
    assert record["first_grasp_step"] == 1
    assert record["carried_target_distance_reduction_m"] == 0.0
    assert record["reached_target_after_grasp"] is True
    assert record["released_on_target"] is True


def test_action_token_append_preserves_raw_ids():
    policy = object.__new__(OpenVLAPolicy)

    class Tokenizer:
        pad_token_id = 0
        bos_token_id = 1
        eos_token_id = 2

    policy.tokenizer = Tokenizer()
    prefix = BatchFeature(data={
        "input_ids": torch.tensor([[0, 1, 29871], [0, 1, 29871]]),
        "attention_mask": torch.tensor([[0, 1, 1], [0, 1, 1]]),
        "pixel_values": torch.zeros((2, 3, 2, 2)),
    })
    actions = torch.tensor([[31744, 31999], [31800, 31801]])
    result = policy._append_action_tokens(prefix, actions)
    assert result["input_ids"][0, -3:-1].tolist() == [31744, 31999]
    assert result["input_ids"][1, -3:-1].tolist() == [31800, 31801]
    assert result["input_ids"][0, -1].item() == 2


def test_action_token_append_keeps_variable_prefixes_left_padded():
    policy = object.__new__(OpenVLAPolicy)

    class Tokenizer:
        pad_token_id = 0
        bos_token_id = 1
        eos_token_id = 2

    policy.tokenizer = Tokenizer()
    prefix = BatchFeature(data={
        "input_ids": torch.tensor([[0, 0, 1, 10, 29871], [1, 20, 21, 22, 29871]]),
        "attention_mask": torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]]),
        "pixel_values": torch.zeros((2, 3, 2, 2)),
    })
    actions = torch.tensor([[31744, 31999], [31800, 31801]])

    result = policy._append_action_tokens(prefix, actions)

    assert result["input_ids"].tolist() == [
        [1, 0, 0, 10, 29871, 31744, 31999, 2],
        [1, 20, 21, 22, 29871, 31800, 31801, 2],
    ]
    assert result["attention_mask"].tolist() == [
        [1, 0, 0, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ]


def test_ema_updates_student_parameters_and_keeps_teacher_frozen():
    policy = object.__new__(OpenVLAPolicy)
    policy.vla = torch.nn.Linear(1, 1, bias=False)
    policy.teacher_vla = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        policy.vla.weight.fill_(2.0)
        policy.teacher_vla.weight.fill_(0.0)
    policy.teacher_vla.weight.requires_grad_(False)

    policy.update_teacher_ema(0.25)

    assert policy.teacher_vla.weight.item() == 0.5
    assert not policy.teacher_vla.weight.requires_grad


def test_rollout_minibatches_keep_feedback_actions_and_joint_logprobs_aligned():
    args = SimpleNamespace(episode_len=2, num_envs=2, buffer_gamma=0.99,
                           buffer_lambda=0.95, buffer_minibatch=2, alg_grpo_fix=False)
    buffer = SeparatedReplayBuffer(args, obs_dim=(1,), act_dim=7)
    buffer.warmup(np.array([[10], [20]], dtype=np.uint8), ["first", "second"])

    for step in range(2):
        records = [SimpleNamespace(timestamp_utc=f"{step}-{env}", text=f"{step}-{env}")
                   for env in range(2)]
        buffer.insert(
            obs=np.array([[11 + step], [21 + step]], dtype=np.uint8),
            actions=np.array([[31744 + step * 2 + env] * 7 for env in range(2)], dtype=np.int32),
            action_log_probs=np.array([[-step * 2 - env - 0.5] for env in range(2)], dtype=np.float32),
            value_preds=np.zeros((2, 1), dtype=np.float32),
            rewards=np.zeros((2, 1), dtype=np.float32),
            masks=np.ones((2, 1), dtype=np.float32),
            feedback=records,
        )
        if step == 0:
            buffer.set_next_observation(np.array([[15], [21]], dtype=np.uint8), ["third", "second"])

    seen = set()
    for batch in buffer.feed_forward_generator(include_feedback=True):
        observations, instructions, actions, _, _, _, old_logprobs, _, feedback = batch
        assert old_logprobs.shape == (2, 1)
        for index, record in enumerate(feedback):
            step, env = map(int, record.text.split("-"))
            assert actions[index, 0] == 31744 + step * 2 + env
            expected_observation = (15 if env == 0 else 21) if step else (10 if env == 0 else 20)
            expected_instruction = "third" if step and env == 0 else "first" if env == 0 else "second"
            assert observations[index, 0] == expected_observation
            assert instructions[index] == expected_instruction
            assert old_logprobs[index, 0] == -step * 2 - env - 0.5
            seen.add((step, env))
    assert seen == {(0, 0), (0, 1), (1, 0), (1, 1)}
