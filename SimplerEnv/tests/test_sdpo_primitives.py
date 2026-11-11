import torch
import numpy as np
from types import SimpleNamespace
from transformers import BatchFeature

from simpler_env.feedback import build_rel_spatial_feedback
from simpler_env.env.simpler_wrapper import completed_episode_info
from simpler_env.policies.openvla.openvla_train import OpenVLAPolicy
from simpler_env.sdpo_distillation import reverse_kl_student_to_teacher
from simpler_env.utils.replay_buffer import SeparatedReplayBuffer


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

    seen = set()
    for batch in buffer.feed_forward_generator(include_feedback=True):
        observations, instructions, actions, _, _, _, old_logprobs, _, feedback = batch
        assert old_logprobs.shape == (2, 1)
        for index, record in enumerate(feedback):
            step, env = map(int, record.text.split("-"))
            assert actions[index, 0] == 31744 + step * 2 + env
            assert observations[index, 0] == (10 if env == 0 else 20) + step
            assert instructions[index] == ("first" if env == 0 else "second")
            assert old_logprobs[index, 0] == -step * 2 - env - 0.5
            seen.add((step, env))
    assert seen == {(0, 0), (0, 1), (1, 0), (1, 1)}
