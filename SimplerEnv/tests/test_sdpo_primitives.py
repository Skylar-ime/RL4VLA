import torch

from simpler_env.feedback import build_rel_spatial_feedback
from simpler_env.sdpo_distillation import reverse_kl_student_to_teacher


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
