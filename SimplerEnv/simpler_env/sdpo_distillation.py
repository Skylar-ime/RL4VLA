"""Model-agnostic SDPO reverse-KL primitive."""

from __future__ import annotations

import torch


def reverse_kl_student_to_teacher(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                                  valid_mask: torch.Tensor) -> torch.Tensor:
    """Mean KL(student || stop-gradient teacher) on aligned action tokens."""
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have identical aligned shapes")
    if valid_mask.shape != student_logits.shape[:-1]:
        raise ValueError("valid_mask must match batch and token dimensions")
    teacher_log_probs = torch.log_softmax(teacher_logits.detach(), dim=-1)
    student_log_probs = torch.log_softmax(student_logits, dim=-1)
    token_kl = (student_log_probs.exp() * (student_log_probs - teacher_log_probs)).sum(dim=-1)
    mask = valid_mask.to(dtype=token_kl.dtype)
    return (token_kl * mask).sum() / mask.sum().clamp_min(1.0)
