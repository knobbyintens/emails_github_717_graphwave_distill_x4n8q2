"""Spectral distillation objective for continual learning."""

import torch
import torch.nn.functional as F
from .wavelets import wavelet_basis


def spectral_distillation(student: torch.Tensor, teacher: torch.Tensor,
                          adjacency: torch.Tensor) -> torch.Tensor:
    basis = wavelet_basis(adjacency)
    student_coeff = torch.einsum("scd,kde->kce", student, basis)
    teacher_coeff = torch.einsum("scd,kde->kce", teacher, basis)
    low = F.mse_loss(student_coeff[0], teacher_coeff[0])
    high = F.mse_loss(student_coeff[1:], teacher_coeff[1:])
    return low + 0.5 * high
