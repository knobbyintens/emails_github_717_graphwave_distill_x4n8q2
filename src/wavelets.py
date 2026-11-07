"""Graph wavelet transforms used by the distillation objective."""

import torch


def normalized_laplacian(adjacency: torch.Tensor) -> torch.Tensor:
    degree = adjacency.sum(dim=-1).clamp(min=1e-6)
    inv_sqrt = degree.pow(-0.5)
    return torch.eye(adjacency.size(-1), device=adjacency.device) - (
        inv_sqrt.unsqueeze(-1) * adjacency * inv_sqrt.unsqueeze(-2)
    )


def wavelet_basis(adjacency: torch.Tensor, scales: int = 3) -> torch.Tensor:
    lap = normalized_laplacian(adjacency)
    basis = [torch.eye(lap.size(-1), device=lap.device)]
    power = lap
    for _ in range(scales):
        basis.append(power)
        power = power @ lap
    return torch.stack(basis, dim=0)
