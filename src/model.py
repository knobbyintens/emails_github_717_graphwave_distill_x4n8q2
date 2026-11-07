"""Molecular property predictor."""

import torch
import torch.nn as nn


class MolecularGNN(nn.Module):
    def __init__(self, atom_dim: int = 64, hidden: int = 256, tasks: int = 12):
        super().__init__()
        self.atom_embedding = nn.Embedding(120, atom_dim)
        self.message = nn.Linear(atom_dim, atom_dim)
        self.head = nn.Linear(atom_dim, tasks)

    def forward(self, atom_ids: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        h = self.atom_embedding(atom_ids)
        h = self.message(adjacency @ h)
        return self.head(h.mean(dim=1))
