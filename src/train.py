"""Continual training over a stream of assays."""

import yaml
import torch
from .model import MolecularGNN
from .distill import spectral_distillation


def train_stream(cfg):
    model = MolecularGNN(tasks=cfg["tasks"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    teacher = MolecularGNN(tasks=cfg["tasks"])
    teacher.load_state_dict(model.state_dict())
    for assay in range(cfg["assays"]):
        atoms = torch.randint(0, 120, (4, 30))
        adjacency = torch.eye(30).repeat(4, 1, 1)
        target = torch.randn(4, cfg["tasks"])
        logits = model(atoms, adjacency)
        loss = (logits - target).pow(2).mean()
        loss = loss + cfg["lambda"] * spectral_distillation(
            model.head.weight.unsqueeze(0).expand(4, -1, -1),
            teacher.head.weight.unsqueeze(0).expand(4, -1, -1),
            adjacency,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"assay={assay} loss={loss.item():.4f}")


if __name__ == "__main__":
    import sys
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        train_stream(yaml.safe_load(handle))
