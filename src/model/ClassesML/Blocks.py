# src/model/ClassesML/Blocks.py
from __future__ import annotations

import torch.nn as nn


class DenseBlock(nn.Module):
    """
    Bloc fully-connected minimal : Linear → (BatchNorm) → Activation → Dropout.

    Conçu pour être ré-utilisé dans `TabularMLP` sans dépendre d’autres modules.
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        activation: nn.Module | None,
        batch_normalization: bool = False,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = [nn.Linear(in_size, out_size)]

        if batch_normalization:
            layers.append(nn.BatchNorm1d(out_size))

        if activation is not None:
            layers.append(activation)

        layers.append(nn.Dropout(dropout_rate))

        self.block = nn.Sequential(*layers)

    # ------------------------------------------------------------------ #
    def forward(self, x):
        return self.block(x)
