# src/ClassesML/TabularMLP.py
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from ClassesML.Blocks import DenseBlock
from Utils.Utilities import Utilities


class TabularMLP(nn.Module):
    """Embeddings + MLP dense pour données tabulaires mixtes."""

    def __init__(self, hyperparams: dict) -> None:
        super().__init__()

        n_cont: int = hyperparams["n_cont"]
        cat_card: List[int] = hyperparams["cat_cardinalities"]
        emb_dims: List[int] = hyperparams["emb_dims"]

        hidden_sizes: List[int] = hyperparams["hidden_layers_size"]
        act = Utilities.get_activation(hyperparams["activation"])
        bn = hyperparams["batch_normalization"]
        drop = hyperparams["dropout_rate"]

        # --- Embeddings -------------------------------------------------- #
        self.embeddings = nn.ModuleList(
            [nn.Embedding(card, dim) for card, dim in zip(cat_card, emb_dims)]
        )
        input_dim = n_cont + sum(emb_dims)

        layers = [
            DenseBlock(
                in_size=input_dim,
                out_size=hidden_sizes[0],
                activation=act,
                batch_normalization=bn,
                dropout_rate=drop,
            )
        ]
        for i in range(len(hidden_sizes) - 1):
            layers.append(
                DenseBlock(
                    in_size=hidden_sizes[i],
                    out_size=hidden_sizes[i + 1],
                    activation=act,
                    batch_normalization=bn,
                    dropout_rate=drop,
                )
            )

        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_sizes[-1], 1)

    # ------------------------------------------------------------------ #
    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        emb = [emb_layer(x_cat[:, i]) for i, emb_layer in enumerate(self.embeddings)]
        x = torch.cat([x_cont] + emb, dim=1)
        x = self.mlp(x)
        return self.head(x)  # logit
