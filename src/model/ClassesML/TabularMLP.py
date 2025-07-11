# Replace the entire TabularMLP class:

from __future__ import annotations
from typing import List
import torch
import torch.nn as nn
from ClassesML.Blocks import DenseBlock
from Utils.Utilities import Utilities

class TabularMLP(nn.Module):
    """Ultra-simple MLP treating categorical as numerical."""

    def __init__(self, hyperparams: dict) -> None:
        super().__init__()

        n_cont: int = hyperparams["n_cont"]
        n_cat: int = len(hyperparams["cat_cardinalities"])
        
        # 🔧 ULTRA SIMPLE: Just basic linear layers
        input_dim = n_cont + n_cat
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        # 🔧 SIMPLE: Just concatenate and ensure no NaN
        x_cont = torch.nan_to_num(x_cont, nan=0.0)  # Replace NaN with 0
        x_cat = torch.nan_to_num(x_cat.float(), nan=0.0)  # Replace NaN with 0
        
        x = torch.cat([x_cont, x_cat], dim=1)
        return self.layers(x)