from __future__ import annotations

import math
from pathlib import Path
from typing import List, Sequence

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset


class WineDataset(Dataset):
    """
    Dataset tabulaire pour la classification binaire `buy_flag`.

    Les données sont lues depuis un fichier **Parquet**.
    """

    def __init__(
        self,
        parquet_path: str | Path,
        cont_cols: Sequence[str],
        cat_cols: Sequence[str],
        label_col: str = "buy_flag",
    ) -> None:
        super().__init__()

        df = pd.read_parquet(parquet_path)
        self.cont_cols: List[str] = list(cont_cols)
        self.cat_cols: List[str] = list(cat_cols)

        self.x_cont: Tensor = torch.tensor(df[self.cont_cols].values, dtype=torch.float32)
        self.x_cat: Tensor = torch.tensor(df[self.cat_cols].values, dtype=torch.long)
        self.y: Tensor = torch.tensor(df[label_col].values, dtype=torch.float32).unsqueeze(1)

    # --------------------------------------------------------------------- #
    #                           Dataset interface                           #
    # --------------------------------------------------------------------- #

    def __len__(self) -> int:
        return self.y.size(0)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.x_cont[idx], self.x_cat[idx], self.y[idx]

    # --------------------------------------------------------------------- #
    #                       Helpers pour les embeddings                      #
    # --------------------------------------------------------------------- #

    def categorical_cardinalities(self) -> list[int]:
        """Cardinalité (max + 1) de chaque feature catégorielle."""
        card = []
        for i, col in enumerate(self.cat_cols):
            card.append(int(self.x_cat[:, i].max().item()) + 1)
        return card

    def default_embedding_dims(self) -> list[int]:
        """min(50, round(1.6 * √card)) pour chaque colonne."""
        return [
            max(4, min(50, round(1.6 * math.sqrt(c))))
            for c in self.categorical_cardinalities()
        ]
