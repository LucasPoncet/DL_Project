from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader, random_split

from ClassesData.WineDataset import WineDataset


class WineDataModule:
    """DataLoader helper (inspiré du DataModule PyTorch-Lightning)."""

    def __init__(
        self,
        parquet_path: str | Path,
        cont_cols: Sequence[str],
        cat_cols: Sequence[str],
        batch_size: int = 256,
        val_split: float = 0.1,
        test_split: float = 0.1,
        num_workers: int = 0,
        seed: int = 42,
    ) -> None:
        full_ds = WineDataset(parquet_path, cont_cols, cat_cols)
        n_total = len(full_ds)
        n_val = int(n_total * val_split)
        n_test = int(n_total * test_split)
        n_train = n_total - n_val - n_test

        self.train_ds, self.val_ds, self.test_ds = random_split(
            full_ds,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(seed),
        )

        self.batch_size = batch_size
        self.num_workers = num_workers

    # ------------------------------------------------------------------ #
    def _loader(self, ds, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds, shuffle=False)
