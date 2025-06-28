# src/ClassesML/TrainerTabular.py
from __future__ import annotations

import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Utils.MetricsTabular import accuracy, f1


class TrainerTabular:
    """Boucle d'entraînement inspirée de TrainerClassifier."""

    def __init__(self, hyperparams: dict) -> None:
        self.hp = hyperparams

    # ------------------------------------------------------------------ #
    def set_model(self, model: nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.model.to(device)

    def set_scope(self, optimizer, scheduler=None) -> None:
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = nn.BCEWithLogitsLoss()

    def set_data(self, train_loader: DataLoader, valid_loader: DataLoader) -> None:
        self.train_loader = train_loader
        self.valid_loader = valid_loader

    # ------------------------------------------------------------------ #
    def _epoch(
        self, loader: DataLoader, train: bool = False
    ) -> Tuple[float, float]:
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        preds, labels = [], []

        for x_cont, x_cat, y in loader:
            x_cont, x_cat, y = (
                x_cont.to(self.device),
                x_cat.to(self.device),
                y.to(self.device),
            )

            logits = self.model(x_cont, x_cat)
            loss = self.criterion(logits, y)

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item() * y.size(0)
            preds.append(torch.sigmoid(logits).detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        preds_np = np.concatenate(preds)
        labels_np = np.concatenate(labels)
        loss_avg = total_loss / len(loader.dataset) # type: ignore[arg-type]
        f1_score = f1(labels_np, preds_np)
        return loss_avg, f1_score

    # ------------------------------------------------------------------ #
    def run(self) -> Tuple[list[float], list[float]]:
        print("Starting training loop...")
        train_f1_hist, val_f1_hist = [], []
        best_f1 = 0.0

        for epoch in range(self.hp["max_epoch"]):
            t0 = time.time()
            train_loss, train_f1 = self._epoch(self.train_loader, train=True)
            val_loss, val_f1 = self._epoch(self.valid_loader, train=False)

            if self.scheduler:
                self.scheduler.step(val_f1)

            train_f1_hist.append(train_f1)
            val_f1_hist.append(val_f1)

            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(self.model.state_dict(), "best_model.pt")

            dt = time.time() - t0
            print(
                f"Epoch {epoch+1:03d}/{self.hp['max_epoch']} "
                f"- train_loss {train_loss:.4f}  train_f1 {train_f1:.3f} "
                f"| val_loss {val_loss:.4f}  val_f1 {val_f1:.3f} "
                f"({dt:.1f}s)"
            )

        return train_f1_hist, val_f1_hist
