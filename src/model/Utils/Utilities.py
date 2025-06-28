# src/model/Utilities/Utilities.py
from __future__ import annotations

import torch
import torch.nn as nn


class Utilities:
    """
    Aide-mémoire réduit : uniquement ce qu’il faut pour TabularMLP.
    """

    @staticmethod
    def get_activation(name: str | None) -> nn.Module | None:
        """
        Convertit une chaîne ('relu', 'sigmoid', 'tanh', 'linear') en module PyTorch.
        Retourne `None` pour 'linear' ou `None`.
        """
        name = (name or "linear").lower()
        match name:
            case "relu":
                return nn.ReLU()
            case "sigmoid":
                return nn.Sigmoid()
            case "tanh":
                return nn.Tanh()
            case "linear" | _:
                return None

    # Facultatif : petite fonction d’accuracy si besoin ailleurs.
    @staticmethod
    def compute_accuracy(y_true, y_pred_logits) -> float:
        """
        Exactitude top-1 pour classification multiclasse ou binaire logits.
        """
        if not isinstance(y_true, torch.Tensor):
            y_true = torch.as_tensor(y_true)
        if not isinstance(y_pred_logits, torch.Tensor):
            y_pred_logits = torch.as_tensor(y_pred_logits)

        if y_pred_logits.dim() > 1 and y_pred_logits.size(1) > 1:
            _, preds = torch.max(y_pred_logits, 1)
        else:  # sigmoid/binary logits
            preds = (torch.sigmoid(y_pred_logits) >= 0.5).long().view(-1)

        correct = (preds == y_true.view(-1)).sum().item()
        return correct / y_true.numel() * 100.0
