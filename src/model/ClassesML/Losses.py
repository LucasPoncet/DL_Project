import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    alpha: Balances the importance of positive/negative examples.
    gamma > 0: Reduces the loss for well-classified examples, focusing on hard examples.
    """
    def __init__(self, alpha: float = 0.8, gamma: float = 2.0, reduction="mean"):
        super().__init__()
        self.alpha = torch.tensor([1.0, alpha])   # poids classe 0 / classe 1
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, target):
        ce_loss = self.ce(logits, target)         # [batch]
        pt = torch.exp(-ce_loss)                  # proba attribuée à la bonne classe
        at = self.alpha.to(logits.device)[target] # α_t
        loss = at * (1.0 - pt) ** self.gamma * ce_loss
        return loss.mean() if self.reduction == "mean" else loss.sum()
