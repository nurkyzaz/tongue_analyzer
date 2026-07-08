"""Multiclass Focal Loss (RTDS uses Focal Loss for class imbalance / label ambiguity)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, class_weight=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("class_weight", class_weight if class_weight is not None else None)

    def forward(self, logits, target, sample_weight=None):
        logp = F.log_softmax(logits, dim=1)
        ce = F.nll_loss(logp, target, weight=self.class_weight, reduction="none")
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        if sample_weight is not None:
            loss = loss * sample_weight
        return loss.mean()
