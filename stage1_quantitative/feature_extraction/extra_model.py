"""Extra-features network (Phase 4): 8 new multi-label tongue features from TCM-Tongue.

Same shared-encoder + mask-guided-pooling design as the main model, but a single multi-label head.
Kept as a SEPARATE model so it can't destabilise the working v3 (5 chars + severity); at inference both
run and their outputs merge. Returns raw logits [B, len(EXTRA_FEATURES)] (apply sigmoid for presence).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from labels import EXTRA_FEATURES


class ExtraFeaturesNet(nn.Module):
    def __init__(self, encoder="resnet34", pretrained=True, dropout=0.2):
        super().__init__()
        self.encoder = timm.create_model(encoder, pretrained=pretrained, features_only=True)
        c = self.encoder.feature_info.channels()[-1]
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(2 * c, len(EXTRA_FEATURES))

    def forward(self, x, mask):
        feat = self.encoder(x)[-1]
        m = F.interpolate(mask, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        m = (m > 0.5).float()
        denom = m.sum(dim=(2, 3)).clamp(min=1.0)
        masked_pool = (feat * m).sum(dim=(2, 3)) / denom
        global_pool = feat.mean(dim=(2, 3))
        z = self.dropout(torch.cat([masked_pool, global_pool], dim=1))
        return self.head(z)
