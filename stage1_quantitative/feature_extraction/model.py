"""Multi-task tongue characteristic network (SSC-Net-inspired).

Shared encoder (timm) -> mask-guided feature masking (background suppressed) -> 5 classification
heads for the key characteristics. The mask is used to pool features only over the tongue region,
which is SSC-Net's core idea for suppressing redundant background. At training the ground-truth mask
is used; at inference the Stage-1 segmentation model supplies the mask.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from labels import KEY_CHARS, NUM_CLASSES


class MultiTaskTongueNet(nn.Module):
    def __init__(self, encoder="resnet34", pretrained=True, dropout=0.2):
        super().__init__()
        self.encoder = timm.create_model(encoder, pretrained=pretrained, features_only=True)
        c = self.encoder.feature_info.channels()[-1]  # channels of the last feature map
        self.dropout = nn.Dropout(dropout)
        # One linear head per characteristic. Input = [masked_pool ; global_pool] (2*c).
        self.heads = nn.ModuleDict(
            {ch: nn.Linear(2 * c, NUM_CLASSES[ch]) for ch in KEY_CHARS}
        )

    def forward(self, x, mask):
        feat = self.encoder(x)[-1]                      # [B,C,h,w]
        m = F.interpolate(mask, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        m = (m > 0.5).float()
        denom = m.sum(dim=(2, 3)).clamp(min=1.0)        # [B,1]
        masked_pool = (feat * m).sum(dim=(2, 3)) / denom            # tongue-region features
        global_pool = feat.mean(dim=(2, 3))                         # whole-image context
        z = self.dropout(torch.cat([masked_pool, global_pool], dim=1))
        return {ch: head(z) for ch, head in self.heads.items()}
