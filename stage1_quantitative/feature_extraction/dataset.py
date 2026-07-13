"""Multi-task dataset: image + tongue mask -> 5 categorical characteristic labels.

Target per characteristic = manual (gold) label when available, else L2 (weak) label. A per-sample
weight upweights gold-derived targets. Ground-truth masks are used during training (at inference the
segmentation model provides the mask).
"""
import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from labels import KEY_CHARS, CLASS_TO_IDX, SEVERITY_KEYS

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MANUAL_CHARS = {"tai", "zhi", "fissure", "tooth_mk"}  # coating has no manual column


def _tf(size, train):
    aug = [A.LongestMaxSize(max_size=size),
           A.PadIfNeeded(size, size, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0)]
    if train:
        aug += [A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(0.08, 0.15, 20, border_mode=cv2.BORDER_CONSTANT,
                                   value=0, mask_value=0, p=0.7),
                A.RandomBrightnessContrast(0.3, 0.3, p=0.6),
                # white-balance robustness: MILD per-channel shifts simulate illuminant/colour-temperature
                # casts, WITHOUT rotating hue (which would corrupt the tai/zhi colour labels themselves).
                A.RGBShift(r_shift_limit=14, g_shift_limit=10, b_shift_limit=14, p=0.5),
                A.HueSaturationValue(hue_shift_limit=6, sat_shift_limit=18, val_shift_limit=12, p=0.4),
                A.RandomGamma((80, 120), p=0.4),
                A.ImageCompression(quality_lower=40, quality_upper=95, p=0.3)]
    aug += [A.Normalize(IMAGENET_MEAN, IMAGENET_STD), ToTensorV2()]
    return A.Compose(aug)


class MultiTaskDataset(Dataset):
    def __init__(self, manifest_csv, data_root, split, size=384, severity_csv=None):
        df = pd.read_csv(manifest_csv)
        df = df[(df.split == split) & (df.has_mask)].reset_index(drop=True)
        self.sev_cols = list(SEVERITY_KEYS)
        if severity_csv and os.path.exists(severity_csv):
            sev = pd.read_csv(severity_csv)
            df = df.merge(sev, on="sid", how="left")
        for c in self.sev_cols:
            if c not in df:
                df[c] = 0.0
        df[self.sev_cols] = df[self.sev_cols].fillna(0.0)
        self.df = df
        self.root = data_root
        self.tf = _tf(size, split == "train")

    def __len__(self):
        return len(self.df)

    def _target(self, row):
        """Return (label_idx[5], weight[5]). Weight encodes supervision: 2.0 expert-manual gold, 1.0
        auto label, **0.0 when a characteristic is unlabeled** (e.g. partial-label TCM-Tongue rows) so
        an unknown characteristic never injects a spurious class-0 target."""
        y = np.zeros(len(KEY_CHARS), dtype=np.int64)
        w = np.zeros(len(KEY_CHARS), dtype=np.float32)
        for k, ch in enumerate(KEY_CHARS):
            val = None
            if ch in MANUAL_CHARS:
                mv = row.get(f"{ch}_manual")
                if isinstance(mv, str) and mv in CLASS_TO_IDX[ch]:
                    val, w[k] = mv, 2.0
            if val is None:
                av = row.get(ch)
                if isinstance(av, str) and av in CLASS_TO_IDX[ch]:
                    val, w[k] = av, 1.0
            y[k] = CLASS_TO_IDX[ch][val] if val is not None else 0
        return y, w

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = cv2.cvtColor(cv2.imread(os.path.join(self.root, row.raw_path)), cv2.COLOR_BGR2RGB)
        mask = (cv2.imread(os.path.join(self.root, row.mask_path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)
        out = self.tf(image=img, mask=mask)
        y, w = self._target(row)
        # TCM-Tongue rows carry no severity annotation -> NaN so the regression loss can mask them out
        # (they'd otherwise be trained toward 0, corrupting fissure/tooth-mark severity on cracked tongues).
        if row.get("src") == "tcm_tongue":
            sev = torch.full((len(self.sev_cols),), float("nan"), dtype=torch.float32)
        else:
            sev = torch.tensor([row[c] for c in self.sev_cols], dtype=torch.float32)
        return out["image"], out["mask"].unsqueeze(0).float(), torch.from_numpy(y), torch.from_numpy(w), sev
