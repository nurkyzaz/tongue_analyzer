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
from labels import KEY_CHARS, CLASS_TO_IDX

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
                A.HueSaturationValue(15, 25, 15, p=0.5),
                A.RandomGamma((70, 130), p=0.4),
                A.ImageCompression(quality_lower=40, quality_upper=95, p=0.3)]
    aug += [A.Normalize(IMAGENET_MEAN, IMAGENET_STD), ToTensorV2()]
    return A.Compose(aug)


class MultiTaskDataset(Dataset):
    def __init__(self, manifest_csv, data_root, split, size=384):
        df = pd.read_csv(manifest_csv)
        self.df = df[(df.split == split) & (df.has_mask)].reset_index(drop=True)
        self.root = data_root
        self.tf = _tf(size, split == "train")

    def __len__(self):
        return len(self.df)

    def _target(self, row):
        """Return (label_idx[5], weight[5]) using gold label when available."""
        y = np.zeros(len(KEY_CHARS), dtype=np.int64)
        w = np.ones(len(KEY_CHARS), dtype=np.float32)
        for k, ch in enumerate(KEY_CHARS):
            val, gold = row.get(ch), None
            if ch in MANUAL_CHARS:
                mv = row.get(f"{ch}_manual")
                if isinstance(mv, str) and mv in CLASS_TO_IDX[ch]:
                    gold, w[k] = mv, 2.0
            val = gold if gold is not None else val
            y[k] = CLASS_TO_IDX[ch].get(val, 0) if isinstance(val, str) else 0
        return y, w

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = cv2.cvtColor(cv2.imread(os.path.join(self.root, row.raw_path)), cv2.COLOR_BGR2RGB)
        mask = (cv2.imread(os.path.join(self.root, row.mask_path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)
        out = self.tf(image=img, mask=mask)
        y, w = self._target(row)
        return out["image"], out["mask"].unsqueeze(0).float(), torch.from_numpy(y), torch.from_numpy(w)
