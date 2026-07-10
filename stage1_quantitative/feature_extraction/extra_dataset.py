"""Dataset for the extra multi-label features (TCM-Tongue): image + precomputed mask -> 8 binary labels."""
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
from labels import EXTRA_FEATURES

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def _tf(size, train):
    aug = [A.LongestMaxSize(size), A.PadIfNeeded(size, size, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0)]
    if train:
        aug += [A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(0.08, 0.12, 15, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.6),
                A.RandomBrightnessContrast(0.25, 0.25, p=0.5),
                A.HueSaturationValue(12, 18, 12, p=0.4),
                A.ImageCompression(quality_lower=45, quality_upper=95, p=0.3)]
    aug += [A.Normalize(MEAN, STD), ToTensorV2()]
    return A.Compose(aug)


class ExtraFeaturesDataset(Dataset):
    def __init__(self, labels_csv, data_root, mask_dir, split, size=384):
        df = pd.read_csv(labels_csv)
        self.df = df[df.split == split].reset_index(drop=True)
        self.root = data_root
        self.mask_dir = mask_dir
        self.tf = _tf(size, split == "train")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = cv2.cvtColor(cv2.imread(os.path.join(self.root, row.img_path)), cv2.COLOR_BGR2RGB)
        stem = os.path.splitext(os.path.basename(row.img_path))[0]
        mp = os.path.join(self.mask_dir, stem + ".png")
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        mask = (m > 127).astype(np.float32) if m is not None else np.ones(img.shape[:2], np.float32)
        out = self.tf(image=img, mask=mask)
        y = torch.tensor([float(row[c]) for c in EXTRA_FEATURES], dtype=torch.float32)
        return out["image"], out["mask"].unsqueeze(0).float(), y
