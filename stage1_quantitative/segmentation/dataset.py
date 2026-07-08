"""Segmentation dataset: Raw image -> binary tongue mask, with real-world augmentation.

Augmentations deliberately simulate smartphone-photo conditions (lighting, color cast, blur,
JPEG artifacts, pose) to close the domain gap from clinical images to app uploads.
"""
import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transforms(size):
    return A.Compose([
        A.LongestMaxSize(max_size=size),
        A.PadIfNeeded(size, size, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.15, rotate_limit=20,
                           border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.7),
        A.RandomBrightnessContrast(0.3, 0.3, p=0.6),
        A.HueSaturationValue(15, 25, 15, p=0.5),
        A.RandomGamma((70, 130), p=0.4),
        A.OneOf([A.GaussianBlur(blur_limit=(3, 7)), A.MotionBlur(blur_limit=7)], p=0.3),
        A.ImageCompression(quality_lower=40, quality_upper=90, p=0.4),
        A.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ToTensorV2(),
    ])


def eval_transforms(size):
    return A.Compose([
        A.LongestMaxSize(max_size=size),
        A.PadIfNeeded(size, size, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
        A.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ToTensorV2(),
    ])


class SegDataset(Dataset):
    def __init__(self, manifest_csv, data_root, split, size=384):
        df = pd.read_csv(manifest_csv)
        self.df = df[(df.split == split) & (df.has_mask)].reset_index(drop=True)
        self.root = data_root
        self.tf = train_transforms(size) if split == "train" else eval_transforms(size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = cv2.cvtColor(cv2.imread(os.path.join(self.root, row.raw_path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.root, row.mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)
        out = self.tf(image=img, mask=mask)
        return out["image"], out["mask"].unsqueeze(0).float()
