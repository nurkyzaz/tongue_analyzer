"""Evaluate a segmentation checkpoint on an images+masks folder (e.g. SM-Tongue real photos).

Measures the real-world domain gap for a model trained elsewhere (e.g. TonguExpert).

    python evaluation/eval_seg.py --seg checkpoints/seg/best.pt \
        --images-dir data/external/sm_tongue/SM-Tongue-2155-anonymized/images \
        --masks-dir  data/external/sm_tongue/SM-Tongue-2155-anonymized/masks
"""
import argparse
import os
import glob
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def load_seg(ckpt, device):
    st = torch.load(ckpt, map_location=device, weights_only=False)
    enc = st["args"].get("encoder", "resnet34")
    m = smp.UnetPlusPlus(encoder_name=enc, encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(st["model"]); m.to(device).eval()
    return m


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--masks-dir", required=True)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_seg(args.seg, device)
    tf = A.Compose([A.LongestMaxSize(max_size=args.size),
                    A.PadIfNeeded(args.size, args.size, border_mode=cv2.BORDER_CONSTANT, value=0),
                    A.Normalize(MEAN, STD), ToTensorV2()])

    imgs = sorted(glob.glob(os.path.join(args.images_dir, "*")))
    if args.limit:
        imgs = imgs[:args.limit]
    dices, ious = [], []
    for ip in imgs:
        name = os.path.splitext(os.path.basename(ip))[0]
        mp = None
        for ext in (".png", ".jpg", ".jpeg"):
            cand = os.path.join(args.masks_dir, name + ext)
            if os.path.exists(cand):
                mp = cand; break
        if mp is None:
            continue
        img = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
        gt = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        x = tf(image=img)["image"].unsqueeze(0).to(device)
        pr = (model(x).sigmoid()[0, 0] > 0.5).cpu().numpy().astype(np.uint8)
        # resize prediction back to gt size for a fair (letterbox-aware) comparison
        pr = cv2.resize(pr, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        g = (gt > 127).astype(np.uint8)
        inter = (pr & g).sum(); uni = (pr | g).sum()
        tp = inter; fp = (pr & (1 - g)).sum(); fn = ((1 - pr) & g).sum()
        dices.append(2 * tp / (2 * tp + fp + fn + 1e-6))
        ious.append(inter / (uni + 1e-6))
    print(f"n={len(dices)}  mean Dice={np.mean(dices):.4f}  mean IoU={np.mean(ious):.4f}  "
          f"Dice<0.9: {np.mean(np.array(dices) < 0.9)*100:.1f}%")


if __name__ == "__main__":
    main()
