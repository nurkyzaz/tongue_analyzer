"""Precompute tongue masks for a set of images using the trained segmentation model.

TCM-Tongue has no masks, and the feature model uses mask-guided pooling, so we generate masks with
seg_combined (real-photo Dice ~0.97) once and cache them. Mask filename mirrors the image stem.

    python scripts/precompute_masks.py --labels data/processed/tcm_tongue_labels.csv \
        --out data/external/tcm_tongue/masks --seg checkpoints/seg_combined/best.pt
"""
import argparse
import os
import cv2
import numpy as np
import pandas as pd
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seg", default="checkpoints/seg_combined/best.pt")
    ap.add_argument("--size", type=int, default=384)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    st = torch.load(args.seg, map_location=device, weights_only=False)
    model = smp.UnetPlusPlus(encoder_name=st["args"].get("encoder", "resnet34"),
                             encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(st["model"]); model.to(device).eval()
    tf = A.Compose([A.LongestMaxSize(args.size),
                    A.PadIfNeeded(args.size, args.size, border_mode=cv2.BORDER_CONSTANT, value=0),
                    A.Normalize(MEAN, STD), ToTensorV2()])

    df = pd.read_csv(args.labels)
    os.makedirs(args.out, exist_ok=True)
    done = 0
    with torch.no_grad():
        for _, row in df.iterrows():
            stem = os.path.splitext(os.path.basename(row.img_path))[0]
            dst = os.path.join(args.out, stem + ".png")
            if os.path.exists(dst):
                done += 1; continue
            img = cv2.imread(row.img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            x = tf(image=img)["image"].unsqueeze(0).to(device)
            m = (model(x).sigmoid()[0, 0] > 0.5).cpu().numpy().astype(np.uint8)
            # letterbox geometry: undo pad+resize -> recover mask at ORIGINAL image size so the
            # dataset's joint image+mask transform aligns them (like the TonguExpert masks).
            s = args.size / max(h, w)
            nh, nw = round(h * s), round(w * s)
            pt, pl = (args.size - nh) // 2, (args.size - nw) // 2
            m = m[pt:pt + nh, pl:pl + nw]
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(dst, m * 255)
            done += 1
            if done % 1000 == 0:
                print(f"  {done}/{len(df)}", flush=True)
    print(f"Done: {done} masks -> {args.out}")


if __name__ == "__main__":
    main()
