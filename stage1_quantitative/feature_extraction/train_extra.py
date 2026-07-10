"""Train the extra-features multi-label model on TCM-Tongue (Phase 4).

    python stage1_quantitative/feature_extraction/train_extra.py \
        --data-root . --labels data/processed/tcm_tongue_labels.csv \
        --mask-dir data/external/tcm_tongue/masks --epochs 25 --out checkpoints/extra_features
"""
import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, average_precision_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from labels import EXTRA_FEATURES
from feature_extraction.extra_dataset import ExtraFeaturesDataset
from feature_extraction.extra_model import ExtraFeaturesNet


def pos_weight(ds, device):
    y = np.stack([ [float(ds.df.iloc[i][c]) for c in EXTRA_FEATURES] for i in range(len(ds)) ])
    pos = y.sum(0); neg = len(y) - pos
    w = np.clip(neg / np.clip(pos, 1, None), 1.0, 15.0)   # up-weight rare positives, capped
    return torch.tensor(w, dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); P, G = [], []
    for img, mask, y in loader:
        p = torch.sigmoid(model(img.to(device), mask.to(device))).cpu().numpy()
        P.append(p); G.append(y.numpy())
    P, G = np.concatenate(P), np.concatenate(G)
    aps, f1s = [], []
    for k in range(len(EXTRA_FEATURES)):
        aps.append(average_precision_score(G[:, k], P[:, k]) if G[:, k].sum() else 0.0)
        f1s.append(f1_score(G[:, k], P[:, k] > 0.5, zero_division=0))
    return np.array(aps), np.array(f1s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=".")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--mask-dir", required=True)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="checkpoints/extra_features")
    ap.add_argument("--max-steps", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tr = ExtraFeaturesDataset(args.labels, args.data_root, args.mask_dir, "train", args.img_size)
    va = ExtraFeaturesDataset(args.labels, args.data_root, args.mask_dir, "val", args.img_size)
    tl = DataLoader(tr, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    vl = DataLoader(va, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    print(f"train={len(tr)} val={len(va)} device={device}", flush=True)

    model = ExtraFeaturesNet(args.encoder, pretrained=True).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight(tr, device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()

    best = 0.0
    for ep in range(args.epochs):
        model.train(); t0, run = time.time(), 0.0
        for step, (img, mask, y) in enumerate(tl):
            if args.max_steps and step >= args.max_steps:
                break
            img, mask, y = img.to(device), mask.to(device), y.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                loss = crit(model(img, mask), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run += loss.item()
        sched.step()
        aps, f1s = evaluate(model, vl, device)
        mAP = aps.mean()
        print(f"ep {ep+1:02d}/{args.epochs} loss={run/(step+1):.3f} mAP={mAP:.3f} meanF1={f1s.mean():.3f} | "
              + " ".join(f"{EXTRA_FEATURES[k][:6]}:{aps[k]:.2f}" for k in range(len(EXTRA_FEATURES)))
              + f" ({time.time()-t0:.0f}s)", flush=True)
        if mAP > best:
            best = mAP
            torch.save({"model": model.state_dict(), "args": vars(args), "mAP": float(mAP)},
                       os.path.join(args.out, "best.pt"))
    print(f"BEST mAP={best:.3f} -> {args.out}/best.pt")


if __name__ == "__main__":
    main()
