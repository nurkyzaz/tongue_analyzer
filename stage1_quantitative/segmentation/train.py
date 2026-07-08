"""Train U-Net++ tongue segmentation (RTDS recipe: ResNet-34 encoder, Dice+BCE loss).

Example:
    python stage1_quantitative/segmentation/train.py \
        --data-root data/raw --manifest data/processed/manifest.csv \
        --epochs 30 --batch-size 16 --img-size 384 --out checkpoints/seg
"""
import argparse
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from dataset import SegDataset, SegManifestDataset


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    inter = union = tp = fp = fn = 0.0
    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)
        pred = (model(img).sigmoid() > 0.5).float()
        inter += (pred * mask).sum().item()
        union += ((pred + mask) >= 1).float().sum().item()
        tp += (pred * mask).sum().item()
        fp += (pred * (1 - mask)).sum().item()
        fn += ((1 - pred) * mask).sum().item()
    iou = inter / (union + 1e-6)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
    return dice, iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="checkpoints/seg")
    ap.add_argument("--unified", action="store_true", help="use unified img_path/mask_path manifest")
    ap.add_argument("--max-steps", type=int, default=0, help="cap steps/epoch for smoke tests")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    val_sources = {}
    if args.unified:
        tr = SegManifestDataset(args.manifest, args.data_root, "train", args.img_size)
        va = SegManifestDataset(args.manifest, args.data_root, "val", args.img_size)
        # per-domain val loaders to watch the real-photo (sm_tongue) score specifically
        for s in ("tonguexpert", "sm_tongue"):
            ds = SegManifestDataset(args.manifest, args.data_root, "val", args.img_size, source=s)
            if len(ds):
                val_sources[s] = DataLoader(ds, args.batch_size, num_workers=args.workers, pin_memory=True)
    else:
        tr = SegDataset(args.manifest, args.data_root, "train", args.img_size)
        va = SegDataset(args.manifest, args.data_root, "val", args.img_size)
    tl = DataLoader(tr, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    vl = DataLoader(va, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    print(f"train={len(tr)} val={len(va)} device={device} encoder={args.encoder}")

    model = smp.UnetPlusPlus(encoder_name=args.encoder, encoder_weights="imagenet",
                             in_channels=3, classes=1).to(device)
    dice_loss = smp.losses.DiceLoss(mode="binary")
    bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()

    best = 0.0
    for ep in range(args.epochs):
        model.train()
        t0, run = time.time(), 0.0
        for step, (img, mask) in enumerate(tl):
            if args.max_steps and step >= args.max_steps:
                break
            img, mask = img.to(device), mask.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(img)
                loss = dice_loss(logits, mask) + bce(logits, mask)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += loss.item()
        sched.step()
        dice, iou = evaluate(model, vl, device)
        n = (step + 1)
        per_dom = "  ".join(f"{s}:{evaluate(model, dl, device)[0]:.4f}" for s, dl in val_sources.items())
        print(f"ep {ep+1:02d}/{args.epochs} loss={run/n:.4f} val_dice={dice:.4f} val_iou={iou:.4f} "
              f"[{per_dom}] ({time.time()-t0:.0f}s)", flush=True)
        if dice > best:
            best = dice
            torch.save({"model": model.state_dict(), "args": vars(args), "val_dice": dice},
                       os.path.join(args.out, "best.pt"))
    print(f"BEST val_dice={best:.4f} -> {args.out}/best.pt")


if __name__ == "__main__":
    main()
