"""Train the multi-task characteristic head (5 key characteristics, Focal Loss, mask-guided pooling).

Example:
    python stage1_quantitative/feature_extraction/train.py \
        --data-root data/raw --manifest data/processed/manifest.csv \
        --epochs 30 --batch-size 32 --img-size 384 --out checkpoints/multitask
"""
import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from labels import KEY_CHARS, NUM_CLASSES, SEVERITY_KEYS
from feature_extraction.dataset import MultiTaskDataset
from feature_extraction.model import MultiTaskTongueNet
from feature_extraction.losses import FocalLoss


def class_weights(loader, device, mode="sqrt", clamp=(0.5, 4.0)):
    """Per-class weights. `sqrt` (mild inverse-sqrt-freq) avoids crushing an extreme majority class
    the way full inverse-freq does; combined with Focal Loss that is enough for imbalance.
    Weights are mean-normalised to ~1 and clamped so no class dominates or vanishes."""
    counts = {ch: torch.zeros(NUM_CLASSES[ch]) for ch in KEY_CHARS}
    for _img, _mask, y, _w, _sev in loader:
        for k, ch in enumerate(KEY_CHARS):
            for c in range(NUM_CLASSES[ch]):
                counts[ch][c] += (y[:, k] == c).sum()
    w = {}
    for ch in KEY_CHARS:
        n = counts[ch].clamp(min=1)
        if mode == "none":
            raw = torch.ones_like(n)
        elif mode == "inv":
            raw = 1.0 / n
        else:  # sqrt
            raw = 1.0 / n.sqrt()
        raw = raw / raw.mean()                      # centre around 1.0
        w[ch] = raw.clamp(clamp[0], clamp[1]).to(device)
    return w


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds = {ch: [] for ch in KEY_CHARS}
    gts = {ch: [] for ch in KEY_CHARS}
    sev_abs, n_sev = 0.0, 0
    for img, mask, y, _, sev in loader:
        out = model(img.to(device), mask.to(device))
        for k, ch in enumerate(KEY_CHARS):
            preds[ch].append(out[ch].argmax(1).cpu().numpy())
            gts[ch].append(y[:, k].numpy())
        sev_abs += (out["severity"].cpu() - sev).abs().sum().item()
        n_sev += sev.numel()
    res = {}
    for ch in KEY_CHARS:
        p, g = np.concatenate(preds[ch]), np.concatenate(gts[ch])
        res[ch] = (accuracy_score(g, p), f1_score(g, p, average="macro", zero_division=0))
    return res, sev_abs / max(n_sev, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default="checkpoints/multitask")
    ap.add_argument("--cw-mode", default="sqrt", choices=["none", "inv", "sqrt"])
    ap.add_argument("--severity-csv", default="data/processed/severity.csv")
    ap.add_argument("--sev-weight", type=float, default=5.0, help="weight for severity regression loss")
    ap.add_argument("--max-steps", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tr = MultiTaskDataset(args.manifest, args.data_root, "train", args.img_size, severity_csv=args.severity_csv)
    va = MultiTaskDataset(args.manifest, args.data_root, "val", args.img_size, severity_csv=args.severity_csv)
    tl = DataLoader(tr, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    vl = DataLoader(va, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    print(f"train={len(tr)} val={len(va)} device={device}", flush=True)

    model = MultiTaskTongueNet(args.encoder, pretrained=True).to(device)
    cw = class_weights(tl, device, mode=args.cw_mode)
    losses = {ch: FocalLoss(gamma=2.0, class_weight=cw[ch]).to(device) for ch in KEY_CHARS}
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()

    best = 0.0
    for ep in range(args.epochs):
        model.train()
        t0, run = time.time(), 0.0
        for step, (img, mask, y, w, sev) in enumerate(tl):
            if args.max_steps and step >= args.max_steps:
                break
            img, mask, y, w, sev = img.to(device), mask.to(device), y.to(device), w.to(device), sev.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                out = model(img, mask)
                cls_loss = sum(losses[ch](out[ch], y[:, k], w[:, k]) for k, ch in enumerate(KEY_CHARS))
                reg_loss = F.smooth_l1_loss(out["severity"], sev)
                loss = cls_loss + args.sev_weight * reg_loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += loss.item()
        sched.step()
        res, sev_mae = evaluate(model, vl, device)
        mean_f1 = np.mean([res[ch][1] for ch in KEY_CHARS])
        summary = " ".join(f"{ch}:{res[ch][0]:.2f}/{res[ch][1]:.2f}" for ch in KEY_CHARS)
        print(f"ep {ep+1:02d}/{args.epochs} loss={run/(step+1):.3f} meanF1={mean_f1:.3f} sevMAE={sev_mae:.3f} | "
              f"[acc/F1] {summary} ({time.time()-t0:.0f}s)", flush=True)
        # select best on classification F1 minus severity error, so both matter
        score = mean_f1 - sev_mae
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "args": vars(args), "mean_f1": mean_f1,
                        "sev_mae": sev_mae}, os.path.join(args.out, "best.pt"))
    print(f"BEST score(meanF1-sevMAE)={best:.3f} -> {args.out}/best.pt")


if __name__ == "__main__":
    main()
