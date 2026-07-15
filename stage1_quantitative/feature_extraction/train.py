"""Train the multi-task characteristic head (5 key characteristics, Focal Loss, mask-guided pooling).

Example:
    python stage1_quantitative/feature_extraction/train.py \
        --data-root data/raw --manifest data/processed/manifest.csv \
        --epochs 30 --batch-size 32 --img-size 384 --out checkpoints/multitask
"""
import argparse
import copy
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, accuracy_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from labels import KEY_CHARS, NUM_CLASSES, SEVERITY_KEYS, CLASS_TO_IDX
from feature_extraction.dataset import MultiTaskDataset, MANUAL_CHARS
from feature_extraction.model import MultiTaskTongueNet
from feature_extraction.losses import FocalLoss


def sample_labels(df):
    """Per-sample class index for each characteristic (manual gold if present, else auto) — mirrors
    MultiTaskDataset._target so the balanced sampler sees exactly the labels training uses."""
    Y = np.zeros((len(df), len(KEY_CHARS)), np.int64)
    for i in range(len(df)):
        row = df.iloc[i]
        for k, ch in enumerate(KEY_CHARS):
            val = row.get(ch)
            if ch in MANUAL_CHARS:
                mv = row.get(f"{ch}_manual")
                if isinstance(mv, str) and mv in CLASS_TO_IDX[ch]:
                    val = mv
            Y[i, k] = CLASS_TO_IDX[ch].get(val, 0) if isinstance(val, str) else 0
    return Y


def rarity_weights(df, clamp=15.0):
    """Sampling weight per image = MAX over characteristics of that image's class-balanced inverse
    frequency (each char's majority class -> 1, minorities scale up by how rare they are). An image
    rare in ANY characteristic (a scarce non-greasy coating, a pale body, a smooth/uncracked tongue) is
    oversampled — directly attacking the collapse to the majority class. Capped so the ~90 non-greasy
    images aren't shown so often they overfit."""
    Y = sample_labels(df)
    w = np.ones(len(df))
    for k, ch in enumerate(KEY_CHARS):
        counts = np.bincount(Y[:, k], minlength=NUM_CLASSES[ch]).astype(float)
        inv = 1.0 / np.clip(counts, 1, None)
        inv = inv / inv.min()                        # majority class -> 1, minorities > 1
        w = np.maximum(w, inv[Y[:, k]])
    return np.clip(w, 1.0, clamp)


class EMA:
    """Exponential moving average of weights — evaluated/saved instead of the raw model for a steadier,
    better-generalising checkpoint."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)


def class_weights(Y, device, mode="sqrt", clamp=(0.5, 4.0)):
    """Per-class loss weights from the TRUE (unsampled) class frequencies, so they complement — not
    double-count — the balanced sampler. `sqrt` is mild inverse-sqrt-freq; `inv` is full inverse-freq.
    Mean-normalised to ~1 and clamped so no class dominates or vanishes."""
    w = {}
    for k, ch in enumerate(KEY_CHARS):
        counts = np.bincount(Y[:, k], minlength=NUM_CLASSES[ch])
        n = torch.tensor(counts, dtype=torch.float).clamp(min=1)
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
    for img, mask, y, w, sev in loader:
        out = model(img.to(device), mask.to(device))
        for k, ch in enumerate(KEY_CHARS):
            lbl = (w[:, k] > 0).numpy()                 # only score characteristics this row actually labels
            preds[ch].append(out[ch].argmax(1).cpu().numpy()[lbl])
            gts[ch].append(y[:, k].numpy()[lbl])
        sd, valid = out["severity"].cpu(), ~torch.isnan(sev)
        sev_abs += (sd[valid] - sev[valid]).abs().sum().item()
        n_sev += int(valid.sum())
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
    ap.add_argument("--cw-clamp-max", type=float, default=4.0, help="max per-class loss weight (raise for stronger minority emphasis)")
    ap.add_argument("--sampler", default="none", choices=["none", "rarity"], help="rarity = oversample images rare in any characteristic")
    ap.add_argument("--ema", action="store_true", help="track+save an EMA of weights")
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--severity-csv", default="data/processed/severity.csv")
    ap.add_argument("--sev-weight", type=float, default=5.0, help="weight for severity regression loss")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--init", default=None, help="checkpoint to warm-start from (fine-tune, e.g. multitask_v5/best.pt)")
    ap.add_argument("--no-wb", action="store_true", help="disable the white-balance colour augmentation (train on the labels' own colours)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wb = not args.no_wb
    tr = MultiTaskDataset(args.manifest, args.data_root, "train", args.img_size, severity_csv=args.severity_csv, wb=wb)
    va = MultiTaskDataset(args.manifest, args.data_root, "val", args.img_size, severity_csv=args.severity_csv, wb=wb)
    Ytr = sample_labels(tr.df)
    if args.sampler == "rarity":
        sw = rarity_weights(tr.df)
        sampler = WeightedRandomSampler(torch.as_tensor(sw, dtype=torch.double), len(sw), replacement=True)
        tl = DataLoader(tr, args.batch_size, sampler=sampler, num_workers=args.workers, pin_memory=True, drop_last=True)
        ng = CLASS_TO_IDX["coating"].get("non_greasy", 0)
        print(f"rarity sampler: weight range [{sw.min():.1f}, {sw.max():.1f}], non-greasy draw share "
              f"~{sw[Ytr[:,0]==ng].sum()/sw.sum():.1%} (raw {(Ytr[:,0]==ng).mean():.1%})", flush=True)
    else:
        tl = DataLoader(tr, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    vl = DataLoader(va, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    print(f"train={len(tr)} val={len(va)} device={device} encoder={args.encoder} size={args.img_size}", flush=True)

    model = MultiTaskTongueNet(args.encoder, pretrained=True).to(device)
    if args.init:                                      # warm-start for fine-tuning on cleaner labels
        st = torch.load(args.init, map_location=device, weights_only=False)
        model.load_state_dict(st["model"], strict=False)
        print(f"warm-started from {args.init}", flush=True)
    cw = class_weights(Ytr, device, mode=args.cw_mode, clamp=(0.5, args.cw_clamp_max))
    losses = {ch: FocalLoss(gamma=2.0, class_weight=cw[ch]).to(device) for ch in KEY_CHARS}
    ema = EMA(model, args.ema_decay) if args.ema else None
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
                valid = ~torch.isnan(sev)                       # mask severity-less (TCM-Tongue) rows
                reg_loss = F.smooth_l1_loss(out["severity"][valid], sev[valid]) if valid.any() else out["severity"].sum() * 0.0
                loss = cls_loss + args.sev_weight * reg_loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                ema.update(model)
            run += loss.item()
        sched.step()
        # evaluate (and save) the EMA weights when enabled, else the live model
        eval_model = model
        if ema is not None:
            eval_model = copy.deepcopy(model)
            eval_model.load_state_dict(ema.shadow, strict=True)
        res, sev_mae = evaluate(eval_model, vl, device)
        mean_f1 = np.mean([res[ch][1] for ch in KEY_CHARS])
        summary = " ".join(f"{ch}:{res[ch][0]:.2f}/{res[ch][1]:.2f}" for ch in KEY_CHARS)
        print(f"ep {ep+1:02d}/{args.epochs} loss={run/(step+1):.3f} meanF1={mean_f1:.3f} sevMAE={sev_mae:.3f} | "
              f"[acc/F1] {summary} ({time.time()-t0:.0f}s)", flush=True)
        # select best on classification F1 minus severity error, so both matter
        score = mean_f1 - sev_mae
        if score > best:
            best = score
            torch.save({"model": eval_model.state_dict(), "args": vars(args), "mean_f1": mean_f1,
                        "sev_mae": sev_mae}, os.path.join(args.out, "best.pt"))
    print(f"BEST score(meanF1-sevMAE)={best:.3f} -> {args.out}/best.pt")


if __name__ == "__main__":
    main()
