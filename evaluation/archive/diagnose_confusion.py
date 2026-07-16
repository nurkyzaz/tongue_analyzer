"""Per-class confusion matrices for the 5 core tongue characteristics — to see WHERE the model is
inaccurate (e.g. pale body read as regular, everything read as greasy), not just an aggregate F1.

Uses the expert L1-manual gold for tai/zhi/fissure/tooth_mk (a pure expert-vs-model check); coating has
no manual column so we fall back to its auto label. Prints, per characteristic, the confusion matrix
(rows = gold, cols = predicted) + per-class recall.

    CUDA_VISIBLE_DEVICES=0 envs/tih/bin/python evaluation/diagnose_confusion.py \
        --mt checkpoints/multitask_v5/best.pt
"""
import argparse, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "stage1_quantitative"))
from labels import KEY_CHARS, LABEL_MAPS, CHAR_DESC
from feature_extraction.dataset import MultiTaskDataset
from feature_extraction.model import MultiTaskTongueNet


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mt", default="checkpoints/multitask_v5/best.pt")
    ap.add_argument("--data-root", default="data/raw")
    ap.add_argument("--manifest", default="data/processed/manifest.csv")
    ap.add_argument("--size", type=int, default=384)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    st = torch.load(args.mt, map_location=device, weights_only=False)
    model = MultiTaskTongueNet(st["args"].get("encoder", "resnet34"), pretrained=False)
    model.load_state_dict(st["model"], strict=False); model.to(device).eval()

    ds = MultiTaskDataset(args.manifest, args.data_root, "test", args.size)
    dl = DataLoader(ds, 32, num_workers=8, pin_memory=True)
    df = ds.df
    preds = {c: [] for c in KEY_CHARS}
    for img, mask, y, w, sev in dl:
        out = model(img.to(device), mask.to(device))
        for ch in KEY_CHARS:
            preds[ch].append(out[ch].argmax(1).cpu().numpy())
    preds = {c: np.concatenate(v) for c, v in preds.items()}

    print(f"model: {args.mt}  |  test images: {len(df)}\n")
    for ch in KEY_CHARS:
        labels = LABEL_MAPS[ch]
        idx = {v: i for i, v in enumerate(labels)}
        col = f"{ch}_manual"
        if col in df and df[col].notna().sum() > 0:
            has = df[col].notna().values
            gt = df.loc[has, col].map(lambda v: idx.get(str(v).strip().lower(), -1)).values
            src = "expert manual"
        else:
            col = ch
            has = df[col].notna().values
            gt = df.loc[has, col].map(lambda v: idx.get(str(v).strip().lower(), -1)).values
            src = "auto label (no manual gold)"
        pr = preds[ch][has]
        keep = gt >= 0
        gt, pr = gt[keep], pr[keep]
        n = len(gt)
        cm = np.zeros((len(labels), len(labels)), int)
        for g, p in zip(gt, pr):
            cm[g, p] += 1
        acc = (gt == pr).mean() if n else 0.0
        print(f"=== {CHAR_DESC[ch]} ({ch})  n={n}  acc={acc:.2f}  [{src}] ===")
        head = "gold\\pred".ljust(14) + "".join(l[:11].ljust(12) for l in labels)
        print(head)
        for i, l in enumerate(labels):
            row = l[:13].ljust(14) + "".join(str(cm[i, j]).ljust(12) for j in range(len(labels)))
            rec = cm[i, i] / max(cm[i].sum(), 1)
            print(f"{row} recall={rec:.2f} (n={cm[i].sum()})")
        print()


if __name__ == "__main__":
    main()
