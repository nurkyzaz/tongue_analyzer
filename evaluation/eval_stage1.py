"""Evaluate the Stage-1 multi-task head on the test split against gold-preferred labels.

Reports per-characteristic accuracy, macro-F1, and per-class F1 (so rare-class behaviour is visible).
Uses ground-truth masks (segmentation is ~0.99 Dice, so predicted≈GT for this metric).

    python evaluation/eval_stage1.py --data-root data/raw --manifest data/processed/manifest.csv \
        --mt checkpoints/multitask_v3/best.pt
"""
import argparse
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stage1_quantitative"))
from labels import KEY_CHARS, LABEL_MAPS
from feature_extraction.dataset import MultiTaskDataset
from feature_extraction.model import MultiTaskTongueNet


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--mt", default="checkpoints/multitask_v3/best.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    st = torch.load(args.mt, map_location=device, weights_only=False)
    model = MultiTaskTongueNet(st["args"].get("encoder", "resnet34"), pretrained=False)
    model.load_state_dict(st["model"]); model.to(device).eval()

    ds = MultiTaskDataset(args.manifest, args.data_root, args.split, args.img_size)
    dl = DataLoader(ds, args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    print(f"Evaluating {len(ds)} {args.split} images | checkpoint={args.mt}\n")

    P = {c: [] for c in KEY_CHARS}; G = {c: [] for c in KEY_CHARS}
    for img, mask, y, _, _sev in dl:
        out = model(img.to(device), mask.to(device))
        for k, ch in enumerate(KEY_CHARS):
            P[ch].append(out[ch].argmax(1).cpu().numpy()); G[ch].append(y[:, k].numpy())

    macro = []
    print(f"{'characteristic':14s} {'acc':>6s} {'macroF1':>8s}   per-class F1")
    for ch in KEY_CHARS:
        p, g = np.concatenate(P[ch]), np.concatenate(G[ch])
        acc = accuracy_score(g, p); mf1 = f1_score(g, p, average="macro", zero_division=0)
        pcf = f1_score(g, p, average=None, labels=list(range(len(LABEL_MAPS[ch]))), zero_division=0)
        macro.append(mf1)
        pcf_s = "  ".join(f"{LABEL_MAPS[ch][i]}={pcf[i]:.2f}" for i in range(len(pcf)))
        print(f"{ch:14s} {acc:6.3f} {mf1:8.3f}   {pcf_s}")
    print(f"\nMEAN macro-F1 = {np.mean(macro):.3f}")


if __name__ == "__main__":
    main()
