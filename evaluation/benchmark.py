"""Grounded accuracy benchmark against EXPERT labels, and a Markdown report.

Two expert-labeled test sets (the categories professionals actually annotated — the most data we have):

  A. TonguExpert 5 key characteristics vs **L1 manual (expert-verified) gold** — evaluated only on the
     held-out test images that carry a manual label, so it is a pure expert-vs-model comparison.
  B. TCM-Tongue 8 pathological categories vs **licensed-practitioner labels** on its held-out 553-image
     test split — per-class Average Precision + P/R/F1.

We deliberately do NOT report "disease" or "constitution" accuracy: our data has no independent expert
labels for those, so any such number would be circular. Constitution mapping is rule-based/educational.

    python evaluation/benchmark.py --mt checkpoints/multitask_v4/best.pt \
        --extra checkpoints/extra_features/best.pt --out docs/PROJECT_HANDBOOK.md
"""
import argparse, os, sys, json
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, average_precision_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stage1_quantitative"))
from labels import KEY_CHARS, LABEL_MAPS, EXTRA_FEATURES, EXTRA_DESC, CHAR_DESC
from feature_extraction.dataset import MultiTaskDataset
from feature_extraction.model import MultiTaskTongueNet
from feature_extraction.extra_dataset import ExtraFeaturesDataset
from feature_extraction.extra_model import ExtraFeaturesNet

MANUAL_CHARS = ["tai", "zhi", "fissure", "tooth_mk"]   # have L1 expert labels


@torch.no_grad()
def benchmark_A(mt_ckpt, data_root, manifest, device, size=384):
    st = torch.load(mt_ckpt, map_location=device, weights_only=False)
    model = MultiTaskTongueNet(st["args"].get("encoder", "resnet34"), pretrained=False)
    model.load_state_dict(st["model"], strict=False); model.to(device).eval()
    ds = MultiTaskDataset(manifest, data_root, "test", size)
    dl = DataLoader(ds, 32, num_workers=8, pin_memory=True)
    df = ds.df
    preds = {c: [] for c in KEY_CHARS}
    for img, mask, y, w, sev in dl:
        out = model(img.to(device), mask.to(device))
        for k, ch in enumerate(KEY_CHARS):
            preds[ch].append(out[ch].argmax(1).cpu().numpy())
    preds = {c: np.concatenate(v) for c, v in preds.items()}

    rows = []
    for ch in MANUAL_CHARS:                      # expert-gold only
        col = f"{ch}_manual"
        mask_has = df[col].notna().values if col in df else np.zeros(len(df), bool)
        if mask_has.sum() == 0:
            continue
        idx = {v: i for i, v in enumerate(LABEL_MAPS[ch])}
        gt = df.loc[mask_has, col].map(lambda v: idx.get(str(v).strip().lower(), -1)).values
        pr = preds[ch][mask_has]
        keep = gt >= 0
        gt, pr = gt[keep], pr[keep]
        acc = accuracy_score(gt, pr)
        p, r, f1, _ = precision_recall_fscore_support(gt, pr, average="macro", zero_division=0)
        rows.append((CHAR_DESC[ch], int(len(gt)), acc, p, r, f1))
    return rows


@torch.no_grad()
def benchmark_B(extra_ckpt, data_root, labels_csv, mask_dir, device, size=384):
    st = torch.load(extra_ckpt, map_location=device, weights_only=False)
    model = ExtraFeaturesNet(st["args"].get("encoder", "resnet34"), pretrained=False)
    model.load_state_dict(st["model"]); model.to(device).eval()
    ds = ExtraFeaturesDataset(labels_csv, data_root, mask_dir, "test", size)
    dl = DataLoader(ds, 32, num_workers=8, pin_memory=True)
    P, G = [], []
    for img, mask, y in dl:
        P.append(torch.sigmoid(model(img.to(device), mask.to(device))).cpu().numpy()); G.append(y.numpy())
    P, G = np.concatenate(P), np.concatenate(G)
    rows = []
    for k, feat in enumerate(EXTRA_FEATURES):
        g, p = G[:, k], P[:, k]
        npos = int(g.sum())
        ap = average_precision_score(g, p) if npos else 0.0
        # best-F1 threshold
        best = (0, 0.5)
        for t in np.linspace(0.2, 0.8, 13):
            pr, rc, f1, _ = precision_recall_fscore_support(g, p > t, average="binary", zero_division=0)
            if f1 > best[0]:
                best = (f1, t)
        pr, rc, f1, _ = precision_recall_fscore_support(g, p > best[1], average="binary", zero_division=0)
        rows.append((EXTRA_DESC[feat], npos, ap, pr, rc, f1))
    return rows


def write_report(A, B, out, mt_ckpt, extra_ckpt):
    L = ["# Accuracy Benchmark — model vs. expert labels\n",
         "_Grounded in the categories professionals actually annotated. We report where we have "
         "**independent expert labels**; we do NOT report disease/constitution accuracy (no such "
         "ground-truth in our data — that would be circular)._\n",
         f"Models: `{mt_ckpt}` (characteristics), `{extra_ckpt}` (extra features).\n",
         "## A. Key characteristics vs TonguExpert **expert (L1 manual)** gold — held-out test",
         "| characteristic | n (expert-labeled) | accuracy | precision | recall | macro-F1 |",
         "|---|---|---|---|---|---|"]
    for name, n, acc, p, r, f1 in A:
        L.append(f"| {name} | {n} | {acc:.3f} | {p:.3f} | {r:.3f} | {f1:.3f} |")
    if A:
        L.append(f"| **mean** |  | **{np.mean([a[2] for a in A]):.3f}** |  |  | **{np.mean([a[5] for a in A]):.3f}** |")
    L += ["", "## B. Pathological categories vs TCM-Tongue **practitioner** labels — held-out 553-img test",
          "_Classes with < 20 test positives are flagged ⚠ (too few to trust the number)._",
          "| category | n positives | Average Precision | precision | recall | F1 |",
          "|---|---|---|---|---|---|"]
    reliable = [b for b in B if b[1] >= 20]
    for name, n, ap, p, r, f1 in B:
        flag = "" if n >= 20 else " ⚠"
        L.append(f"| {name}{flag} | {n} | {ap:.3f} | {p:.3f} | {r:.3f} | {f1:.3f} |")
    if reliable:
        L.append(f"| **mean (mAP, n≥20 only)** |  | **{np.mean([b[2] for b in reliable]):.3f}** |  |  |  |")
    L += ["", "## Method & grounding",
          "- **Expert labels:** (A) TonguExpert `L1_Labels_Manual` (human-verified); (B) TCM-Tongue "
          "licensed-practitioner annotations, held-out test split.",
          "- **Held-out:** models never saw these test images in training.",
          "- **Literature anchors:** SSC-Net reports ~0.85 F1 on the 5 characteristics; TCM-Tongue's own "
          "YOLO benchmarks and published constitution models (~0.71 acc, 'junior-practitioner level') are "
          "the comparison points. Numbers here are directly comparable for the characteristic/feature task.",
          "- **Not benchmarked (honest scope):** TCM pattern / 9-constitution / disease outputs are a "
          "rule-based educational mapping (grounded in ICD-11 / CCMQ / Maciocia); we have no independent "
          "expert-labeled images for them, so we do not claim an accuracy number.\n"]
    with open(out, "w") as f:
        f.write("\n".join(L))
    print("\n".join(L))
    print(f"\nWrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mt", default="checkpoints/multitask_v4/best.pt")
    ap.add_argument("--extra", default="checkpoints/extra_features/best.pt")
    ap.add_argument("--data-root", default=".")
    ap.add_argument("--te-root", default="data/raw")
    ap.add_argument("--manifest", default="data/processed/manifest.csv")
    ap.add_argument("--labels", default="data/processed/tcm_tongue_labels.csv")
    ap.add_argument("--mask-dir", default="data/external/tcm_tongue/masks")
    ap.add_argument("--out", default="docs/PROJECT_HANDBOOK.md")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    A = benchmark_A(args.mt, args.te_root, args.manifest, device)
    B = benchmark_B(args.extra, args.data_root, args.labels, args.mask_dir, device)
    write_report(A, B, args.out, args.mt, args.extra)


if __name__ == "__main__":
    main()
