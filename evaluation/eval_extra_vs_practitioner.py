"""Validate the extra-features model against the PRACTITIONER labels (TCM-Tongue / shezhenv3-txt),
on the held-out TEST split the model never trained on. Answers "how good are we at detecting each
extra feature?" — the features for which we actually have independent labels.

    python evaluation/eval_extra_vs_practitioner.py [--dir "shezhen datasets/shezhenv3-txt/test"]

Reports, per feature: prevalence, AP (area under PR), and precision/recall/F1 at 0.5. The dataset
annotates every visible attribute, so "no box of class C" = that feature is ABSENT (binary presence).
"""
import argparse, glob, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stage1_quantitative"))
import cv2, torch
from infer import Stage1Pipeline, _preprocess
from labels import EXTRA_FEATURES

# YOLO class id (shezhenv3-txt.yaml) -> our extra-feature name. Only the classes the extra head predicts.
YOLO_TO_EXTRA = {
    1: "peeled_coating",   # botaishe 薄苔 — thin coating; matches the model's (misnamed) peeled_coating head
    2: "red_tongue",       # hongshe 红舌
    3: "purple_body",      # zishe 紫舌
    4: "swollen",          # pangdashe 胖大舌
    5: "thin",             # shoushe 瘦舌
    6: "red_dots",         # hongdianshe 红点舌
    11: "black_coating",   # heitaishe 黑苔
    12: "slippery_coating",# huataishe 滑苔
}
IDX = {f: EXTRA_FEATURES.index(f) for f in YOLO_TO_EXTRA.values()}


def gold_presence(label_path):
    present = set()
    if os.path.exists(label_path):
        for line in open(label_path):
            parts = line.split()
            if parts:
                c = int(float(parts[0]))
                if c in YOLO_TO_EXTRA:
                    present.add(YOLO_TO_EXTRA[c])
    return present


def average_precision(pairs):
    """pairs: list of (prob, gold01). Area under the precision-recall curve (interpolation-free AP)."""
    pairs = sorted(pairs, key=lambda x: -x[0])
    P = sum(g for _, g in pairs)
    if P == 0:
        return float("nan")
    tp = 0; ap = 0.0
    for i, (_, g) in enumerate(pairs, 1):
        if g:
            tp += 1
            ap += tp / i          # precision@i at each true positive
    return ap / P


def prf(pairs, thr=0.5):
    tp = sum(1 for p, g in pairs if p >= thr and g)
    fp = sum(1 for p, g in pairs if p >= thr and not g)
    fn = sum(1 for p, g in pairs if p < thr and g)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1, tp, fp, fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="shezhen datasets/shezhenv3-txt/test")
    ap.add_argument("--seg", default="checkpoints/seg_combined/best.pt")
    ap.add_argument("--mt", default="checkpoints/multitask_v5/best.pt")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    pipe = Stage1Pipeline(args.seg, args.mt)
    if pipe.extra is None:
        print("no extra-features head loaded — aborting"); return
    imgs = sorted(glob.glob(os.path.join(args.dir, "images", "*.jpg")))
    if args.limit:
        imgs = imgs[:args.limit]
    per_feat = {f: [] for f in YOLO_TO_EXTRA.values()}
    n = 0
    for ip in imgs:
        stem = os.path.splitext(os.path.basename(ip))[0]
        lp = os.path.join(args.dir, "labels", stem + ".txt")
        img = cv2.imread(ip)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = _preprocess(img, pipe.size).to(pipe.device)
        with torch.no_grad():
            mask = (pipe.seg(x).sigmoid() > 0.5).float()
            probs = torch.sigmoid(pipe.extra(x, mask))[0].cpu().tolist()
        gp = gold_presence(lp)
        for f in YOLO_TO_EXTRA.values():
            per_feat[f].append((probs[IDX[f]], 1 if f in gp else 0))
        n += 1
        if n % 100 == 0:
            print(f"  ...{n}/{len(imgs)}", file=sys.stderr)

    print(f"\nExtra-features vs PRACTITIONER labels — {n} test images ({args.dir})\n")
    print(f"{'feature':17} {'prev':>6} {'AP':>6} {'P@.5':>6} {'R@.5':>6} {'F1':>6}   (tp/fp/fn)")
    print("-" * 68)
    rows = []
    for f in EXTRA_FEATURES:
        if f not in per_feat:
            continue
        pairs = per_feat[f]
        pos = sum(g for _, g in pairs)
        prev = pos / len(pairs) if pairs else 0
        AP = average_precision(pairs)
        p, r, f1, tp, fp, fn = prf(pairs)
        rows.append((f, prev, AP, p, r, f1))
        print(f"{f:17} {prev:6.2f} {AP:6.2f} {p:6.2f} {r:6.2f} {f1:6.2f}   ({tp}/{fp}/{fn})")
    macro = [r[2] for r in rows if r[2] == r[2]]
    print("-" * 68)
    print(f"macro-mAP over {len(macro)} features: {sum(macro)/len(macro):.3f}" if macro else "no positives")


if __name__ == "__main__":
    main()
