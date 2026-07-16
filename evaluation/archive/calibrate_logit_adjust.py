"""Post-hoc logit-adjustment calibration for the imbalanced characteristic heads (no retraining).

The heads collapse to the majority class (coating->greasy, body->regular) because argmax is dominated by
the class prior. Logit adjustment subtracts tau*log(prior) from each class logit; we fit tau per
characteristic on VAL (maximising macro-F1) and report VAL+TEST before/after. This is deployable in
infer.py immediately and is reversible.

    CUDA_VISIBLE_DEVICES=0 envs/tih/bin/python evaluation/calibrate_logit_adjust.py --mt checkpoints/multitask_v5/best.pt
"""
import argparse, os, sys
import numpy as np, torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "stage1_quantitative"))
from labels import KEY_CHARS, LABEL_MAPS, CHAR_DESC
from feature_extraction.dataset import MultiTaskDataset
from feature_extraction.model import MultiTaskTongueNet


@torch.no_grad()
def collect(model, ds, dev):
    dl = DataLoader(ds, 32, num_workers=8)
    logits = {c: [] for c in KEY_CHARS}
    for img, mask, y, w, sev in dl:
        o = model(img.to(dev), mask.to(dev))
        for c in KEY_CHARS:
            logits[c].append(o[c].cpu().numpy())
    return {c: np.concatenate(v) for c, v in logits.items()}, ds.df


def gt_of(df, ch):
    idx = {v: i for i, v in enumerate(LABEL_MAPS[ch])}
    return df[ch].map(lambda v: idx.get(str(v).strip().lower(), -1)).values


def macro_f1(gt, pred, k):
    return precision_recall_fscore_support(gt, pred, labels=list(range(k)), average="macro", zero_division=0)[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mt", default="checkpoints/multitask_v5/best.pt")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    st = torch.load(args.mt, map_location=dev, weights_only=False)
    model = MultiTaskTongueNet(st["args"].get("encoder", "resnet34"), pretrained=False)
    model.load_state_dict(st["model"], strict=False); model.to(dev).eval()

    root, man = "data/raw", "data/processed/manifest.csv"
    val_lg, vdf = collect(model, MultiTaskDataset(man, root, "val", 384), dev)
    test_lg, tdf = collect(model, MultiTaskDataset(man, root, "test", 384), dev)
    tr = __import__("pandas").read_csv(man)
    tr = tr[tr.split == "train"]

    print(f"model: {args.mt}\n{'char':16s}{'val F1 base->cal':22s}{'test F1 base->cal':22s} tau  minority-recall test base->cal")
    for ch in KEY_CHARS:
        k = len(LABEL_MAPS[ch])
        prior = np.array([(tr[ch].astype(str).str.lower() == l).sum() for l in LABEL_MAPS[ch]], float)
        prior = np.clip(prior / prior.sum(), 1e-6, None)
        logp = np.log(prior)
        vgt, tgt = gt_of(vdf, ch), gt_of(tdf, ch)
        vm, tm = vgt >= 0, tgt >= 0
        # fit tau on val
        best = (macro_f1(vgt[vm], val_lg[ch][vm].argmax(1), k), 0.0)
        for tau in np.linspace(0, 3, 31):
            pred = (val_lg[ch] - tau * logp).argmax(1)
            f = macro_f1(vgt[vm], pred[vm], k)
            if f > best[0]:
                best = (f, tau)
        tau = best[1]
        vbase = macro_f1(vgt[vm], val_lg[ch][vm].argmax(1), k)
        vcal = macro_f1(vgt[vm], (val_lg[ch] - tau * logp)[vm].argmax(1), k)
        tbase = macro_f1(tgt[tm], test_lg[ch][tm].argmax(1), k)
        tcal = macro_f1(tgt[tm], (test_lg[ch] - tau * logp)[tm].argmax(1), k)
        # minority (rarest by train prior) recall on test
        mc = int(prior.argmin())
        tp = test_lg[ch][tm]; g = tgt[tm]
        rb = ((tp.argmax(1) == mc) & (g == mc)).sum() / max((g == mc).sum(), 1)
        rc = (((tp - tau * logp).argmax(1) == mc) & (g == mc)).sum() / max((g == mc).sum(), 1)
        print(f"{ch:16s}{vbase:.2f}->{vcal:.2f}{'':13s}{tbase:.2f}->{tcal:.2f}{'':13s}{tau:.1f}  {LABEL_MAPS[ch][mc]}: {rb:.2f}->{rc:.2f}")


if __name__ == "__main__":
    main()
