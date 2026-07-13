"""Select a diverse, minority-covering image set for HUMAN labeling (a trustworthy eval set).

Runs v5 over a pool from all three sources and greedily picks images to fill per-class quotas across the
weak axes (body colour incl. pale, coating incl. non-greasy/thick, cracks, tooth-marks) so the labeled
set actually contains minority classes instead of 90% greasy. v5 predictions are used ONLY to stratify —
they are NOT shown to the labeler.

    CUDA_VISIBLE_DEVICES=0 envs/tih/bin/python evaluation/select_eval_set.py --n 40 --out /tmp/evalset
"""
import argparse, glob, json, os, random, shutil, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stage1_quantitative"))
from labels import KEY_CHARS
from infer import Stage1Pipeline

# per-(axis,class) target counts — biased toward the minority classes we most need to measure
QUOTA = {
    ("zhi", "light"): 9, ("zhi", "regular"): 6, ("zhi", "dark"): 6,
    ("coating", "non_greasy"): 8, ("coating", "greasy_thick"): 8, ("coating", "greasy"): 5,
    ("tai", "yellow"): 5, ("tai", "white"): 4,
    ("fissure", "severe"): 6, ("tooth_mk", "severe"): 6,
}


def pool(seed=11, exclude=None):
    random.seed(seed)
    exclude = set(exclude or [])
    items = []
    m = __import__("pandas").read_csv("data/processed/manifest.csv")
    te = m[(m.split == "test") & (m.has_mask)].sample(min(220, len(m)), random_state=seed)
    items += [("data/raw/" + p, "TE") for p in te.raw_path]
    sm = [p for p in glob.glob("data/external/sm_tongue/**/*.png", recursive=True) if "mask" not in p.lower() and "overlay" not in p.lower()]
    random.shuffle(sm); items += [(p, "SM") for p in sm[:160]]
    tc = glob.glob("data/external/tcm_tongue/shezhenv3-txt/train/images/*.jpg")
    random.shuffle(tc); items += [(p, "TCM") for p in tc[:160]]
    items = [it for it in items if it[0] not in exclude]     # drop already-labeled images
    random.shuffle(items)
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default="/tmp/evalset")
    ap.add_argument("--seg", default="checkpoints/seg_combined/best.pt")
    ap.add_argument("--mt", default="checkpoints/multitask_v5/best.pt")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--exclude", default=None, help="meta.json of an existing set whose images to skip")
    args = ap.parse_args()
    pipe = Stage1Pipeline(args.seg, args.mt)

    exclude = [e["path"] for e in json.load(open(args.exclude))] if args.exclude else []
    scored = []
    for path, src in pool(seed=args.seed, exclude=exclude):
        try:
            s1 = json.loads(pipe(path).to_json())
        except Exception:
            continue
        pred = {ch: s1["key_characteristics"][ch]["value"] for ch in KEY_CHARS}
        scored.append((path, src, pred))
    print(f"scored pool: {len(scored)}", flush=True)

    filled = {k: 0 for k in QUOTA}
    picks, seen = [], set()
    # greedy: repeatedly take the image that fills the most still-unmet quotas
    while len(picks) < args.n:
        best, best_gain = None, 0
        for path, src, pred in scored:
            if path in seen:
                continue
            gain = sum(1 for (ax, cl), q in QUOTA.items() if pred[ax] == cl and filled[(ax, cl)] < q)
            if gain > best_gain:
                best, best_gain = (path, src, pred), gain
        if best is None or best_gain == 0:
            break
        path, src, pred = best
        seen.add(path); picks.append((path, src, pred))
        for (ax, cl), q in QUOTA.items():
            if pred[ax] == cl and filled[(ax, cl)] < q:
                filled[(ax, cl)] += 1
    # top up with random unseen to reach n (representativeness)
    for path, src, pred in scored:
        if len(picks) >= args.n:
            break
        if path not in seen:
            seen.add(path); picks.append((path, src, pred))

    os.makedirs(args.out, exist_ok=True)
    man = []
    for i, (path, src, pred) in enumerate(picks):
        dst = os.path.join(args.out, f"t{i:02d}.jpg")
        shutil.copy(path, dst)
        man.append({"id": f"t{i:02d}", "src": src, "path": path, "v5_pred": pred})
    json.dump(man, open(os.path.join(args.out, "meta.json"), "w"))
    print(f"selected {len(picks)} -> {args.out}")
    print("quota fill:", {f"{a}={c}": f"{filled[(a,c)]}/{q}" for (a, c), q in QUOTA.items()})


if __name__ == "__main__":
    main()
