"""Score a characteristic model against the EXPERT (TonguExpert L1 manual) labels on the held-out test
split, and by image SOURCE on the human eval (TE clinical vs SM/TCM real-photo) — the honest standard now
that we trust the expert labels (docs/LABEL_IMPROVEMENT_PLAN.md). Use to compare v5 vs the SSL-fine-tuned
model, esp. on the real-photo (SM) domain gap.

    CUDA_VISIBLE_DEVICES=0 envs/tih/bin/python evaluation/eval_vs_expert.py --mt checkpoints/multitask_v5/best.pt
"""
import argparse, json, os, sys
from collections import defaultdict
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stage1_quantitative"))
from infer import Stage1Pipeline

ap = argparse.ArgumentParser()
ap.add_argument("--mt", required=True)
ap.add_argument("--seg", default="checkpoints/seg_combined/best.pt")
ap.add_argument("--limit", type=int, default=300)
args = ap.parse_args()
pipe = Stage1Pipeline(args.seg, args.mt)

# --- A) vs EXPERT L1 manual, held-out test split ---
m = pd.read_csv("data/processed/manifest.csv")
te = m[m.split == "test"]
acc = {c: [0, 0] for c in ["tai", "zhi", "fissure", "tooth_mk"]}
n = 0
for _, r in te.iterrows():
    if not any(pd.notna(r.get(c + "_manual")) for c in acc):
        continue
    src = "data/raw/" + str(r.raw_path)
    if not os.path.exists(src):
        continue
    kc = json.loads(pipe(src).to_json())["key_characteristics"]; n += 1
    for c in acc:
        gv = r.get(c + "_manual")
        if pd.notna(gv):
            acc[c][0] += kc[c]["value"] == gv; acc[c][1] += 1
    if n >= args.limit:
        break
print(f"\n=== {args.mt} ===")
print(f"A) vs EXPERT (L1 manual), held-out test, n={n}:")
for c in acc:
    print(f"   {c:9} {acc[c][0]}/{acc[c][1]} = {acc[c][0]/max(acc[c][1],1):.0%}")

# --- B) by SOURCE on the human eval (real-photo domain gap) ---
by = defaultdict(lambda: defaultdict(lambda: [0, 0]))
for s in ("human40", "human40b"):
    g = json.load(open(f"evaluation/{s}_labels.json"))
    meta = {e["id"]: e for e in json.load(open(f"data/eval/{s}/meta.json"))}
    for i, lab in g.items():
        p = f"data/eval/{s}/{i}.jpg"
        if not os.path.exists(p):
            continue
        src = meta.get(i, {}).get("src", "?")
        kc = json.loads(pipe(p).to_json())["key_characteristics"]
        for ch in ("tai", "zhi", "coating"):
            if lab.get(ch):
                by[src][ch][0] += kc[ch]["value"] == lab[ch]; by[src][ch][1] += 1
print("B) by source on human eval (tai / zhi / coating):")
for src in ("TE", "TCM", "SM"):
    a = by[src]
    def pc(c): return f"{a[c][0]/max(a[c][1],1):.0%}"
    print(f"   {src:4} n={a['tai'][1]:>3}  tai {pc('tai')}  zhi {pc('zhi')}  coating {pc('coating')}")
