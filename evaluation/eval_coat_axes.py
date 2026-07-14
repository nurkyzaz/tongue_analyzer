"""Measure the split coating axes (thickness × texture) against the user's human40 coating labels.

Decomposes each human `coating` label into the two axes and compares to the model's derived axes:
  thickness gold: greasy_thick -> thick ; else thin
  texture   gold: non_greasy   -> smooth; else greasy
Reports per-axis accuracy + confusion, next to the original conflated 3-way accuracy, so we can see
whether either axis is more reliable than the combined label (the point of the split)."""
import json, os, sys
from collections import Counter
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stage1_quantitative"))
from infer import Stage1Pipeline

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--labels", default="evaluation/human40_labels.json")
ap.add_argument("--images", default="data/eval/human40")
ap.add_argument("--mt", default="checkpoints/multitask_v5/best.pt")
args = ap.parse_args()
gold = json.load(open(args.labels))
pipe = Stage1Pipeline("checkpoints/seg_combined/best.pt", args.mt)

def to_thick(c): return "thick" if c == "greasy_thick" else "thin"
def to_tex(c):   return "smooth" if c == "non_greasy" else "greasy"

conf3 = [0, 0]; th = [0, 0]; tx = [0, 0]
th_cm = Counter(); tx_cm = Counter()
for iid in sorted(gold):
    p = f"{args.images}/{iid}.jpg"
    if not os.path.exists(p): continue
    g = gold[iid].get("coating")
    if not g: continue
    s = json.loads(pipe(p).to_json())["key_characteristics"]
    pc = s["coating"]["value"]
    pth = s["coat_thickness"]["value"]; ptx = s["coat_texture"]["value"]
    conf3[0] += (pc == g); conf3[1] += 1
    th[0] += (pth == to_thick(g)); th[1] += 1
    tx[0] += (ptx == to_tex(g));   tx[1] += 1
    th_cm[(to_thick(g), pth)] += 1
    tx_cm[(to_tex(g), ptx)] += 1

print(f"conflated 3-way coating : {conf3[0]}/{conf3[1]} = {conf3[0]/conf3[1]:.0%}")
print(f"thickness (thin/thick)  : {th[0]}/{th[1]} = {th[0]/th[1]:.0%}")
print(f"texture   (smooth/greasy): {tx[0]}/{tx[1]} = {tx[0]/tx[1]:.0%}")
print("\nthickness confusion (gold->pred):", dict(th_cm))
print("texture   confusion (gold->pred):", dict(tx_cm))
