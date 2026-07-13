"""Validate the measured 'extra' features against the user's human40 extra labels:
  - red_tip: zoning.tip_redness_delta vs gold none/mild/strong (ordinal)
  - red_dots: extra-features model's red_dots prob vs gold none/few/many (ordinal)
Reports mean signal per gold class (does it separate?) and a suggested red_tip threshold.
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stage1_quantitative"))
import torch, torch.nn.functional as F
from infer import Stage1Pipeline
from zoning import analyze
from labels import EXTRA_FEATURES

gold = json.load(open("evaluation/human40_extra_labels.json"))
pipe = Stage1Pipeline("checkpoints/seg_combined/best.pt", "checkpoints/multitask_v5/best.pt")
RD = EXTRA_FEATURES.index("red_dots")

rows = []
for iid in sorted(gold):
    p = None
    for ext in (".jpg", ".png", ".jpeg"):
        q = os.path.join("data/eval/human40", iid + ext)
        if os.path.exists(q): p = q; break
    if not p: continue
    o, m, disp = pipe(p, return_mask=True), None, None
    # need mask+disp for zoning; re-run with return_mask
    _, m, disp = pipe(p, return_mask=True)
    z = analyze(disp, m)
    # extra model red_dots prob
    import cv2, numpy as np
    from infer import _preprocess
    img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
    x = _preprocess(img, pipe.size).to(pipe.device)
    with torch.no_grad():
        mask = (pipe.seg(x).sigmoid() > 0.5).float()
        rd_prob = float(torch.sigmoid(pipe.extra(x, mask))[0, RD].cpu()) if pipe.extra is not None else None
    rows.append({"id": iid, "tipd": z.get("tip_redness_delta"),
                 "g_tip": gold[iid]["red_tip"], "rd": rd_prob, "g_dots": gold[iid]["red_dots"]})

def mean(vs): vs=[v for v in vs if v is not None]; return sum(vs)/len(vs) if vs else float("nan")

print("=== red_tip: zoning tip_redness_delta by gold class ===")
for c in ("none", "mild", "strong"):
    sub = [r["tipd"] for r in rows if r["g_tip"] == c]
    print(f"  {c:7} n={len(sub):2}  mean tipΔred={mean(sub):+.2f}")
# threshold sweep for present = mild|strong
gold_pos = [r for r in rows if r["g_tip"] in ("mild", "strong")]
print("  threshold sweep (present = mild|strong):")
for thr in (0.0, 1.0, 1.5, 2.0, 2.5):
    tp = sum(1 for r in rows if r["tipd"] is not None and r["tipd"] > thr and r["g_tip"] in ("mild","strong"))
    fp = sum(1 for r in rows if r["tipd"] is not None and r["tipd"] > thr and r["g_tip"] == "none")
    fn = sum(1 for r in rows if (r["tipd"] is None or r["tipd"] <= thr) and r["g_tip"] in ("mild","strong"))
    prec = tp/(tp+fp) if tp+fp else 0; rec = tp/(tp+fn) if tp+fn else 0
    print(f"    thr={thr:>3}: precision={prec:.2f} recall={rec:.2f} (tp={tp} fp={fp} fn={fn})")

print("\n=== red_dots: extra-model prob by gold class ===")
for c in ("none", "few", "many"):
    sub = [r["rd"] for r in rows if r["g_dots"] == c]
    print(f"  {c:5} n={len(sub):2}  mean red_dots_prob={mean(sub):.3f}")
