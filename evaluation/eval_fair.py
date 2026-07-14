"""Fairer accuracy on the human labels. Exact 3-way match penalises the fuzzy severity boundary
(light vs severe) that even experts disagree on. So we also report:
  - within-1-grade: prediction is at most one ordinal step off (the honest metric for graded signs)
  - presence: for none/light/severe signs, did we agree it's THERE vs not (ignoring the severity grade)
This separates "the model can't see it" from "it graded severity differently".

    CUDA_VISIBLE_DEVICES=0 envs/tih/bin/python evaluation/eval_fair.py
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stage1_quantitative"))
from labels import KEY_CHARS, LABEL_MAPS
from infer import Stage1Pipeline

ORDER = LABEL_MAPS                                  # each feature's ordinal order
PRESENCE_FEATS = {"fissure", "tooth_mk", "coating"}  # first class = 'absent/none/non_greasy'

pipe = Stage1Pipeline("checkpoints/seg_combined/best.pt", "checkpoints/multitask_v5/best.pt")
agg = {ch: {"exact": 0, "within1": 0, "presence": 0, "n": 0} for ch in KEY_CHARS}

for setname in ("human40", "human40b"):
    labels = f"evaluation/{setname}_labels.json"
    if not os.path.exists(labels):
        continue
    gold = json.load(open(labels))
    for iid, g in gold.items():
        p = f"data/eval/{setname}/{iid}.jpg"
        if not os.path.exists(p):
            continue
        s = json.loads(pipe(p).to_json())["key_characteristics"]
        for ch in KEY_CHARS:
            gv = g.get(ch)
            if gv is None or gv not in ORDER[ch]:
                continue
            pv = s[ch]["value"]
            gi, pi = ORDER[ch].index(gv), ORDER[ch].index(pv)
            a = agg[ch]; a["n"] += 1
            a["exact"] += (pv == gv)
            a["within1"] += (abs(pi - gi) <= 1)
            if ch in PRESENCE_FEATS:
                a["presence"] += ((gi == 0) == (pi == 0))   # both 'none' or both 'present'

print(f"\n{'feature':10} {'exact':>7} {'within-1':>9} {'presence':>9}   n")
print("-" * 48)
for ch in KEY_CHARS:
    a = agg[ch]; n = max(a["n"], 1)
    pres = f"{a['presence']/n:>8.0%}" if ch in PRESENCE_FEATS else "     — "
    print(f"{ch:10} {a['exact']/n:>7.0%} {a['within1']/n:>9.0%} {pres:>9}   {a['n']}")
tot = {k: sum(agg[ch][k] for ch in KEY_CHARS) for k in ("exact", "within1", "n")}
print("-" * 48)
print(f"{'OVERALL':10} {tot['exact']/tot['n']:>7.0%} {tot['within1']/tot['n']:>9.0%}       —    {tot['n']}")
