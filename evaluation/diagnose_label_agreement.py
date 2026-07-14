"""Diagnose: do the USER's human labels agree with PROFESSIONAL labels on the SAME images?
Reads the merged label store and, per feature, compares source=human vs source in {expert, practitioner}
on the same image_path. Prints agreement + every disagreement (so we can eyeball who's right).

    envs/tih/bin/python evaluation/diagnose_label_agreement.py
"""
import pandas as pd
from collections import defaultdict

df = pd.read_csv("data/processed/label_store.csv")
# pivot: (image, feature) -> {source: value}
by = defaultdict(dict)
for _, r in df.iterrows():
    by[(r.image_path, r.feature)][r.source] = str(r.value)

# graded features share a vocabulary between human and TonguExpert-expert -> compare directly.
# for TCM 'practitioner' (present/absent), map the human graded value to presence.
GRADED = ["tai", "zhi", "fissure", "tooth_mk"]
def to_presence(feat, v):
    if v in ("present", "absent"):
        return v
    return "present" if v not in ("none", "no") else "absent"

for pro in ("expert", "practitioner"):
    print(f"\n================  HUMAN  vs  {pro.upper()}  ================")
    for feat in GRADED + (["red_dots"] if pro == "practitioner" else []):
        pairs = []
        for (img, f), d in by.items():
            if f != feat or "human" not in d or pro not in d:
                continue
            hv, pv = d["human"], d[pro]
            if pro == "practitioner":              # TCM is presence-style
                a, b = to_presence(feat, hv), to_presence(feat, pv)
            else:                                   # expert graded -> direct
                a, b = hv, pv
            pairs.append((img, hv, pv, a == b))
        if not pairs:
            continue
        agree = sum(p[3] for p in pairs)
        print(f"\n  {feat}: agree {agree}/{len(pairs)} = {agree/len(pairs):.0%}")
        for img, hv, pv, ok in pairs:
            if not ok:
                print(f"      DISAGREE  human={hv:14} {pro}={pv:10}  {img.split('/')[-1]}")
