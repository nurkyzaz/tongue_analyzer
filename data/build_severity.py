"""Build continuous severity regression targets from TonguExpert's phenotype measurements.

These give the model a graded 0-1 "degree" signal per feature (the key to the sensitivity fix in
IMPROVEMENT_PLAN.md), complementing the categorical labels. NA in a phenotype means the feature is
absent -> severity 0, which is semantically correct.

    python data/build_severity.py   ->   data/processed/severity.csv  (SID + 3 severity targets)

Signals (validated monotonic with the ordinal labels):
  fissure_sev  = fissure_num * fissure_avgArea_rela   (0 when absent)   / 10, clipped
  toothmk_sev  = tooth_mk_num * tooth_mk_avgArea_rela  (0 when absent)   / 10, clipped
  coating_cov  = tai_div_zhi (coating/body area ratio)                  robust-scaled
"""
import os
import numpy as np
import pandas as pd

P = "data/raw/Phenotypes/"


def _rd(f):
    return pd.read_csv(os.path.join(P, f), sep="\t", na_values=["NA", ""])


def build(out="data/processed/severity.csv"):
    tai = _rd("P22_Tai_Shape.txt")
    fis = _rd("P42_Fissure_Shape.txt")
    tm = _rd("P52_Toothmark_Shape.txt")

    df = pd.DataFrame({"sid": tai["SID"]})
    fis_sev = (fis.fissure_num.fillna(0) * fis.fissure_avgArea_rela.fillna(0))
    tm_sev = (tm.tooth_mk_num.fillna(0) * tm.tooth_mk_avgArea_rela.fillna(0))
    df = df.merge(pd.DataFrame({"sid": fis.SID, "fissure_sev": np.clip(fis_sev / 10.0, 0, 1)}), on="sid", how="left")
    df = df.merge(pd.DataFrame({"sid": tm.SID, "toothmk_sev": np.clip(tm_sev / 10.0, 0, 1)}), on="sid", how="left")
    # coating coverage: robust-scale (10th->1st pct anchor, /200) then clip
    df = df.merge(pd.DataFrame({"sid": tai.SID,
                                "coating_cov": np.clip((tai.tai_div_zhi - 10.0) / 200.0, 0, 1)}), on="sid", how="left")
    df = df.fillna(0.0)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {out}  n={len(df)}")
    for c in ["fissure_sev", "toothmk_sev", "coating_cov"]:
        print(f"  {c:12s} mean={df[c].mean():.3f}  p50={df[c].median():.3f}  p90={df[c].quantile(.9):.3f}")


if __name__ == "__main__":
    build()
