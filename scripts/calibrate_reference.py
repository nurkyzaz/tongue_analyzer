"""Calibrate population reference stats so interpretation highlights what's DISTINCTIVE per person.

Runs the pipeline over a mixed sample and records, per feature: mean/percentiles of severity (graded)
and value frequencies (categorical) + extra-feature base rates. The interpreter then weights each
feature's pattern vote by how UNUSUAL it is (rare/high => informative), so common features (e.g. a
slightly greasy coating that ~everyone has) don't drive everyone to the same pattern.

Also emits data-grounded severity-band cut points (addresses the 'thresholds lack data support' review).

    python scripts/calibrate_reference.py  ->  stage2_interpretation/knowledge_base/reference_stats.json
"""
import warnings, glob, json, random, os
warnings.filterwarnings("ignore")
import numpy as np
import cv2
import sys
sys.path.insert(0, ".")
from pipeline import FullPipeline
sys.path.insert(0, "stage1_quantitative")
from labels import KEY_CHARS, EXTRA_FEATURES

GRADED = ["coating", "fissure", "tooth_mk"] + EXTRA_FEATURES


def main():
    pipe = FullPipeline("checkpoints/seg_combined/best.pt", "checkpoints/multitask_v3/best.pt")
    imgs = glob.glob("data/external/sm_tongue/SM-Tongue-2155-anonymized/images/*.png")
    imgs += glob.glob("data/raw/TongueImage/Raw/*.jpg")
    random.seed(0); random.shuffle(imgs)
    sev = {k: [] for k in GRADED}
    catfreq = {c: {} for c in ("tai", "zhi")}
    n = 0
    for p in imgs[:250]:
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        res, _, _ = pipe.analyze_array(img)
        if not res["quantitative"]["quality"]["accepted"]:
            continue
        kc = res["quantitative"]["key_characteristics"]
        for c in ("tai", "zhi"):
            v = kc[c]["value"]; catfreq[c][v] = catfreq[c].get(v, 0) + 1
        for f in res["interpretation"]["features"]:
            if f["key"] in sev:
                sev[f["key"]].append(f["severity"])
        n += 1

    stats = {"n": n, "graded": {}, "categorical": {}}
    qs = list(range(0, 101, 5))          # 0,5,...,100 -> quantile curve for percentile-rank lookup
    for k in GRADED:
        a = np.array(sev[k]) if sev[k] else np.array([0.0])
        stats["graded"][k] = {"mean": round(float(a.mean()), 4),
                              "q": [round(float(np.percentile(a, p)), 4) for p in qs]}
    for c in ("tai", "zhi"):
        tot = sum(catfreq[c].values()) or 1
        stats["categorical"][c] = {v: round(cnt / tot, 4) for v, cnt in catfreq[c].items()}

    out = "stage2_interpretation/knowledge_base/reference_stats.json"
    with open(out, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"calibrated on n={n}; wrote {out}")
    print("graded means:", {k: v["mean"] for k, v in stats["graded"].items()})
    print("categorical freqs:", stats["categorical"])


if __name__ == "__main__":
    main()
