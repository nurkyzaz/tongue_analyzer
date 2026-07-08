"""Unified segmentation manifest combining TonguExpert (clinical) + SM-Tongue (real photos).

Paths are stored relative to the project root (~/tongue). Reuses TonguExpert's existing split;
SM-Tongue gets a fresh reproducible 80/10/10 split. This is what fixes the real-photo domain gap.

    python data/build_seg_manifest.py
"""
import argparse
import os
import glob
import pandas as pd

TE_ROOT = "data/raw"
SM_ROOT = "data/external/sm_tongue/SM-Tongue-2155-anonymized"


def build(te_manifest, out, seed=42):
    rows = []

    # TonguExpert: reuse existing split; paths are relative to data/raw
    te = pd.read_csv(te_manifest)
    for _, r in te[te.has_mask].iterrows():
        rows.append({"img_path": os.path.join(TE_ROOT, r.raw_path),
                     "mask_path": os.path.join(TE_ROOT, r.mask_path),
                     "source": "tonguexpert", "split": r.split})

    # SM-Tongue: fresh split
    sm = pd.DataFrame({"img_path": sorted(glob.glob(os.path.join(SM_ROOT, "images", "*.png")))})
    sm["mask_path"] = sm.img_path.str.replace("/images/", "/masks/", regex=False)
    sm = sm[sm.mask_path.apply(os.path.exists)].reset_index(drop=True)
    sm = sm.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(sm)
    sm["split"] = "train"
    sm.loc[: int(n * 0.1), "split"] = "test"
    sm.loc[int(n * 0.1): int(n * 0.2), "split"] = "val"
    for _, r in sm.iterrows():
        rows.append({"img_path": r.img_path, "mask_path": r.mask_path,
                     "source": "sm_tongue", "split": r.split})

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {out}  total={len(df)}")
    print(pd.crosstab(df.source, df.split))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--te-manifest", default="data/processed/manifest.csv")
    ap.add_argument("--out", default="data/processed/seg_manifest.csv")
    args = ap.parse_args()
    build(args.te_manifest, args.out)
