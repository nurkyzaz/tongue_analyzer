"""Build a unified manifest for the TonguExpert dataset.

Joins raw images + masks + the 5 key-characteristic labels (L2 predicted = weak, all 5,992;
L1 manual = gold, sparse) into one CSV with a reproducible train/val/test split.

Usage:
    python data/build_manifest.py --root <DATA_ROOT> --out data/processed/manifest.csv

<DATA_ROOT> is the extracted TonguExpertDatabase dir containing TongueImage/ and Phenotypes/.
"""
import argparse
import os
import pandas as pd

# 5 key characteristics (SSC-Net's clinically important set), from L2_Labels_Predict columns.
KEY_CHARS = ["coating", "tai", "zhi", "fissure", "tooth_mk"]
# L1 manual file uses `labels_<char>`; note it lacks a `coating` column.
MANUAL_CHARS = ["tai", "zhi", "fissure", "tooth_mk"]


def _read_tsv(path):
    # Only "NA"/empty are missing. "None" is a real negative class (e.g. no fissure / no tooth-mark),
    # so it is kept as the literal category "none" (see _norm below).
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=["NA", ""])


def _norm(series):
    """Lowercase category strings; map explicit 'None'/'none' negatives to 'none'."""
    return series.where(series.isna(), series.str.strip().str.lower())


def build(root: str, out: str, seed: int = 42):
    raw_dir = os.path.join(root, "TongueImage", "Raw")
    mask_dir = os.path.join(root, "TongueImage", "Mask")
    pheno = os.path.join(root, "Phenotypes")

    raw_files = {os.path.splitext(f)[0]: f for f in os.listdir(raw_dir)}
    mask_files = {os.path.splitext(f)[0]: f for f in os.listdir(mask_dir)}

    df = pd.DataFrame({"sid": sorted(raw_files)})
    df["raw_path"] = df["sid"].map(lambda s: os.path.join("TongueImage", "Raw", raw_files[s]))
    df["mask_path"] = df["sid"].map(
        lambda s: os.path.join("TongueImage", "Mask", mask_files[s]) if s in mask_files else ""
    )
    df["has_mask"] = df["mask_path"] != ""

    # L2 weak labels (all images) -> columns coating,tai,zhi,fissure,tooth_mk
    l2 = _read_tsv(os.path.join(pheno, "L2_Labels_Predict.txt"))
    l2 = l2.rename(columns={"SID": "sid", **{f"{c}_label": c for c in KEY_CHARS}})
    for c in KEY_CHARS:
        l2[c] = _norm(l2[c])
    df = df.merge(l2[["sid"] + KEY_CHARS], on="sid", how="left")

    # L1 gold labels (sparse) -> columns <char>_manual
    l1 = _read_tsv(os.path.join(pheno, "L1_Labels_Manual.txt"))
    l1 = l1.rename(columns={"SID": "sid", **{f"labels_{c}": f"{c}_manual" for c in MANUAL_CHARS}})
    for c in MANUAL_CHARS:
        l1[f"{c}_manual"] = _norm(l1[f"{c}_manual"])
    df = df.merge(l1[["sid"] + [f"{c}_manual" for c in MANUAL_CHARS]], on="sid", how="left")
    df["has_manual"] = df[[f"{c}_manual" for c in MANUAL_CHARS]].notna().any(axis=1)

    # Reproducible split. Bias manual-labeled samples toward val/test so eval uses gold labels.
    rng = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    manual = rng[rng.has_manual].index.tolist()
    other = rng[~rng.has_manual].index.tolist()
    split = pd.Series("train", index=rng.index)

    def assign(idxs, val_frac, test_frac):
        n = len(idxs)
        n_val, n_test = int(n * val_frac), int(n * test_frac)
        for i in idxs[:n_test]:
            split[i] = "test"
        for i in idxs[n_test:n_test + n_val]:
            split[i] = "val"

    assign(manual, 0.30, 0.30)   # gold set: lots of val/test for reliable metrics
    assign(other, 0.05, 0.05)
    rng["split"] = split
    df = rng.sort_values("sid").reset_index(drop=True)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)

    # Stats
    print(f"Total: {len(df)} | with mask: {df.has_mask.sum()} | with manual: {df.has_manual.sum()}")
    print("Split:\n", df.split.value_counts())
    print("\nL2 label coverage (non-null):")
    for c in KEY_CHARS:
        print(f"  {c:10s}: {df[c].notna().sum():5d}  classes={df[c].dropna().nunique()}")
    print("\nExample class distribution — coating:\n", df["coating"].value_counts().head(10))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="data/processed/manifest.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build(args.root, args.out, args.seed)
