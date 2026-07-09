"""Assemble a diverse folder of test tongue images + verify the pipeline produces varied, graded output.

Picks images spanning distinct features (normal, pale, dark, yellow-coating, thick-greasy, cracks,
tooth-marks, and subtle/mild cases to check sensitivity), runs the full pipeline on each, copies the
image to test_images/ with a descriptive name, and writes a README. Doubles as a regression check.

    python scripts/make_test_set.py
"""
import os
import sys
import shutil
import cv2
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline import FullPipeline

OUT = os.path.join(ROOT, "test_images")
DATA = "data/raw"

# (name, filter dict on manifest columns/manual) — one representative each, picked from held-out test split
PICKS = [
    ("01_balanced",        {"coating": "non_greasy", "fissure": "none", "tooth_mk": "none", "zhi": "regular"}),
    ("02_balanced_alt",    {"fissure": "none", "tooth_mk": "none", "zhi": "regular"}),
    ("03_pale_toothmarks", {"zhi": "light", "tooth_mk": "severe"}),
    ("04_dark_body",       {"zhi": "dark"}),
    ("05_yellow_coating",  {"tai": "yellow"}),
    ("06_thick_greasy",    {"coating": "greasy_thick"}),
    ("07_deep_cracks",     {"fissure": "severe"}),
    ("08_toothmarks",      {"tooth_mk": "severe"}),
    ("09_mild_cracks",     {"fissure": "light"}),      # subtle -> sensitivity check
    ("10_mild_toothmarks", {"tooth_mk": "light"}),     # subtle -> sensitivity check
]


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    df = pd.read_csv("data/processed/manifest.csv")
    test = df[df.split == "test"]
    pipe = FullPipeline("checkpoints/seg_combined/best.pt", "checkpoints/multitask_v3/best.pt")

    lines = ["# Test tongue images\n",
             "Diverse held-out tongues for trying the demo. Each row shows the model's reading.\n",
             "Upload any of these at the demo URL.\n",
             "| file | model reading (graded features) | top pattern |", "|---|---|---|"]
    used = set()
    for name, filt in PICKS:
        cand = test
        for col, val in filt.items():
            cand = cand[(cand[col] == val) | (cand.get(f"{col}_manual") == val)]
        row = next((r for _, r in cand.iterrows() if r.sid not in used), None)
        if row is None:
            print(f"[skip] {name}: no match"); continue
        used.add(row.sid)
        img = cv2.cvtColor(cv2.imread(os.path.join(DATA, row.raw_path)), cv2.COLOR_BGR2RGB)
        res, _, _ = pipe.analyze_array(img)
        fname = f"{name}.jpg"
        shutil.copy(os.path.join(DATA, row.raw_path), os.path.join(OUT, fname))
        it = res["interpretation"]
        feats = "; ".join(f"{f['value']}({f['severity']:.2f})" for f in it["features"]
                          if f.get("mentioned") or f["key"] in ("tai", "zhi"))
        top = it["patterns"][0]["tcm_name"] if it["patterns"] else "-"
        lines.append(f"| {fname} | {feats} | {top} |")
        print(f"[ok] {name} <- {row.sid}: {feats} -> {top}")

    with open(os.path.join(OUT, "README.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {OUT}/ ({len(used)} images) + README.md")


if __name__ == "__main__":
    main()
