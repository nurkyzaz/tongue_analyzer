"""Pick clear teaching examples from the dataset for the demo's 'What different tongues mean' gallery.

For each target sign it finds a held-out (test-split) image whose label shows that sign clearly AND
on which the model confidently agrees, copies the image into the demo's static folder, and records
the model's reading + a grounded explanation (via Stage 2).

    python scripts/build_examples.py
Images are de-identified TonguExpert samples (public research dataset) — attributed in the UI.
"""
import os
import sys
import json
import shutil
import cv2
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "stage1_quantitative"))
sys.path.insert(0, os.path.join(ROOT, "stage2_interpretation"))
from pipeline import FullPipeline
from interpret import interpret

# (focus characteristic, target value, teaching title)
TARGETS = [
    ("tooth_mk", "severe", "Pronounced tooth marks"),
    ("coating", "greasy_thick", "Thick greasy coating"),
    ("tai", "yellow", "Yellow coating"),
    ("zhi", "light", "Pale tongue body"),
    ("zhi", "dark", "Red / dark tongue body"),
    ("fissure", "severe", "Deep cracks / fissures"),
]
OUT_DIR = os.path.join(ROOT, "deployment", "api", "static", "examples")
DATA_ROOT = "data/raw"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv("data/processed/manifest.csv")
    test = df[df.split == "test"]
    pipe = FullPipeline("checkpoints/seg_combined/best.pt", "checkpoints/multitask_v3/best.pt")

    examples = []
    used = set()
    for focus, val, title in TARGETS:
        # candidate SIDs whose label (manual preferred) shows the target sign
        col_m = f"{focus}_manual"
        cand = test[(test.get(col_m) == val) | (test[focus] == val)] if col_m in test else test[test[focus] == val]
        best = None
        for _, row in cand.iterrows():
            if row.sid in used:
                continue
            img = cv2.cvtColor(cv2.imread(os.path.join(DATA_ROOT, row.raw_path)), cv2.COLOR_BGR2RGB)
            res, mask, disp = pipe.analyze_array(img)
            kc = res["quantitative"]["key_characteristics"]
            if not res["quantitative"]["quality"]["accepted"]:
                continue
            if kc[focus]["value"] != val:            # model must agree the sign is present
                continue
            conf = kc[focus]["confidence"]
            if best is None or conf > best[0]:
                best = (conf, row.sid, row.raw_path, kc, res)
        if best is None:
            print(f"[skip] no confident example for {focus}={val}")
            continue
        conf, sid, raw_path, kc, res = best
        used.add(sid)
        fname = f"{focus}_{val}.jpg"
        shutil.copy(os.path.join(DATA_ROOT, raw_path), os.path.join(OUT_DIR, fname))
        interp = res["interpretation"]
        fr = next((c for c in interp["features"] if c["key"] == focus), {})
        focus_meaning = ((fr.get("plain") or "") if fr else "")
        examples.append({
            "file": fname, "title": title, "focus": focus,
            "characteristics": {k: {"value": v["value"], "confidence": v["confidence"],
                                    "description": v["description"]} for k, v in kc.items()},
            "focus_meaning": focus_meaning,
            "patterns": interp["patterns"],
            "combined": interp["combined"],
        })
        print(f"[ok] {focus}={val} -> {sid} ({fname}), conf={conf:.2f}")

    with open(os.path.join(OUT_DIR, "examples.json"), "w", encoding="utf-8") as f:
        json.dump({"attribution": "De-identified samples from the TonguExpert public dataset (biosino.org).",
                   "examples": examples}, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {len(examples)} examples to {OUT_DIR}/examples.json")


if __name__ == "__main__":
    main()
