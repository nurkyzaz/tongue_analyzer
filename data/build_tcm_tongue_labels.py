"""Convert the TCM-Tongue (shezhen) YOLO detection labels -> per-image multi-label presence CSV.

Each YOLO label file lists boxes `class cx cy w h`; a class present in the file => that attribute is
present in the image. We keep the NEW features we can't currently detect and map them to readable names.

    python data/build_tcm_tongue_labels.py
    -> data/processed/tcm_tongue_labels.csv  (img_path, split, + binary columns per new feature)
"""
import os
import glob
import pandas as pd

ROOT = "data/external/tcm_tongue/shezhenv3-txt"

# YOLO class id -> our new feature name (only the ones adding NEW vocabulary; overlaps w/ our 5 skipped)
NEW_FEATURES = {
    1: "peeled_coating",    # botaishe   -> Stomach-Yin deficiency
    2: "red_tongue",        # hongshe    -> Heat
    3: "purple_body",       # zishe      -> Blood stasis
    4: "swollen",           # pangdashe  -> Dampness / Qi deficiency
    5: "thin",              # shoushe    -> Blood / Yin deficiency
    6: "red_dots",          # hongdianshe-> Heat
    11: "black_coating",    # heitaishe  -> extreme Cold/Heat
    12: "slippery_coating", # huataishe  -> Dampness
}
COLS = list(dict.fromkeys(NEW_FEATURES.values()))


def build(out="data/processed/tcm_tongue_labels.csv"):
    rows = []
    for split in ("train", "val", "test"):
        img_dir = os.path.join(ROOT, split, "images")
        lab_dir = os.path.join(ROOT, split, "labels")
        for img in sorted(glob.glob(os.path.join(img_dir, "*.jpg"))):
            stem = os.path.splitext(os.path.basename(img))[0]
            lab = os.path.join(lab_dir, stem + ".txt")
            present = {c: 0 for c in COLS}
            if os.path.exists(lab):
                with open(lab) as f:
                    for line in f:
                        parts = line.split()
                        if not parts:
                            continue
                        cid = int(float(parts[0]))
                        if cid in NEW_FEATURES:
                            present[NEW_FEATURES[cid]] = 1
            rows.append({"img_path": img, "split": split, **present})
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {out}  n={len(df)}  (train/val/test = "
          f"{(df.split=='train').sum()}/{(df.split=='val').sum()}/{(df.split=='test').sum()})")
    print("Positive counts per new feature:")
    for c in COLS:
        print(f"  {c:16s} {int(df[c].sum()):5d}  ({df[c].mean()*100:.1f}%)")


if __name__ == "__main__":
    build()
