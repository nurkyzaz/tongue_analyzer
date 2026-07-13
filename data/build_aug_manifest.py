"""Build an AUGMENTED multitask manifest = TonguExpert (5-char) + harvested TCM-Tongue minority examples.

TonguExpert is 89% greasy / 2% non-greasy, which collapses the coating head to "greasy". TCM-Tongue's
`botaishe` (薄苔 = thin coating) class supplies 416 real NON-GREASY coating examples — verified thin (not
peeled): 91% of them co-occur with a coating-colour box, impossible for an absent/peeled coating. We add
them as PARTIAL labels: coating=non_greasy (+ tai colour when a white/yellow-coating box is also present),
all other characteristics left blank so the dataset supervises only what we actually know (weight 0 else).
Paths are made repo-root-relative and a `src` column is added; train with --data-root .

    python data/build_aug_manifest.py   ->  data/processed/manifest_aug.csv
"""
import os, glob
import numpy as np
import pandas as pd

BASE = "data/external/tcm_tongue/shezhenv3-txt"
MASKS = "data/external/tcm_tongue/masks"
THIN, WHITE, YELLOW = 1, 9, 10   # botaishe(thin=non_greasy), baitaishe(white), huangtaishe(yellow)


def tonguexpert():
    df = pd.read_csv("data/processed/manifest.csv")
    df["raw_path"] = "data/raw/" + df["raw_path"].astype(str)
    df["mask_path"] = "data/raw/" + df["mask_path"].astype(str)
    df["src"] = "tonguexpert"
    return df


def tcm_tongue(cols):
    rows = []
    for split in ("train", "val", "test"):
        for lf in sorted(glob.glob(os.path.join(BASE, split, "labels", "*.txt"))):
            cls = set(int(float(l.split()[0])) for l in open(lf) if l.split())
            if THIN not in cls:
                continue
            stem = os.path.splitext(os.path.basename(lf))[0]
            img = os.path.join(BASE, split, "images", stem + ".jpg")
            mask = os.path.join(MASKS, stem + ".png")
            if not (os.path.exists(img) and os.path.exists(mask)):
                continue
            # Harvest ONLY the coating-thickness label (non_greasy). We deliberately do NOT cross-label
            # tai (coating colour) from the co-occurring white/yellow boxes: that box doesn't reliably
            # match TonguExpert's light_yellow/yellow boundary and it regressed tai accuracy (v7 test).
            row = {c: np.nan for c in cols}
            row.update({"sid": "TCM_" + stem, "raw_path": img, "mask_path": mask, "has_mask": True,
                        "coating": "non_greasy", "has_manual": False, "split": split, "src": "tcm_tongue"})
            rows.append(row)
    return pd.DataFrame(rows, columns=cols)


def main():
    te = tonguexpert()
    tc = tcm_tongue(list(te.columns))
    out = pd.concat([te, tc], ignore_index=True)
    out.to_csv("data/processed/manifest_aug.csv", index=False)
    print(f"TonguExpert {len(te)} + TCM-Tongue non_greasy {len(tc)} = {len(out)} rows -> data/processed/manifest_aug.csv")
    for split in ("train", "val", "test"):
        s = out[out.split == split]
        ng = (s.coating == "non_greasy").sum()
        ngt = ((s.coating == "non_greasy") & (s.src == "tcm_tongue")).sum()
        print(f"  {split}: {len(s)} rows | coating=non_greasy {ng} (+{ngt} added from TCM-Tongue)")


if __name__ == "__main__":
    main()
