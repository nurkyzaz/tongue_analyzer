"""Build a training manifest from the USER's human labels (the target-convention labels), so we can
fine-tune on them. For each labeled image it runs the segmenter to get an aligned 384-letterboxed
image+mask pair (what the multitask dataset expects) and writes the 5 core labels.

    CUDA_VISIBLE_DEVICES=0 envs/tih/bin/python data/build_human_manifest.py \
        --sets human40:train,human40b:val --out data/processed/human_manifest.csv

Add human_train (once labeled) to --sets to grow it. Split policy is up to the caller (keep an
independent set for validation, e.g. human40b, so we measure generalisation not fit).
"""
import argparse, json, os, sys
import cv2
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stage1_quantitative"))
from labels import KEY_CHARS, LABEL_MAPS
from infer import Stage1Pipeline

OUT_IMG = "data/eval/human_masks"       # aligned 384 image+mask pairs live here


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True, help="comma list of set:split, e.g. human40:train,human40b:val")
    ap.add_argument("--out", default="data/processed/human_manifest.csv")
    ap.add_argument("--seg", default="checkpoints/seg_combined/best.pt")
    ap.add_argument("--mt", default="checkpoints/multitask_v5/best.pt")
    args = ap.parse_args()
    os.makedirs(OUT_IMG, exist_ok=True)
    pipe = Stage1Pipeline(args.seg, args.mt)

    rows = []
    for spec in args.sets.split(","):
        setname, split = spec.split(":")
        labs = json.load(open(f"evaluation/{setname}_labels.json"))
        for iid, lab in labs.items():
            src = f"data/eval/{setname}/{iid}.jpg"
            if not os.path.exists(src):
                continue
            _, m, disp = pipe(src, return_mask=True)          # aligned 384 mask + letterboxed image
            imgp = f"{OUT_IMG}/{setname}_{iid}_img.jpg"
            maskp = f"{OUT_IMG}/{setname}_{iid}_mask.png"
            cv2.imwrite(imgp, cv2.cvtColor(disp, cv2.COLOR_RGB2BGR))
            cv2.imwrite(maskp, (m * 255).astype("uint8"))
            row = {"sid": f"{setname}_{iid}", "raw_path": imgp, "mask_path": maskp, "has_mask": 1, "split": split}
            for ch in KEY_CHARS:
                v = lab.get(ch)
                row[ch] = v if (v in LABEL_MAPS[ch]) else None
            rows.append(row)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}  n={len(df)}  splits={dict(df.split.value_counts())}")
    for ch in KEY_CHARS:
        print(f"  {ch:9} labeled {int(df[ch].notna().sum())}")


if __name__ == "__main__":
    main()
