"""Convenient model testing against the unified label store (data/processed/label_store.csv).

Pick a label SOURCE and it scores the model on every image that source labels, per feature:
  - categorical features (coating/tai/zhi/fissure/tooth_mk with graded gold) -> exact-match accuracy
  - presence features (gold is present/absent) -> precision / recall / F1 (model output binarized)

  # honest gold: 5 core + extras on the user's 38 hand-labeled images
  CUDA_VISIBLE_DEVICES=0 envs/tih/bin/python evaluation/eval_model.py --source human

  # large PROFESSIONAL test: fissure / tooth-marks / red-dots presence on the TCM test split
  #   (test split only -> the model did not train on these)
  CUDA_VISIBLE_DEVICES=0 envs/tih/bin/python evaluation/eval_model.py \
      --source practitioner --split test --features fissure,tooth_mk,red_dots --limit 400
"""
import argparse, json, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stage1_quantitative"))
import pandas as pd
from infer import Stage1Pipeline
from zoning import analyze

CORE = {"coating", "tai", "zhi", "fissure", "tooth_mk"}
# feature -> how to turn a prediction into "present"/"absent" for presence-style gold
PRESENCE = {
    "fissure":          lambda s, z: s["key_characteristics"]["fissure"]["value"] != "none",
    "tooth_mk":         lambda s, z: s["key_characteristics"]["tooth_mk"]["value"] != "none",
    "red_dots":         lambda s, z: s["extra_characteristics"].get("red_dots", {}).get("value") == "present",
    "red_tongue":       lambda s, z: s["extra_characteristics"].get("red_tongue", {}).get("value") == "present",
    "purple_body":      lambda s, z: s["extra_characteristics"].get("purple_body", {}).get("value") == "present",
    "swollen":          lambda s, z: s["extra_characteristics"].get("swollen", {}).get("value") == "present",
    "thin_body":        lambda s, z: s["extra_characteristics"].get("thin", {}).get("value") == "present",
    "thin_coating":     lambda s, z: s["extra_characteristics"].get("peeled_coating", {}).get("value") == "present",
    "black_coating":    lambda s, z: s["extra_characteristics"].get("black_coating", {}).get("value") == "present",
    "slippery_coating": lambda s, z: s["extra_characteristics"].get("slippery_coating", {}).get("value") == "present",
    "red_tip":          lambda s, z: (z.get("tip_redness_delta") or -9) > 2.0,   # strong-tip flag
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mt", default="checkpoints/multitask_v5/best.pt")
    ap.add_argument("--seg", default="checkpoints/seg_combined/best.pt")
    ap.add_argument("--store", default="data/processed/label_store.csv")
    ap.add_argument("--source", required=True, choices=["human", "expert", "auto", "practitioner"])
    ap.add_argument("--split", default=None, help="restrict to a dataset split (e.g. test)")
    ap.add_argument("--features", default=None, help="comma list; default = all this source labels")
    ap.add_argument("--limit", type=int, default=None, help="cap #images (per feature set) for speed")
    args = ap.parse_args()

    df = pd.read_csv(args.store)
    df = df[df.source == args.source]
    if args.split:
        df = df[df.split == args.split]
    if args.features:
        want = set(args.features.split(","))
        df = df[df.feature.isin(want)]
    if df.empty:
        print("no labels match that filter"); return

    # need zoning only if red_tip is in scope
    need_zoning = "red_tip" in set(df.feature)
    imgs = sorted(df.image_path.unique())
    if args.limit:
        imgs = imgs[:args.limit]
    gold = defaultdict(dict)
    for _, r in df[df.image_path.isin(imgs)].iterrows():
        gold[r.image_path][r.feature] = r.value

    pipe = Stage1Pipeline(args.seg, args.mt)
    # cat: [correct,total]; pres: [tp,fp,fn,tn]
    cat = defaultdict(lambda: [0, 0])
    pres = defaultdict(lambda: [0, 0, 0, 0])
    n_ok = 0
    for i, path in enumerate(imgs):
        if not os.path.exists(path):
            continue
        if need_zoning:
            out, m, disp = pipe(path, return_mask=True)
            z = analyze(disp, m)
        else:
            out, z = pipe(path), {}
        s = json.loads(out.to_json())
        if not s["quality"]["accepted"]:
            continue
        n_ok += 1
        for feat, gval in gold[path].items():
            # presence-style scoring: TCM gold is present/absent; human ordinal gold (none/few/many,
            # none/mild/strong) is normalised to absent/present so it scores too.
            if feat in PRESENCE and (gval in ("present", "absent") or gval in ("none", "few", "many", "mild", "strong", "no", "yes")):
                pv = PRESENCE[feat](s, z)
                gv = gval in ("present", "few", "many", "mild", "strong", "yes")
                idx = 0 if (pv and gv) else 1 if (pv and not gv) else 2 if (not pv and gv) else 3
                pres[feat][idx] += 1
            elif feat in CORE:
                pv = s["key_characteristics"][feat]["value"]
                cat[feat][0] += (pv == gval); cat[feat][1] += 1
        if args.limit and (i + 1) % 100 == 0:
            print(f"  ...{i+1}/{len(imgs)}")

    print(f"\n=== {args.mt}  vs  source={args.source}"
          f"{' split='+args.split if args.split else ''}  ({n_ok} images scored) ===")
    if cat:
        print("categorical (exact-match accuracy):")
        tc = tn = 0
        for f in sorted(cat):
            c, n = cat[f]; tc += c; tn += n
            print(f"  {f:14} {c:4}/{n:<4} = {c/max(n,1):.0%}")
        print(f"  {'OVERALL':14} {tc:4}/{tn:<4} = {tc/max(tn,1):.0%}")
    if pres:
        print("presence (precision / recall / F1  @  positive rate):")
        for f in sorted(pres):
            tp, fp, fn, tn = pres[f]
            prec = tp/(tp+fp) if tp+fp else 0
            rec = tp/(tp+fn) if tp+fn else 0
            f1 = 2*prec*rec/(prec+rec) if prec+rec else 0
            base = (tp+fn)/max(tp+fp+fn+tn, 1)
            print(f"  {f:16} P={prec:.2f} R={rec:.2f} F1={f1:.2f}  (pos={tp+fn}/{tp+fp+fn+tn}={base:.0%})")


if __name__ == "__main__":
    main()
