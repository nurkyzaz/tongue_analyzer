"""Multimodal image->syndrome benchmark: run OUR image pipeline on tongue photos and score it against
expert answers — the closest thing to a real image->syndrome check of the vision stack.

Layout (ZhongJing-OMNI `TongueDiagnosis/` convention, so it runs on that set the day it is released):
    <data_dir>/images/<id>.png
    <data_dir>/answers/<id>_answer.txt      # expert answer (free Chinese text: tongue signs + syndrome)
    <data_dir>/questions/<id>_question.txt   # optional

For each image we:
  1. run the production pipeline (seg + v5 characteristics + extra features) -> our tongue features + KB pattern,
  2. parse the expert answer's tongue text with the SAME Chinese parser used in benchmark_syndrome.py
     -> the expert-stated features, and map its syndrome words -> a coarse TCM axis,
  3. report (a) FEATURE agreement (does our vision read the same body-colour / coating / marks the expert
     described?) and (b) AXIS consistency (does our pattern point the expert's thermal/deficiency way?).

Honest status: ZhongJing-OMNI's multimodal tongue data is announced but NOT publicly released (both its
GitHub and HF repos ship only README + a single `demo.png`). So this runs as a worked **n=1** example on
that one public image today; the harness is general and scales to the full set with no code change.

    CUDA_VISIBLE_DEVICES=0 envs/tih/bin/python evaluation/benchmark_multimodal.py \
        --data data/external/ZhongJing-OMNI/TongueDiagnosis \
        --seg checkpoints/seg_combined/best.pt --mt checkpoints/multitask_v5/best.pt
"""
import argparse, glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "stage1_quantitative"))
sys.path.insert(0, os.path.join(ROOT, "stage2_interpretation"))

# reuse the exact Chinese tongue parser + axis maps from the text benchmark (single source of truth)
from benchmark_syndrome import parse_tongue, GOLD_AXIS, PATTERN_AXIS, load_kb  # noqa: E402

# features we can compare head-to-head between our vision output and the expert's stated tongue
CORE = {"zhi": "body colour", "tai": "coating colour", "coating": "coating thickness"}
MARKS = {"fissure": "cracks", "tooth_mk": "tooth-marks"}
EXTRA = {"swollen": "swollen", "thin": "thin", "red_tongue": "red body", "purple_body": "purple/dusky",
         "red_dots": "red dots", "peeled_coating": "peeled coating"}


def expert_axis(text):
    """Coarse TCM axis(es) named in the expert answer (syndrome words anywhere in the free text)."""
    return {ax for kw, ax in GOLD_AXIS if kw in text}


def expert_features(answer):
    """The tongue features the expert explicitly stated, via the shared Chinese parser."""
    chars, extra = parse_tongue(answer)
    feats = dict(chars)
    for k in extra:
        feats[k] = "present"
    return feats


def our_features(result):
    """Flatten our pipeline output to the same comparable feature space."""
    q = result["quantitative"]["key_characteristics"]
    feats = {k: v["value"] for k, v in q.items()}
    for k, v in result["quantitative"].get("extra_characteristics", {}).items():
        if v.get("value") == "present":
            feats[k] = "present"
    return feats


def compare_features(ours, exp):
    """Only score features the expert actually mentioned (they don't describe every axis)."""
    rows, agree, total = [], 0, 0
    for feat, label in {**CORE, **MARKS, **EXTRA}.items():
        if feat not in exp:
            continue
        total += 1
        ev = exp[feat]
        if feat in MARKS or feat in EXTRA:            # presence features: expert-present vs our value != absent/none
            ok = ours.get(feat, "none") not in ("none", "absent")
        else:                                          # categorical: exact class match
            ok = ours.get(feat) == ev
        agree += ok
        rows.append(f"    {label:16s} expert={ev!s:12s} ours={ours.get(feat, '-')!s:12s} {'OK' if ok else 'x'}")
    return rows, agree, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="ZhongJing TongueDiagnosis-layout dir")
    ap.add_argument("--seg", default="checkpoints/seg_combined/best.pt")
    ap.add_argument("--mt", default="checkpoints/multitask_v5/best.pt")
    args = ap.parse_args()

    from pipeline import FullPipeline
    pipe = FullPipeline(args.seg, args.mt)

    imgs = sorted(glob.glob(os.path.join(args.data, "images", "*.png")) +
                  glob.glob(os.path.join(args.data, "images", "*.jpg")))
    if not imgs:
        print(f"No images under {args.data}/images/ — ZhongJing-OMNI's tongue set is not yet released "
              f"(only demo.png is public). Harness is ready; drop in the images to run.")
        return

    n = ax_hit = feat_agree = feat_total = 0
    for img in imgs:
        sid = os.path.splitext(os.path.basename(img))[0]
        ans_path = os.path.join(args.data, "answers", f"{sid}_answer.txt")
        if not os.path.exists(ans_path):
            continue
        answer = open(ans_path, encoding="utf-8").read()
        result = pipe.analyze(img)
        ours = our_features(result)
        pats = result["interpretation"].get("patterns", [])
        our_pat = pats[0]["id"] if pats else "balanced"
        our_ax = PATTERN_AXIS.get(our_pat, set())
        exp = expert_features(answer)
        eax = expert_axis(answer)

        n += 1
        rows, a, t = compare_features(ours, exp)
        feat_agree += a; feat_total += t
        consistent = bool(our_ax & eax) if eax else None
        if consistent:
            ax_hit += 1
        print(f"[{sid}] our pattern={our_pat} axis={sorted(our_ax)} | expert axis={sorted(eax)} "
              f"-> {'CONSISTENT' if consistent else ('miss' if eax else 'no-axis')}")
        print("\n".join(rows))
        print(f"    feature agreement: {a}/{t}\n")

    print("=" * 60)
    print(f"Images scored: {n}")
    if n:
        print(f"Axis consistency (our pattern axis ∈ expert axis): {ax_hit}/{n}")
        print(f"Feature agreement (expert-mentioned features): {feat_agree}/{feat_total} "
              f"= {feat_agree/max(feat_total,1):.0%}")


if __name__ == "__main__":
    main()
