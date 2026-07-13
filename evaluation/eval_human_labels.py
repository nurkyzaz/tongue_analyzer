"""Score a characteristic model against HUMAN labels on a small image set — a clean read that bypasses
the noisy TonguExpert auto-labels. Runs the full deployment path (seg mask -> characteristic model).

    CUDA_VISIBLE_DEVICES=0 envs/tih/bin/python evaluation/eval_human_labels.py \
        --mt checkpoints/multitask_v5/best.pt --labels human_labels.json --images /tmp/lblset
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stage1_quantitative"))
from labels import KEY_CHARS, CHAR_DESC
from infer import Stage1Pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mt", required=True)
    ap.add_argument("--seg", default="checkpoints/seg_combined/best.pt")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    gold = json.load(open(args.labels))
    pipe = Stage1Pipeline(args.seg, args.mt, size=args.size)

    per_char = {ch: [0, 0] for ch in KEY_CHARS}   # [correct, total]
    print(f"\n=== {args.mt} ===")
    for iid in sorted(gold):
        path = None
        for ext in (".jpg", ".png", ".jpeg"):
            p = os.path.join(args.images, iid + ext)
            if os.path.exists(p):
                path = p; break
        if not path:
            print(f"  {iid}: image not found"); continue
        s1 = json.loads(pipe(path).to_json())
        pred = {ch: s1["key_characteristics"][ch]["value"] for ch in KEY_CHARS}
        marks = []
        for ch in KEY_CHARS:
            g = gold[iid].get(ch)
            if g is None:
                continue
            ok = pred[ch] == g
            per_char[ch][0] += ok; per_char[ch][1] += 1
            marks.append(f"{ch}:{'ok' if ok else pred[ch]+'≠'+g}")
        if not args.quiet:
            print(f"  {iid}: " + "  ".join(marks))
    print("  --- per-characteristic accuracy vs human ---")
    tot_c = tot_n = 0
    for ch in KEY_CHARS:
        c, n = per_char[ch]; tot_c += c; tot_n += n
        print(f"  {CHAR_DESC[ch]:34s} {c}/{n} = {c/max(n,1):.0%}")
    print(f"  {'OVERALL':34s} {tot_c}/{tot_n} = {tot_c/max(tot_n,1):.0%}")


if __name__ == "__main__":
    main()
