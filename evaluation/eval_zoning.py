"""Validate the zoned-colour analyzer against the human-40 labels: does 'tip redder than body' and
'pale coated centre' actually show up in the measurements, and do they correlate with the labels the
model gets wrong (body colour, greasy_thick)?"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stage1_quantitative"))
from infer import Stage1Pipeline
from zoning import analyze

ap = argparse.ArgumentParser()
ap.add_argument("--labels", default="evaluation/human40_labels.json")
ap.add_argument("--images", default="data/eval/human40")
ap.add_argument("--seg", default="checkpoints/seg_combined/best.pt")
ap.add_argument("--mt", default="checkpoints/multitask_v5/best.pt")
args = ap.parse_args()
gold = json.load(open(args.labels))
pipe = Stage1Pipeline(args.seg, args.mt)

rows = []
print(f"{'id':4} {'coating':13} {'zhi':8} | {'tipΔred':>8} {'grad':>6} {'ctrCoat':>8} {'ctrDesat':>8}")
print("-" * 66)
for iid in sorted(gold):
    p = None
    for ext in (".jpg", ".png", ".jpeg"):
        q = os.path.join(args.images, iid + ext)
        if os.path.exists(q):
            p = q; break
    if not p:
        continue
    _, m, disp = pipe(p, return_mask=True)
    z = analyze(disp, m)
    if not z.get("ok"):
        print(f"{iid:4} (mask too small)"); continue
    g = gold[iid]
    r = {"id": iid, "coating": g["coating"], "zhi": g.get("zhi", "?"),
         "tip": z.get("tip_redness_delta"), "grad": z.get("redness_gradient"),
         "coat": z.get("center_coating"), "desat": z.get("center_desat")}
    rows.append(r)
    print(f"{iid:4} {g['coating']:13} {g.get('zhi','?'):8} | "
          f"{r['tip']:8.2f} {r['grad']:6.2f} {r['coat']:8.2f} {r['desat']:8.2f}")

# --- aggregate checks ---
def mean(vs): vs = [v for v in vs if v is not None]; return sum(vs)/len(vs) if vs else float('nan')
print("\n--- does 'tip redder than body' show up? (tip_redness_delta by body colour) ---")
for z in ("light", "regular", "dark"):
    sub = [r for r in rows if r["zhi"] == z]
    print(f"  zhi={z:8} n={len(sub):2}  mean tipΔred={mean([r['tip'] for r in sub]):+.2f}  mean grad={mean([r['grad'] for r in sub]):+.2f}")
print(f"  tongues with tipΔred > +2 (clear red tip): {sorted(r['id'] for r in rows if r['tip'] and r['tip']>2)}")
print("\n--- does 'pale coated centre' track thick coating? (center_coating by coating) ---")
for c in ("non_greasy", "greasy", "greasy_thick"):
    sub = [r for r in rows if r["coating"] == c]
    print(f"  {c:13} n={len(sub):2}  mean center_coating(L+)={mean([r['coat'] for r in sub]):+.2f}  mean desat={mean([r['desat'] for r in sub]):+.2f}")
