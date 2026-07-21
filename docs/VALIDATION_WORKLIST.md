# Validation worklist — the "unvalidated heuristics" (for you)

_Created 2026-07-21._ Three new/weak detectors feed the no-LLM rules but aren't yet checked against
labelled data. This is the convenient, prioritised list of what to do about each. **Everything here runs
on casper** (needs the model + datasets); this note just tells you the exact steps.

TL;DR: **only ONE thing actually needs you to label — `red_sides`.** The other two already have labels.

---

## 🅰 `red_sides` (Liver/GB zone) — **needs your labels** ⭐ the only real labeling task

Why: it's a brand-new geometry signal (`side_redness_delta` → `red_sides` → qi-stagnation) and there is **no
existing label for "are the side edges red"** anywhere (the practitioner "zone" labels are shape, not colour).
Its threshold (`RED_SIDE_THRESH = 2.5` in `stage1_quantitative/infer.py`) is a guess.

It now slots straight into your existing tool — I already added the `red_sides` field + rubric.

**Fast path (~10 min, human-40 set):**
1. Rebuild the label tool (now includes the `red_sides` field):
   ```bash
   python3 evaluation/build_label_tool.py --mode extra      # -> evaluation/label_human40.html
   ```
2. Open `evaluation/label_human40.html`, go through the 38 tongues, set **red_sides = none / mild / strong**
   for each (rubric in `docs/LABELING_GUIDE.md` → Extras: *left+right edges redder than the middle? ignore the tip*).
   Export → overwrite `evaluation/human40_extra_labels.json`.
3. Calibrate the threshold — the sweep is already wired (mirrors how red_tip's 2.0 was chosen):
   ```bash
   python3 evaluation/eval_extra_features.py
   ```
   It prints `side_redness_delta` by class + a precision/recall/F1 sweep and a **suggested `RED_SIDE_THRESH`**.
   Tell me the number (or I read it off) and I update `infer.py`.

**Want more positives (red sides is rare — human-40 may only have a few):** rank a bigger set by the signal
and label just the top candidates. On casper:
```bash
python3 - <<'PY'
import os, glob, json, sys
sys.path.insert(0, "stage1_quantitative")
from infer import Stage1Pipeline; from zoning import analyze
pipe = Stage1Pipeline("checkpoints/seg_combined/best.pt", "checkpoints/multitask_v5/best.pt")
scored = []
for p in glob.glob("data/eval/**/*.jpg", recursive=True)[:800]:   # point at any image dir
    try:
        _, m, disp = pipe(p, return_mask=True)
        z = analyze(disp, m)
        if z.get("side_redness_delta") is not None: scored.append((z["side_redness_delta"], p))
    except Exception: pass
scored.sort(reverse=True)
print("TOP red-sided candidates to eyeball/label:")
for s, p in scored[:40]: print(f"  {s:+.2f}  {p}")
PY
```
Copy the top files into a folder and point the label tool at it (`--dir <that folder>`), or just eyeball them
to sanity-check that high-score = genuinely red edges.

---

## 🅱 `black_coating` (灰黑苔) — **already labelled, no work from you**

The **practitioner set (TCM-Tongue, class 11 → `black_coating`)** already labels this on 6,528 images in the
label store. It needs a **casper eval run** against the TCM-Tongue **test split** (553 imgs the extra model never
trained on), not new labels. Ping me and I'll wire the check (precision/recall of the `black_coating` head vs the
practitioner label); if it's weak we soften the two grey-black rules' deltas. Rules affected:
`black_moist_extreme_cold`, `black_dry_extreme_heat`.

## 🅲 `slippery_coating` (滑苔) — **already labelled, no work from you**

Same story — TCM-Tongue class 12 → `slippery_coating`. Same casper eval, no labeling. Rule affected:
`slippery_coat_cold_damp`.

---

## What I do with the results
- **`red_sides`:** set `RED_SIDE_THRESH` to the swept value; if precision stays low even at the best threshold,
  I demote the vote (smaller delta) or gate it behind the refine question instead of the base reading.
- **`black`/`slippery`:** if the detector precision is low on the test split, I shrink those rules' boosts (they
  already fire only when the detector triggers, so low base-rate limits the harm — this just right-sizes it).

**Priority order:** 🅰 `red_sides` (only one needing you) → then 🅱/🅲 whenever we're next on casper.
See also `docs/LABELING_GUIDE.md` (rubric) and `docs/PLAN.md` §7-B.
