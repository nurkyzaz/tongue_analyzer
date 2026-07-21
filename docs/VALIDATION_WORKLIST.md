# Validation worklist — the "unvalidated heuristics" (for you)

_Created 2026-07-21._ Three new/weak detectors feed the no-LLM rules but aren't yet checked against
labelled data. This is the convenient, prioritised list of what to do about each. **Everything here runs
on casper** (needs the model + datasets); this note just tells you the exact steps.

TL;DR: **only ONE thing actually needs you to label — `red_sides`.** The other two already had labels and
are now **validated (results below)** — the result forced a fix.

---

## ✅ Results — extra features vs practitioner labels (553-img held-out test split, 2026-07-21)

Ran `evaluation/eval_extra_vs_practitioner.py` (extra-features model vs TCM-Tongue/shezhenv3-txt test labels):

| feature | prevalence | AP | F1@.5 | read |
|---|---|---|---|---|
| red_dots | 0.14 | **0.68** | 0.68 | usable ✓ (matches our "reliable" tag) |
| red_tongue | 0.27 | **0.61** | 0.50 | usable ✓ |
| thin | 0.05 | 0.58 | 0.51 | moderate |
| slippery_coating | 0.02 | 0.33 | 0.23 | **weak** — precision 0.14, over-fires |
| swollen | 0.06 | 0.19 | 0.26 | over-predicts (137 FP) — but our rules gate it on co-occurrence |
| purple_body | 0.03 | 0.17 | 0.17 | weak (known low-n) |
| **black_coating** | 0.01 | **0.05** | **0.00** | **broken** — 0 TP / 11 FP; every firing is a false positive |

macro-mAP 0.45 (consistent with the long-documented ~0.46). **`peeled_coating` skipped** — only 1 positive in test.

**Action taken (this is the point of validating):**
- **Disabled both grey-black rules** (`black_moist_extreme_cold`, `black_dry_extreme_heat`) via a new
  `"enabled": false` flag — the detector's "present" is ~always a false positive, so the rules only added
  noise. Kept in the KB with a `disabled_note`; TCM mapping is correct, re-enable when detection works.
- **Halved the `slippery_coat_cold_damp` deltas** (weak detector; co-occurrence gate limits the harm).
- Removed the two grey-black cases from the mapping eval (they tested now-disabled rules) → **34/34**.
- Left `swollen`-based rules as-is: precision is low alone, but every rule using it also requires a
  co-occurring sign (pale+wet, or yellow), which filters the false positives.

Net: the two rules that rested on a **non-functional detector are off**; the ones on **usable detectors
(red_dots, red_tongue, thin, tooth_mk, fissure, tai, zhi)** stand.

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

## 🅱 `black_coating` (灰黑苔) — ✅ **validated → rules disabled** (see Results)

AP 0.05, F1 0.00 on the 553-img test split. The two grey-black rules are now `"enabled": false`.

## 🅲 `slippery_coating` (滑苔) — ✅ **validated → deltas halved** (see Results)

AP 0.33, precision 0.14. `slippery_coat_cold_damp` deltas halved; still gated on pale/white/wet co-occurrence.

---

## What I do with the results
- **`red_sides`:** set `RED_SIDE_THRESH` to the swept value; if precision stays low even at the best threshold,
  I demote the vote (smaller delta) or gate it behind the refine question instead of the base reading.
- **`black`/`slippery`:** if the detector precision is low on the test split, I shrink those rules' boosts (they
  already fire only when the detector triggers, so low base-rate limits the harm — this just right-sizes it).

**Priority order:** 🅰 `red_sides` (only one needing you) → then 🅱/🅲 whenever we're next on casper.
See also `docs/LABELING_GUIDE.md` (rubric) and `docs/PLAN.md` §7-B.
