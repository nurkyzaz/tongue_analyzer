# Splitting coating into thickness × texture (2026-07-13)

## Why
The single `coating` label (`non_greasy` / `greasy` / `greasy_thick`) conflates two things TCM grades
separately: **thickness** (薄/厚 thin/thick) and **texture** (腻/滑 greasy/slippery vs smooth). The user
found greasiness genuinely hard to judge because surface papillae patterns mimic a greasy coating — but
thickness is easier to read. Conflating them into one 3-way axis loses that.

## What we have (and don't)
No source gives clean, independent thickness- and texture-*labels*: TonguExpert's `L2` coating field is
the conflated 3-way, and its Shape/Texture phenotype files are raw pixel features (coating area, LBP/HOG/
GLCM), not gradable labels. So a two-head **retrain has no independent ground truth to learn from**.

## What we did — faithful derivation (no retrain)
The 3-way label is compositional, so we derive two axes from the model's existing class probabilities:

```
thickness: P(thick)  = P(greasy_thick)              -> thick if >0.5 else thin
texture:   P(greasy) = P(greasy) + P(greasy_thick)  -> greasy if >0.5 else smooth
```

Exact, interpretable, immediately checkable. Implemented in `labels.derive_coat_axes` and surfaced by
`infer.py` as `coat_thickness` / `coat_texture` in `key_characteristics` (the conflated `coating` stays
in the output for backward-compat and still drives Stage-2 pattern voting unchanged).

## Result — the split is worth it (vs the user's 38 hand labels)

| axis | accuracy | `evaluation/eval_coat_axes.py` |
|------|---------:|--------------------------------|
| conflated 3-way coating | 55% | (baseline) |
| **thickness (thin/thick)** | **82%** | reliable |
| texture (smooth/greasy) | 68% | the hard axis |

Thickness is much more reliable than the combined label; texture is where the difficulty lives, and its
dominant error is **smooth→greasy (9/38)** — exactly the papillae-pattern confound. So the split lets the
product show a confident thickness reading and be honest about greasiness instead of forcing one blurry
call.

## Demo
Stage 2 (`interpret.py`) now emits two display cards — **Coating thickness** and **Coating texture** —
each with its own score, and hides the conflated `coating` card (`display=False`; it still votes). The
texture card's plain text notes that patterns can mimic greasiness. Pattern voting is unchanged (no
regression). Frontend `index.html` renders both as graded bars. Live on the demo (v5, port 7860 + tunnel).

## If we want to push accuracy further
Retrain two binary heads using the extra independent signal we *do* have: TCM `botaishe` (thin) and
`huataishe` (slippery/greasy) presence labels, plus the derived axes — via the existing partial-label
training path. That would add real information for the thin/greasy distinction (especially the missing
"thick + smooth" combo the current label can't express).
