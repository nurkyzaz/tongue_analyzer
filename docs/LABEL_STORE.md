# Unified label store & model testing (2026-07-13)

Goal: stop juggling per-dataset label files. Merge every label we hold into one table, keyed on the
underlying image, so any model can be tested against any source with a single command.

## Build

```
python data/build_label_store.py    # -> data/processed/label_store.csv  (server-side; needs the datasets)
```

Long format: `image_path, dataset, split, feature, value, source`. One row per (image, feature, value).
Because human-40 ids are linked to their real dataset path via `meta.json`, a human label and a
practitioner label for the **same tongue** land on the same `image_path` (18 such cross-check images).

### What's in it (104,760 rows, 12,534 images)

| source | what | coverage |
|--------|------|----------|
| `human` | the user's hand labels (5 core + 5 extra) | 38 images × 10 features — **gold** |
| `expert` | TonguExpert manual grading (no coating; sparse) | fissure 572, tooth_mk 656, tai 378, zhi 339 |
| `auto` | TonguExpert auto labels (the model's **train** signal) | 5,992 images × 5 core |
| `practitioner` | TCM-Tongue YOLO categories → our schema | 6,528 images: fissure, tooth_mk, red_dots, red_tongue, purple, swollen, thin body/coating, black/slippery coating, tai, zoning |

Splits are carried through, so eval can use the TCM **test** split (553 imgs) the extra-features model
never trained on. Fixed the old `botaishe`→`peeled` bug: class 1 = **thin coating**, mapped to
`thin_coating` (the extra model's misnamed `peeled_coating` output is the matching prediction).

## Test a model

```
# honest gold — 5 core features on the user's 38 hand-labeled images
python evaluation/eval_model.py --source human

# professional presence test — TCM test split (model didn't train on it)
python evaluation/eval_model.py --source practitioner --split test \
       --features fissure,tooth_mk,red_dots,red_tongue,swollen
```
Categorical gold → exact-match accuracy; presence gold (incl. human ordinal none/few/many) →
precision / recall / F1 with the model output binarized.

## Current production model (v5) results

**vs human gold (38 imgs)** — categorical: coating 55 / fissure 58 / tai 66 / tooth_mk 66 / zhi 62,
**overall 61%**. Presence: red_dots P0.84 R0.73 F1**0.78**; red_tip (strong flag) P0.92 R0.39 F1 0.55.

**vs practitioner (TCM test split, 504 scored)** — presence P/R/F1:

| feature | P | R | F1 | positive rate |
|---------|--:|--:|---:|--------------:|
| fissure | 0.60 | 0.84 | **0.70** | 37% |
| red_dots | 0.52 | 0.71 | 0.60 | 8% |
| red_tongue | 0.70 | 0.36 | 0.48 | 25% |
| tooth_mk | 0.30 | 0.26 | 0.28 | 6% |
| swollen | 0.16 | 0.76 | 0.26 | 7% |

Reads: **fissure and red_dots hold up on a large professional set**; **red_tongue under-detected**
(low recall); **swollen and tooth_mk are weak** (swollen barely trained; tooth_mk's presence
binarization over-fires on the rare TCM-positive class even though it scores 66% on the graded human
set). red_dots validated twice now (human F1 0.78, practitioner F1 0.60 — the gap is partly leakage:
some human-40 images are TCM-train, the TCM-test number is the clean one).

`label_store.csv` (9 MB) is a git-ignored build artifact; the builder + eval harness are tracked.
