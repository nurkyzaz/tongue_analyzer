# Phase 4 — integrating TCM-Tongue for a wider feature vocabulary

_Goal: detect more traditional tongue features than our current 5, using the TCM-Tongue dataset
(6,719 images, 20 practitioner-verified categories, CC BY 4.0, Dryad `10.5061/dryad.1c59zw48r`)._

## The 20 categories → our schema

| # | category (pinyin) | meaning | status |
|---|---|---|---|
| 0 | jiankangshe | healthy tongue | reference |
| 1 | botaishe | **peeled / mirror coating** | **NEW** → Stomach-Yin deficiency |
| 2 | hongshe | **red tongue** | **NEW** (stronger than our "dark") → Heat |
| 3 | zishe | **purple tongue** | **NEW** → Blood stasis |
| 4 | pangdashe | **swollen / chubby** | **NEW** → Dampness / Qi deficiency |
| 5 | shoushe | **thin tongue** | **NEW** → Blood / Yin deficiency |
| 6 | hongdianshe | **red dots / prickles** | **NEW** → Heat |
| 7 | liewenshe | cracked | have (`fissure`) |
| 8 | chihenshe | tooth-marked | have (`tooth_mk`) |
| 9 | baitaishe | white coating | have (`tai`) |
| 10 | huangtaishe | yellow coating | have (`tai`) |
| 11 | heitaishe | **black/grey coating** | **NEW** → extreme Cold/Heat |
| 12 | huataishe | **slippery/wet coating** | **NEW** → Dampness |
| 13-19 | shen/gandan/piwei/xinfei ao/tu | organ-subregion depression/protrusion | defer (esoteric) |

**New features to add:** `peeled_coating`, `red_tongue`, `purple_body`, `swollen`, `thin`, `red_dots`,
`black_coating`, `slippery_coating`. (Organ subregions deferred.)

## Integration approach (robust, no destabilizing v3)

TCM-Tongue is object-detection (YOLO boxes); the categories are whole-tongue attributes, so we
collapse **detections → per-image multi-label presence** (category box present ⇒ attribute present).

TCM-Tongue has **no segmentation masks**, and our model uses mask-guided pooling → generate masks for
its images with our seg model (Dice ~0.97) as a preprocessing step.

Train a single **v4** model with a shared encoder and **partial-label supervision**:
- TonguExpert samples → supervise 5 char heads + severity (mask the extra-features head).
- TCM-Tongue samples → supervise a new **multi-label `extra_features` head** (mask char/severity heads).
- Shared encoder learns from both; each batch mixes sources, loss ignores unlabeled heads.

Then extend the KB (new feature entries + patterns: blood-stasis for purple, yin-deficiency for peeled,
heat for red/red-dots) and the interpreter/inference to surface the new features with dual-language.

## Steps
1. ✅ Download from Dryad (2.34 GB) → `data/external/tcm_tongue/`.
2. Convert YOLO labels → per-image multi-label CSV (`data/processed/tcm_tongue_labels.csv`).
3. Generate seg masks for TCM-Tongue images (batch, our seg model).
4. Add `extra_features` multi-label head + partial-label loss; unified multi-source dataset.
5. Train v4; verify no regression on the 5 chars + severity, plus new-feature AP.
6. Extend KB + interpreter + inference for the new features; regenerate examples; update demo.
