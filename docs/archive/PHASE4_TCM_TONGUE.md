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

## Steps — ✅ DONE (2026-07-10)
1. ✅ Dataset obtained (Dryad behind anti-bot → user browser-downloaded; `-txt`/YOLO format on server).
2. ✅ `data/build_tcm_tongue_labels.py`: YOLO → per-image multi-label CSV (6,528 imgs, 8 new features).
3. ✅ `scripts/precompute_masks.py`: cached seg masks for all TCM-Tongue images.
4. ✅ **Separate** mask-guided multi-label model (`extra_model.py` + `train_extra.py`) — chosen over a
   partial-label joint model so it can't destabilise v3; runs alongside v3 at inference.
5. ✅ Trained: **val mAP 0.452** — strong: peeled 0.70, red-dots 0.62, thin 0.61, red 0.60 AP;
   weaker (rare/subjective): purple 0.27, swollen 0.25, slippery 0.15 → fire conservatively.
6. ✅ KB `extra_features` (map to existing patterns) + `interpret.py` readings/voting + `infer.py`
   auto-loads the model → new features surface in reports with dual-language + severity. Demo updated.

**Design note:** went with two models (v3 + extra) instead of one joint partial-label model — lower
risk, cleaner, ~10 ms total. A future optimization could distil both into one shared-encoder model.
