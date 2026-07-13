# Characteristic-model accuracy investigation (2026-07-13)

_Why the model misreads (pale→normal, coating always "greasy"), what we tried, and what actually
moves the needle. Model in production remains **v5** (`multitask_v5`); nothing here beat it._

## How we measured
- **Confusion matrices** vs TonguExpert **expert-manual** gold (`evaluation/diagnose_confusion.py`) —
  per-class, not just aggregate accuracy (which hides minority collapse).
- **False-positive check** on the full test set's auto labels (cracks/tooth-marks/greasy on negatives).
- **Human labels** on real, diverse images (`evaluation/eval_human_labels.py`) — the honest metric, since
  the training auto-labels are themselves noisy.

## Root-cause diagnosis (v5)
| Symptom | Number | Cause |
|---|---|---|
| Coating → always "greasy" | non-greasy recall 40%, 60% of non-greasy flipped | training is **89% greasy / 2% non-greasy** (4092 vs 90 imgs) |
| Pale body → "regular/dark" | `light` recall 59% | model weakness + noisy auto-labels (zhi training *is* balanced) |
| Cracks/tooth-marks over-fire | 9% / 13% FP on smooth tongues | expert test set had **zero negatives** → we were blind to it |

Post-hoc **logit-adjustment calibration does not help** (`calibrate_logit_adjust.py`, τ=0 optimal): the
errors are genuine, not a threshold artifact. The model trains on TonguExpert **auto-labels** and we only
have ~100–160 expert-manual labels per feature to check against — label quality is the ceiling.

## What we tried (all lost to v5)
| Model | Change | Result |
|---|---|---|
| **v6** | ConvNeXt V2 @512 + rebalance | **Regressed** (pale 0.59→0.43). Train loss → 0.017 = memorised the noisy labels. Bigger ≠ better. |
| **v7** | resnet34@384 + TCM-Tongue non-greasy data (90→437) + rarity sampler + EMA | Pale ✓ (0.59→0.65), fissure ✓, but I injected noisy `tai` cross-labels (0.92→0.87). |
| **v7b** | v7 with coating-only harvest | `tai` partly recovered; coating still overshot. |
| **v8** | resnet34@384 + EMA + **gold weight 4.0** (expert L1 labels dominate) | Val meanF1 0.730→0.745, val zhi 0.72→0.75. But on **human-40**: tooth_mk 66→74 ✓, yet coating 55→42 ✗ (coating has NO gold → up-weighting the other 4 chars pulled the shared encoder), fissure 58→53, zhi 62→59. **OVERALL 61→59 — not promoted.** Same non-transfer lesson: val/gold gains don't reach the honest human metric. |

**WS1 (training-signal) conclusion (2026-07-13):** re-weighting the *few* gold labels we have (v8) does
NOT beat v5 on the honest human eval — it improves the chars that have gold at the cost of coating (which
has none). The training-signal fix therefore needs **more/cleaner labels or per-characteristic training**,
not a loss-weight knob. v5 remains production. `TIH_GOLD_WEIGHT` env added (default 2.0 = v5 behaviour).

### Verdict on human labels (10 real images)
| | v5 | v7 | v7b |
|---|---|---|---|
| Coating | **50%** | 10% | 10% |
| Coating color | **70%** | 40% | 50% |
| Body color (pale) | 30% | **50%** | **50%** |
| Fissures | 40% | 40% | 20% |
| Tooth-marks | 50% | 50% | 50% |
| **Overall** | **48%** | 38% | 36% |

## Key findings
1. **The bottleneck is label/data quality and *measurement*, not model capacity or class imbalance.**
   Two independent experiments (bigger model; rebalancing) both made things worse by fitting noise.
2. **"Coating collapse" partly reflects reality.** Humans labeled **9/10 real tongues as coated**
   (greasy/thick), so "mostly greasy" is closer to truth than assumed; the non-greasy oversampling
   *overshot* and made the model call coated tongues bare — worse for real photos. The real coating gap
   is **greasy vs greasy-thick granularity**, not non-greasy detection.
3. **Pale is genuinely improvable** (30%→50% vs human, 0.59→0.65 expert-manual) via EMA + a gentle
   body-colour rebalance — but not at the cost of breaking coating, as v7/v7b did.
4. **Models score ~48% vs a human**, far below the 87% auto-benchmark. A real share of the gap is
   **subjective threshold disagreement** (faint crack = "light" or "none"?) — needs a labeling standard,
   not more training.

## What actually moves the needle (next)
- **A trustworthy human/expert-labeled eval set** — the single highest-value step. Seeded here:
  `data/eval/human10` (labeled) + `data/eval/human40` (stratified to cover minorities, awaiting labels).
  Select more with `evaluation/select_eval_set.py`. Without this, every retrain optimises toward noise.
- Then **targeted, measured** fixes: body-colour-only rebalance for pale; greasy-vs-thick granularity;
  agree crack/tooth-mark severity thresholds.

## Side bug found
`data/build_tcm_tongue_labels.py` maps TCM-Tongue class 1 (`botaishe` = 薄苔 **thin** coating) to
`peeled_coating` — wrong (91% co-occur with a coating colour, impossible for a peeled/absent coating).
So the extra-features model's `peeled_coating` output actually detects **thin** coating, and the KB maps
peeled→yin-deficiency → spurious yin votes. Tracked as a separate fix.

## Reusable infra added
Backward-compatible training flags in `stage1_quantitative/feature_extraction/train.py`:
`--sampler rarity` (oversample images rare in any characteristic), `--ema`, `--cw-clamp-max`; dataset
now supports **partial labels** (weight 0 on unlabeled characteristics) + severity masking, enabling
cross-source data like TCM-Tongue. `data/build_aug_manifest.py` builds the augmented manifest.
