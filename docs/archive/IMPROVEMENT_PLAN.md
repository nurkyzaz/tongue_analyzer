# Improvement Plan — sensitivity, dual-language interpretation, follow-up flow

_2026-07-09. Companion to [TCM_RESEARCH.md](TCM_RESEARCH.md). Educational framing throughout: we
report features and the traditional associations a practitioner would note — never a diagnosis._

## 0. Diagnosis: why the demo "feels generic regardless of input"

Verified from the code + label stats, not guessed. Three compounding causes:

1. **Argmax over 3 classes throws away degree.** Each characteristic is a single categorical head
   (`fissure ∈ {none,light,severe}`). A faint crack scores highest on `none`, so it is reported as
   "no cracks." The model literally cannot say "a small crack." Every near-normal tongue collapses to
   the same handful of majority labels.
2. **Class imbalance biases toward "normal."** Training distribution is skewed (coating 89% greasy,
   fissure 68% none, tooth_mk 57% none). Even with Focal Loss the model defaults to the majority class
   on ambiguous inputs, erasing subtle findings.
3. **We discard the richest signal we already own.** TonguExpert ships **~1,300 continuous phenotypes**
   (crack depth, coating-coverage %, per-subregion color in RGB/LAB/HSV). We trained on the 5 *categorical*
   labels only. The continuous scores **are** the degree information — unused.
4. **The interpreter fires the same 2–3 patterns.** It keys off argmax classes, so similar inputs →
   identical text.

**Consequence:** the fix is not "a bigger model" — it is **predict and report degree**, and **key the
interpretation off graded severities**.

---

## A. Sensitivity & degree (highest priority — directly fixes the complaint)

**A1. Add regression heads for continuous severity.** Extend the multi-task head with per-feature
regression outputs trained on TonguExpert's continuous phenotypes (normalize each to 0–1). Output becomes,
per feature: a **class** *and* a **continuous severity** (e.g. `fissure: severity 0.28`). Loss = existing
Focal (categorical) + Smooth-L1 (regression) + a monotonicity/consistency term so class and severity agree.

**A2. Ordinal modeling for graded features.** `fissure`, `tooth_mk`, `coating` are ordinal
(none < light < severe). Replace independent softmax with **ordinal regression** (CORN/CORAL heads) so the
model respects the ordering and interpolates degree instead of hard-switching.

**A3. Report everything present, with graded language.** Lower the "worth mentioning" threshold: map
severity → words (`0–.15 none · .15–.35 faint · .35–.6 mild · .6–.8 moderate · .8–1 pronounced`). So a
0.3 crack surfaces as *"faint cracks"* with its meaning, instead of being dropped.

**A4. Calibrate confidence.** Temperature-scale the logits on the validation set so the % shown to users
is meaningful (currently over/under-confident). Report a calibrated confidence per feature.

**A5. Consistency test-time augmentation.** Average predictions over flips/crops to stabilize subtle-feature
detection and give a variance estimate (used to hedge language when uncertain).

_Deliverable:_ Stage-1 output gains `severity` + `severity_band` per feature; interpreter reads severity,
not just class. Measured on TonguExpert's continuous ground truth (MAE per phenotype) + the existing macro-F1.

---

## B. Dual-language interpretation (TCM term + plain gloss)

Rewrite the interpreter around the grounded reference set (schema in TCM_RESEARCH §4). For **each detected
feature** emit both registers:

> **Faint cracks (severity 0.3).** *TCM:* early Yin/fluid depletion. *In plain terms:* often associated
> with dryness, insufficient rest, or not enough fluids.

For **dampness/greasy coating** specifically (the user's example): *TCM:* Dampness/Phlegm the Spleen isn't
transforming → *plain:* **sluggish digestion, bloating, heaviness after meals.** Plain glosses come from
**SymMap** (TCM-symptom → modern-symptom mapping) so they're grounded, not invented.

Pattern names use **WHO ICD-11 Chapter 26** standardized terms (citable) + a friendly plain name
("Spleen-qi deficiency pattern" → "low digestive energy / poor digestion tendency").

---

## C. Optional follow-up-question flow (grounded in CCMQ)

Framed as *"exploring what this traditional framework would ask next,"* never diagnosis.

1. From graded features, compute a **prior** over the 9 CCMQ constitutions / classic patterns (each feature
   `points_to` patterns with weights; severities weight the vote).
2. When one pattern's prior crosses a threshold **and** 1–2 short questions would move it, ask them —
   drawn from the **validated CCMQ items** (e.g. phlegm-dampness item *"Do you often feel your body is heavy
   or not relaxed?"*; Spleen-qi item *"Do you get bloated after eating?"*).
3. Update confidence with a transparent **Bayesian/log-odds** step (each answer has a published-item weight).
4. Present as: *"Your tongue shows signs traditionally linked to **X**. In this tradition, X often goes with
   Y — does that match you?"* → answer refines the confidence bar. Always with the disclaimer.

This reuses a **validated instrument** rather than inventing questions, which is exactly the "reputable
institution" grounding requested.

---

## D. Knowledge base (build once, both interpreter + follow-up read it)

Replace/extend `tcm_knowledge.json` with the schema in TCM_RESEARCH §4, populated from:
- **feature → severity → tcm_term + plain_gloss + icd11 code + points_to** (Maciocia/Kirschbaum + ICD-11 + SymMap)
- **pattern → tongue_signs, associated_symptoms (plain), followup_questions (CCMQ), recommendations
  (constitution-specific diet/lifestyle from Wang Qi guidance).**
Recommendations are **specific to the sign**, not generic ("for a greasy coating with bloating: favour
warm cooked foods, reduce cold/raw and greasy foods" — not just "tonify Qi").

Every entry carries a `source` field. This is authorable now (no model retraining needed) and is the
fastest path to non-generic, richer output.

---

## E. Data & model-quality plan (training-data first, per the ask)

Priority is **richer labels, not just more images**:
1. **Use TonguExpert continuous phenotypes** (already downloaded) → enables A1/A2. _Biggest win, zero new data._
2. **Integrate TCM-Tongue** (6,719, 20 practitioner-verified categories, CC BY 4.0 pending license check)
   → adds features we can't currently detect (red dots, peeled/mirror coating, purple body, thin/swollen shape,
   organ subregions). Requires mapping its 20 categories onto our schema + a detection→classification adapter.
3. **Keep SM-Tongue** for real-photo segmentation robustness (already integrated; NC — eval/demo only).
4. **Color calibration preprocessing** (grey-world / learned white-balance, optional in-frame card) — reduces
   phone-camera color variance, which otherwise corrupts color-based features (`tai`, `zhi`).
5. **Backbone upgrade** for subtle texture: try ConvNeXt-T / Swin-T vs current ResNet-34 (the phenotype
   regression benefits from finer features), keep it mobile-exportable.
6. **Expand the eval set** with real phone photos across skin tones/lighting to measure true sensitivity.

---

## F. Phased roadmap (with the fast wins first)

| Phase | Work | Needs retraining? | Impact on "generic" complaint |
|---|---|---|---|
| **1 (days)** | D: rebuild knowledge base (dual-language, ICD-11/CCMQ/SymMap grounded) + B: interpreter reads it | No | **High** — richer, differentiated text immediately |
| **2 (days)** | A1/A2: regression + ordinal heads on TonguExpert phenotypes; A3 graded language; A4 calibration | Yes (Stage-1 head only) | **High** — subtle features finally surface |
| **3 (1–2 wk)** | C: CCMQ follow-up flow (prior → questions → Bayesian update) in API + demo | No | Medium-high — interactive refinement |
| **4 (1–2 wk)** | E2: integrate TCM-Tongue (license permitting) → wider feature vocabulary; E5 backbone | Yes | Medium — new detectable features |
| **5 (ongoing)** | E4 color calibration, E6 real-photo eval set, TTA | Partly | Robustness / trust |

**Recommended start: Phase 1 + Phase 2 in parallel** — Phase 1 needs no GPU and fixes the demo's tone
today; Phase 2 retrains only the Stage-1 head (hours on GPU 0) and fixes the underlying sensitivity.

## G. Guardrails (unchanged, reinforced)
Educational framing; "traditionally associated with" language; every pattern/recommendation carries a
source; quality gate rejects unusable photos; no disease claims; follow-up questions are exploratory, not
diagnostic. Licensing tracked in RESOURCES.md before anything commercial.
