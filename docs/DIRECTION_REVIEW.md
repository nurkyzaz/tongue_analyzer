# Direction review & execution plan (2026-07-13)

Assessment of whether the project is on track to be "the most accurate model to detect insightful
information from a single tongue image," and the plan that follows from it.

## Verdict
Architecture and rigor are **right**; the risks are in the training *signal*, the single-image
*constraint*, and real-world *domain gap* — not in the two-stage design. "Accurate" (feature detection,
has ground truth) and "insightful" (patterns/constitution, no ground truth, tongue = 1 of 4 exams) are
different problems with different ceilings and different levers.

### What's right (keep)
- Two-stage split: learnable/measurable feature detection, then transparent rule-based interpretation.
  Correctly rejects black-box image→syndrome (circular, unvalidatable).
- Honest measurement: real human-vs-model ~55–61% (not the optimistic 0.87 auto-benchmark); refuses to
  claim disease/constitution accuracy.
- Correct bottleneck diagnosis: label/data quality, not model capacity (v6/v7 regressed by fitting noise).
- Strong, citable grounding (ICD-11, CCMQ, SymMap, Maciocia, Delphi); mapping now auditable.

### Ranked risks / gaps
1. **Training signal is still noisy auto-labels.** The human eval set diagnoses but doesn't fix the
   ceiling — the model can't exceed its training-label quality. **#1 accuracy lever.**
2. **Single image leaves the strongest signals unused:** sublingual veins (top objective blood-stasis
   sign, AUC 0.917), moisture (wet/dry — separates yang-def from yin-def), zoning (data exists, unwired).
3. **Real-world domain gap under-measured:** eval images are curated pools, not diverse phone captures;
   color features (drive Heat/Damp-Heat) are what phone casts corrupt; color calibration shipped OFF.
4. **Mapping is additive / context-blind:** swelling means different things by body colour — needs
   combination rules (test harness now exists).
5. **Power + licensing:** ~78 eval images = wide error bars; SM-Tongue (CC-BY-NC) + unverified
   TonguExpert terms block commercial shipping.

---

## Execution plan (workstreams, sequenced by value × tractability)

### WS1 — Training signal (attacks the accuracy ceiling) · P0
- **1a** Audit L1 manual-gold coverage. ✅ (tai 378 / zhi 339 / fissure 572 / tooth_mk 656; no coating)
- **1b** Build a **gold-preferred** training manifest: manual labels high-weight, auto labels low-weight,
  per-characteristic (partial-label infra already supports it).
- **1c** Retrain **v8** (resnet34@384 + EMA + gold weight 4.0), eval on human40. ✅ done
- **1d** Promote if it beats v5. ❌ **v8 not promoted** — val↑ (0.745) but human-40 61→59% (lifted
  tooth_mk, regressed coating which has no gold). **Finding: the few gold labels we have can't be
  re-weighted into a win; WS1 now needs MORE/CLEANER labels or per-characteristic training** (see
  ACCURACY_INVESTIGATION §v8). Redirect: (i) harvest more expert-graded coating data; (ii) per-char
  fine-tuning that freezes coating; (iii) grow the human train/val split itself.

### WS2 — Unused high-insight signals (single-image) · P1
- **2a** **Moisture** from specular gloss on the mask. ✅ done — `zoning.py` measures the specular-
  highlight fraction; asserts only **wet** (validated: high-gloss tongues are genuinely moist, e.g. t14).
  **"dry" deliberately NOT inferred** from low gloss (that's just diffuse lighting — 26/38 would falsely
  read dry; honest gap, needs controlled capture/texture model). Surfaced in `zoned_analysis.moisture`,
  wired in `interpret.py` (wet → yang_deficiency/phlegm_dampness). Mapping test +1 case, 11/11. Live on
  the demo. Added `moisture` to the next-round labeling schema for validation.
- **2b** Wire **red_tip → Heart-heat** vote (detected & validated this session; currently doesn't vote).
- **2c** **Zone-route red_dots** via zoning; add mask **deviation** (asymmetry) heuristic.
- **2d** Scope the **sublingual-vein** second-capture module (design only this round).

### WS3 — Real-world robustness · P1
- **3a** Real-phone eval set across skin tones/lighting (needs user-supplied photos). ⬜ blocked on data.
- **3b** Color calibration ✅ measured (synthetic warm/cool casts + strength sweep, `eval_color_calib.py`).
  Finding: helps `tai` everywhere + rescues both colours under casts, but costs `zhi` ~8pp on clean
  images (grey-world over-corrects the warm face). **Decision: keep OFF by default** (doesn't beat the
  honest clean metric), lowered default strength 0.6→**0.35** (the knee), **recommend enabling once
  real-phone photos confirm net benefit**. See `docs/COLOR_CALIBRATION.md`.

### WS4 — Mapping quality · P2
- **4a** Expand the mapping test set with more grounded combinations. ✅ 12 cases incl. flip contrasts.
- **4b** **Combination rules** ✅ done — `interpret.apply_combination_rules` + 9 grounded rules in
  `tcm_knowledge.json['combination_rules']` (read RAW detections via `present_features`, so context is
  seen even for non-distinctive features). Fixes the additive-voting blind spot: the **same swelling now
  flips** — pale+swollen+wet → yang_deficiency, red+swollen+yellow → damp-heat. Also: wet/pale suppress
  heat; greasy-white=damp vs greasy-yellow=damp-heat; dark+red=heat vs dark+purple=stasis; red+cracks+
  scanty=yin. Mapping test 12/12; live on the demo.

### WS5 — Rigor & shipping · P2
- **5a** Grow human eval + second labeler (inter-rater agreement = irreducible-noise floor).
- **5b** Resolve commercial licensing (SM-Tongue, TonguExpert).

**Executing now:** WS1 (gold-preferred v8, background) + WS2b (red_tip voting, contained). Status tracked
in the session task list.
