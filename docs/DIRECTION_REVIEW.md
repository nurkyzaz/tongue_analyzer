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
- **1c** Retrain **v8** (resnet34@384 + EMA + gold-preferred), eval on human40 + human40b.
- **1d** Promote if it beats v5 on the honest human eval; recalibrate `reference_stats.json`.

### WS2 — Unused high-insight signals (single-image) · P1
- **2a** **Moisture (wet/dry)** signal from specular reflection on the mask (new measured feature).
- **2b** Wire **red_tip → Heart-heat** vote (detected & validated this session; currently doesn't vote).
- **2c** **Zone-route red_dots** via zoning; add mask **deviation** (asymmetry) heuristic.
- **2d** Scope the **sublingual-vein** second-capture module (design only this round).

### WS3 — Real-world robustness · P1
- **3a** Real-phone eval set across skin tones/lighting (needs user-supplied photos).
- **3b** Turn on **color calibration** once the model is WB-robust; measure tai/zhi effect.

### WS4 — Mapping quality · P2
- **4a** Expand the mapping test set with more grounded combinations.
- **4b** **Combination rules** (body-colour-conditioned weights) against the test set.

### WS5 — Rigor & shipping · P2
- **5a** Grow human eval + second labeler (inter-rater agreement = irreducible-noise floor).
- **5b** Resolve commercial licensing (SM-Tongue, TonguExpert).

**Executing now:** WS1 (gold-preferred v8, background) + WS2b (red_tip voting, contained). Status tracked
in the session task list.
