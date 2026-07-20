# KB recalibration campaign — book-frequency weights vs hand-tuned (2026-07-16)

**Question:** the rule engine (`interpret.vote_patterns`) votes with hand-tuned `points_to` weights in
`tcm_knowledge.json`. Now that we've parsed 3 books into the KG, can we re-derive those weights from the
**empirical book-citation frequency** (kg_graph.json micro layer) and beat the hand-tuned v1?

**Verdict: NO — v1 (hand-tuned) stays production.** Book *citation frequency* is not a good weight
signal. Documented here as a known limitation.

## Method

`stage2_interpretation/kg/recalibrate.py` generates `tcm_knowledge_v2.json`: for each of v1's existing
feature→pattern edges it blends v1's weight-share with the empirical citation-share (restricted to v1's
patterns, magnitude preserved): `share_v2 = (1-λ)·share_v1 + λ·share_emp`. Only v1's **existing** pairs
are re-weighted — v1's edge set is TCM-vetted, and the book extraction's noise lives in spurious *pairs*
(e.g. it cites `pale→yin-deficiency` 5×, which is TCM-wrong and v1 correctly omits), so restricting to
v1's pairs filters that out. Toggle with `TIH_KB_VERSION=v2` (default `v1`), instant rollback.

## Results

Scored on **TCMEval-SDT** (109 tongue-bearing cases, directional consistency vs gold expert syndrome —
the only pattern-level gold we have) + the 17 mapping regression cases. *(human40's 40 images can't
score this: they carry FEATURE labels, and the KB re-weighting changes PATTERNS, not feature detection —
the 61% is a fixed Stage-1 vision metric. Same reason α couldn't be scored there.)*

| KB | mapping (17) | TCMEval-SDT accuracy |
|---|---|---|
| **v1 (hand-tuned, production)** | 17/17 | **69.7%** (76/109) |
| v2 λ=0.35 (conservative blend) | 17/17 | 69.7% (tie — no case flipped) |
| v2 λ=0.7 | 15/17 ✗ | 47.7% ✗ |
| v2 λ=1.0 (pure book shares) | 15/17 ✗ | 49.5% ✗ |

No λ beats v1: a conservative blend ties it, and trusting book frequency more *crashes* accuracy
(69.7% → 49.5%) and regresses mapping.

## Why (the known limitation)

**Citation frequency measures how often books *discuss* a link, not its *diagnostic discriminativeness*.**
A sign mentioned often toward a common pattern gets over-weighted; a decisive-but-rarely-restated link
(e.g. purple→blood-stasis) gets under-weighted. Hand-tuned weights encode diagnostic importance; raw
mention-counts don't. So the books' real value is the **cited edges themselves** (grounding + citations,
already used by the graph-RAG ensemble) and the disambiguation nuance — **not** re-deriving scalar
weights. A future data-derived recalibration would need *practitioner-labeled feature↔pattern
co-occurrence* (PLAN.md §WS-B step 4), not book-mention frequency.

## Status
- `kg/recalibrate.py` + the `TIH_KB_VERSION` toggle are kept (cheap, reversible, re-runnable if we ever
  get a better empirical signal). `tcm_knowledge_v2.json` is a git-ignored rebuildable artifact.
- **Production KB = v1.** No change to serving.
