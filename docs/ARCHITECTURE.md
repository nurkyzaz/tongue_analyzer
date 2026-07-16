# TongueInsight — Architecture (v2, 2026-07-16)

Current system as built. The master plan for *remaining* work is [PLAN.md](PLAN.md); this describes how
the pieces fit together today. Educational, **non-diagnostic** — every output is *"傳統上多與… /
traditionally associated with…"*, never a diagnosis.

## Pipeline

```
                 ┌──────────────── Stage 1: Quantitative (FROZEN) ────────────────┐
 photo ─► quality gate ─► SEGMENTATION (U-Net++, seg_combined) ─► masked ROI ─┐
                                                                              ├─► multitask_v5: 5 core signs (cls) + severity (reg)
                                                                              ├─► extra_features: 8 multi-label signs
                                                                              └─► zoning.py (training-free geometry): red-tip, moisture, zones
                                                                                       │  structured JSON (features + confidences)
                 ┌──────────────────── Stage 2: Interpretation ──────────────────────┐  │
                 │  RULE ENGINE (interpret.py + tcm_knowledge.json)                   │◄─┘
                 │    distinctiveness-weighted votes + combination rules  → patterns  │
                 │  KNOWLEDGE GRAPH (kg/, macro-micro from licensed books) ── being    │
                 │    integrated: grounded matcher (cite-or-abstain) + RAG retrieval  │
                 │  REFINEMENT (WS-B): user answers → symptom evidence → re-score      │
                 │  NARRATOR: grounded RAG+LLM  →  falls back to template if offline   │
                 └────────────────────────────────────────────────────────────────────┘
                                          │  reading JSON  ─►  app (see §Output contract)
```

## Stage 1 — Quantitative (done, frozen — not the current focus)

- **Segmentation** `checkpoints/seg_combined` — U-Net++ (ResNet-34), real-photo Dice ~0.975 (TonguExpert
  + SM-Tongue closed the app-photo domain gap).
- **Characteristics** `checkpoints/multitask_v5` — mask-guided multitask CNN: 5 core signs
  (`coating, tai, zhi, fissure, tooth_mk`) as graded categories + severity regression. **v6/v7/v8 all
  failed to beat v5 on the honest human metric** — auto-labels are the ceiling. `checkpoints/extra_features`
  adds 8 multi-label signs (peeled/red/purple/swollen/thin/red-dots/black/slippery).
- **Geometry** `zoning.py` — training-free, rotation-robust: red-tip (Δ vs body), moisture (specular
  gloss, asserts *wet* only), zoning. **These are the only signs that localize** — `multitask_v5` is
  classification+severity and does not, which the UI honours (only geometry/local signs get a hero ring).
- **Honest accuracy:** ~61% exact 3-way vs human labels but **97% within-one-grade**; presence is strong.
  We are *not* chasing Stage-1 accuracy further (label ceiling); the ±1-grade band is surfaced in the UI.

## Stage 2 — Interpretation (the active work — PLAN.md §3)

- **Input:** Stage-1 JSON (+ optional user answers from the refinement pass).
- **Rule engine** (`interpret.py` + `knowledge_base/tcm_knowledge.json`) — the auditable/testable
  backbone: distinctiveness-weighted feature votes + **combination rules** (context flips readings, e.g.
  swollen+pale+wet→Yang-def vs swollen+yellow→Damp-heat). Tested by `evaluation/eval_mapping.py`.
- **Knowledge graph** (`kg/`, being integrated) — a macro-micro typed graph that unifies the rule KB
  (seed layer, parity-verified superset), the book section hierarchy (macro), and LLM-extracted cited
  triplets (micro, from the licensed literature). Substrate for:
  - **Grounded matcher (WS-C):** LLM proposes patterns with evidence triples + citations, *cite-or-abstain*;
    runs in shadow mode alongside the rule engine first, promoted on the numbers.
  - **Interactive refinement (WS-B):** the inverse `symptom/question → pattern` edges let a user's
    follow-up answers re-enter as symptom evidence and re-score patterns; questions are chosen by
    information gain (best-separating the top-2 candidates), so there are only ever ~2.
- **Retrieval (RAG):** hybrid semantic (`nomic-embed` via Ollama) + lexical (TF-IDF, RRF) over the cited
  corpus; grounds the narrator, degrades to lexical offline.
- **Narrator:** grounded RAG+LLM re-expresses *only* grounded facts; falls back to a deterministic
  **template** when the LLM is offline (the always-on path — see the "Degraded" UI state).
- **Confidence** is computed (detection × distinctiveness × evidence-convergence), reproducible.

## Output contract (what the app renders — see the design bundle)

The reading JSON drives: an annotated **hero** (segmentation mask + rings for localizable signs only);
a **headline** (findings); **linkage cards** (傳統上多與 — patterns with evidence chips ← and confidence
bars, top-2); a **refine strip** (shown only when top-2 are close); the **six signs** on a Mild→Pronounced
track with a **±1-grade uncertainty band**; a **trend** across recent readings; **宜/忌** advice; a
**Sources** sheet (citation-only, or citation+snippet where licensing allows); and a disclaimer. Three
data conditions (ambiguous top-2 / clear top-1 / nothing-notable) and a degraded (narrator-offline) state.

## Deployment (PLAN.md §WS-E, docs/DEPLOYMENT.md)

Inference is **CPU-fast (~0.34s)** — no GPU to serve. Target: containerized FastAPI on a cheap CPU box
(bake the 3 checkpoints), narrator served off-box once WS-C/WS-D clear (template is the always-on
fallback). **Licensing before paid ship:** seg trained partly on SM-Tongue (CC-BY-NC) → retrain without
it or license; surface cited snippets only if the book grant allows (architecture is safe either way).

## Non-goals / guardrails

Not a medical device. Educational framing, explicit disclaimers, no diagnosis. Image-quality gate rejects
unusable photos. The rule engine stays auditable; the LLM is always grounded (cite-or-abstain).
Copyrighted book texts are processed locally, never committed or vendored.
