# TongueInsight — Master Plan (v2, 2026-07-16)

**This is the single source of truth for what's left to build.** It supersedes the two founding
plans (`Tongue analysis ML.md`, `…(1).md`) and the scattered `*_PLAN` / `*_IMPROVEMENT` / `DIRECTION_REVIEW`
docs (to be archived — see §6). Deep history stays in `docs/PROGRESS.md` and `git log`.

The project is an **educational, non-diagnostic** wellness feature (計五味 / Savor): tongue photo →
detected features → a grounded TCM "constitution leaning," framed as *"traditionally associated with…"*.

---

## 1. Where we are (honest status)

**Stage 1 (vision) — essentially DONE and frozen.**
- Segmentation `seg_combined` (U-Net++), real-photo Dice ~0.975.
- Characteristics `multitask_v5` (5 core signs + severity) — **production**; v6/v7/v8 all failed to beat it
  on the human metric. Extra features + training-free geometry (`zoning.py`: red-tip, moisture).
- Honest accuracy: ~61% exact 3-way vs human labels, but **97% within-one-grade**; presence is strong.
- **We are not chasing Stage-1 accuracy further** — auto-labels are the ceiling; more label data is the
  only real lever, and that's on hold. Vision is good enough to ship.

**Stage 2 (interpretation) — the current focus.** Works today (rule-engine votes + combination rules +
a grounded RAG/LLM narrator over a 102-chunk self-authored corpus), but this is exactly what we're
overhauling with the newly-licensed literature (§3).

**Deployment — decided:** cheap CPU server, thin-client app. See `docs/DEPLOYMENT.md`.

### What the original plan promised that we're now DROPPING (and why)
| Dropped item | Why |
|---|---|
| SAM-based segmentation | Replaced by U-Net++ (done, better fit for real photos) |
| Full **773-phenotype** extraction | Evidence shows auto-labels cap accuracy; not worth the complexity |
| **MMIR-TCM** adaptation / MedTCM / TDEU | No code or data ever released — we built our own Stage 2 |
| Grad-CAM explainability | RAG **citations** are a better, cheaper explainability story |
| ONNX / INT8 / distillation / LoRA | Unnecessary: inference is already 0.34 s on CPU. Revisit only for a future on-device build |

---

## 2. The target architecture for Stage 2

Adapted from **TCM-DiffRAG** (`tongue_lit/TCM_Rag.pdf`), constrained to cheap-CPU serving. Key insight:
their expensive query-time step (a fine-tuned model that decomposes free-text vignettes into triples) is
**unnecessary for us — Stage-1 output is already structured triples.** We inherit the KG + retrieval
gains; per-request cost stays one LLM call.

```
Stage-1 JSON (features + confidences)          ← unchanged, code-computed
      │
      ▼   entry nodes = detected features (deterministic; no CoT decomposer needed)
MACRO-MICRO TCM KNOWLEDGE GRAPH  (built offline from the licensed books)
   macro = book chapter/section hierarchy   → explanatory context, semantic integrity
   micro = LLM-extracted triplets w/ source citation → precise feature→pattern locking
      │
      ▼   macro-micro integrated + hybrid (semantic+lexical) retrieval
GROUNDED MATCHER (LLM, cite-or-abstain, structured output)
   proposes patterns + evidence triples + citations
   rule engine runs alongside as a computed prior/referee (shadow → ensemble)
      │
      ▼   confidence = detection × distinctiveness × evidence-convergence  (computed, reproducible)
READING  →  [optional] interactive refinement pass (§ WS-B)  →  output (§5)
```

---

## 3. Workstreams (the actual work left)

### WS-A · Macro-micro knowledge graph from the licensed literature  *(foundation — do first)*
Sources now in `tongue_lit/` (we have usage rights):
- **Gerlach, *TCM Tongue Diagnosis Explained*** — modern, feature-organized, clean section hierarchy → the backbone.
- **Maciocia, *Tongue Diagnosis in Chinese Medicine*** — depth/authority.
- **WHO International Standard Terminologies on TCM (2022)** — the **controlled vocabulary** to normalize node names (bilingual).
- **Oriental Tongue Diagnosis** — supplementary (zoning/meridians).

Steps:
1. **Ontology spine** from the WHO terminology → canonical bilingual names for features & patterns.
2. **Macro layer** — parse each book's hierarchy into title nodes (Gerlach's numbering parses directly).
3. **Micro layer** — LLM-extract triplets offline (on casper): `feature→pattern`, `feature+context→pattern`,
   `pattern→symptom`, `pattern→recommendation`, `pattern↔pattern` disambiguation — **each with a source
   citation + snippet** (traceability).
4. **Seed as a SUPERSET of today's `tcm_knowledge.json`** so day-one it reproduces current rules, then grows.
5. Store: graph (nodes/edges + adjacency) + a snippet store. Build script offline; artifact shipped read-only.

**Licensing check-in:** confirm the grant covers surfacing *short cited snippets to end users* vs internal
processing only. Architecture stays safe either way (triplets + short attributed snippets).

### WS-B · Interactive evidence refinement  *(your follow-up-questions idea — the "personalization" layer)*
Replaces the paper's training-time personalized KG with an inference-time, data-free equivalent.
1. **Two-pass reading.** Pass 1: tongue features → candidate patterns + the questions that would most
   *disambiguate* them. Pass 2: user's yes/no (or slider) answers enter as **symptom evidence** and
   re-run retrieval+reasoning over the *same* KG (`symptom→pattern` edges).
2. **Question selection by information gain** — pick the item that best separates the top-2 candidates,
   not a fixed per-pattern list. Draw from validated CCMQ items where possible.
3. Keep the transparent log-odds `refine()` as the fallback/interim.
4. *(optional, later)* A lightweight **data-derived "stylized" KG**: mine our practitioner-labeled sets
   (TCM-Tongue) for feature↔pattern co-occurrence → grounding cards from *our own data*. Closest analog
   to the paper's approach that our data actually supports.

### WS-C · Grounded matcher + honest confidence
1. LLM matcher, **cite-or-abstain**, temperature 0, structured JSON out (patterns + evidence + citations).
2. **Shadow mode first:** run alongside the rule engine on the gallery; log disagreements; score both.
3. Promote to **full-LLM or ensemble** based on the numbers (don't decide in advance).
4. **Add raw confidence % back** into the output (currently overwritten by a word) — small, independent task.

### WS-D · Evaluation & safety  *(the gate — nothing promotes without this)*
1. Adopt **RAGAS** (faithfulness, context precision/recall) like the paper — this *is* the hallucination gate.
2. Expand `eval_mapping.py` well past 12 combos; keep TCMEval-SDT (rules baseline 69.7%) and `eval_rag.py`.
3. **Hallucination-rate check before defaulting the LLM ON** in production.
4. (Later, pre-commercial) expert TCM blind review of narrative quality.

### WS-E · Deployment  *(per `docs/DEPLOYMENT.md`)*
1. Containerize FastAPI on the cheap CPU box (CPU torch, bake the 3 checkpoints, pin deps).
2. Narrative served **off-box** (casper or hosted) once WS-C/WS-D clear; template is the always-on fallback.
3. **Licensing blocker for paid ship:** retrain seg without SM-Tongue (CC-BY-NC) or license it.

### WS-F · Output / UX design  *(user-led — scaffold in §5)*

### WS-G · Repo cleanup  *(§6)*

---

## 4. Sequence & dependencies

```
WS-A (KG build)  ──►  WS-C (matcher, shadow)  ──►  WS-D (eval gate)  ──►  WS-E (deploy narrator on)
      │                     ▲
      └──►  WS-B (refinement, uses symptom edges from the KG)
WS-F (output design) runs in parallel (user).   WS-G (cleanup) is independent, do anytime.
Quick win, anytime: raw confidence % (part of WS-C.4).
```

---

## 5. Next design phase — how the output is presented (scaffold for WS-F)

Everything the pipeline **can** surface (you decide what's in / out / how grouped):

**Currently shown:** headline · findings text · ranked findings (five-flavour tags) · graded severity bars ·
constitution-leaning chip · 宜/忌 recommendation · "you might also notice" symptoms · distinctiveness hooks ·
confidence note · per-sign detail · top-3 patterns · follow-up questions.

**Computed but NOT shown (candidates):** raw per-class confidence % · full pattern distribution (not just
top-3) · which combination rules fired ("why we read it this way") · the **RAG citations/snippets** for
this reading · suppressed low-distinctiveness features · shareable "tongue-type" card (built, currently
unwired) · history/trend across visits (needs app-side storage).

**Design principles to hold (from the brand + our honesty discipline):**
- Educational, **non-diagnostic** — "傳統上多與…有關", never a disease conclusion.
- **Honest uncertainty** — show the confidence note; flag tentative reads; never over-claim faint signs.
- **Progressive disclosure** — one-line headline → findings → detail on tap. Don't dump everything.
- Almanac / five-flavour brand; pressure-free Cantonese voice; result optionally shareable.

**Open questions for you:** how much clinical detail to expose vs hide? Show citations to users (trust) or
keep them internal? Surface the follow-up flow inline or as an optional "refine" step? Include history/trend
in v1 or defer? Reinstate the tongue-type share card?

---

## 6. Repo cleanup (DONE 2026-07-16)

Executed. `docs/` went from 26 → 11 living files; `evaluation/` from 23 → 12 living scripts.
All moves were `git mv` (history preserved); nothing was destroyed.

**Removed (OS junk):** `__MACOSX/`.
**Archived → `docs/archive/`** (historical, superseded by this plan): the two `Tongue analysis ML*.md`,
`DIRECTION_REVIEW`, `IMPROVEMENT_PLAN`, `KB_IMPROVEMENT_PLAN`, `LABEL_IMPROVEMENT_PLAN`, `OUTPUT_REDESIGN_PLAN`,
`ACCURACY_INVESTIGATION`, `LABEL_QUALITY_DIAGNOSIS`, `CONSTITUTION_BENCHMARK`, `COATING_SPLIT`,
`COLOR_CALIBRATION`, `FEATURE_FEEDBACK_2026-07-13`, `PHASE4_TCM_TONGUE`, `BENCHMARK_DESIGN`, `TCM_RESEARCH`,
`RAG_LLM_INTERPRETATION`.
**Living docs (kept):** `PLAN.md` (this), `PROGRESS.md`, `HANDOFF.md`, `ARCHITECTURE.md`, `DEPLOYMENT.md`,
`LABELING_GUIDE.md`, `LABEL_STORE.md`, `BENCHMARK.md`, `FEATURE_MAPPING_REFERENCE.md`,
`FEATURE_PATTERN_MAPPING.md`, `RESOURCES.md`.
**Archived one-off eval scripts → `evaluation/archive/`:** `diagnose_confusion`, `diagnose_label_agreement`,
`calibrate_logit_adjust`, `eval_memory_sam`, `find_professional_labels`, `benchmark_multimodal`,
`eval_human_labels`, `eval_fair`, `eval_vs_expert`, `eval_color_calib`, `eval_zoning`.
**Living harnesses (kept):** `eval_model`, `eval_mapping`, `eval_rag`, `eval_seg`, `eval_stage1`,
`eval_coat_axes`, `eval_extra_features`, `benchmark`, `benchmark_syndrome`, `build_gallery`,
`build_label_tool`, `select_eval_set`.

**Left in place (your call):** `design.zip` (15 KB, harmless — the unzipped `design/` is untracked too).
**Still TODO:** `docs/ARCHITECTURE.md` needs a refresh (still describes the dropped SAM/MMIR-TCM plan).
