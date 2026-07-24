# TongueInsight (TIH) — Project Handbook

Single onboarding + reference doc. Audience: engineers who know TCM. Educational tool, **not** a medical
device — every output is framed "traditionally associated with…", never a diagnosis.

Machine-readable companions kept beside their code: `stage2_interpretation/knowledge_base/sources.json`
(licence registry), `stage2_interpretation/kg/README.md` (KG build), `design/` (demo UI design pack).

---

## 1. What it is

- Input: one tongue photo. Output: detected visual features → a grounded TCM **pattern/constitution
  leaning** + 宜/忌 advice, as JSON for the Savor app.
- Two stages:
  - **Stage 1 (vision):** segmentation + feature CNNs → structured JSON of graded features. **Frozen.**
  - **Stage 2 (interpretation):** rule engine + knowledge graph + optional LLM narrative → the reading.
- Positioning: educational/wellness. No disease names. Rejects bad photos.

---

## 2. Transfer checklist (what changes hands beyond GitHub)

- **Checkpoints** (the product; not in git, large): casper `~/tongue/checkpoints/{seg_combined,multitask_v5,extra_features}/best.pt`.
- **casper server**: `192.168.1.184`, GPU 0; venv `envs/tih/`; project at `~/tongue/`. SSH creds + account.
- **Datasets** (not in git): TonguExpert, TCM-Tongue (`shezhen datasets/`), SM-Tongue, BioHit — see §4.1 for links.
- **Licensed books + the permission grant** (IP-sensitive): `tongue_lit/` (git-ignored, copyrighted). Record who holds the licence and its terms.
- **LLM backend**: Ollama on casper (`gemma3`, `nomic-embed-text`, `qwen2.5:14b`); optional shared vLLM `:8000` (Qwen2.5-VL, API-key).
- **Secrets / env**: see §12 env table (`TIH_API_KEY`, backend URLs…).
- **Deployment**: cloudflared HTTPS tunnel + demo URL; Docker image (`deployment/`).
- **External accounts**: HuggingFace (SM-Tongue), Dryad, biosino, Apple Developer (if started), Savor app-team contacts.
- **Regenerable, don't ship manually**: RAG index (`rag.py --build`), `corpus.jsonl`, gallery/label HTML.

---

## 3. Repo structure

- `stage1_quantitative/` — vision.
  - `infer.py` — loads seg + feature CNNs; runs the full Stage-1 pass → `Stage1Output` JSON. Holds the geometry thresholds.
  - `labels.py` — `KEY_CHARS`, `LABEL_MAPS`, `EXTRA_FEATURES` (the label vocabulary).
  - `zoning.py` — training-free colour-by-region (red_tip, red_sides, moisture).
  - `schema.py` — `Stage1Output` dataclass / JSON shape.
  - `segmentation/`, `feature_extraction/` — training code.
- `stage2_interpretation/` — interpretation.
  - `interpret.py` — **the rule engine** + output assembly (features→votes, combination rules, CCMQ crosswalk, 宜/忌, headline, follow-ups).
  - `knowledge_base/` — `tcm_knowledge.json` (features/patterns/rules), `tcm_patterns.json`, `knowledge_cards.json` (RAG cards), `sources.json` (licences), `corpus.jsonl` (built RAG index input).
  - `rag.py` — embed + retrieve over the corpus. `build_corpus.py` — builds `corpus.jsonl`.
  - `llm_client.py` — LLM backend wrapper (Ollama/OpenAI-compatible).
  - `kg/` — knowledge graph: `build_kg.py`, `parse_book.py`, `micro_extract.py`, `who_terms.py`, `normalize.py`, `graph.py`, `retrieval.py`, `matcher.py` (cite-or-abstain), `ensemble.py` (rule+matcher blend), `refine_engine.py` (follow-up questions).
- `deployment/api/` — `app.py` (FastAPI), `service.py` (framing + capture-quality gates + overlay), `static/index.html` (phone demo).
- `evaluation/` — eval harnesses (§11) + labeling tools.
- `data/` — dataset prep/loaders (`build_label_store.py`, `build_tcm_tongue_labels.py`).
- `checkpoints/`, `tongue_lit/` (books, git-ignored), `shezhen datasets/` (TCM-Tongue), `design/`, `docs/`.

---

## 4. Data & models (why 3 models on different data)

### 4.1 Datasets

| Dataset | Link | Used for | Licence |
|---|---|---|---|
| TonguExpert | [biosino.org](https://www.biosino.org/) | seg masks + **5 core feature** labels | ⚠️ commercial terms unverified |
| TCM-Tongue (btbu) | [Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.1c59zw48r) · [GitHub](https://github.com/btbuIntelliSense/Intelligent-tongue-diagnosis-detection-dataset) | **extra pathological features** + 553-img practitioner benchmark | CC-BY 4.0 |
| SM-Tongue | [HF](https://huggingface.co/datasets/Mark-CHAE/SM-Tongue-Public-Original512) | real-**phone** seg pairs | ⚠️ CC-BY-NC (ship-blocker) |
| BioHit | [GitHub](https://github.com/BioHit/TongeImageDataset) | 300 imgs + manual masks — **available add-on, not in the final seg manifest** (kept for future refinement) | no explicit licence |
| human40 / human40b | internal | honest real-world check + threshold calibration | owned |

### 4.2 Model A — segmentation (U-Net++, `seg_combined`)

- Trained on TonguExpert masks **+ SM-Tongue (real phone)** — the unified `data/processed/seg_manifest.csv`
  is 8,147 pairs (5,992 TonguExpert + 2,155 SM-Tongue). (BioHit is an available add-on but was **not** added
  to this manifest — the combined set already reached target Dice; see §4.1.)
- Why combined: TonguExpert is clinic-lit; SM-Tongue adds phone photos → fixes the clinic→phone domain gap.
- Out: binary tongue mask (real-photo Dice ~0.975). Feeds the crop + mask-guidance for the feature CNNs.

### 4.3 Model B — 5 core features (multi-task CNN, `multitask_v5`)

- Trained on **TonguExpert** — the one set that labels all 5: `coating`, `tai` (coat colour), `zhi` (body colour), `fissure`, `tooth_mk`.
- Multi-task: shared backbone, one classification head per feature (graded none/light/severe or the colour classes) + a severity regression head. Mask-guided (features masked to the tongue).
- Accuracy vs expert gold (held-out test, `benchmark.py --mt …v5`): tai 0.92, cracks 0.92, zhi 0.81, tooth_mk 0.85; ~59% exact 3-way but **97% within one grade** (`eval_fair.py`).
- v6/v7/v8 all lost to v5 on the human metric → frozen.

### 4.4 Model C — extra features (multi-label CNN, `extra_features`)

- Trained on **TCM-Tongue** (20 practitioner-verified categories) — **because TonguExpert doesn't label the pathological signs**: `peeled_coating`, `red_tongue`, `purple_body`, `swollen`, `thin`, `red_dots`, `slippery_coating` (+ `black_coating`, now dropped).
- Multi-label sigmoid; presence prob doubles as severity.
- Validated on the 553-img TCM-Tongue **test split** (`eval_extra_vs_practitioner.py`): red_dots AP 0.68, red_tongue 0.61, thin 0.58, slippery 0.33, swollen 0.19, purple 0.17, **black_coating 0.05 → removed**.

### 4.5 Geometry (no training) — `zoning.py`

- `red_tip` (Heart/upper-jiao, thresh 2.0), `red_sides` (Liver/GB, thresh 1.5, calibrated on human40), `moisture=wet` (only "wet" asserted; "dry" is an honest gap).

---

## 5. Pipeline

```
photo
 └─ seg (U-Net++) ──► mask
        │
        ├─ crop + mask ──► CNN-B (5 core)   ─┐
        ├─ crop + mask ──► CNN-C (extras)   ─┤─► Stage-1 JSON
        └─ mask + pixels ─► zoning geometry ─┘        │
                                                      ▼
         service.py gates: framing (coverage) + capture-quality (blur/exposure)
                                                      │  (bad photo → error, no reading)
                                                      ▼
         interpret.py: feature votes → combination rules → patterns
                        → CCMQ constitution → 宜/忌 → headline → follow-ups
                        (+ optional KG matcher ensemble, + optional LLM narrative)
                                                      ▼
                                            /analyze JSON → app
```

Gates (`deployment/api/service.py`): coverage tiers (`<5%` = no tongue → error; `<12%` = warn/unreliable;
~36% fills the oval) + `capture_quality()` (blur via Laplacian variance; exposure via luma+clipping). A hard
fail sets `framing.status="error"`; the frontend refuses to render a reading (fixed 2026-07-21).

---

## 6. Example JSON

### 6.1 Stage-1 output (per feature; abridged)

```json
{
  "key_characteristics": {
    "zhi": {"value": "light", "confidence": 0.68, "severity": 0.40,
            "probs": {"light": 0.68, "regular": 0.27, "dark": 0.05}},
    "coating": {"value": "greasy_thick", "confidence": 0.81, "severity": 0.86,
            "probs": {"non_greasy": 0.05, "greasy": 0.14, "greasy_thick": 0.81}}
  },
  "extra_characteristics": {"swollen": {"value": "present", "severity": 0.72}},
  "zoned_analysis": {"moisture": {"value": "wet", "severity": 0.6}},
  "quality": {"mask_coverage": 0.31, "accepted": true, "reasons": []}
}
```

### 6.2 `/analyze` output (Stage-2; abridged, real)

```json
{
  "headline": "The sign that stands out most is your tooth marks … leans toward low digestive energy.",
  "patterns": [
    {"id": "spleen_qi_deficiency", "kind": "pattern",
     "constitution": {"id": "qi_deficiency", "zh": "气虚质", "via": "syndrome"},
     "confidence": 0.607, "confidence_pct": 61},
    {"id": "phlegm_dampness", "kind": "pattern",
     "constitution": {"id": "phlegm_dampness", "zh": "痰湿质"}, "confidence_pct": 57}
  ],
  "constitution_leaning": {"id": "qi_deficiency", "zh": "气虚质", "via": "syndrome"},
  "recommendation": {
    "conditions": [{"name": "low digestive energy", "confidence": 0.61}],
    "actions": ["favour warm, well-cooked foods; go easy on cold/raw & iced drinks"],
    "avoid": ["cold and raw food, iced drinks", "skipping meals or overeating"]
  },
  "followup": [{"pattern_id": "spleen_qi_deficiency", "questions": [ … ]}],
  "framing": {"status": "ok", "coverage": 0.31, "reliable": true},
  "quality_gate": {"status": "ok", "checks": {"sharpness": …, "mean_luma": …}}
}
```

Full result also carries: `features` (surfaced signs with `points_to`), `findings`, `symptoms`,
`confidence_note`, `reasoning` (fired rules), `sources`, `report` (narrative), `disclaimer`. External
citations are stripped unless `TIH_SHOW_CITATIONS=true`.

---

## 7. Rule engine (`interpret.py`) — how the reading is computed

- **Per-feature votes.** Each feature value carries `points_to` weights in `tcm_knowledge.json`. Example
  `zhi:light` → `{blood_deficiency: 1.3, spleen_qi_deficiency: 0.4, yang_deficiency: 0.4}`.
- **Distinctiveness weighting.** A vote is scaled by how *far from the population norm* the sign is
  (`rel`, from `reference_stats`) — a barely-above-average sign votes weakly.
- **Extra features vote by measured AP.** `EXTRA_RELIABILITY` = each extra's practitioner-test AP
  (red_dots 0.68 … purple 0.17); `black_coating` is in `_REMOVED_EXTRA` (not surfaced, not voted).
- **Combination rules** (`apply_combination_rules`). Fixed context boosts on top of the additive votes,
  because a sign flips meaning by context. Rule shape:
  `{when:{feat:val}, any?:{feat:val}, boost:{pattern:±delta}, enabled?, cite, note}`.
  - `when` = all must match; `any` = at least one; `boost` = add/subtract (clamped ≥0); `enabled:false` = kept but skipped.
  - Examples: `swollen+pale+wet → yang_deficiency +0.6, damp_heat −0.4`; `pale+thin → blood_deficiency +0.5`;
    `wet → damp_heat −0.4, yin_deficiency −0.3`. 21 rules; the 2 grey-black ones are disabled (detector broken).
- **Aggregate** (`vote_patterns`): sum votes + rule boosts → top-K; low totals fall back to `balanced`;
  `confidence_pct` is the raw score.
- **Regression guard**: `evaluation/eval_mapping.py` — 34 canonical/adversarial cases, currently 34/34.

---

## 8. 9 constitutions + 2 patterns

- Tongue reads a **pattern (证, today's state)**; we map it to a **CCMQ body-constitution (体质, baseline)**
  via `CCMQ_CONSTITUTION` in `interpret.py`, for Savor's constitution/food/seasonality features.
- Each card has `kind` = `pattern` or `constitution`, and `constitution` = the mapped 体质.
- 8 of our labels map 1:1 to 8 constitutions. **2 are syndromes, not constitutions**, both → 气虚质:
  - `spleen_qi_deficiency` (脾气虚证) — pale + tooth-marks.
  - `blood_deficiency` (血虚证) — pale + thin body.
- Constitution → tongue signature: 平和 pale-red+thin-white · 气虚 pale+tooth-marks · 阳虚 pale+swollen+wet ·
  阴虚 red+peeled+cracks · 痰湿 thick-greasy-white+swollen · 湿热 red+greasy-yellow+red-dots · 血瘀 dark/purple ·
  气郁 red sides · 特禀 no tongue sign (set by inquiry).
- Daily layer: pipeline emits 今日宜/忌 per reading, so constitution/food features update per photo instead of a one-time quiz.

---

## 9. Knowledge graph + LLM matcher (`kg/`)

- **KG** = feature/pattern nodes + cited edges, built **offline** from the licensed books; shipped read-only.
  - *Macro*: book chapter/section hierarchy (`parse_book.py`).
  - *Micro*: LLM-extracted `feature→pattern` triplets, each with a source citation + snippet
    (`micro_extract.py`, run with qwen2.5:14b on casper). ~605 nodes / 1245 edges / 282 cited edges, 0 junk.
  - *Spine*: WHO-IST 2022 canonical bilingual names (`who_terms.py`).
- **Matcher** (`matcher.py`): given detected features, proposes patterns **with citations, cite-or-abstain**
  (never invents an edge). Temperature 0, structured output.
- **Ensemble** (`ensemble.py`): `blended = (1−α)·rule + α·matcher`, α=0.2 (swept on human40). Rule engine is
  the auditable backbone; the matcher adds cited evidence + honest confidence. Toggle `TIH_WSC_ENSEMBLE`
  (default on); falls back to pure rules if the matcher is unavailable.
- Role in pipeline: runs **after** `vote_patterns`, re-ranks/annotates patterns with citations. Adds no runtime model weight (graph lookups).

---

## 10. RAG + LLM narrative

- **What RAG grounds**: only the **prose narrative**, not the pattern choice. Patterns come from the rule
  engine (+ KG matcher); the narrator just re-expresses grounded facts.
- **Corpus** (`corpus.jsonl`, 119 chunks) built by `build_corpus.py` from: KB chunks (`tcm_knowledge.json`)
  + authored cards (`knowledge_cards.json`, our own summaries of licensed sources). **Never raw book text.**
  Each chunk carries provenance (source, licence, usage) resolved against `sources.json`.
- **Retrieval** (`rag.py`): embed with `nomic-embed-text` (Ollama); retrieve top-k for the detected
  features. (`nomic` is weak on Chinese → cards are authored in English; swap to `bge-m3` if CN cards are added.)
- **Narrator** (`llm_client.py`): optional. Default **off** → deterministic template report. When on, a small
  local model rewrites the grounded facts into prose. `SHOW_CITATIONS` off strips book/author names from the
  public JSON (grounding stays internal). Gate: faithfulness check before defaulting on.
- **Matcher vs narrator**: matcher = *which patterns* (+ citations); narrator = *the words*; RAG = the
  grounding the narrator retrieves over.

---

## 11. Evaluation & the honest-metric rule

- Promote a change only if it beats v5 / the rules on the **human** or **practitioner** metric — never on val/auto labels.
- Harnesses (`evaluation/`): `eval_mapping.py` (feature→pattern, 34/34), `eval_extra_vs_practitioner.py`
  (extra CNN vs practitioner labels), `eval_extra_features.py` (red_tip/red_sides thresholds vs human40),
  `benchmark.py` / `benchmark_syndrome.py`, `eval_seg.py`, `eval_rag.py` (retrieval hit@k), `eval_stage1.py`.
- Labeling: `build_label_tool.py` → self-contained HTML labeler; rubric was in the (removed) labeling guide —
  the field help text now lives in the tool itself. `human40_extra_labels.json` = gold extras.

---

## 12. Run it & configure

On casper:
```bash
source envs/tih/bin/activate
uvicorn deployment.api.app:app --host 0.0.0.0 --port 7860   # demo + API
python evaluation/eval_mapping.py                           # rule-engine regression
python stage2_interpretation/build_corpus.py && python stage2_interpretation/rag.py --build   # rebuild RAG
```

Endpoints: `GET /health`, `POST /analyze`, `POST /refine`, `GET /examples`, `GET /` (demo).

Env vars:

| Var | Purpose |
|---|---|
| `TIH_SEG_CKPT`, `TIH_MT_CKPT` | checkpoint paths |
| `TIH_LLM_BACKEND`, `TIH_LLM_BASE_URL`, `TIH_LLM_MODEL`, `TIH_LLM_API_KEY` | LLM narrator (default off) |
| `TIH_EMBED_URL`, `TIH_EMBED_MODEL` | RAG embedder |
| `TIH_WSC_ENSEMBLE`, `TIH_WSC_ALPHA` | KG matcher ensemble (default on, α=0.2) |
| `TIH_SHOW_CITATIONS` | expose book/author citations (default false) |
| `TIH_FAITHFULNESS_MIN`, `TIH_GOLD_WEIGHT`, `TIH_KB_VERSION` | eval/KB knobs |
| `TIH_COLOR_CALIB`, `TIH_CC_STRENGTH` | colour calibration |
| `TIH_API_KEY`, `TIH_CORS_ORIGINS` | API auth + CORS |

---

## 13. Known issues / TODO

- **SM-Tongue is CC-BY-NC** → retrain seg on commercial-clean data (or license) before charging.
- **TonguExpert commercial terms unverified** → confirm before paid ship.
- Moisture: only "wet" detected; "dry" unmeasurable from gloss (needs controlled capture or a texture model).
- `red_sides` threshold calibrated on only 21 labels → widen with more.
- `black_coating`, and to a lesser degree `purple`/`swollen`, are weak detectors — black removed, others down-weighted.
- RAG embedder weak on Chinese → English cards only until `bge-m3` swap.
- Narrator stays off until the faithfulness gate is wired in production.

---

## 14. Glossary

- **证 (zhèng)** pattern/syndrome — current state; what the tongue reads. **体质 (tǐzhì)** constitution — stable baseline (CCMQ 9).
- **CCMQ** — Wang Qi 9-constitution questionnaire (source of the 9 + the follow-up questions).
- **tai** 苔 coating colour · **zhi** 质 body colour · **fissure** cracks · **tooth_mk** scalloped edges.
- **Stage 1 / Stage 2** — vision / interpretation. **KG** — knowledge graph. **Matcher** — cite-or-abstain pattern proposer. **Ensemble** — rule+matcher blend.
- **TIH** — TongueInsight Hybrid (project codename). **WS-A…G** — old workstream labels in git history.
