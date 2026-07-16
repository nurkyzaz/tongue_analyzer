# Progress Board

Living task board. ✅ done · 🔄 in progress · ⬜ todo · ⏸ blocked

## Phase 0 — Setup & research
- ✅ Verify external resources exist ([RESOURCES.md](RESOURCES.md))
- ✅ SSH access to casper; GPU audit (GPU0 free, GPU1 shared vLLM)
- ✅ Repo skeleton + docs (README, ARCHITECTURE, RESOURCES)
- 🔄 Server env: venv `~/tongue/envs/tih` + PyTorch/CV stack
- ⬜ GitHub remote (needs token/gh auth)

## Phase 1 — Data pipeline
- ✅ Download + extract TonguExpert (5,992 imgs+masks, labels, phenotypes)
- 🔄 Build manifest (SID → raw, mask, labels, features) + train/val/test split
- ⬜ Data loaders (seg + multitask) with augmentation
- ⬜ Dataset stats report

## Phase 2 — Stage 1 quantitative
- ✅ 2.1 Segmentation (U-Net++ ResNet-34) — **val Dice 0.994** (`checkpoints/seg/best.pt`)
- ✅ 2.2 Multi-task head (5 key chars, mask-guided pooling, Focal Loss) — **test mean macro-F1 0.735** (`checkpoints/multitask_v2/best.pt`)
- ✅ 2.3 End-to-end inference → `Stage1Output` JSON (`stage1_quantitative/infer.py`) + test eval (`evaluation/eval_stage1.py`)
- ✅ 2.4 Real-photo domain gap fixed: added SM-Tongue (2,155 real pairs); combined U-Net++ →
  sm_tongue Dice **0.975** (was 0.749 clinical-only). See seg table below.
- ✅ 2.5 Memory-SAM implemented (SAM2 + DINOv2 fallback; DINOv3 gated) → real-photo Dice **0.980**,
  1148ms/img (`evaluation/eval_memory_sam.py`). Verdict: U-Net++ for production, Memory-SAM as auto-labeler.
- ⬜ 2.2b Regression heads for continuous phenotypes (P*.txt) — richer output
- ⬜ 2.6 Push char-model accuracy (bigger backbone, higher-res, EMA/TTA) — macro-F1 0.735 → target higher

### Segmentation on REAL photos (SM-Tongue, 215-img test)
| model | Dice | latency | params |
|---|---|---|---|
| U-Net++ (TonguExpert only) | 0.749 | ~5ms | 24M |
| U-Net++ (TonguExpert + SM-Tongue) | **0.975** | ~5ms | 24M |
| Memory-SAM (SAM2 + DINOv2) | **0.980** | 1148ms | ~1B |

### Stage-1 test metrics (695 imgs, gold-preferred labels)
| char | acc | macroF1 |  | char | acc | macroF1 |
|---|---|---|---|---|---|---|
| coating | 0.88 | 0.56 |  | fissure | 0.86 | 0.84 |
| tai | 0.82 | 0.78 |  | tooth_mk | 0.77 | 0.77 |
| zhi | 0.74 | 0.72 |  | **MEAN** | | **0.735** |

## Phase 3 — Stage 2 interpretation
- ✅ LLM adapter (`llm_client.py`) — backend-agnostic (none / OpenAI-compatible vLLM/API via env)
- ✅ Seed TCM knowledge base + rule-based retrieval (`knowledge_base/tcm_patterns.json`)
- ✅ JSON→report generator (`interpret.py`) — deterministic template + graded language + pattern voting
- ✅ **Combination rules** for context-dependent mapping (`interpret.apply_combination_rules`, tested by
  `evaluation/eval_mapping.py`)
- ✅ **Grounded RAG+LLM narrative** — rule backbone + a TRUE vector RAG (faiss + nomic-embed via Ollama +
  TF-IDF hybrid) over a 102-chunk cited corpus (`knowledge_cards.json`); retrieval hit@4 96%
  (`evaluation/eval_rag.py`). See `docs/archive/RAG_LLM_INTERPRETATION.md`.
- 🔄 **WS-A macro-micro knowledge graph** (`stage2_interpretation/kg/`, the Stage-2 overhaul — PLAN.md §3):
  seed layer from `tcm_knowledge.json` with `--verify` **superset parity**; macro layer = Gerlach's 184
  book sections (`parse_book.py`); graph = 359 nodes / 427 edges / 10 rules incl. inverse `evidence_for`
  edges (WS-B lever). Micro layer: offline extractor `micro_extract.py` (casper, free) + `normalize.py`;
  **qwen2.5:14b chosen** over gemma3:4b on Gerlach ch.2 (more faithful, 0 junk). Next: full ch.2–4 run +
  `add_micro_layer`.
- ⬜ WS-C grounded cite-or-abstain matcher (shadow → ensemble) · ⬜ WS-B refinement engine (symptom evidence
  + information-gain questions) · ⬜ WS-D RAGAS eval gate.

## Phase 4 — Integration
- ✅ `pipeline.py` orchestrator (image[+metadata] → quantitative JSON + report) — **end-to-end works**
- ✅ Quality gate (mask-coverage reject) wired in
- ✅ FastAPI service + web demo (`deployment/api`): live-camera **framing guide oval**, upload,
  **visible mask overlay**, framing feedback, characteristic bars, wellness report. Live on `:7860`.
- ✅ **WS-F phone demo (2026-07-16):** `deployment/api/static/index.html` rebuilt to the
  `TongueInsight.dc.html` design — 393-wide phone shell, Chinese-first (Noto Sans TC), pure-black ground.
  Screens: **Capture** (宜/忌 guide) → **Analysing** (6-sign ticker) → **Reading** (annotated hero,
  傳統上多與 linkage cards with **evidence chips derived from `feature.points_to`** + cited 為何·Why rows +
  confidence bars, 六項舌徵 banded tracks, 今日宜忌 advice, degraded-narrator banner) → **Refine**
  (info-gain questions, one per screen → re-render with 已補問·refined marker) → **Sources sheet**
  (per-pattern citations, WHO-IST note). Wired to existing `/analyze` + `/refine`. Adds `?demo=1`
  offline fixture (renders the flagship state with **no model box** — for local review + a community
  fallback). Verified via DOM: all 4 screens + refine + sources flows.

## Phase 5–7 — Optimize / evaluate / deploy
- ⬜ ONNX export + benchmarks · metrics · API hardening + disclaimers
- ⬜ Deployment: containerize FastAPI (CPU-fast, 0.34s/img) → thin client / PWA; on-device is v2

---

## Current state (2026-07-16) — see `docs/PLAN.md` (SoT) + `docs/HANDOFF.md`
**Stage 1 FROZEN:** seg_combined + **multitask_v5** (v6/v7/v8 all lost to v5 on the honest human metric) +
extra_features + `zoning.py` geometry. Honest accuracy ~61% exact / **97% within-one-grade** vs human. Not
chasing further (label ceiling).
**Stage 2 = the active work (PLAN.md §3):** rule engine (combination rules, mapping test 12/12) +
grounded RAG+LLM narrative (retrieval 96%), now being overhauled onto a **macro-micro knowledge graph**
(`kg/`) built from the newly-licensed books in `tongue_lit/`. WS-A graph is built (seed+macro, parity
verified); micro extraction underway (qwen2.5:14b on casper, free).
**App design:** a Claude Design handoff bundle for the Savor 舌 tab lives in `prompt-execution-request/` —
maps 1:1 onto WS-B (refine), WS-C (linkage cards), and the Sources sheet.

**Blocked on user data:** real-phone photos for color calibration + true real-world accuracy.
**Next (PLAN.md sequence):** finish WS-A micro layer → WS-C matcher (shadow) → WS-B refinement → WS-D
RAGAS gate → WS-E deploy. **Repo cleanup done** (docs 26→11 living; archives under `docs/archive/` +
`evaluation/archive/`).

## Was waiting on team (resolved / moot)
- vLLM `:8000` key — not obtained; using **local Ollama** (gemma3 + nomic-embed, no auth) instead.
- GPU0 has been usable throughout. Stage-2 LLM backend = Ollama (env-swappable).
