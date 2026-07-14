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
  (`evaluation/eval_rag.py`). See `docs/RAG_LLM_INTERPRETATION.md`.

## Phase 4 — Integration
- ✅ `pipeline.py` orchestrator (image[+metadata] → quantitative JSON + report) — **end-to-end works**
- ✅ Quality gate (mask-coverage reject) wired in
- ✅ FastAPI service + web demo (`deployment/api`): live-camera **framing guide oval**, upload,
  **visible mask overlay**, framing feedback, characteristic bars, wellness report. Live on `:7860`.

## Phase 5–7 — Optimize / evaluate / deploy
- ⬜ ONNX export + benchmarks · metrics · API hardening + disclaimers
- ⬜ Deployment: containerize FastAPI (CPU-fast, 0.34s/img) → thin client / PWA; on-device is v2

---

## Current state (2026-07-13) — see `docs/HANDOFF.md` for the full picture
**Production:** seg_combined + **multitask_v5** (v6/v7/v8 all lost to v5 on the honest human metric) +
extra_features. **Honest accuracy ~61% vs human** (`eval_model.py --source human`); coating weakest.
**Stage-1 additions:** coating split thickness×texture (thickness 82% vs conflated 55%), red-tip &
moisture signals (`zoning.py`). **Stage-2:** combination rules (mapping test 12/12) + graded/honest
report (headline, distinctiveness, per-sign confidence) + grounded RAG+LLM (Ollama gemma3 + nomic-embed,
102-chunk corpus, retrieval 96%). **Testing:** human40 (labeled), human40b (40 imgs, NOT yet labeled),
merged `label_store`, mapping_testset, eval_rag, labeled `data/eval/gallery/`. **Demo** live on :7860 with
LLM+RAG.

**Blocked on user data:** label human40b (`evaluation/label_human40b.html`); real-phone photos for color
calibration + true real-world accuracy. **Next-value:** grow the human eval → re-verify; cleaner coating
labels (not loss re-weighting); grow RAG corpus; hallucination-rate eval before LLM-default; deploy.

## Was waiting on team (resolved / moot)
- vLLM `:8000` key — not obtained; using **local Ollama** (gemma3 + nomic-embed, no auth) instead.
- GPU0 has been usable throughout. Stage-2 LLM backend = Ollama (env-swappable).
