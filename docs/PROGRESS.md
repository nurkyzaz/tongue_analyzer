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
- ⬜ 2.2b Regression heads for continuous phenotypes (P*.txt) — richer output
- ⬜ 2.4 Real-photo eval set (domain-gap check) + tune coating rare class

### Stage-1 test metrics (695 imgs, gold-preferred labels)
| char | acc | macroF1 |  | char | acc | macroF1 |
|---|---|---|---|---|---|---|
| coating | 0.88 | 0.56 |  | fissure | 0.86 | 0.84 |
| tai | 0.82 | 0.78 |  | tooth_mk | 0.77 | 0.77 |
| zhi | 0.74 | 0.72 |  | **MEAN** | | **0.735** |

## Phase 3 — Stage 2 interpretation
- ⬜ LLM adapter (vLLM / local / API) — **decision pending team answers**
- ⬜ TCM knowledge base + FAISS RAG
- ⬜ JSON→report generator + prompts

## Phase 4 — Integration
- ⬜ `run_full_analysis(image, metadata)` orchestrator
- ⬜ FastAPI service + Gradio demo + quality gate

## Phase 5–7 — Optimize / evaluate / deploy
- ⬜ ONNX export + benchmarks · metrics · API hardening + disclaimers

## Waiting on team
- vLLM `:8000` API key + shared-capacity confirmation
- GPU0 exclusivity confirmation
- Preferred Stage-2 LLM backend
