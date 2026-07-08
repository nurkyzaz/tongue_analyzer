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
- ⬜ 2.1 Segmentation (U-Net++ ResNet-34, Raw→Mask) — **first vertical slice**
- ⬜ 2.2 Multi-task head (5 key chars cls + phenotype reg, mask feature-masking, Focal Loss)
- ⬜ 2.3 Train/eval; export Stage1Output JSON + inference script

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
