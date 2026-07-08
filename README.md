# TongueInsight Hybrid (TIH)

A two-stage system for quantitative tongue analysis + clinically-grounded interpretation,
designed to run behind a mobile app: **user uploads a tongue photo → structured analysis + written report.**

```
photo ──► [Stage 1: Quantitative]  ──► structured JSON ──► [Stage 2: Interpretation] ──► report
             segmentation + phenotypes                        RAG over TCM knowledge (LLM)
```

## Status

Early implementation. See [docs/PROGRESS.md](docs/PROGRESS.md) for the live task board and
[docs/RESOURCES.md](docs/RESOURCES.md) for the audit of external datasets/code (what actually
exists and what doesn't).

## Repository layout

| Path | Purpose |
|---|---|
| `stage1_quantitative/` | Segmentation + multi-task phenotype/characteristic model |
| `stage2_interpretation/` | RAG-based clinical report generator (LLM-agnostic) |
| `data/` | Data prep + loaders (raw data is **not** committed) |
| `evaluation/` | Metrics (Dice/IoU, mAP/F1, regression error, report quality) |
| `deployment/` | FastAPI service + Gradio demo |
| `scripts/` | One-off + cluster job scripts |
| `docs/` | Architecture, resources/risk register, progress |

## Compute

Training/inference run on **casper** (`192.168.1.184`), 2× RTX 3090 24GB.
Use **GPU 0** (GPU 1 hosts a shared Qwen2.5-VL vLLM server). See [docs/RESOURCES.md](docs/RESOURCES.md).

## Key adaptation vs. the original plan

The Stage-2 reference (MMIR-TCM) has **no released code/weights/data**, so Stage 2 is built as our
own **LLM-agnostic RAG layer** (text-only: Stage-1 JSON → report), not a clone. Stage 1 is built on
the **TonguExpert** dataset (raw images + masks + 5 key labels + hundreds of phenotypes), which is
downloadable, with architecture recipes from **RTDS** and **SSC-Net**.

> **Medical disclaimer:** This is a wellness/informational tool, not a medical device. Output must
> carry clear disclaimers and must not be presented as diagnosis.
