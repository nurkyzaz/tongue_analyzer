# TongueInsight Hybrid (TIH) — `tongue_analyzer`

An **educational** image-analysis tool that detects visual features on a tongue photo (coating
texture/color, cracks, shape, tooth-marks) and maps them to the corresponding concepts in the
**traditional-Chinese-medicine (TCM) tongue-reading tradition**, described in both the TCM term and a
plain-language gloss.

> **This is not a medical device or a diagnostic instrument.** Every output is framed as *"traditionally
> associated with…"* — one framework's perspective, not a diagnosis. See the disclaimer in every report.

```
photo ─► [Stage 1: feature detection]  ─► structured JSON ─► [Stage 2: interpretation] ─► educational report
            segmentation + graded features                    grounded TCM knowledge base (+ optional LLM)
```

## Status (2026-07-13)

**Working end-to-end**, honestly measured. **New agents / contributors: start with
[docs/HANDOFF.md](docs/HANDOFF.md).**

| Component | State |
|---|---|
| Segmentation (U-Net++, TonguExpert + SM-Tongue) | ✅ **0.975 real-photo Dice** (`seg_combined`) |
| Feature model (5 core + 8 extra, multi-task) | ✅ **production = `multitask_v5`**; ~61% vs HUMAN labels (auto-benchmark 0.87 is optimistic) |
| Stage-1 signals | ✅ coating split thickness×texture, red-tip, moisture (`zoning.py`) |
| Interpretation (rule engine + KB) | ✅ distinctiveness-weighted votes + **combination rules** (context-aware) + honest graded report |
| Grounded RAG + LLM narrative | ✅ rule backbone + **true vector RAG** (102-chunk cited corpus, retrieval 96%) + local-LLM narrative (Ollama) |
| FastAPI service + web demo | ✅ live on casper `:7860` (camera guide, mask overlay, examples), cloudflared HTTPS tunnel |
| **Honest metric** | promote a change only if it beats v5 on the **human** eval / mapping test — not on val/auto |

## Docs (read these)

- **[docs/HANDOFF.md](docs/HANDOFF.md)** — self-contained project state, where it runs, and next steps. **Start here.**
- **[docs/DIRECTION_REVIEW.md](docs/DIRECTION_REVIEW.md)** — the assessment + workstream plan (WS1–6) with statuses.
- **[docs/RAG_LLM_INTERPRETATION.md](docs/RAG_LLM_INTERPRETATION.md)** — the hybrid rule+RAG+LLM interpretation layer.
- **[docs/ACCURACY_INVESTIGATION.md](docs/ACCURACY_INVESTIGATION.md)** — why the model misreads and what actually helps.
- **[docs/IMPROVEMENT_PLAN.md](docs/IMPROVEMENT_PLAN.md)** — the original sensitivity/interpretation plan (largely delivered).
- **[docs/TCM_RESEARCH.md](docs/TCM_RESEARCH.md)** — how TCM literature categorizes features, and the
  reputable open resources (WHO ICD-11, CCMQ, SymMap) we ground the knowledge base in.
- **[docs/RESOURCES.md](docs/RESOURCES.md)** — audit of every external dataset/model, with licensing flags.
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** · **[docs/PROGRESS.md](docs/PROGRESS.md)**

## Repository layout

| Path | Purpose |
|---|---|
| `stage1_quantitative/` | Segmentation + multi-task feature model + inference → JSON |
| `stage2_interpretation/` | Grounded TCM knowledge base + interpreter (LLM-optional) |
| `data/` | Data prep + loaders (raw data **not** committed) |
| `evaluation/` | Metrics (Dice/IoU, macro-F1, Memory-SAM benchmark) |
| `deployment/api/` | FastAPI service + single-page web demo |
| `scripts/`, `docs/` | Cluster/one-off scripts · documentation |

## Compute

Training/inference run on **casper** (`192.168.1.184`), 2× RTX 3090 24GB; use **GPU 0**. Code is synced
there under `~/tongue/`; data/venv/checkpoints live on the server (not in git). See RESOURCES.md.

## Reproduce (on casper)

```bash
source envs/tih/bin/activate
python data/build_manifest.py --root data/raw                 # TonguExpert labels → manifest
python data/build_seg_manifest.py                              # + SM-Tongue → unified seg manifest
python stage1_quantitative/segmentation/train.py --unified ... # U-Net++ segmentation
python stage1_quantitative/feature_extraction/train.py ...     # multi-task feature head
uvicorn deployment.api.app:app --host 0.0.0.0 --port 7860      # demo + API
```
