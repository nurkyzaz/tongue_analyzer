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

## Status (2026-07-09)

**Working end-to-end**, with a clear improvement path.

| Component | State |
|---|---|
| Segmentation (U-Net++, TonguExpert + SM-Tongue) | ✅ Dice 0.994 clinical / **0.975 real photos** |
| Feature model (5 characteristics, multi-task) | ✅ test macro-F1 **0.735** — *categorical only (see below)* |
| Interpretation (grounded TCM KB, LLM-optional) | ✅ per-feature + combined patterns |
| FastAPI service + web demo (camera guide, mask overlay, examples gallery) | ✅ runs on casper |
| **Known limitation** | outputs feel generic on near-normal tongues — **fix designed in [IMPROVEMENT_PLAN](docs/IMPROVEMENT_PLAN.md)** (predict *degree*, not just class) |

## Docs (read these)

- **[docs/IMPROVEMENT_PLAN.md](docs/IMPROVEMENT_PLAN.md)** — why outputs feel generic + the plan to fix
  sensitivity, dual-language interpretation, and the optional follow-up-question flow. **Start here.**
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
