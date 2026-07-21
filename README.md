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

## Status (2026-07-16)

**Working end-to-end**, honestly measured. Stage 1 is **frozen**; the active work is the **Stage 2
knowledge-graph + RAG overhaul**. **New agents / contributors: read [docs/PLAN.md](docs/PLAN.md) — the
single source of truth — then [docs/PROGRESS.md](docs/PROGRESS.md) for the live log.**

| Component | State |
|---|---|
| Segmentation (U-Net++, TonguExpert + SM-Tongue) | ✅ **frozen** — 0.975 real-photo Dice (`seg_combined`) |
| Feature model (5 core + 8 extra, multi-task) | ✅ **frozen** — `multitask_v5`; ~61% exact / 97% within-one-grade vs human labels |
| Stage-1 signals | ✅ coating split thickness×texture, red-tip, moisture (`zoning.py`) |
| Interpretation (rule engine + KB) | ✅ distinctiveness-weighted votes + **combination rules** (context-aware) + honest graded report |
| Grounded RAG + LLM narrative | ✅ rule backbone + true vector RAG (cited corpus, retrieval 96%) + local-LLM narrative (Ollama) |
| **Macro-micro knowledge graph** (`stage2_interpretation/kg/`) | 🔄 seed+macro built (parity-verified); micro extraction from licensed books underway (PLAN.md §3) |
| FastAPI service + web demo | ✅ live on casper `:7860` (camera guide, mask overlay), cloudflared HTTPS tunnel |
| **Honest metric** | promote a change only if it beats v5 on the **human** eval / mapping test — not on val/auto |

## Docs (read these)

- **[docs/PLAN.md](docs/PLAN.md)** — the master plan / single source of truth for remaining work. **Start here.**
- **[docs/PROGRESS.md](docs/PROGRESS.md)** — the live implementation log (most recent work first).
- **[stage2_interpretation/kg/README.md](stage2_interpretation/kg/README.md)** — the knowledge-graph layers, model, and build.
- **[docs/RESOURCES.md](docs/RESOURCES.md)** — audit of every external dataset/model, with licensing flags.
- **[docs/CORPUS.md](docs/CORPUS.md)** — RAG corpus + source registry (licenses/permissions).
- **[docs/VALIDATION_WORKLIST.md](docs/VALIDATION_WORKLIST.md)** — detector validation results + what still needs labels.
- Mapping/label/metric refs: `FEATURE_PATTERN_MAPPING.md`, `LABEL_STORE.md`, `LABELING_GUIDE.md`, `BENCHMARK.md`.
- Product/UX: **[docs/design/00_INDEX.md](docs/design/00_INDEX.md)** and `design/`. Decision memo: `CEO_MARKET_REGULATORY_QA.md`.
- Deep history lives in `git log` (superseded planning/investigation notes were pruned 2026-07-21).

## Repository layout

| Path | Purpose |
|---|---|
| `stage1_quantitative/` | Segmentation + multi-task feature model + inference → JSON |
| `stage2_interpretation/` | Interpreter + rule KB + RAG; `kg/` = macro-micro knowledge graph (WS-A) |
| `data/` | Data prep + loaders (raw data **not** committed) |
| `evaluation/` | Living eval harnesses (one-offs under `evaluation/archive/`) |
| `deployment/api/` | FastAPI service + single-page web demo |
| `design/` | Claude Design handoff bundle — the Savor 舌 tab app comps |
| `docs/` | Documentation (`PLAN.md` = SoT; superseded notes pruned — see `git log`) |

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
