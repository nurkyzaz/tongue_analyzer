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

## Status (2026-07-21)

**Working end-to-end**, honestly measured. Stage 1 is **frozen**. **New engineers: read
[docs/PROJECT_HANDBOOK.md](docs/PROJECT_HANDBOOK.md) — the single onboarding + reference doc.**

| Component | State |
|---|---|
| Segmentation (U-Net++, TonguExpert + SM-Tongue) | ✅ **frozen** — 0.975 real-photo Dice (`seg_combined`) |
| Feature model (5 core + 8 extra, multi-task) | ✅ **frozen** — `multitask_v5`; ~61% exact / 97% within-one-grade vs human labels |
| Stage-1 signals | ✅ coating split thickness×texture, red-tip, moisture (`zoning.py`) |
| Interpretation (rule engine + KB) | ✅ distinctiveness-weighted votes + **combination rules** (context-aware) + honest graded report |
| Grounded RAG + LLM narrative | ✅ rule backbone + true vector RAG (cited corpus, retrieval 96%) + local-LLM narrative (Ollama) |
| **Macro-micro knowledge graph** (`stage2_interpretation/kg/`) | ✅ built from licensed books (~605 nodes / 1245 edges / 282 cited); cite-or-abstain matcher + rule ensemble live |
| FastAPI service + web demo | ✅ live on casper `:7860` (camera guide, mask overlay), cloudflared HTTPS tunnel |
| **Honest metric** | promote a change only if it beats v5 on the **human** eval / mapping test — not on val/auto |

## Docs

- **[docs/PROJECT_HANDBOOK.md](docs/PROJECT_HANDBOOK.md)** — everything: transfer checklist, repo map, data/models, pipeline, rule engine, KG/matcher, RAG, eval, run/config. **Start here.**
- **[stage2_interpretation/kg/README.md](stage2_interpretation/kg/README.md)** — knowledge-graph build detail (beside its code).
- `design/` — the demo's UI design pack (BRAND, FEATURE_SPEC, prototype `mockup.html`).
- `stage2_interpretation/knowledge_base/sources.json` — machine-readable licence registry.
- Historical planning/progress/analysis notes were consolidated into the handbook and pruned — see `git log`.

## Repository layout

| Path | Purpose |
|---|---|
| `stage1_quantitative/` | Segmentation + multi-task feature model + inference → JSON |
| `stage2_interpretation/` | Interpreter + rule KB + RAG; `kg/` = macro-micro knowledge graph (WS-A) |
| `data/` | Data prep + loaders (raw data **not** committed) |
| `evaluation/` | Living eval harnesses (one-offs under `evaluation/archive/`) |
| `deployment/api/` | FastAPI service + single-page web demo |
| `design/` | Claude Design handoff bundle — the Savor 舌 tab app comps |
| `docs/` | Single handbook (`PROJECT_HANDBOOK.md`) + `design/`; history in `git log` |

## Compute

Training/inference run on **casper** (`192.168.1.184`), 2× RTX 3090 24GB; use **GPU 0**. Code is synced
there under `~/tongue/`; data/venv/checkpoints live on the server (not in git). See the handbook §2/§4.

## Reproduce (on casper)

```bash
source envs/tih/bin/activate
python data/build_manifest.py --root data/raw                 # TonguExpert labels → manifest
python data/build_seg_manifest.py                              # + SM-Tongue → unified seg manifest
python stage1_quantitative/segmentation/train.py --unified ... # U-Net++ segmentation
python stage1_quantitative/feature_extraction/train.py ...     # multi-task feature head
uvicorn deployment.api.app:app --host 0.0.0.0 --port 7860      # demo + API
```
