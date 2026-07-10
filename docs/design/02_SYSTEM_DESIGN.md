# System Design

Two stages behind one API. Stage 1 is the **image model** (detects features); Stage 2 is the
**knowledge layer** (turns features into a grounded, dual-language educational report). They talk
through a stable JSON contract, so either can evolve independently.

```
photo ─► [Stage 1: image model] ─► Stage-1 JSON ─► [Stage 2: knowledge layer] ─► report
           segmentation                             feature→pattern reasoning
           + feature detection                      over tcm_knowledge.json
           + severity regression
```

## Components & files

### Stage 1 — quantitative (`stage1_quantitative/`)
| File | Role |
|---|---|
| `segmentation/` | U-Net++ (ResNet-34) train/dataset — tongue mask. `checkpoints/seg_combined`. |
| `feature_extraction/model.py` | `MultiTaskTongueNet`: shared encoder → mask-guided pooling → 5 classification heads **+ a severity regression head**. |
| `feature_extraction/train.py` | Multi-task training (Focal loss + Smooth-L1 severity). `checkpoints/multitask_v3`. |
| `labels.py` | The 5 characteristics, their classes, and the severity keys. |
| `infer.py` | `Stage1Pipeline`: photo → mask → per-feature class **probs + graded severity** → `Stage1Output` JSON. |
| `schema.py` | The Stage-1 → Stage-2 JSON contract. |

**The 5 characteristics:** `coating` (thickness/greasiness), `tai` (coating color), `zhi` (body color),
`fissure` (cracks), `tooth_mk` (tooth-marks). Each returns a value, calibrated-ish confidence,
per-class probabilities, and a 0–1 **severity**.

### Stage 2 — interpretation (`stage2_interpretation/`)
| File | Role |
|---|---|
| `knowledge_base/tcm_knowledge.json` | **The knowledge base** — features, patterns, symptoms, questions, recommendations, sources. |
| `interpret.py` | Severity→bands, dual-language readings, **severity-weighted pattern voting**, `refine()` follow-up. |
| `llm_client.py` | Optional LLM backend (RAG **rephrases** grounded content; off by default). |

### Orchestration & serving
| File | Role |
|---|---|
| `pipeline.py` | `FullPipeline.analyze_array(img)` → runs Stage 1 + Stage 2, returns the full result (+ mask for overlay). |
| `deployment/api/app.py` | FastAPI: `/analyze`, `/refine`, `/examples`, `/health`, `/` (demo). CORS + optional API key. |
| `deployment/api/service.py` | Adds mask overlay + framing feedback. |
| `deployment/api/static/index.html` | Single-page demo: camera guide, mask overlay, graded feature cards, pattern cards, interactive follow-up. |

### Evaluation & data
| File | Role |
|---|---|
| `data/build_manifest.py` / `build_seg_manifest.py` / `build_severity.py` | Build training manifests (labels, unified seg, severity targets). |
| `evaluation/eval_seg.py` / `eval_stage1.py` / `eval_memory_sam.py` | Segmentation, characteristic, and Memory-SAM benchmarks. |
| `scripts/make_test_set.py` / `build_examples.py` | Diverse test folder + demo gallery. |

## Design principles
- **Transparent, not black-box:** the image model only detects features; the "diagnosis" reasoning is
  explicit, weighted voting over an editable KB — you can trace every conclusion to its inputs.
- **Grounded:** every statement cites Maciocia/ICD-11/CCMQ/SymMap; the LLM (if enabled) may only
  rephrase grounded content.
- **Degree-aware:** severity regression + bands make it sensitive to subtle features (the fix for
  "generic" output).
- **Deployable:** Stage-1 is a 24M-param CNN (≈5 ms, mobile-exportable); heavy models (SAM2/DINO) are
  used only as offline tools, not in the serving path.
- **Educational guardrails:** wellness framing, "traditionally associated with…", quality gate,
  disclaimers, no diagnosis claims.

## Compute / deployment
Runs on **casper** (`192.168.1.184`, GPU 0). Served via `uvicorn` on `:7860`; public HTTPS via a
Cloudflare quick-tunnel (so the camera works anywhere). Checkpoints/data live on the server (not git).
