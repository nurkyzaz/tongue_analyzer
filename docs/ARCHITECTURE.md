# TongueInsight Hybrid — Architecture (v1.1, reality-adjusted)

## Pipeline

```
                 ┌─────────────────────── Stage 1: Quantitative ───────────────────────┐
 photo ─► quality gate ─► SEGMENTATION (U-Net++) ─► masked ROI ─► MULTI-TASK HEAD ─┐
                              (Raw→Mask)                          ├─ 5 key chars (cls) │
                                                                  ├─ phenotypes (reg)  │
                                                                  └─ quality score     │
                                                                                       ▼
                                                                          structured JSON
                                                                                       │
                 ┌──────────────────── Stage 2: Interpretation ─────────────────────┐  │
 metadata ─────► │ prompt-build ◄── RAG (FAISS over TCM KB) ── LLM (text) ─► report │◄─┘
                 └──────────────────────────────────────────────────────────────────┘
```

## Stage 1 — Quantitative (buildable now on TonguExpert)

**Data:** TonguExpert 5,992 (Raw+Mask). Labels: `L2_Labels_Predict` (5 key chars, all 5,992, weak) +
`L1_Labels_Manual` (1,747 gold, sparse). Continuous phenotypes: `P*.txt` (color/shape/texture/CNN per region).

**1.1 Segmentation** — U-Net++ (RTDS recipe) with ImageNet-pretrained ResNet-34 encoder, Dice+BCE loss.
Trained directly on Raw→Mask. Heavy real-world augmentation to close the app-photo domain gap.
(SAM optional later for mask refinement; masks are already provided so U-Net++ is the pragmatic core.)

**1.2 Multi-task head** (SSC-Net design) — shared encoder (ResNet-34 / Swin-tiny via `timm`):
- Segmentation mask **masks background features** before pooling (SSC-Net key idea).
- Heads:
  - **5 categorical** classifiers: `coating, tai, zhi, fissure, tooth_mk` (Focal Loss for imbalance).
  - **Regression** head for selected continuous phenotypes (normalized).
  - **Quality** score.
- Train on L2 (weak) → fine-tune/weight L1 (gold). Manual labels used for eval.

**Output:** `stage1_quantitative/schema.py` `Stage1Output` → JSON (key chars + confidences, phenotype
vector, mask ref, overall quality).

## Stage 2 — Interpretation (our own RAG; MMIR-TCM has no code)

- **Input:** Stage-1 JSON (+ optional metadata: complaint, symptoms, history).
- **RAG:** FAISS over a curated TCM knowledge base (pattern cards, characteristic→pattern mappings),
  built by us. Retrieves relevant TCM patterns for the observed characteristics.
- **LLM:** text-only, via an **adapter** (`LLMClient`) that can target: the shared vLLM `:8000`,
  a local small model on GPU0, or a hosted API. ~3B text model is sufficient.
- **Output:** structured report (observations → pattern(s) → plain-language explanation → disclaimer),
  with retrieved citations for grounding.

## Deployment

FastAPI `POST /analyze` (image [+ metadata]) → `{quantitative: {...}, report: "...", disclaimers: [...]}`.
Stage 1 exportable to ONNX for on-device/edge; Stage 2 via cloud LLM. Gradio demo for internal testing.

## Non-goals / guardrails

Not a medical device. Wellness/informational framing, explicit disclaimers, no diagnosis claims,
image quality gate rejects unusable photos.
