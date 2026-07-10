# Data, Models & Research

## Datasets

| Dataset | Size | What it gives | Status | License |
|---|---|---|---|---|
| **TonguExpert** | 5,992 | images + masks + 5 labels + **~1,300 continuous phenotypes** (crack area, coating %, color) | ✅ in use (features + **severity targets**) | verify terms |
| **SM-Tongue** | 2,155 real | real-photo segmentation masks | ✅ in use (fixed real-photo domain gap: seg 0.75→0.975) | CC-BY-NC-4.0 (non-commercial) |
| **TCM-Tongue** | 6,719 | **20 practitioner-verified categories** (adds purple, red-dots, peeled, swollen, thin, red…) | ⏳ Phase 4 — Dryad download blocked by anti-bot; needs manual fetch | CC BY 4.0 (Dryad) |
| **RTDS** | 2,100 | code only (U-Net++, Swin-hybrid, Focal Loss) — data private | reference recipe | — |
| Memory-SAM / SM-Tongue tooling | — | SAM2+DINO training-free segmenter | benchmarked (auto-labeler, not in serving path) | SAM2 Apache-2.0; DINOv3 gated |

Full audit incl. UTongue/BioHit/TOM and licensing flags: [../RESOURCES.md](../RESOURCES.md).

## Models & current metrics

| Model | Arch | Trained on | Metric |
|---|---|---|---|
| Segmentation `seg_combined` | U-Net++ / ResNet-34 | TonguExpert + SM-Tongue | Dice **0.994** clinical / **0.975** real photos |
| Features `multitask_v3` | shared ResNet-34 + mask-guided pooling; 5 cls heads + severity head | TonguExpert (labels + continuous phenotypes) | macro-F1 **0.738**, severity MAE **0.11** |
| (benchmark) Memory-SAM | SAM2 + DINOv2 retrieval-to-prompt | training-free | Dice 0.980 real (but ~1B params, 1.1 s/img) — U-Net++ chosen for serving |

## Why outputs were "generic" (and the fix)
Root cause: the model emitted only a 3-way class per feature via argmax, discarding degree, and the
**continuous phenotypes were unused**. Fix (done): a **severity regression head** on those phenotypes +
expected-ordinal severity from class probs → graded output; the interpreter reads severity → bands →
"faint/mild/moderate/pronounced". Result: subtle features (e.g. a faint crack at 0.19) now surface,
while a balanced tongue reads "Balanced". Detail: [../IMPROVEMENT_PLAN.md](../IMPROVEMENT_PLAN.md).

## Research grounding (for the knowledge layer)
How TCM literature categorizes features (body: color/shape/cracks/tooth-marks; coating:
color/thickness/quality; organ sub-regions) and the reputable open resources we ground in (WHO ICD-11,
CCMQ, SymMap, Maciocia/Kirschbaum): [../TCM_RESEARCH.md](../TCM_RESEARCH.md) and
[03_KNOWLEDGE_SOURCES.md](03_KNOWLEDGE_SOURCES.md).

## Phase 4 (next) — wider feature vocabulary
Integrate **TCM-Tongue** to detect purple body (blood stasis), red dots (heat), peeled/mirror coating
(Stomach-Yin), swollen & thin shape, red tongue. Plan (partial-label multi-task head on both datasets,
no destabilizing v3): [../PHASE4_TCM_TONGUE.md](../PHASE4_TCM_TONGUE.md). **Blocker:** getting the 2.34 GB
dataset onto the server (Dryad is behind Cloudflare anti-bot; GitHub mirror is Baidu-Cloud only).

## Roadmap (fast wins → longer)
1. ✅ Grounded dual-language KB + graded interpreter + follow-up flow
2. ✅ Severity regression (sensitivity fix), v3 model
3. ⏳ Phase 4: TCM-Tongue new features (blocked on data fetch)
4. ⬜ Color-calibration preprocessing (phone-camera color robustness)
5. ⬜ Real-photo eval set across skin tones/lighting; confidence calibration
6. ⬜ Optional LLM backend for richer prose (grounded RAG)
