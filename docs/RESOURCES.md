# Resource Audit & Risk Register

_Last verified: 2026-07-08._ This is the "what actually exists" table that the implementation
plan glossed over. Verified live before committing compute.

## External datasets & code

| Resource | Paper | Code | Data | Status / notes |
|---|---|---|---|---|
| **TonguExpert** | biosino.org platform | platform algos (SAM/ResNet/YOLO/VGG) | ✅ **Downloadable** ~302MB | **Primary data.** v1.1 (Jul 2024): 5,992 subjects, 1,353 phenotypes. Includes `TongueImage/Raw` + `TongueImage/Mask` + manual & predicted labels. No login. |
| **RTDS** | Signal Image & Video Processing 2026 | ✅ [github.com/byzhang811/RTDS-Tongue_Analysis](https://github.com/byzhang811/RTDS-Tongue_Analysis) (U-Net++, Swin-hybrid, Focal Loss) | ❌ 2,100 imgs **private** | Reuse architecture/recipe only. |
| **SSC-Net** | Digital Health 2025 ([PMC12099091](https://pmc.ncbi.nlm.nih.gov/articles/PMC12099091/)) | availability unconfirmed | ❌ BUCM 1,500 imgs not public | Reuse multi-task design (mask→feature-masking, ROI fusion, 5 key chars). |
| **MMIR-TCM** | [arXiv 2607.01814](https://arxiv.org/abs/2607.01814) (Jul 2026) | ❌ "Coming soon" — docs only | ❌ MedTCM not released | **Cannot clone.** Build our own Stage-2 RAG instead. |
| **Memory-SAM** | [arXiv 2510.15849](https://arxiv.org/abs/2510.15849) | ✅ [github.com/jw-chae/memory-sam](https://github.com/jw-chae/memory-sam) (SAM2 + DINOv3 retrieval-to-prompt) | ✅ **SM-Tongue 2,155 real 512² pairs** (HF `Mark-CHAE/SM-Tongue-Public-Original512`) | The real Memory-SAM. Cloned to `repos/memory-sam`. Implemented with **DINOv2 fallback** (DINOv3 gated). SM-Tongue is our real-photo data. |
| **TCM-Tongue** | [arXiv 2507.18288](https://arxiv.org/abs/2507.18288) (Jul 2025) | benchmarks (YOLO/SSD/MobileNet) | ✅ **6,719 imgs, 20 practitioner-verified categories** ([GitHub](https://github.com/btbuIntelliSense/Intelligent-tongue-diagnosis-detection-dataset) → Baidu Cloud; [Dryad DOI](https://datadryad.org/dataset/doi:10.5061/dryad.1c59zw48r)) | **Best new dataset.** Object-detection boxes (COCO/TXT/XML). Adds features we lack (red dots, peeled, purple, thin/chubby, organ subregions). Dryad=CC BY 4.0; GitHub mirror restates none. |
| **UTongue** (UESTC) | — | ✅ repo (MobileNetV2/ResNet) | ~3,000, body color/coating/9-class shape | Baseline pipeline + extra classification data. |
| **BioHit** | — | — | [300 imgs + manual masks](https://github.com/BioHit/TongeImageDataset) | Small seg add-on. |
| **TOM** | [arXiv 2508.14932](https://arxiv.org/abs/2508.14932) | ✅ open-source seg tool (95.2% mIoU) | — | Alternative seg (multi-teacher distillation). Our combined U-Net++ already ≈ this. |

### 🧭 TCM knowledge resources (for the grounded reference set — see TCM_RESEARCH.md)
| Resource | Source | Use | Access |
|---|---|---|---|
| **WHO ICD-11 Ch.26** | World Health Organization | 196 standardized TM **pattern** names/codes | public [icd.who.int](https://icd.who.int/) |
| **CCMQ (9 constitutions)** | Wang Qi, Beijing Univ. of Chinese Medicine | validated questionnaire → **follow-up-question flow** | items in peer-reviewed papers |
| **SymMap** | *Nucleic Acids Research* 2019 | TCM symptom → **plain-language / modern symptom** mapping | [symmap.org](http://www.symmap.org) |
| **Maciocia / Kirschbaum** | standard TCM texts | per-feature interpretation (basis of `tcm_knowledge.json`) | reference (cite) |

### ⚠️ Licensing flags for COMMERCIAL use (important for the mobile app)
- **SM-Tongue**: CC-BY-NC-4.0 (**non-commercial**) — OK for research/eval/demo; needs commercial license (or eval-only use, not production training) before shipping.
- **DINOv3**: Meta gated research license — we use non-gated **DINOv2** as fallback; verify DINOv2 license (Apache-2.0 for code; weights license) for production.
- **SAM2**: Apache-2.0 (commercial OK). **TonguExpert**: verify dataset license/terms before commercial training.

## TonguExpert dataset structure (from README/Header)

```
TonguExpertDatabase/
├── Phenotypes/
│   ├── L1_Labels_Manual.txt      # human-verified gold labels
│   ├── L2_Labels_Predict.txt     # model-predicted labels
│   ├── P11_Tg_Color / P12_Shape / P13_Texture / P14_CNN
│   ├── P2*_Tai_* (coating) / P3*_Zhi_* (body)
│   ├── P4*_Fissure_* / P5*_Toothmark_* / P60_Subregions
├── TongueImage/{Raw, Mask}/
└── README.txt
```

**5 key characteristics** (categorical `*_label` + confidence `*_score`):
`coating` (greasy), `tai` (coating color), `zhi` (body color), `fissure`, `tooth_mk`.
Plus hundreds of continuous color/texture phenotypes (RGB/LAB/HSV histograms, subregions) → regression.

## Compute — casper (192.168.1.184)

- Ubuntu 22.04, Python 3.10, 20 cores, 31GB RAM, ~126GB disk free.
- 2× RTX 3090 24GB. NVIDIA driver 580 / CUDA 13.0 (no system CUDA toolkit; use PyTorch cu124 wheels).
- **GPU 0 = ours** (free). **GPU 1 = shared** Qwen2.5-VL-7B-Instruct-AWQ vLLM on `:8000` (API-key protected), ~22GB. Ollama also on `:11434` (localhost).
- Project root on server: `~/tongue/` (`data/`, `envs/tih/` venv, `checkpoints/`, `repos/`, `logs/`).

## Open questions for the team (blocking Stage 2 backend only)

1. Is the `:8000` Qwen2.5-VL vLLM server production/shared? API key? Spare capacity / rate limits?
2. Is GPU 0 always free for our training + inference?
3. Preferred LLM/API target for Stage 2 (Stage 2 needs only a **text** model, ~3B is enough)?

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| MMIR-TCM code never released | High | Med | Own RAG layer, LLM-agnostic — **done as design choice** |
| TonguExpert labels are model-*predicted* (noisy), not all manual | Med | Med | Prefer `L1_Labels_Manual`; treat predicted as weak labels; Focal Loss |
| No real smartphone-photo data (RTDS private) | High | High | Heavy domain augmentation; collect small internal eval set of phone photos |
| Domain gap: clinical vs. app photos | High | High | Robust segmentation + quality gate (reject poor images) |
| Clinical/medical liability | — | High | Wellness framing + disclaimers, no diagnosis claims, human-in-loop |
| Shared GPU contention | Med | Low | Confined to GPU 0; checkpoint often |
