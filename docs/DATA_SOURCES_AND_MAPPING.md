# Data, Sources & Feature→Constitution Mapping — reference

_Updated 2026-07-21._ The single reference for **what data/literature we use, for what, under which
licence**, the **exact features we detect**, and **how they map to the 9 CCMQ body-constitutions** (via two
tongue *patterns*). Machine-readable source registry: `stage2_interpretation/knowledge_base/sources.json`.

---

## 1. Image datasets (for the vision models)

| Dataset | Link | What we use it for | Size | Licence / copyright |
|---|---|---|---|---|
| **TonguExpert** | [biosino.org](https://www.biosino.org/) | **Primary** Stage-1 training (segmentation + 5 core features) **and** the expert-gold benchmark | 5,992 subjects, 1,353 phenotypes (~302 MB) | ⚠️ **terms unverified for commercial** — verify before paid ship |
| **TCM-Tongue** (btbu) | [Dryad DOI](https://datadryad.org/dataset/doi:10.5061/dryad.1c59zw48r) · [GitHub](https://github.com/btbuIntelliSense/Intelligent-tongue-diagnosis-detection-dataset) | Trains + **validates the "extra" pathological features**; its **553-img test split** is our practitioner benchmark | 6,719 imgs, 20 practitioner categories | ✅ **CC-BY 4.0** (Dryad) |
| **SM-Tongue** | [HuggingFace](https://huggingface.co/datasets/Mark-CHAE/SM-Tongue-Public-Original512) | Real-**phone** segmentation (fixes the clinic→phone domain gap) | 2,155 real 512² pairs | ⚠️ **CC-BY-NC 4.0 (non-commercial)** — **ship-blocker**: retrain or license before charging |
| **BioHit** | [GitHub](https://github.com/BioHit/TongeImageDataset) | Small segmentation add-on (masks) | 300 imgs + masks | ⚠️ repo states no explicit licence — check terms |
| **Our human labels** (`human40` + `human40b`) | internal | **Honest real-world** feature check + red_tip/red_sides threshold calibration | 76 imgs, hand-labeled | internal / owned |

**Architecture-only (no data used):** RTDS, SSC-Net, TOM, Memory-SAM — we reused recipes/ideas, not their
(private) data. MMIR-TCM/MedTCM never released → we built our own Stage-2 instead.

---

## 2. Books, standards & literature (for the TCM interpretation)

Grounding rule: for copyrighted works we **author our own summaries and cite them** — we never paste their
text. `usage: owned` = we hold usage rights; `open-access` = CC-BY (attribution only).

| Source | What we use it for | Type | Licence / copyright | Usage |
|---|---|---|---|---|
| **Maciocia — *Tongue Diagnosis in Chinese Medicine*** | Depth/authority for feature→pattern interpretation | textbook | copyrighted | owned (licensed) |
| **Gerlach — *TCM Tongue Diagnosis Explained*** (World Scientific 2025) | Feature-organized backbone of the knowledge graph | textbook | copyrighted | owned (licensed) |
| **Oriental Tongue Diagnosis** (ed. Dubounet) | Zoning / meridian (organ-region) perspective | textbook | copyrighted | owned (licensed) |
| **Kirschbaum — *Atlas of Chinese Tongue Diagnosis*** | Visual atlas correlations | textbook | copyrighted | owned |
| **朱文锋《中医诊断学》** (Zhu Wenfeng) | Dense CN feature→pattern rules (greasy/curdy coat, coating-colour heat depth, red-thorn zones) | textbook | copyrighted | owned (licensed 2026-07-21) |
| **李灿东《中医诊断学》/《舌诊》** (Li Candong) | Modern CN clinical rules (moisture axis, grey-black, swollen-vs-thin) | textbook | copyrighted | owned (licensed 2026-07-21) |
| **《中医舌诊研究与临床应用》** (Shanghai Sci-Tech) | Clinical tongue atlas / case correlations | textbook | copyrighted | owned (licensed 2026-07-21) |
| **WHO IST on TCM (2022)** | Canonical **bilingual** feature/pattern names (ontology spine) | standard | CC-BY-NC-SA-3.0-IGO | standard |
| **ISO 23961-1:2021** ([iso.org](https://www.iso.org/standard/77468.html)) | Tongue-specific bilingual vocabulary (finer spine than WHO) | standard | ISO (proprietary) | licensed 2026-07-21 |
| **CCMQ** (Wang Qi, 9-constitution questionnaire) | The **9 body-constitutions** + the info-gain follow-up questions | instrument | published instrument | owned |
| **SymMap** ([symmap.org](http://www.symmap.org)) | TCM symptom → plain-language/modern-symptom mapping | database | academic | owned |
| **Sacred Lotus** ([sacredlotus.com](https://www.sacredlotus.com/)) | Educational web cross-reference | web | web reference | public-web-reference |
| **Thomson Medical** | Constitution patient-education framing | web | web reference | public-web-reference |
| Reliability study (JMIR, [PMC7380897](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7380897/)) | Calibrates our **confidence/hedge** copy | paper | **CC-BY** | open-access |
| Tongue-coating microbiome ([PMC8932003](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8932003/)) | "Modern view" cards (yellow coat ↔ microbiome) | paper | **CC-BY** | open-access |
| Yin-deficiency indices ([PMC5449755](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5449755/)) | Grounds our weakest axis (yin-def → red/scant coat) | paper | **CC-BY** | open-access |
| TCM standardization review ([PMC7914658](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7914658/)) | Honest-limits language | paper | **CC-BY** | open-access |
| TCM-Tongue dataset paper ([arXiv 2507.18288](https://arxiv.org/abs/2507.18288)) | Citable pathological-category vocabulary | paper | **CC-BY-4.0** | open-access |
| Delphi expert consensus ([PMC8983216](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8983216/)) | Cross-check for feature→pattern direction | paper | open-access | reference |
| Authored summaries (`authored`, `clinical_lit`, `food_therapy`) | Our own wording of the above (the RAG cards) | — | own | authored-summary |

---

## 3. Features we actually detect

**5 core graded features** (Stage-1 multi-task model; strongest signals):

| Feature | Values | Benchmark acc (vs expert gold) |
|---|---|---|
| `coating` (split into thickness × texture) | non_greasy / greasy / greasy_thick | ~0.87 texture; coating-colour below |
| `tai` — coating colour | white / light_yellow / yellow | **0.92** |
| `zhi` — body colour | light / regular / dark | **0.81** |
| `fissure` — cracks | none / light / severe | **0.92** |
| `tooth_mk` — tooth-marks | none / light / severe | 0.85 |

**Extra multi-label features** (weaker; each **votes in proportion to its measured AP** on the practitioner
test split — `EXTRA_RELIABILITY`):

| Feature | Val AP | Status |
|---|---|---|
| `red_dots` | 0.68 | kept — reliable |
| `red_tongue` | 0.61 | kept |
| `thin` (body) | 0.58 | kept |
| `peeled_coating` | ~0.55* | kept (*too few positives in test to measure) |
| `slippery_coating` | 0.33 | kept, down-weighted |
| `swollen` | 0.19 | kept (load-bearing in gated rules) |
| `purple_body` | 0.17 | kept (main blood-stasis sign) |
| ~~`black_coating`~~ | **0.05** | **REMOVED** — undetectable (0 TP / 11 FP) |

**Training-free geometry** (`zoning.py`, colour-by-region): `red_tip` (Heart/upper-jiao heat, thresh 2.0,
P≈0.92), `red_sides` (Liver/GB zone, thresh 1.5, calibrated 2026-07-21), `moisture=wet` (only "wet"
asserted; "dry" is an honest gap).

---

## 4. How features map to the 9 constitutions (via patterns)

The tongue reads a **pattern / syndrome (证 — your state *today*)**; we then map that to a **CCMQ
body-constitution (体质 — your stable baseline)** for Savor. Crosswalk lives in `interpret.py`
(`CCMQ_CONSTITUTION`); single-feature grounding in `docs/FEATURE_PATTERN_MAPPING.md`.

### The 9 CCMQ body-constitutions and their tongue signature

| # | Constitution (体质) | Our label → | Key tongue features that indicate it |
|---|---|---|---|
| 1 | 平和质 Balanced | `balanced` | pale-red body, thin white coat, no marks |
| 2 | 气虚质 Qi-deficiency | `spleen_qi_deficiency` **or** `blood_deficiency` (see §5) | pale body + tooth-marks (+ swollen); or pale + thin |
| 3 | 阳虚质 Yang-deficiency | `yang_deficiency` | pale + **swollen + moist/wet**, white slippery coat |
| 4 | 阴虚质 Yin-deficiency | `yin_deficiency` | **red** body + **peeled/little** coat + cracks + dry |
| 5 | 痰湿质 Phlegm-dampness | `phlegm_dampness` | **thick greasy white** coat + swollen |
| 6 | 湿热质 Damp-heat | `damp_heat` | red body + **greasy yellow** coat + red dots |
| 7 | 血瘀质 Blood-stasis | `blood_stasis` | **dark / purple** body |
| 8 | 气郁质 Qi-stagnation | `qi_stagnation` | **red sides** (Liver/GB zone) — the tongue's only, weak handle on it |
| 9 | 特禀质 Special-diathesis | `special_diathesis` | *no reliable tongue sign* (allergic/inherited) — set by inquiry, not the tongue |

The crux is **context**: the *same* feature means different things by co-feature (pale + tooth-marks →
qi-def; pale + thin → blood-def; pale + swollen + moist → yang-def). This is why detection feeds
**combination rules**, not just additive votes.

## 5. The 2 "patterns" that aren't constitutions

Two of our labels are TCM **syndromes (证)**, *not* one of Wang Qi's 9 constitutions — but they're the most
**tongue-legible** patterns, so we keep them and fold both into the **气虚质 (Qi-deficiency)** constitution
for Savor:

| Our label | It is a… | Tongue signature | Maps to constitution |
|---|---|---|---|
| `spleen_qi_deficiency` (脾气虚证) | zang-fu **syndrome** | pale body + tooth-marks, often swollen | 气虚质 Qi-deficiency |
| `blood_deficiency` (血虚证) | **syndrome** | pale + **thin** body (± tooth-marks) | 气虚质 Qi-deficiency (血→气 in TCM: qi generates blood) |

So the reading is honest about which is a **constitution (体质)** and which is a **pattern (证)** via each
card's `kind` field, and always resolves to one of the 9 constitutions for the constitution/food/seasonality
features.
