# TCM Tongue-Reading — Research Synthesis & Knowledge-Base Landscape

_Compiled 2026-07-09. Framing: this project is an **educational tool** that maps observed tongue
features to concepts in the traditional-Chinese-medicine (TCM) tongue-reading tradition. It is **not
a diagnostic instrument**; all outputs are "traditionally associated with…", never claims of disease._

This document answers three questions the project needs settled before building further:
1. How does TCM literature categorize tongue features and their traditional interpretations?
2. What high-quality **open resources already exist** (so we don't build a knowledge base from scratch)?
3. What **datasets** should feed a good training set?

---

## 1. How TCM literature categorizes tongue features

The standard references (Maciocia, *Tongue Diagnosis in Chinese Medicine*; Kirschbaum, *Atlas of
Chinese Tongue Diagnosis*; Li Naimin) converge on a consistent taxonomy. It is also the taxonomy
encoded by the research datasets (SSC-Net's 5 characteristics; TCM-Tongue's 20 categories):

**A. Tongue body (质 zhì) — reflects Blood, Qi, Yang, and the Organs**
| Dimension | Values | Traditional association |
|---|---|---|
| Body color | pale · pale-red (normal) · red · dark-red/crimson · purple/dusky | pale→Deficiency (Qi/Blood/Yang); red→Heat; purple→Blood stasis |
| Shape | normal · thin · swollen/chubby · long/short | thin→Blood/Yin deficiency; swollen→Dampness/Qi deficiency |
| Texture | tooth-marks (scalloped) · cracks/fissures · red dots/prickles · smooth/mirror | tooth-marks→Spleen-Qi deficiency+Damp; cracks→Yin/fluid depletion; prickles→Heat |

**B. Tongue coating (苔 tāi) — reflects Stomach/Spleen and pathogenic factors**
| Dimension | Values | Traditional association |
|---|---|---|
| Coating color | white · yellow · grey · black | white→Cold; yellow→Heat; grey/black→extreme Cold or Heat |
| Thickness | none/thin · thick | thin→normal; thick→accumulation (Dampness/Phlegm/food) |
| Quality | greasy/sticky · dry · peeled/geographic · rootless | greasy→Damp-Phlegm; dry→fluid damage; peeled→Stomach-Yin deficiency |

**C. Sub-regions map to organ systems** (used by TonguExpert subregions & TCM-Tongue's organ classes):
tip → Heart/Lung · center → Spleen/Stomach · sides → Liver/Gallbladder · root → Kidney/Intestines.

**Key point for our "sensitivity" problem:** every dimension above is a **spectrum**, not binary. TCM
practitioners grade signs (faint→pronounced). Our current model collapses each to a 3-way class and
usually predicts the "normal" class — which is exactly why subtle features vanish (see IMPROVEMENT_PLAN.md).

---

## 2. Existing open resources (use these before building from scratch)

The user asked us to look for quality reference sets from reputable institutions **before** authoring one.
We found four that we should adopt rather than reinvent:

| Resource | Institution / venue | What it gives us | Access |
|---|---|---|---|
| **WHO ICD-11 Chapter 26** (Traditional Medicine, Module 1) | **World Health Organization** | **196 standardized TM patterns + 150 disorders** with official codes — an authoritative, language-neutral vocabulary for pattern names (e.g. Spleen qi deficiency pattern, Dampness pattern, Blood stasis pattern). Lets us name patterns with a citable standard. | Public: [ICD-11 MMS Ch.26](https://icd.who.int/) · [PDF](https://gaipa.ufc.br/wp-content/uploads/2022/01/icd11-mms-en-26.pdf) |
| **CCMQ — Constitution in Chinese Medicine Questionnaire** | **Wang Qi, Beijing Univ. of Chinese Medicine** | A **validated 60-item questionnaire** classifying people into **9 body constitutions** (balanced, Qi-deficiency, Yang-deficiency, Yin-deficiency, phlegm-dampness, damp-heat, blood-stasis, qi-stagnation, special/allergic). Used in Chinese national health surveys (100,000s of adults); short forms (~23 items) exist. **This is the scientifically-grounded backbone for our follow-up-question flow.** | Items published in peer-reviewed papers ([PPA 2022](https://www.tandfonline.com/doi/full/10.2147/PPA.S373512), [PMC10617149](https://pmc.ncbi.nlm.nih.gov/articles/PMC10617149/)) |
| **SymMap** | Peer-reviewed, *Nucleic Acids Research* 2019 | Integrative DB mapping **TCM symptoms → modern-medicine symptoms → diseases** via UMLS with expert consensus. **This is our "plain-language gloss" bridge** — it translates a TCM sign into terms a lay user understands. | [symmap.org](http://www.symmap.org) (browse + bulk download) |
| **Maciocia / Kirschbaum classical texts** | Standard TCM education | Per-feature interpretations and pattern synthesis (already the basis of our `tcm_knowledge.json`). | Reference texts (cite, don't redistribute) |

**Emerging scientific correlation (for the plain-language layer, cited honestly as *associations*):**
tongue-coating microbiome studies link **thick/greasy coating ↔ digestive/gut state** (e.g. Frontiers
Endocrinology 2026 on dampness patterns in MASLD; tongue-coating↔gastric microbiome correlation,
[PMC8932003](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8932003/)). Pale tongue + tooth-marks is
repeatedly described alongside **Spleen-Qi deficiency / iron-deficiency-anemia presentations** in the
literature. We present these as *traditional associations with some modern correlational interest*, never causation.

> **Note on MMIR-TCM's "RAG over hundreds of sources":** that corpus was never released (code "coming
> soon"). The four resources above are a stronger, citable, institution-backed substitute we can actually obtain.

---

## 3. Datasets for training-data quality

Ranked by usefulness to *this* project. See RESOURCES.md for the full audit incl. licensing.

| Dataset | Size | Labels | For us | License |
|---|---|---|---|---|
| **TonguExpert** (have it) | 5,992 | 5 categorical **+ ~1,300 continuous phenotypes** (crack depth, coating %, RGB/LAB/HSV, 5 subregions) | **Continuous phenotypes are the key unused asset for the sensitivity fix.** | verify terms |
| **TCM-Tongue** ([btbuIntelliSense](https://github.com/btbuIntelliSense/Intelligent-tongue-diagnosis-detection-dataset), [Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.1c59zw48r)) | 6,719 | **20 pathological categories**, practitioner-verified, object-detection boxes (COCO/TXT/XML) | Adds red-dots, peeled/mirror, purple, thin/chubby, organ subregions — features we don't yet detect. | Dryad states **CC BY 4.0**; GitHub mirror on Baidu Cloud restates none — confirm before commercial use |
| **SM-Tongue** (have it) | 2,155 real | segmentation masks | real-photo seg (domain gap) | CC-BY-NC-4.0 (non-commercial) |
| **UTongue** (UESTC) | ~3,000 | body color, coating, 9-class shape | extra classification data + baseline pipeline | GitHub, check terms |
| **BioHit** ([TongeImageDataset](https://github.com/BioHit/TongeImageDataset)) | 300 | manual seg masks | small seg add-on | GitHub |
| **TongueSAM / TongueSet3** | ~1,000 | seg masks (labelme) | extra seg | open |
| **RTDS** | 2,100 | code only (U-Net++, Swin-hybrid, Focal Loss) — **data private** | architecture/recipe reference | code MIT-ish |

**Training-data conclusion:** we already hold the two most valuable pieces (TonguExpert phenotypes +
SM-Tongue real photos). The highest-leverage *addition* is **TCM-Tongue** (practitioner-verified, wider
feature vocabulary) — pending a license check for commercial use. The right move is **not** "more images"
but **richer labels + using the continuous phenotypes we already have**.

---

## 4. The reference set we should build (schema)

Adopt the resources above into one structured JSON the interpreter and follow-up flow both read:

```jsonc
{
  "features": {                         // grounded in Maciocia/Kirschbaum + TCM-Tongue vocab
    "<feature>": { "<severity_band>": {
        "tcm_term": "...", "plain_gloss": "...",     // e.g. "Dampness" / "sluggish digestion, heaviness"
        "icd11_pattern": "SF7C ...",                 // WHO standardized code where applicable
        "points_to": ["phlegm_dampness", "spleen_qi_deficiency"] } } },
  "patterns": {                         // = the 9 CCMQ constitutions + classic patterns, ICD-11-named
    "<pattern>": {
        "tcm_name": "...", "plain_name": "poor digestion tendency",
        "tongue_signs": {...}, "associated_symptoms": [...],   // SymMap-mapped plain language
        "followup_questions": [ {"q":"Do you often feel bloated after meals?","weight":0.4} ],  // CCMQ items
        "recommendations": { "diet": [...], "lifestyle": [...] } } }   // constitution-specific, from Wang Qi guidance
}
```

This is the concrete artifact the IMPROVEMENT_PLAN builds. It is **grounded** (every field traces to
ICD-11 / CCMQ / SymMap / Maciocia) and **honest** (plain glosses are "traditionally associated", framed
as one framework's perspective).
