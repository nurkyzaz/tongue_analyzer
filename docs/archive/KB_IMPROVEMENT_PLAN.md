# Knowledge-Base Improvement Plan (from the expert TCM review)

_2026-07-10. Response to a practitioner review of `tcm_knowledge.json`. The review is excellent and
correct: our KB covers the tongue **body + coating** globally but is missing whole dimensions of
classical tongue diagnosis. This plan maps each gap to our architecture with an honest feasibility
call (KB-only = fast, no model; needs-data/model = requires new training data or capture)._

## What the review found (and our disposition)

| # | Gap | Feasibility | Priority |
|---|---|---|---|
| 1 | **Tongue zoning** (tip→Heart/Lung, centre→Spleen/Stomach, sides→Liver/GB, root→Kidney) | **Feasible now** — TCM-Tongue already has organ-region classes (ids 13–19) we deferred; plus TonguExpert has 5-subregion phenotypes | **High** |
| 2 | **Tongue movement** (stiff, flaccid, tremor, deviation) | **Hard** — tremor/stiffness need *video*; deviation detectable from a still (asymmetry). Needs a capture + model change | Med (roadmap) |
| 3 | **Sublingual vessels** (under-tongue veins) | **New module** — needs an *under-tongue* photo + a model + data (Yuan Hongxia atlas). Separate capture step | Med (roadmap) |
| 4 | **Peeled/mirror coating → primary feature** | **Easy** — already detected by the extra model at **AP 0.70 (our best)**; just promote it | **High — do now** |
| 5 | **severity_bands lack data support** | **Partly done** — we now calibrate population percentiles (`reference_stats.json`); can adopt them + the review's ROC thresholds | **High** |
| 6 | **Terminology alignment (ISO 23961-1 / GB)** | **Docs** — add ISO English + Chinese term names + source per feature | Med |
| 7 | Expand to all **9 CCMQ constitutions** (add qi-stagnation, special-diathesis) | **Easy (KB)** | Med |
| 8 | Coating **moisture/dryness**, **tongue spirit** | KB + partial model | Med |
| 9 | **Modern-medicine correlations** in patterns (educational) | **Easy (KB)** | Low |

## Gap-by-gap plan

### 1. Tongue zoning (visceral correspondence) — HIGH
- **KB (now):** add a `regions` block mapping `tip/centre/sides/root → organ system`, with the classical
  meaning of a sign appearing in each zone (e.g. red tip → Heart/Lung heat; cracks in centre → Stomach).
- **Model (next):** train regional detection from **TCM-Tongue's organ-region classes** (shen/gandan/
  piwei/xinfei depression·protrusion, ids 13–19) — we already downloaded these, just excluded them. Add
  them to the extra-features head as `region_*` outputs. Also usable: TonguExpert `P60_Subregions` (580
  per-subregion color/texture phenotypes) to say *where* a color/coating change concentrates.
- Grounds in: Song Weijiang *Holistic Theory of Tongue Diagnosis* (source #10), Maciocia zoning.

### 2. Tongue movement — MED (needs video)
- A single photo can't see tremor/stiffness. **Deviation** (tongue pulled to one side) is detectable from
  the mask's **left/right asymmetry** — a cheap heuristic we can add now (no new data).
- Full movement (stiff/flaccid/tremor) → roadmap: capture a 2–3 s **video** and analyze motion. KB entries
  for movement values can be authored now (Cao Bingzhang *Guide to Tongue Differentiation*, source #2) so
  they're ready.

### 3. Sublingual vessels — MED (new capture + module)
- Requires the user to lift the tongue and photograph the **underside**. Plan: add an optional second
  capture; train a small model (needs data — Yuan Hongxia *Atlas of Sublingual Vessel Diagnosis*, or the
  ISO 23961-1 sublingual terms) to grade vein color/distension → Blood-stasis signal. Roadmap.

### 4. Promote peeled/mirror coating — HIGH, do now
- It's our most reliable extra feature (AP 0.70) and clinically important (Stomach-Yin deficiency). Action:
  keep detection in the extra model but treat it as a **first-class feature** in the KB/report (already
  wired via `extra_features`; elevate its wording + always surface when present).

### 5. Data-grounded severity thresholds — HIGH
- We now compute population **percentiles** per feature (`reference_stats.json`: p50/p75/p90). Action:
  derive `severity_bands` cut-points from these percentiles instead of fixed 0.15/0.35/0.60/0.80, so
  "moderate/pronounced" means "top-quartile/decile vs. the population."
- Adopt the review's **ROC-validated color thresholds** (source #8: pale-white g ≥ 28.45%, AUC 0.93;
  crimson a ≥ 20.32, AUC 0.94) as a **calibration cross-check** for the color heads, and cite them.

### 6. Terminology alignment (ISO 23961-1 / GB) — MED
- Add to each feature: `iso_term` (official English), `zh` (Chinese), and `standard` source. Cross-check
  our names against ISO 23961-1 (source #4) and note divergences in `_meta` (source #6 comparison study).

### 7–9. Constitutions, moisture/spirit, modern correlations
- Add **qi-stagnation** and **special-diathesis** patterns → full CCMQ 9 (source: CCMQ).
- Add a coating **moisture** dimension (moist/dry) — partly covered by `slippery_coating`; add `dry`.
- Add an educational **`modern_correlation`** field to patterns (e.g. pale ↔ low haemoglobin) — clearly
  labelled non-diagnostic (SymMap + source #8).

## The 10 sources → how each is used
| Source | Use in the KB |
|---|---|
| Zhu Wenfeng, *Diagnostics of TCM* (#1) | baseline term definitions (align education-system terms) |
| Cao Bingzhang, *Guide to Tongue Differentiation* (#2) | tongue-movement classification + meanings |
| Liang Yuyu, *Correction of Tongue Diagnosis* (#3) | fine coating-color × quality combinations |
| ISO 23961-1:2021 (#4) + GB transposition (#5) | official term names (EN/ZH), classification hierarchy |
| Terminology comparison study (#6) | declare our adopted standard + note divergences in `_meta` |
| Wang Yanhui, *Atlas* (#7) | mild/moderate/severe semi-quantitative anchors + reference images |
| Expert-consensus standard library (#8) | **data-grounded severity thresholds** (ROC ranges) |
| Automated tongue-diagnosis review (#9) | capture/pre-processing guidance (color calibration) |
| Song Weijiang, *Holistic Theory* (#10) | tongue-surface **zoning** → organ correspondence |

## Sequenced roadmap
- **Now (KB-only, no training):** #4 promote peeled coating · #1 add `regions` map + deviation-from-mask
  heuristic · #5 percentile-based bands · #6 ISO/ZH term fields · #7 two extra constitutions · #9 modern
  correlations. → richer, standards-aligned, better-calibrated with **zero new data**.
- **Next (needs the data we already have):** train **regional detection** from TCM-Tongue organ classes +
  TonguExpert subregions; add coating moisture/dry.
- **Roadmap (needs new capture/data):** sublingual-vessel module (under-tongue photo) · tongue-movement
  via short video · adopt GB standard + reference images when published.

## Cross-cutting: color calibration (ties #5, #9, and the demo's Damp-heat skew)
Real-photo color casts bias the color features (why uncalibrated photos over-read yellow/red → Damp-heat).
Add **grey-world / learned white-balance** preprocessing (optionally an in-frame grey card), per source #9.
This improves `tai`/`zhi` reliability and reduces spurious Damp-heat — complements the distinctiveness
weighting we just shipped.
