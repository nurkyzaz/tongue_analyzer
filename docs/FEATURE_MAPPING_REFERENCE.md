# Complete feature → pattern mapping reference (with citations)

Every feature the system reads, each value, the pattern(s) it votes for **with the actual weight the KB
uses** (`stage2_interpretation/knowledge_base/tcm_knowledge.json`), and the source that grounds it.

### How to read this
- **Direction vs weight.** The *direction* of every mapping (feature → which pattern) comes from the
  cited TCM sources. The *numeric weight* is our own calibration (magnitude of the vote), tuned with
  population statistics (`reference_stats.json`) so common findings count less than distinctive ones —
  it is **not** taken verbatim from a source. Where a weight encodes a product choice rather than a
  source claim, it's flagged **[calibration]**.
- **How votes combine.** Voting is **additive and distinctiveness-weighted**: each present feature adds
  `weight × how-distinctive-this-finding-is` to each pattern; the top pattern wins. (Limitation: this is
  independent per feature — it can't yet express context rules like "swelling means yang-def *with* a
  pale/moist tongue but damp-heat *with* red/yellow"; see `FEATURE_PATTERN_MAPPING.md` §3–4.)
- **Patterns** (ids used below) map to CCMQ constitutions — see `FEATURE_PATTERN_MAPPING.md` §1.
- **Sources** are keyed [SL]=Sacred Lotus, [Mac]=Maciocia, [Delphi]=PMC8983216, [WangQi]=CCMQ,
  [TCM-T]=TCM-Tongue dataset. Full URLs at the bottom.

---

## A. Core detected features (5)

### 1–2. Coating — thickness × texture (from the `coating` model output)
The `coating` prediction drives one vote and is shown as two axes (see `COATING_SPLIT.md`).

| shown as | value | leads to | basis |
|----------|-------|----------|-------|
| **Coating texture** | greasy / slippery | `phlegm_dampness` **1.0** | greasy/slippery coat = Dampness/Phlegm [SL][Mac]; thick/greasy coat = phlegm-damp & damp-heat constitutions [Delphi] |
| | | `damp_heat` **0.3** | greasy **+ heat colour** = damp-heat [SL] (the heat comes from `tai`, hence the smaller weight) |
| | smooth | (normal — no vote) | thin/smooth even coat is normal [SL] |
| **Coating thickness** | thick | (via the same `coating` vote) | thick coat = more internal accumulation / stronger pathogen [SL]; "coating-texture-thick" is the top damp-heat & phlegm-damp item [Delphi] |
| | thin | (normal — no vote) | thin coat is normal [SL] |

### 3. Coating colour (`tai`)
| value | leads to | basis |
|-------|----------|-------|
| white | `yang_deficiency` 0.2 **[calibration]** | thin white = normal; thick white + moist + pale = Damp-Cold / Yang [SL]. Small weight because white is the common default |
| light_yellow | `damp_heat` 0.6 | yellowing coat = emerging interior Heat [SL] |
| yellow | `damp_heat` **1.0** | yellow coat = interior Heat / Damp-Heat [SL][Mac]; "coating-colour-yellow" = top damp-heat item, importance 4.70/5 [Delphi] |

### 4. Body colour (`zhi`)
| value | leads to | basis |
|-------|----------|-------|
| light (pale) | `blood_deficiency` **1.3**, `spleen_qi_deficiency` 0.4, `yang_deficiency` 0.4 | pale = Qi and/or Blood deficiency, or Cold [SL][Mac]. **The split across three is [calibration]** — the sources say "pale = broadly deficiency/cold"; which one needs the co-features (see combination rules). Leading blood-def is a product choice under review |
| regular (pale-red) | `balanced` 1.0 | pale-red = the normal/balanced tongue [SL][WangQi] |
| dark (red / dark) | `blood_stasis` 0.8, `damp_heat` 0.5, `yin_deficiency` 0.5 | red = Heat (excess or deficient); dark-red/purple = Blood stasis [SL][Mac]. Split is **[calibration]** — needs co-features (coat, purple) to disambiguate |

### 5. Fissures / cracks (`fissure`)
| value | leads to | basis |
|-------|----------|-------|
| light / severe (present) | `yin_deficiency` **1.0** | cracks (horizontal / ice-floe) = Yin deficiency [SL][Mac]. **Caveat:** a *central vertical* crack = Spleen/Stomach Qi deficiency, and a midline-to-tip crack = Heart — we don't yet read crack *location*, so we vote the most common (Yin) [SL] |

### 6. Tooth marks (`tooth_mk`)
| value | leads to | basis |
|-------|----------|-------|
| light / severe (present) | `spleen_qi_deficiency` **1.3**, `phlegm_dampness` 0.5, `yang_deficiency` 0.3 | tooth marks (scalloped edges) = Spleen **Qi** deficiency failing to move fluids → Dampness; pale/moist variant = Spleen **Yang** deficiency [SL][Mac]. (Corrected 2026-07-13: previously mis-voted blood-deficiency, which no source supports) |

---

## B. Extra detected features (8) — from the TCM-Tongue model
Noisier; each vote is additionally down-weighted by the feature's measured reliability
(`EXTRA_RELIABILITY` in `interpret.py`).

| feature | leads to | basis |
|---------|----------|-------|
| peeled / mirror coating | `yin_deficiency` **1.0** | peeled/mirror coat = Stomach/Kidney **Yin** damage (red body) or Qi/Blood deficiency (pale body) [SL][Mac]; [TCM-T] `botaishe`/peeled category |
| red tongue body | `damp_heat` 0.6, `yin_deficiency` 0.4 | red body = Heat — excess (→damp-heat with coat) or deficient (→yin-def, shiny/little coat) [SL][Mac] |
| purple / dusky body | `blood_stasis` **1.0** | purple/dusky = Blood stasis [SL][Mac]; "tongue-colour-dark-purple" = top blood-stasis item, importance 4.60/5 [Delphi] |
| swollen / enlarged | `phlegm_dampness` 0.8, `spleen_qi_deficiency` 0.6 | swollen = Phlegm/Damp/Water retention; pale+moist variant = Spleen/Kidney **Yang** def [SL][Mac]. (Yang-def branch needs moisture/colour context we don't yet vote on) |
| thin body | `blood_deficiency` 0.8, `yin_deficiency` 0.5 | thin body = Qi & Blood deficiency (can't fill the tongue); dark-red+dry variant = Yin-deficient fire [SL] |
| red dots / prickles | `damp_heat` 0.7 | red dots/spots = Heat toxins in Blood [SL][Mac]. **Zone matters** (tip=Heart/Lung, sides=Liver/GB, root=Kidney) — we vote flat "heat" for now [SL] |
| grey-black coating | `damp_heat` 0.4 **[calibration]** | black/grey coat = extreme — Heat if dry, Cold if wet [SL]. Low weight & heat-leaning because we don't read wet/dry to tell which |
| wet / slippery coating | `phlegm_dampness` 0.7 | slippery/wet coat = Dampness [SL][Mac] |

---

## C. Zoned features

### Red tip (`red_tip`, from `zoning.py`) — detected, not yet voting
| value | should lead to | basis | status |
|-------|----------------|-------|--------|
| present (strong) | Heart heat (upper jiao) → `yin_deficiency` / `damp_heat` | red tip = Heat in the Heart [SL][Mac] | **detected & shown, but not wired into voting yet** (campaign backlog #3) |

### Tongue zoning (educational; global detection only for now)
Where a sign appears maps to an organ system (classical zoning) [SL][Mac][Song Weijiang]:

| zone | organ system | example |
|------|--------------|---------|
| tip | Heart / Lung | red tip / dots at tip → Heart-Lung heat (stress, poor sleep) |
| centre | Spleen / Stomach | central crack or thick central coat → digestion |
| sides | Liver / Gallbladder | red/purple sides → Liver (tension, stress) |
| root | Kidney / Intestines | thick root coat → lower-body damp; pale root → Kidney reserve |

---

## D. Not captured (highest-value missing mappings)
- **Sublingual (under-tongue) veins → Blood stasis.** The single strongest *objective* blood-stasis sign
  — top item in the Delphi nomogram (AUC 0.917) [Delphi]. We don't image the underside at all.
- **Moisture (wet / dry).** Distinguishes Yang-def (moist swelling) from Yin-def (dry, cracked) and
  resolves several ambiguous coats [SL]. Not measured.
- **Feature *location*** (crack position, dot zone) — would sharpen Heart/Spleen/Liver/Kidney routing.

---

## Sources
- **[SL]** Sacred Lotus, *Tongue Diagnosis in Chinese Medicine* — https://www.sacredlotus.com/go/diagnosis-chinese-medicine/get/tongue-diagnosis-chinese-medicine (single-feature → pattern grounding for most rows)
- **[Mac]** Maciocia G., *Tongue Diagnosis in Chinese Medicine* & *Foundations of Chinese Medicine* — https://giovanni-maciocia.com/tongue-gallery/
- **[Delphi]** Objective facial/tongue features for TCM constitution — Delphi consensus + nomogram, PMC8983216 — https://pmc.ncbi.nlm.nih.gov/articles/PMC8983216/ (constitution-level features; importance ratings; sublingual-vein AUC)
- **[WangQi]** Wang Qi, CCMQ 9-constitution model (standard ZYYXH/T157-2009) — https://www.thomsonmedical.com/blog/body-constitution
- **[TCM-T]** TCM-Tongue standardized dataset with pathological annotations — https://arxiv.org/pdf/2507.18288 (the 20 categories our extra-features model is trained on)
- **[Song Weijiang]** *Holistic Theory of Tongue Diagnosis* — tongue zoning
