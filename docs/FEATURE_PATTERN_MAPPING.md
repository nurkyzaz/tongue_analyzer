# Feature → pattern mapping: what we map to, and the grounded rules (2026-07-13)

Features are now fairly good (see `LABEL_STORE.md`). The next campaign is the **mapping** — turning
detected features into a result — and making that result correct and grounded. This document answers
"what are we even mapping to?" and gives source-backed feature→pattern rules to build the campaign on.

## 1. What we map TO — constitution vs syndrome

TCM has two different targets, and we should be deliberate about which:

- **Body constitution (体质, tǐzhì)** — a *stable, whole-person tendency* (how you run warm/cold, damp,
  etc.). Prof. Wang Qi's **CCMQ 9-constitution model** is the standardized version (derived from 21,948
  people; Chinese national standard ZYYXH/T157-2009): balanced, qi-deficiency, yang-deficiency,
  yin-deficiency, phlegm-dampness, damp-heat, blood-stasis, qi-stagnation, special/inherited. It is
  **explicitly not a diagnosis** — it's a wellness portrait. citeturn2search0turn2search2
- **Syndrome / pattern (证, zhèng)** — a *clinical state at a point in time* used to prescribe treatment.
  Diagnosing it requires the full four examinations (looking, listening, asking, pulse); tongue alone is
  insufficient.

**Recommendation:** the product should **output a CCMQ constitution tendency** ("leaning toward
phlegm-dampness"), never a syndrome/diagnosis — it's the validated, wellness-appropriate, lower-liability
target, and it's what tongue+questionnaire can actually support. Use **syndrome logic as the internal
bridge**: classical tongue knowledge is written in syndrome terms, so features → syndrome tendency →
constitution. Mapping between them:

| our KB pattern | CCMQ constitution |
|----------------|-------------------|
| spleen_qi_deficiency, blood_deficiency | **qi-deficiency** (气虚质) |
| yang_deficiency | yang-deficiency (阳虚质) |
| yin_deficiency | yin-deficiency (阴虚质) |
| phlegm_dampness | phlegm-dampness (痰湿质) |
| damp_heat | damp-heat (湿热质) |
| blood_stasis | blood-stasis (血瘀质) |
| qi_stagnation | qi-stagnation (气郁质) |
| special_diathesis | special/inherited (特禀质) |
| balanced | balanced (平和质) |

(Our KB currently mixes syndrome names — `spleen_qi_deficiency`, `blood_deficiency` — with constitution
names. The table above is the alignment to make explicit.)

## 2. Single-feature → pattern (source-backed)

From standard tongue-diagnosis references (Sacred Lotus; Maciocia, *Tongue Diagnosis in Chinese
Medicine*; Delphi expert consensus PMC8983216). "→ constitution" in brackets.

| feature | value | indicates |
|---------|-------|-----------|
| **body color (zhi)** | pale | Qi and/or Blood deficiency, or Cold [qi-def / blood → qi-def] |
| | red | Heat — excess (thick yellow coat) or deficient (shiny, little coat → yin-def) |
| | dark-red / crimson | deeper Heat (Ying/Xue level) or Blood stasis |
| | purple (bluish) | Cold + Blood stasis; (reddish) Heat + Blood stasis [blood-stasis] |
| | **red tip** | Heat in the **Heart** (upper jiao) |
| **coating color (tai)** | thin white | normal; thick white + moist → Damp-Cold |
| | yellow | interior Heat; + greasy → Damp-Heat [damp-heat] |
| | grey / black | extreme — Cold if wet, Heat if dry |
| **coating thickness** | thick | more internal accumulation / stronger pathogen [phlegm-damp] |
| **coating texture** | greasy / slippery | **Dampness / Phlegm** [phlegm-damp] |
| | peeled / mirror | Stomach-Yin damage (red body) or Qi/Blood deficiency (pale body) [yin-def] |
| **shape** | swollen + pale + moist | Spleen/Kidney **Yang** deficiency |
| | swollen + red + greasy-yellow | Spleen/Stomach **Damp-Heat** |
| | thin | Qi and Blood deficiency [blood-def] |
| | tooth-marked (normal/pale) | **Spleen Qi (or Yang) deficiency** + Dampness [qi-def] |
| **cracks** | horizontal / ice-floe | Yin deficiency [yin-def] |
| | central vertical | Spleen/Stomach Qi deficiency |
| | midline to tip | Heart |
| **red dots / prickles** | by zone | Heat in Blood — tip=Heart/Lung, sides=Liver/GB, root=Kidney [damp-heat] |

## 3. Combination rules — the actual crux

The **same feature means different things in context**. This is the heart of "make the mapping better",
and it's where a purely additive per-feature vote fails:

| combination | result | why |
|-------------|--------|-----|
| pale + **tooth marks** | Spleen **Qi** deficiency | tooth marks = Spleen can't move fluids |
| pale + **swollen + moist** | Spleen/Kidney **Yang** deficiency | swelling + wet + pale = cold/yang |
| pale + **thin** body | **Blood** deficiency | too little substance to fill the tongue |
| red + greasy **yellow** coat | **Damp-Heat** | heat (red/yellow) + damp (greasy) |
| thick greasy **white** coat + swollen | **Phlegm-Dampness** | damp without the heat |
| red + **peeled/little** coat + cracks | **Yin deficiency** (empty heat) | fluids/coat consumed |
| **dark-purple** + sublingual varicosity | **Blood stasis** | strongest stasis picture |
| greasy **yellow** coat + cracks | Damp-Heat consuming fluids | contradictory signs resolved by heat |

Note how **swelling flips meaning by body color/moisture** (pale+moist → yang-def; red+yellow →
damp-heat), and **pale flips by co-feature** (tooth marks → qi-def; thin → blood-def; swollen+moist →
yang-def). Independent additive votes cannot express this; the campaign needs conditional/combination
rules.

## 4. Gap analysis vs our current KB (campaign backlog, ranked)

1. **Additive independence loses context** (biggest). Our KB votes each feature independently, so it
   can't say "swollen means yang-def *with* pale+moist but phlegm-damp *with* red+yellow." → add
   combination rules or body-color-conditioned weights. The mapping test set (below) is the harness.
2. **Sublingual veins are missing** — the single strongest objective blood-stasis sign (Delphi nomogram
   AUC 0.917, the top-weighted item). We don't capture them at all. citeturn0PMC8983216
3. **red_tip is detected (zoning) but doesn't vote.** Wire it → Heart-heat (upper-jiao) contribution.
4. **red_dots is zone-agnostic.** We have organ zoning; route dots by zone (tip/sides/root → different
   organs) instead of one flat "heat" vote.
5. **Moisture (wet/dry) is not measured.** It's what separates yang-def (moist swelling) from yin-def
   (dry, cracked). A wet/dry signal would resolve several ambiguous cases.
6. **`pale` was over-attributed to blood-deficiency.** Fixed the clearest instance (tooth marks now →
   Spleen-Qi, not blood-def); the broader pale weighting is still additive (see #1).

## 5. Test harness — how we measure the campaign

- **`evaluation/mapping_testset.json` + `eval_mapping.py`** (NEW): 10 canonical feature-combinations from
  the references, each with the pattern(s) the sources accept. Runs them through the *product's own* KB
  voting. **Baseline 80% → 100%** after the one grounded fix in #6 (tooth_mk → Spleen-Qi/Yang, not
  blood-def). This is the campaign scoreboard — add cases as we encode combination rules.
- **`evaluation/benchmark_syndrome.py`** (existing): tests the mapping on *real clinical cases*
  (TCMEval-SDT), tongue-clause → syndrome axis, currently 69.7%.

## Sources
- Sacred Lotus, *Tongue Diagnosis in Chinese Medicine* — https://www.sacredlotus.com/go/diagnosis-chinese-medicine/get/tongue-diagnosis-chinese-medicine
- Maciocia G., *Tongue Diagnosis in Chinese Medicine* — https://giovanni-maciocia.com/tongue-gallery/
- Delphi consensus, objective facial/tongue features for TCM constitution (nomogram, blood-stasis) — https://pmc.ncbi.nlm.nih.gov/articles/PMC8983216/
- Wang Qi CCMQ 9-constitution model — https://www.attiliodalberto.com/chinese-food-therapy/constitutions/ ; https://www.thomsonmedical.com/blog/body-constitution
- TCM-Tongue standardized dataset (feature categories) — https://arxiv.org/pdf/2507.18288
