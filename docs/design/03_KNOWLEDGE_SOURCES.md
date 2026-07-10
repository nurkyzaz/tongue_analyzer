# Knowledge Sources — the databases behind the "diagnoses"

The tool never invents patterns. Everything it says is grounded in the file
[`tcm_knowledge.json`](../../stage2_interpretation/knowledge_base/tcm_knowledge.json), which is in turn
built from four reputable, citable sources. This doc explains each source, what we take from it, and
how to extend it.

## The four grounding sources

| Source | Institution / venue | What we take | Where it lands in the KB |
|---|---|---|---|
| **WHO ICD-11, Chapter 26** (Traditional Medicine, Module 1) | **World Health Organization** | Standardized **pattern names/codes** — 196 TM patterns (e.g. *Spleen qi deficiency pattern*, *Dampness pattern*, *Blood stasis pattern*). Gives every pattern a citable, language-neutral name. | `patterns.<id>.tcm_name` / `.icd11` |
| **CCMQ — Constitution in Chinese Medicine Questionnaire** | **Wang Qi, Beijing University of Chinese Medicine** | The **9-constitution model** (balanced, qi-deficiency, yang-deficiency, yin-deficiency, phlegm-dampness, damp-heat, blood-stasis, qi-stagnation, special) and its **validated questionnaire items** — used in Chinese national health surveys (100,000s of adults). | `patterns` set + `patterns.*.followup_questions` |
| **SymMap** | *Nucleic Acids Research* 2019 (peer-reviewed) | Mapping **TCM symptom → modern-medicine symptom** (via UMLS, expert-verified). This is our **plain-language gloss** bridge (e.g. dampness → "sluggish digestion, bloating"). | `features.*.present_plain`, `patterns.*.associated_symptoms` |
| **Maciocia / Kirschbaum** | Standard TCM texts (*Tongue Diagnosis in Chinese Medicine*; *Atlas of Chinese Tongue Diagnosis*) | Per-feature **interpretations** (what a pale body, a greasy coating, cracks, tooth-marks traditionally mean) and pattern synthesis. | `features.*` meanings, `patterns.*.explanation` |

> **Note on MMIR-TCM's "RAG over hundreds of sources":** that corpus was never released. The four
> sources above are a stronger, obtainable, institution-backed substitute.

## Supporting scientific context (used cautiously, as *associations* only)
- **Tongue-coating microbiome ↔ digestion:** peer-reviewed work links thick/greasy coating to gut/
  gastric microbiome state (Frontiers Endocrinology 2026 on dampness patterns; tongue↔gastric
  microbiome correlation). Lets us say "traditionally associated with sluggish digestion" honestly.
- **Pale tongue + tooth-marks ↔ Spleen-qi deficiency / iron-deficiency presentations:** repeatedly
  described in the literature; we surface it as a *tradition-level* association plus a gentle "consider
  checking iron/ferritin with a doctor" — never a diagnosis.

## The KB schema (what you edit)

```jsonc
{
  "severity_bands": [ {"max":0.15,"word":"none","mention":false}, ... ],   // degree wording

  "features": {                        // one entry per detectable feature
    "<char>": {
      "kind": "graded_value" | "categorical",
      "tcm_term": "...", "present_plain": "...", "absent_plain": "...",     // dual language
      "points_to": { "<pattern_id>": <weight> }                            // how it votes
    }
  },

  "patterns": {                        // the 'diagnosis' catalogue (CCMQ constitutions, ICD-11-named)
    "<pattern_id>": {
      "tcm_name": "...", "icd11": "...", "plain_name": "...",
      "explanation": "...",
      "associated_symptoms": ["plain-language symptoms (SymMap-grounded)"],
      "followup_questions": [ {"q":"...(CCMQ item)...","weight":0.5} ],
      "recommendations": { "diet":[...], "lifestyle":[...] }               // specific, per-sign
    }
  },

  "sources": ["...citations shown in every report..."]
}
```

## How the pieces connect (worked example)
1. Model detects **tooth-marks, severity 0.76** and a **pale body**.
2. KB `features.tooth_mk.points_to` = `{spleen_qi_deficiency:1.0, phlegm_dampness:0.6, blood_deficiency:0.4}`;
   `features.zhi.values.light.points_to` = `{blood_deficiency:1.0, spleen_qi_deficiency:0.6, ...}`.
3. Votes (weighted by severity/confidence) sum → **Spleen qi deficiency** wins.
4. Report pulls that pattern's `plain_name` ("low digestive energy"), `associated_symptoms`
   ("bloating after meals, loose stools, low-iron tendency"), `recommendations`, and offers its
   `followup_questions` ("Do you feel bloated after eating?").

Every arrow above is a line in the JSON — fully auditable and editable.
