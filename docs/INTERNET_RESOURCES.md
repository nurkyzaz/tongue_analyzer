# Internet & Knowledge Resources — usable sources for the community tool

_Created 2026-07-16._ TongueInsight is being built as a **free community tool** — anyone can use it to
help make sense of their own tongue, framed educationally ("傳統上多與… / traditionally associated
with"), never as a diagnosis. We have (or are receiving) permission to use internet resources for the
grounded knowledge base. This file is the running list of **what we may pull from**, with license notes.

**Guiding rules**
- **Ship weightlessly.** Everything here is consumed **offline at build time** and compiled into the
  static knowledge graph / JSON that ships read-only. No source is queried at request time, so nothing
  here adds runtime cost or app weight (see [PLAN.md](PLAN.md) §7 deployment-light lens).
- **Cite everything.** Every edge the KG gains carries a source citation (book + location, or URL +
  retrieval date). The Sources sheet surfaces these; snippets shown only where the license allows.
- **Prefer open / standard vocabularies** so output can be bilingual and merged cleanly.
- **Record license + access date** for each source in the KG `sources` field when we ingest it.

## A · Controlled vocabularies & ontologies (the spine)
| Resource | Use | Access / license |
|---|---|---|
| **WHO International Standard Terminologies on TCM (2022)** | canonical bilingual node names → normalise every feature/pattern (`spleen_qi_deficiency` → WHO code) | WHO publication; ingest terms as ontology spine |
| **WHO ICD-11 Chapter 26 (Traditional Medicine)** | 196 standardized TM **pattern** names + codes | public — [icd.who.int](https://icd.who.int/) |
| **Wikidata / Wikipedia (zh + en)** | bilingual labels, synonyms, cross-links for terms not in WHO IST | CC0 (Wikidata) / CC-BY-SA (Wikipedia) — labels/facts only |

## B · Symptom & questionnaire mapping (for WS-B refinement)
| Resource | Use | Access / license |
|---|---|---|
| **SymMap** ([symmap.org](http://www.symmap.org)) | TCM symptom → plain-language / modern-symptom mapping | public academic DB |
| **CCMQ (Wang Qi 9-constitution questionnaire)** | validated items → the information-gain follow-up questions | items published in peer-reviewed papers |
| **PMC / PubMed open-access articles** | feature→pattern prevalence, empirical weights, symptom edges | OA subset only (CC-BY / public domain) |

## C · Reference texts (per-feature interpretation — the micro layer)
| Resource | Use | Status |
|---|---|---|
| **Gerlach, _TCM Tongue Diagnosis Explained_** | modern feature-organized backbone | licensed, in `tongue_lit/`; ch.2–4 extracted |
| **Maciocia, _Tongue Diagnosis in Chinese Medicine_** | depth/authority; alternative interpretations | licensed — parse next |
| **_Oriental Tongue Diagnosis_ (Kirschbaum)** | classical zoning / meridian perspective | to parse |
| **Chinese textbooks** (朱文锋《中医诊断学》, 李灿东《舌诊》) | rules absent from English sources (greasy-coat subtypes) | **pending usage-rights confirmation** |

## D · Public datasets (feature co-occurrence & empirical weights, not runtime)
| Resource | Use | License |
|---|---|---|
| **TCM-Tongue** (Dryad DOI 10.5061/dryad.1c59zw48r) | 6,719 practitioner-verified imgs → data-derived feature↔pattern co-occurrence (WS-B stylized KG) | **CC-BY 4.0** ✅ |
| **TonguExpert** | primary Stage-1 data | verify dataset terms before commercial use |
| **BioHit tongue image dataset** | small seg add-on | check repo terms |
| **SM-Tongue** (HF `Mark-CHAE/SM-Tongue-Public-Original512`) | real-photo seg | **CC-BY-NC** — research/demo only, not commercial ship |

## E · Community-tool posture
- **Open by default:** grounded on open standards (WHO, ICD-11, Wikidata) + citable literature so any user
  can trace a reading to its source.
- **Feedback loop:** an optional "was this helpful?" 👍/👎 + free-text (WS-D.9) gives a real-world,
  user-contributed signal to recalibrate KB weights over time — the community improves the tool.
- **Privacy:** refinement answers stay on-device and only adjust the current reading (per the design's
  Refine copy); photos are processed, not retained, unless the user opts into history/trend.

## Ingestion checklist (per new source)
1. Confirm license permits our use (educational, possibly commercial later) → record it.
2. Parse offline (`kg/parse_book.py` for texts; a small fetcher for web/ontology sources).
3. Micro-extract triplets **cite-or-abstain** (`kg/micro_extract.py`), normalise to canonical vocab.
4. Merge as `cond.layer="micro"` edges with `sources` + optional `snippet`; keep candidates for review.
5. Never ship the raw copyrighted text — only metadata, offsets, triplets, and short attributed snippets
   where the grant allows.
