# RAG corpus & source registry

The vector-RAG corpus (`knowledge_base/corpus.jsonl`) is the grounding the **LLM narrative** retrieves
over (Option B). It's built from our own grounded content — never copyrighted text — by
`stage2_interpretation/build_corpus.py`:

- **KB chunks** (features, patterns, combination rules, zoning) from `tcm_knowledge.json`.
- **Authored cards** (`knowledge_base/knowledge_cards.json`) — our summaries of the reasoning a
  practitioner uses to tell similar pictures apart (the high-value RAG content).

Current: **119 chunks** (74 cards + 45 KB). Each chunk carries provenance:
`{id, text, source, source_keys, license, usage, lang, type, tags}`.
_2026-07-21: +8 authored cards from the newly-cleared CN textbooks + ISO 23961-1 (below). Re-embed on
casper (`rag.py --build`) still pending before they're retrievable._

```bash
python stage2_interpretation/build_corpus.py            # -> corpus.jsonl
python stage2_interpretation/build_corpus.py --stats    # counts by type / license / usage / lang
python stage2_interpretation/build_corpus.py --validate # lint provenance (fails on unregistered/pending)
python stage2_interpretation/rag.py --build             # re-embed the index (on casper; Ollama)
python evaluation/eval_rag.py                           # retrieval hit@4
```

## Source registry — the permission spine

`knowledge_base/sources.json` registers every source with its **license** and **usage**. `build_corpus.py`
resolves each chunk's `source` string against the registry aliases and stamps `license` + `usage`
(most-restrictive-wins), so we always know what may be surfaced vs. what is internal-only. **This ties
into `TIH_SHOW_CITATIONS`:** internally the graph/corpus stay fully cited; the public surface is stripped.

**Workflow to add a source:** (1) register it in `sources.json` with its license + `usage`; (2) add
authored cards to `knowledge_cards.json` citing it; (3) `--validate` — it hard-fails on any card whose
source is unregistered or still `permission-pending`.

`usage` values: `owned` · `open-access` (CC-BY, free w/ attribution) · `open-access-nc` (CC-BY-NC) ·
`standard` (ISO/GB/WHO) · `authored-summary` (our wording) · `public-web-reference` · `permission-pending`.

## Recommended sources to grow the corpus

Give the go-ahead per row; I expand once a source's `usage` flips off `permission-pending`.
**CC-BY / open-access rows need NO permission** — attribution only — so they're the fastest wins.

### A — Usable now (open-access, permission-free — attribution only)
| Source | What it adds | License |
|---|---|---|
| *Comparative Analysis of Tongue Indices in Self-Reported Yin Deficiency* (PMC5449755) | **Quantitative** yin-def → red body / scant coat data — grounds our weakest axis | CC-BY *(verify)* |
| *Can TCM Diagnosis Be Parameterized & Standardized? A Narrative Review* (PMC7914658) | Standardization framing; honest-limits language | CC-BY |
| *Intra/Inter-rater reliability of tongue-coating dx via smartphones — Quasi-Delphi* (JMIR, PMC7380897) | Real-world **reliability** numbers → calibrates our confidence/hedge copy | CC-BY |
| *Tongue-coating microbiome* (Sci Reports, PMC8932003) | "Modern view" cards (yellow coat ↔ microbiome) | CC-BY |
| *TCM-Tongue dataset paper* (arXiv 2507.18288) | Pathological-category definitions = a citable feature vocabulary | CC-BY-4.0 |

*(We only ever author our OWN summaries of these — never paste their text — so even the "verify" row is low-risk.)*

### B — Standards (terminology usable)
| Source | What it adds | Status |
|---|---|---|
| **ISO 23961-1:2021 — TCM Vocabulary for Diagnostics, Part 1: Tongue** | THE tongue-specific term standard — a better **ontology spine** than the general WHO IST | ✅ **cleared 2026-07-21** (`usage: standard`) — 1 vocabulary-spine card added |
| WHO IST 2022 (already the spine) | canonical bilingual pattern/feature names | owned |

### C — Copyrighted books (✅ permission granted 2026-07-21 — `usage: owned`)
We author our OWN English summaries citing them; we never paste their text.
| Source | What it adds | Lang | Status |
|---|---|---|---|
| **朱文锋《中医诊断学》** (national planning textbook, 舌诊 chapter) | Dense CN feature→pattern rules absent from EN sources | zh | ✅ cleared — cards added (greasy-vs-curdy, coating-colour depth, red-thorns-by-zone, tooth-marks) |
| **李灿东《中医诊断学》/《舌诊》** | Modern CN clinical tongue rules | zh | ✅ cleared — cards added (slippery-vs-dry, grey-black bidirectional, swollen-vs-thin) |
| **《中医舌诊研究与临床应用》** (Shanghai Sci-Tech) | Clinical tongue atlas / case correlations | zh | ✅ cleared — cited (red-thorns card) |
| Kirschbaum — *Atlas of Chinese Tongue Diagnosis* (already owned) | Visual atlas correlations | en | owned |

**Done 2026-07-21:** A (open-access) + ISO 23961-1 (spine) + the CN textbooks (C) are all cleared and a
first batch of 8 grounded English-summary cards is in. **Next:** (1) re-embed on casper so they're
retrievable (`rag.py --build`); (2) deeper CN extraction — the greasy-coat sub-types and more zoned
red-dot/crack rules the English sources lack are the biggest remaining content gain; (3) optionally swap
the embedder to `bge-m3` if we later add native-Chinese cards (nomic-embed is weak on CN — see below).

## Bilingual (Chinese) note
The schema already supports `lang` per card, so Chinese cards can be added once CN sources are cleared.
Caveat: the current embedder (`nomic-embed-text`) is weak on Chinese — for real CN retrieval we'd swap
to a multilingual embedder (e.g. `bge-m3`) at index time. Flagged, not yet done.
