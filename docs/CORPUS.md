# RAG corpus & source registry

The vector-RAG corpus (`knowledge_base/corpus.jsonl`) is the grounding the **LLM narrative** retrieves
over (Option B). It's built from our own grounded content — never copyrighted text — by
`stage2_interpretation/build_corpus.py`:

- **KB chunks** (features, patterns, combination rules, zoning) from `tcm_knowledge.json`.
- **Authored cards** (`knowledge_base/knowledge_cards.json`) — our summaries of the reasoning a
  practitioner uses to tell similar pictures apart (the high-value RAG content).

Current: **108 chunks** (63 cards + 45 KB). Each chunk carries provenance:
`{id, text, source, source_keys, license, usage, lang, type, tags}`.

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

### B — Standards (terminology usable; purchase the text)
| Source | What it adds | Status |
|---|---|---|
| **ISO 23961-1:2021 — TCM Vocabulary for Diagnostics, Part 1: Tongue** | THE tongue-specific term standard — a better **ontology spine** than the general WHO IST | purchase to use codes |
| WHO IST 2022 (already the spine) | canonical bilingual pattern/feature names | owned |

### C — Copyrighted books (need your permission grant)
| Source | What it adds | Lang |
|---|---|---|
| **朱文锋《中医诊断学》** (national planning textbook, 舌诊 chapter) | Dense CN feature→pattern rules absent from EN sources | zh |
| **李灿东《中医诊断学》/《舌诊》** | Modern CN clinical tongue rules | zh |
| **《中医舌诊研究与临床应用》** (Shanghai Sci-Tech) | Clinical tongue atlas / case correlations | zh |
| Kirschbaum — *Atlas of Chinese Tongue Diagnosis* (already owned) | Visual atlas correlations | en |

**Recommendation:** start with **A** (free, immediate — especially the yin-deficiency indices and the
reliability study, which directly harden our weakest area and our confidence copy) and **ISO 23961-1**
as the spine upgrade; pursue the **Chinese textbooks (C)** for the biggest content gain once permission
lands — they carry feature→pattern detail the English sources don't.

## Bilingual (Chinese) note
The schema already supports `lang` per card, so Chinese cards can be added once CN sources are cleared.
Caveat: the current embedder (`nomic-embed-text`) is weak on Chinese — for real CN retrieval we'd swap
to a multilingual embedder (e.g. `bge-m3`) at index time. Flagged, not yet done.
