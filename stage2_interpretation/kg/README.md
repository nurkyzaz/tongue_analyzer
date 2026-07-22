# Macro-micro TCM knowledge graph (WS-A)

The grounded substrate for the Stage 2 overhaul (see [`docs/PROJECT_HANDBOOK.md`](../../docs/PROJECT_HANDBOOK.md) §2–3).
Replaces the split between a rule KB (`tcm_knowledge.json`) and a flat card list
(`knowledge_cards.json`) with one typed graph that retrieval + the grounded matcher (WS-C) and the
interactive refinement pass (WS-B) both query.

## Layers (all merge into one graph)

| Layer | Source | rel / nodes added | Status |
|---|---|---|---|
| **seed** | `tcm_knowledge.json` | every fact, re-typed | ✅ done (`build_kg.py`) |
| **who-spine** | WHO IST 2022 (`who_terms.py` → `who_spine.json`) | `props.who` code + 中文 + pinyin on 25 nodes | ✅ done |
| **macro** | 3-book hierarchies (`parse_book.py` → `book_sections.json`) | `section_of` + `section:` nodes (406) | ✅ done |
| **micro** | qwen2.5:14b triplets from 3 books, cited + snippet | `points_to`/`argues_against` + `attested_in` + `snippets` | ✅ done (282 cited edges; 98 candidates held for review) |

## Current graph (seed + who-spine + macro + micro)

```
nodes 605   edges 1245   rules 15   snippets 282   who-tagged nodes 25
edges: points_to 283 (27 seed + 256 book-cited), argues_against 26, evidence_for 94,
       has_symptom 55, recommends 39, section_of 406, attested_in 282, probes 21, ...
```

Micro layer folds **three licensed books** (each mined independently into its own
`book_triplets_<id>.json`, then merged by `build_kg.py`, which globs all per-book files):
- **gerlach** (`--mode decimal`, ch.2–7) — 121 edges. Modern, feature-organised backbone.
- **oriental** (`--mode title`, all sections) — 93 edges. Classical zoning perspective (Dubounet ed.).
- **maciocia** (`--mode title`, all sections) — 68 edges. Adds sublingual-vein / purple / Yin-Xu nuance
  (slide-deck PDF → `pdftotext -layout`; noisier sections, but cite-or-abstain keeps only real triplets).

Books with no decimal numbering are parsed by `--mode title` (flush-left capitalised heading lines);
each heading becomes its own "chapter" so `micro_extract.py --chapters all` mines the whole book.

## WHO-IST ontology spine (`who_terms.py` → `who_spine.json`)

The WHO *International Standard Terminologies on TCM* (2022) is a numbered bilingual glossary. We use
it **not as a triplet source** but as an **ontology spine**: `who_terms.py` extracts every entry
header (code, English, 简体中文, pīnyīn — CC BY-NC-SA terminology, definitions dropped) into a
git-ignored index `who_terms.json`; the hand-verified `who_spine.json` maps each of our 10 patterns +
tongue-sign features/values to a WHO code + bilingual name. `build_kg.add_who_spine` tags those onto
`node.props.who` (25 nodes) so the graph can emit **bilingual output** and future books merge onto
stable WHO codes instead of ad-hoc English strings. `balanced` / `special_diathesis` are CCMQ 体质
(constitution) terms — absent from WHO IST — so they carry `source:"ccmq"` and no `who_code`. This is
metadata only (no edges/weights) → parity is untouched.

```bash
pdftotext -layout tongue_lit/9789240042322-eng.pdf who.txt
python stage2_interpretation/kg/who_terms.py --src who.txt              # -> who_terms.json (index)
python stage2_interpretation/kg/who_terms.py --src who.txt --find "pale tongue"   # look up a term
# who_spine.json is the tracked curated map; build_kg folds it automatically
```

## WS-C graph-RAG retrieval (`retrieval.py`)

`GraphRAG.retrieve(present_features)` returns a **connected 2-hop subgraph** around the detected
feature nodes (via `KnowledgeGraph.neighborhood`), ranks the patterns it reaches by signed evidence
(`points_to` − `argues_against`), and attaches the cited book edges (source + verbatim snippet) plus
the 2-hop symptom/rec/question context. `Retrieval.context_cards()` flattens it into grounded,
citation-tagged lines ready to drop into the WS-C matcher prompt — relationships, not isolated facts.
An interim Stage-1→graph vocab alias bridges the coating-split keys until the WHO-IST spine lands.
Scoring is calibrated per §7-A: **sublinear corroboration** (many book citations of one edge add
log-scaled, not linearly) + a **gentle IDF distinctiveness** multiplier (bounded 0.65–1.55; specific
features count more) — applied only here, the seed rule-engine weights stay parity-locked. Sanity
gate: `evaluation/eval_graph_rag.py` (4/4 unambiguous cases).

## WS-C grounded cite-or-abstain matcher (`matcher.py`)

`GroundedMatcher.match(present)` feeds the graph-RAG `context_cards()` to an LLM (JSON-mode via
`llm_client.response_format`, temperature 0) that selects the best-supported patterns **using only the
retrieved cited facts**. Cite-or-abstain is enforced in code: a proposed pattern is dropped unless it
was reachable in the retrieved subgraph AND grounded in a detected feature; if none survive, it
abstains. Falls back to the graph-RAG ranking when no LLM is configured (the degraded path). Runs in
**shadow mode** — `matcher.py --shadow` scores it against the rule engine's vote (top-1 agreement +
Jaccard) so promotion is on the numbers, not decided in advance. The rule engine stays production.

**Shadow run on real Stage-1 output** (`evaluation/eval_shadow_matcher.py`, human40, 40 imgs):
hallucination-rate **0.0**, top-1 agreement **0.50** vs the rule engine, mean Jaccard 0.48, abstain 0.0
— disagreements almost all within-family (phlegm ↔ spleen-qi ↔ yin ↔ damp-heat). Verdict: safe, but
not a clear win → **ensemble** (matcher for cited evidence + a second-opinion prior), don't replace the
rule ranker.

## WS-C ensemble (`ensemble.py`) — rule prior + cited evidence

`ensemble_cards(rule_cards, matcher_out)` blends the auditable rule vote (the prior) with the matcher's
cite-or-abstain confidence: `blended = (1-α)·rule + α·matcher`, **α=0.2** (rule-dominant, env
`TIH_WSC_ALPHA`). Two safety properties by construction: **abstention is neutral** (a rule pattern the
matcher didn't name keeps its prior, not a penalty), and **matcher-only hints are capped at α** (a
grounded pattern the rule missed can appear as a cited secondary, never overturn a confident rule lead).
When the rule engine leads with `balanced`, the ensemble stays out of the way. Each surfaced pattern gets
book **citations** (source + snippet), the matcher's `why`, and an honest `confidence_pct`. Wired into
`interpret.py` behind `TIH_WSC_ENSEMBLE` (**default OFF** until promoted); fully degrading (no graph /
LLM down → rule cards).

**Eval** (`evaluation/eval_ensemble.py`, human40) — α sweep decided the default:

| α | top-1 stability vs rule | lead-cited | narrative faithfulness (WS-D) |
|---|---|---|---|
| 0.35 | 0.75 | 0.925 | 0.868 |
| **0.2** (default) | **0.85** | **0.90** | **0.929** |
| 0 (rule-only) | 1.0 | ~0 | 0.936 |

Citations attach whenever the matcher grounds a pattern — **independent of α** — so lowering α costs
almost no grounding (0.925→0.90) but recovers faithfulness to the rule-only baseline (0.868→0.929) and
raises stability. matcher-added 0.0, hallucination 0.0 at both. **WS-D gate PASS** at α=0.2 (0.929 ≥
0.85). At α=0.2 the faithfulness tradeoff is gone, so `TIH_WSC_ENSEMBLE` was **promoted default-ON (live on the
casper demo, qwen2.5:14b)** — verified end-to-end (`/analyze` returns cited, %-scored ensemble patterns).

Micro edges are tagged `cond.layer="micro"`, carry a book citation + a `snippet` id, and COEXIST with
the seed rule weights (rule engine keeps using seed edges; the WS-C matcher can prefer cited micro
edges). 49 `candidate` triplets — real signs our detector can't observe (sublingual veins) or Gerlach
patterns outside our 10 (ira, repletio hepatici) — are held in `_meta.micro_candidates` for the
ontology-spine step, not merged.

## Building the macro layer

```bash
# decimal-numbered book (Gerlach):
python stage2_interpretation/kg/parse_book.py \
    --book tongue_lit/874856627-TCM-Tongue-Diagnosis-Explained.txt --id gerlach \
    --title "Gerlach O., TCM Tongue Diagnosis Explained (World Scientific, 2025)"
# title-heading books (no decimal numbering) — Oriental (.txt), Maciocia (pdftotext -layout the PDF):
python stage2_interpretation/kg/parse_book.py --mode title \
    --book tongue_lit/371253413-Oriental-Tongue-Diagnosis.txt --id oriental \
    --title "Oriental Tongue Diagnosis (ed. Dubounet)"
python stage2_interpretation/kg/parse_book.py --mode title \
    --book tongue_lit/tongue-diagnosis-maciocia-online.txt --id maciocia \
    --title "Maciocia G., Tongue Diagnosis in Chinese Medicine (online)"
# -> book_sections.json (tracked: section metadata + char offsets, NOT the copyrighted text)
python stage2_interpretation/kg/build_kg.py --verify   # folds macro + all micro books in automatically
```

## Micro layer (offline on casper, one file per book)

```bash
# inspect the extraction prompt without an LLM:
python stage2_interpretation/kg/micro_extract.py --dry-run --book-id oriental --chapters all
# run with the LLM env set (casper). Each book -> book_triplets_<id>.json (never clobbers the others):
python stage2_interpretation/kg/micro_extract.py --book-id gerlach  --chapters 2,3,4
python stage2_interpretation/kg/micro_extract.py --book-id oriental --chapters all
python stage2_interpretation/kg/micro_extract.py --book-id maciocia --chapters all
```

`micro_extract.py` gives the LLM our canonical feature/pattern vocabulary (so triplets map into our
ontology), enforces **cite-or-abstain** (a triplet is dropped unless its snippet is verbatim in the
section), runs at temperature 0, and writes a reviewable `book_triplets.json`. The graph merge
(`build_kg.add_micro_layer`) is written after a first run, validated against real extractions.

The seed layer is a strict **superset** of the KB — `build_kg.py --verify` asserts every pattern,
feature, `points_to` weight, symptom, recommendation, follow-up question and combination rule is
reachable in the graph. So enabling the graph changes **no** current behaviour on day one; macro/micro
only *add*.

## Build

```bash
python stage2_interpretation/kg/build_kg.py --verify   # -> knowledge_base/kg_graph.json (git-ignored)
```

The compiled `kg_graph.json` is a rebuildable artifact (git-ignored, like `corpus.jsonl`). Its
**sources** are tracked: `tcm_knowledge.json` today, plus triplet/section files once macro+micro land.

## Model (see `graph.py` docstring for the full node-id + rel conventions)

- **Nodes** are typed (`pattern`, `feature`, `value`, `symptom`, `recommendation`, `question`,
  `region`, `organ`, later `section`), each with `name`, `props`, and a `sources` citation list.
- **Edges** carry `rel`, optional `weight`, `cond`, `sources`, and a `snippet` id (micro layer).
- **`rules`** holds the combination rules verbatim (context-conditioned hyperedges — kept as-is
  because they gate multi-feature boosts the rule engine already applies).
- **`snippets`** is the attributed short-quote store (traceability for cite-or-abstain in WS-C).

## Why the inverse `evidence_for` edges matter (WS-B)

Seeding also emits `symptom -> pattern` and `question -> pattern` **`evidence_for`** edges (the
inverse of `has_symptom` / `probes`). That is the lever for the personalization idea: a user's
follow-up answers re-enter as *symptom evidence* and re-score patterns over the same graph, and
`questions_for_pattern` lets us pick the question that best separates the top candidates by
information gain instead of a fixed list.

## Query API (`KnowledgeGraph`)

```python
from kg.graph import KnowledgeGraph
g = KnowledgeGraph.load()
g.patterns_for_value("zhi", "light")       # forward:  detected feature -> [(pattern, weight, sources)]
g.patterns_for_symptom("symptom:bloating") # inverse:  user evidence  -> [(pattern, weight, sources)]  (WS-B)
g.questions_for_pattern("pattern:...")     # candidate disambiguation probes (WS-B)
g.symptoms_for_pattern / g.recs_for_pattern / g.snippet / g.edges_from / g.edges_to / g.stats()
```
