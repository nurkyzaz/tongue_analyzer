# Macro-micro TCM knowledge graph (WS-A)

The grounded substrate for the Stage 2 overhaul (see [`docs/PLAN.md`](../../docs/PLAN.md) §2–3).
Replaces the split between a rule KB (`tcm_knowledge.json`) and a flat card list
(`knowledge_cards.json`) with one typed graph that retrieval + the grounded matcher (WS-C) and the
interactive refinement pass (WS-B) both query.

## Layers (all merge into one graph)

| Layer | Source | rel / nodes added | Status |
|---|---|---|---|
| **seed** | `tcm_knowledge.json` | every fact, re-typed | ✅ done (`build_kg.py`) |
| **macro** | Gerlach hierarchy (`parse_book.py` → `book_sections.json`) | `section_of` + `section:` nodes (184) | ✅ done |
| **micro** | qwen2.5:14b triplets from Gerlach ch.2–7, cited + snippet | `points_to`/`argues_against` + `attested_in` + `snippets` | ✅ done (120 cited edges; 50 candidates held for review) |

## Current graph (seed + macro + micro)

```
nodes 363   edges 671   rules 10   snippets 120
edges: points_to 135 (27 seed + 108 book-cited), argues_against 12, evidence_for 76,
       has_symptom 55, recommends 39, section_of 184, attested_in 120, probes 21, ...
```

Micro layer covers Gerlach ch.2–4 (feature chapters) + ch.5–7 (clinical cases) — 88 new triplets
from ch.5–7 folded in 2026-07-16.

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
rule ranker. Narrative faithfulness gate (`evaluation/eval_faithfulness.py`): **0.936 → PASS**.

Micro edges are tagged `cond.layer="micro"`, carry a book citation + a `snippet` id, and COEXIST with
the seed rule weights (rule engine keeps using seed edges; the WS-C matcher can prefer cited micro
edges). 49 `candidate` triplets — real signs our detector can't observe (sublingual veins) or Gerlach
patterns outside our 10 (ira, repletio hepatici) — are held in `_meta.micro_candidates` for the
ontology-spine step, not merged.

## Building the macro layer

```bash
python stage2_interpretation/kg/parse_book.py \
    --book tongue_lit/874856627-TCM-Tongue-Diagnosis-Explained.txt --id gerlach \
    --title "Gerlach O., TCM Tongue Diagnosis Explained (World Scientific, 2025)"
# -> book_sections.json (tracked: section metadata + char offsets, NOT the copyrighted text)
python stage2_interpretation/kg/build_kg.py --verify   # folds macro in automatically
```

## Micro layer (next, offline on casper)

```bash
# inspect the extraction prompt without an LLM:
python stage2_interpretation/kg/micro_extract.py --dry-run --book-id gerlach
# run with the LLM env set (casper): mines chapters 2-4, cite-or-abstain, -> book_triplets.json
python stage2_interpretation/kg/micro_extract.py --book-id gerlach --chapters 2,3,4
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
