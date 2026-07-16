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
| **micro** | qwen2.5:14b triplets from Gerlach ch.2–4, cited + snippet | `points_to`/`argues_against` + `attested_in` + `snippets` | ✅ done (72 cited edges; 49 candidates held for review) |

## Current graph (seed + macro + micro)

```
nodes 360   edges 572   rules 10   snippets 72
edges: points_to 93 (27 seed + 66 book-cited), argues_against 6, evidence_for 76,
       has_symptom 55, recommends 39, section_of 184, attested_in 72, probes 21, ...
```

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
