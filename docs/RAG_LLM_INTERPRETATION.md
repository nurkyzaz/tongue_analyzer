# Hybrid interpretation: grounded rules + LLM narrative (2026-07-13)

Answers "do we need RAG + an LLM to interpret the features?" The honest answer: **a hybrid**, not a
replacement. The rule engine stays as the accuracy backbone; an LLM adds fluent, combination-aware
language on top — grounded so it can't wander.

## Why hybrid (not pure-LLM)
- A hand-coded rule set can't gracefully cover every feature *combination*, and an LLM writes far richer,
  more nuanced, more personalised text. **That's real and worth having.**
- BUT an ungrounded LLM's main failure is **hallucination** — inventing TCM claims, over-confidence,
  fabricated conditions. For a health-adjacent product that's the dangerous direction.
- AND "better than any human" is capped by the **tongue's information ceiling**: a pale tongue is
  inherently qi/blood/yang-ambiguous without pulse+symptoms (see `CONSTITUTION_BENCHMARK.md`). An LLM
  improves the *expression*, not the *evidence*.

So: **rules decide the pattern (deterministic, auditable, testable); the LLM only re-expresses the
supplied grounding into a warm, woven read.**

## Architecture
```
features ─► [rule engine]  ─► pattern + confidence  ─┐  (deterministic, testable via mapping_testset.json)
            votes + combination rules                │
                                                     ▼
        grounding = {detected signs (+reliability/distinctiveness),
                     why_together (the combination rules that FIRED, cited),
                     leaning_pattern (rule-computed, w/ symptoms+recs+modern view),
                     sources}
                                                     │
                                                     ▼
                              [LLM narrative]  ─► warm grounded report
                    (interpret._llm_narrative; forbidden to add signs/patterns/claims or diagnose;
                     hedges 'less certain' reads; says "balanced" when no pattern leads)
                                                     │  falls back to the deterministic template if the
                                                     ▼  LLM is unavailable or returns nothing
                                              user-facing report
```

## Retrieval: structured facts + a TRUE vector RAG
Two layers of retrieval feed the LLM:
1. **Structured retrieval** — the exact grounded facts for this tongue (detected signs, the combination
   rules that fired, the rule-computed leaning pattern + its symptoms/recs/modern-view). Deterministic,
   no retrieval hallucination.
2. **Vector RAG** (`rag.py`, `build_corpus.py`) — a real semantic retriever over an embedded knowledge
   **corpus** (`knowledge_base/corpus.jsonl`): 48 cited chunks = the KB's feature/pattern/rule/zoning
   content PLUS **authored disambiguation cards** (the reasoning that tells similar pictures apart:
   "damp-heat vs phlegm-damp", "why a pale tongue is ambiguous", "yang-def vs phlegm-damp when swollen",
   "what the tongue cannot tell you"). Chunks are embedded with a local model (**nomic-embed-text** via
   Ollama, 768-d, no auth) into a **faiss** cosine index. At inference we embed a query built from the
   tongue's signs + leaning pattern and retrieve the top-k cited chunks into the grounding as
   `reference_notes` — so the LLM reasons over *combinations and disambiguations* from sourced material,
   not its parametric memory. Retrieval quality is strong (e.g. "pale swollen wet tongue" → the
   yang-def-vs-phlegm-damp card @0.81). Degrades gracefully: no index/embedder → no `reference_notes`,
   structured grounding still stands.

   **The corpus is the crux, and it's legitimately-sourced:** our own authored summaries citing public
   references — NOT copyrighted texts (Maciocia etc. can't be vendored verbatim). It starts small (48
   chunks) and is the thing to grow: add more cited cards + open-access literature (ICD-11 / CCMQ
   descriptions, CC-BY papers) and rebuild the index. Value scales with corpus quality.

   Build: `python stage2_interpretation/build_corpus.py && python stage2_interpretation/rag.py --build`
   (index cached to `corpus_vecs.npy`; git-ignored artifact).

## How to enable
LLM is OFF by default (deterministic template). Enable by pointing the backend-agnostic `LLMClient` at any
OpenAI-compatible endpoint via env:
```
TIH_LLM_BACKEND=openai
TIH_LLM_BASE_URL=http://localhost:11434/v1   # local Ollama (no auth)
TIH_LLM_MODEL=gemma3:latest                   # 4.3B, runs on GPU0; ~1.5–2s/report
TIH_LLM_API_KEY=ollama
```
Currently wired to the **local Ollama `gemma3:4B`** (no key needed, unlike the gated shared vLLM). Live on
the demo. If the LLM errors/times out, `chat()` returns None → the grounded template is served instead, so
the product never breaks.

## Validated
- Grounded to the computed pattern (t30→dampness, t17→empty-heat), weaves signs into one story, uses the
  distinctiveness + reliability tags, hedges the "less certain" greasy read.
- **Balanced-safety:** when no pattern leads, `leaning_pattern` is emptied so the model says "a balanced
  picture" and treats mild signs as nuances instead of inventing a pattern (t36).
- ~1.6–2.0s total including the LLM.

## Honest limitations & next steps
- It's a **4B local model** — good, not perfect; grounding is strong but not airtight (minor elaborations).
  Before defaulting it ON in production: (a) run a **hallucination-rate check** (does the LLM text ever add
  a sign/claim not in the grounding?) across many images; (b) consider a stronger model.
- The **mapping accuracy** (which pattern) is unchanged — still the rule backbone + the tongue's info
  ceiling. To make the *diagnosis* better: keep expanding the grounded combination rules (testable), and —
  the real test of "can an LLM map better than the rules?" — score LLM reasoning on **TCMEval-SDT**
  (`benchmark_syndrome.py`, currently 69.7% for the rules) and only trust it if it beats them.
- Keep the rule engine regardless: it's the auditable, testable backbone and the trust/legal story.
