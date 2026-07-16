"""WS-C graph-RAG retrieval over the macro-micro knowledge graph (docs/PLAN.md §7-C).

The old RAG retrieved flat cited chunks (`knowledge_cards.json`). This retrieves a **connected
2-hop subgraph** around the detected feature nodes, so the grounded matcher (and the LLM narrator)
sees *relationships* — `value:tai=yellow --points_to--> pattern:damp_heat --has_symptom--> symptom`,
plus the inverse `symptom --evidence_for--> pattern` links and any `argues_against` disambiguations —
not isolated facts. Pure stdlib; the graph is a static artifact so this adds no runtime weight beyond
dictionary lookups (fits the cheap-deploy budget).

    from kg.retrieval import GraphRAG
    rag = GraphRAG.load()
    r = rag.retrieve({"tai": "yellow", "coat_texture": "greasy", "fissure": "present"})
    r.patterns          # ranked [{id, name, score, evidence:[...], symptoms, recs, sections}]
    r.context_cards()   # grounded, citation-tagged lines ready to drop into the matcher prompt
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kg.graph import KnowledgeGraph  # noqa: E402

# edges we walk out from the entry features to reach the grounded neighbourhood
_RAG_RELS = {"points_to", "argues_against", "is_value_of", "has_symptom", "recommends",
             "probes", "evidence_for", "attested_in", "section_of", "maps_to"}

# Interim Stage-1 -> graph-vocab bridge. Stage-1 emits the coating-split keys and a few different
# value words than the seed KB. A proper fix is the WHO-IST ontology spine (WS-A); until then this
# small alias + a presence-fallback lets graph-RAG run on real Stage-1 output. Keep it explicit.
_FEATURE_ALIAS = {"coat_texture": "coating", "coat_thickness": "coating", "coat_color": "tai",
                  "coating_color": "tai", "body_color": "zhi", "cracks": "fissure",
                  "teeth_marks": "tooth_mk", "teethmark": "tooth_mk"}
# features the graph models as a single presence node (any positive value -> "present")
_BINARY = {"coating", "fissure", "tooth_mk", "red_dots", "swollen", "thin", "peeled_coating",
           "red_tongue", "purple_body", "black_coating", "slippery_coating"}
_VALUE_ALIAS = {("zhi", "pale"): "light", ("zhi", "red"): "dark", ("zhi", "purple"): "dark",
                ("tai", "light-yellow"): "light_yellow", ("tai", "grey"): "yellow"}
# Stage-1 "not present" values we never turn into an entry node
_ABSENT = {"none", "absent", "no", "normal", "regular", "0", "", None}


class Retrieval:
    """Result of a graph-RAG query: a ranked pattern view + the raw subgraph + citation snippets."""
    def __init__(self, entries, patterns, nodes, edges, snippets, missing):
        self.entries = entries          # [value/feature node ids used as seeds]
        self.patterns = patterns        # ranked list of pattern dicts (see _rank)
        self.nodes = nodes              # set of reached node ids (the subgraph)
        self.edges = edges              # induced edges within the subgraph
        self.snippets = snippets        # {snippet_id: {text, source, location}} referenced here
        self.missing = missing          # detected (feature,value) with no node in the graph

    def context_cards(self, max_per_pattern=6):
        """Flatten the top patterns into grounded, citation-tagged lines for a matcher/LLM prompt.
        Every line is traceable: a cited micro edge shows its book source (+ short snippet); a seed
        edge shows (rule KB). Nothing here is model-generated."""
        cards = []
        for p in self.patterns:
            cards.append("PATTERN %s (%s) — support %.2f" % (p["name"], p["id"].split(":", 1)[-1], p["score"]))
            for ev in p["evidence"][:max_per_pattern]:
                src = "; ".join(ev["sources"]) if ev["sources"] else "rule KB"
                tag = "argues AGAINST" if ev["polarity"] == "argues_against" else "supports"
                line = "   - %s=%s %s (w=%.2f) [%s]" % (ev["feature"], ev["value"], tag, ev["weight"], src)
                if ev.get("snippet"):
                    line += ' "%s"' % ev["snippet"]
                cards.append(line)
            if p["symptoms"]:
                cards.append("   often with: " + ", ".join(p["symptoms"][:6]))
        return "\n".join(cards)

    def as_dict(self):
        return {"entries": self.entries, "missing": self.missing,
                "patterns": self.patterns,
                "subgraph": {"n_nodes": len(self.nodes), "n_edges": len(self.edges)},
                "snippets": self.snippets}


class GraphRAG:
    def __init__(self, graph):
        self.g = graph

    @classmethod
    def load(cls, path=None):
        return cls(KnowledgeGraph.load(path) if path else KnowledgeGraph.load())

    # ---- entry nodes ----------------------------------------------------
    def _entry_ids(self, present):
        """Map detected {feature: value} to graph seed nodes. Normalises Stage-1 keys/values to the
        graph vocab (interim alias), then prefers the exact value node → a presence node → the
        feature node. Skips 'absent/normal' readings. Returns (unique entry_ids, missing)."""
        entries, seen, missing = [], set(), []
        for feat0, val0 in present.items():
            val = (str(val0).strip().lower() if val0 is not None else "")
            if val in _ABSENT:
                continue
            feat = _FEATURE_ALIAS.get(feat0, feat0)
            val = _VALUE_ALIAS.get((feat, val), val)
            if feat in _BINARY:
                val = "present"
            for cand in ("value:%s=%s" % (feat, val), "value:%s=present" % feat, "feature:%s" % feat):
                if cand in self.g.nodes:
                    if cand not in seen:
                        seen.add(cand)
                        entries.append(cand)
                    break
            else:
                missing.append((feat0, val0))
        return entries, missing

    # ---- ranking --------------------------------------------------------
    def _rank(self, entries, entry_set, reached, induced):
        """Score every pattern reachable from the entries by summed signed evidence, and attach the
        cited edges + 2-hop context (symptoms/recs/questions/sections) that justify it."""
        patterns = {}
        for e in induced:
            if e["rel"] not in ("points_to", "argues_against"):
                continue
            if e["src"] not in entry_set:          # only count evidence coming from a detected feature
                continue
            pid = e["dst"]
            if not pid.startswith("pattern:"):
                continue
            feat, val = self._decode(e["src"])
            w = e.get("weight", 1.0)
            signed = w if e["rel"] == "points_to" else -w
            p = patterns.setdefault(pid, {"id": pid, "name": self._name(pid), "score": 0.0,
                                          "evidence": [], "symptoms": [], "recs": [], "questions": [],
                                          "sections": []})
            p["score"] += signed
            snip = self.g.snippet(e["snippet"]) if e.get("snippet") else None
            p["evidence"].append({"feature": feat, "value": val, "weight": w,
                                  "polarity": e["rel"], "sources": e.get("sources", []),
                                  "snippet": (snip or {}).get("text", "") if snip else "",
                                  "layer": (e.get("cond") or {}).get("layer", "seed")})
        # 2-hop context per surviving pattern (dedupe evidence: same feature+snippet+source once)
        for pid, p in patterns.items():
            seen, uniq = set(), []
            for ev in p["evidence"]:
                k = (ev["feature"], ev["value"], ev["polarity"], ev["snippet"], tuple(ev["sources"]))
                if k in seen:
                    continue
                seen.add(k)
                uniq.append(ev)
            p["evidence"] = sorted(uniq, key=lambda x: (-x["weight"], x["polarity"] == "argues_against"))
            p["symptoms"] = [self._name(s) for s in self.g.symptoms_for_pattern(pid) if s in reached]
            p["recs"] = [self._name(r) for r in self.g.recs_for_pattern(pid) if r in reached]
            p["questions"] = [self._name(q) for q, _ in self.g.questions_for_pattern(pid) if q in reached]
            p["sections"] = [self._name(e["dst"]) for e in self.g.edges_from(pid, "attested_in")]
        return sorted(patterns.values(), key=lambda x: x["score"], reverse=True)

    # ---- public ---------------------------------------------------------
    def retrieve(self, present, hops=2):
        """present: {feature_key: value} (interpret.present_features shape). Returns a Retrieval."""
        entries, missing = self._entry_ids(present)
        entry_set = set(entries)
        reached, induced = self.g.neighborhood(entries, hops=hops, rels=_RAG_RELS)
        patterns = self._rank(entries, entry_set, reached, induced)
        snippets = {}
        for e in induced:
            sid = e.get("snippet")
            if sid and sid in self.g.snippets:
                snippets[sid] = self.g.snippets[sid]
        return Retrieval(entries, patterns, reached, induced, snippets, missing)

    # ---- helpers --------------------------------------------------------
    def _decode(self, node_id):
        if node_id.startswith("value:"):
            fv = node_id[len("value:"):]
            feat, _, val = fv.partition("=")
            return feat, (val or "present")
        if node_id.startswith("feature:"):
            return node_id[len("feature:"):], "present"
        return node_id, ""

    def _name(self, node_id):
        n = self.g.node(node_id)
        if not n:
            return node_id
        nm = n.get("name")
        if isinstance(nm, dict):
            return nm.get("en") or nm.get("zh") or next(iter(nm.values()), node_id)
        return nm or node_id


if __name__ == "__main__":
    import json
    rag = GraphRAG.load()
    # a damp-heat-ish case: yellow greasy coat + cracks
    demo = {"tai": "yellow", "coat_texture": "greasy", "coat_thickness": "thick", "fissure": "present"}
    r = rag.retrieve(demo)
    print("entries:", r.entries)
    print("missing:", r.missing)
    print("subgraph: %d nodes, %d edges, %d snippets" % (len(r.nodes), len(r.edges), len(r.snippets)))
    print("\ntop patterns:")
    for p in r.patterns[:4]:
        print("  %-34s score=%.2f  evidence=%d  symptoms=%d  cited=%d"
              % (p["name"], p["score"], len(p["evidence"]),
                 len(p["symptoms"]), sum(1 for e in p["evidence"] if e["sources"])))
    print("\n--- context cards (matcher prompt) ---")
    print(r.context_cards())
