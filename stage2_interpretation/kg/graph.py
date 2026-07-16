"""Macro-micro TCM knowledge graph — data model + queries (WS-A, docs/PLAN.md §3).

A single typed graph unifies three layers so Stage 2 retrieval/reasoning has ONE grounded
substrate instead of a rule KB + a flat card list:

  * SEED  layer  — every fact in tcm_knowledge.json, re-expressed as typed nodes/edges so the
                   graph is a strict SUPERSET of today's rule engine (parity is enforced by
                   build_kg.py --verify). Guarantees day-one behaviour is unchanged.
  * MACRO layer  — book chapter/section title nodes (context, semantic integrity). Added later
                   from the licensed literature (Gerlach backbone). rel="section_of".
  * MICRO layer  — LLM-extracted triplets from the books, each carrying a citation + snippet id.
                   Same edge types as the seed layer, so they merge transparently.

Storage is a plain JSON document (nodes dict + edges list + verbatim combination rules + a snippet
store). This module is the read/query API over it; build_kg.py is the writer. Pure stdlib so it runs
in any env (local or casper).

Node id conventions:
  pattern:<id>            e.g. pattern:spleen_qi_deficiency
  feature:<name>          e.g. feature:coating, feature:red_tongue
  value:<feature>=<value> e.g. value:tai=yellow, value:fissure=present
  symptom:<slug>          e.g. symptom:bloating-or-fullness-after-eating
  rec:<slug>              e.g. rec:diet:favour-warm-cooked-meals
  question:<slug>         a follow-up / disambiguation probe
  region:<zone> / organ:<slug>
  section:<book>:<path>   (macro layer)

Edge relations (`rel`):
  is_value_of   value   -> feature
  points_to     value|feature -> pattern     (weight; the core feature->pattern signal)
  has_symptom   pattern -> symptom
  evidence_for  symptom|question -> pattern   (weight; the INVERSE lever WS-B needs: user answers
                                               enter as symptom evidence and re-score patterns)
  probes        pattern -> question           (weight; which question separates candidates)
  recommends    pattern -> rec
  maps_to       region  -> organ
  section_of    node    -> section            (macro layer)
"""
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GRAPH = os.path.join(HERE, "..", "knowledge_base", "kg_graph.json")


class KnowledgeGraph:
    def __init__(self, nodes=None, edges=None, rules=None, snippets=None, meta=None):
        self.nodes = nodes or {}          # id -> {type, name, props, sources}
        self.edges = edges or []          # list of {src, rel, dst, weight, cond, sources, snippet}
        self.rules = rules or []          # verbatim combination_rules (hyperedges, kept as-is)
        self.snippets = snippets or {}    # snippet_id -> {text, source, location}
        self.meta = meta or {}
        self._index()

    # ---- indexing -------------------------------------------------------
    def _index(self):
        self._out = defaultdict(list)     # src -> [edge]
        self._in = defaultdict(list)      # dst -> [edge]
        self._by_type = defaultdict(list)  # type -> [node_id]
        for e in self.edges:
            self._out[e["src"]].append(e)
            self._in[e["dst"]].append(e)
        for nid, n in self.nodes.items():
            self._by_type[n.get("type", "?")].append(nid)

    # ---- io -------------------------------------------------------------
    @classmethod
    def load(cls, path=DEFAULT_GRAPH):
        with open(path) as f:
            d = json.load(f)
        return cls(nodes=d.get("nodes", {}), edges=d.get("edges", []),
                   rules=d.get("rules", []), snippets=d.get("snippets", {}),
                   meta=d.get("_meta", {}))

    def to_dict(self):
        return {"_meta": self.meta, "nodes": self.nodes, "edges": self.edges,
                "rules": self.rules, "snippets": self.snippets}

    def save(self, path=DEFAULT_GRAPH):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=1)

    # ---- mutation (used by builders) ------------------------------------
    def add_node(self, node_id, node_type, name=None, props=None, sources=None):
        if node_id in self.nodes:
            # merge sources; keep first name/props but union sources
            existing = self.nodes[node_id]
            for s in (sources or []):
                if s and s not in existing.setdefault("sources", []):
                    existing["sources"].append(s)
            if props:
                existing.setdefault("props", {}).update(props)
            return node_id
        self.nodes[node_id] = {"type": node_type, "name": name or {},
                               "props": props or {}, "sources": list(sources or [])}
        self._by_type[node_type].append(node_id)
        return node_id

    def add_edge(self, src, rel, dst, weight=None, cond=None, sources=None, snippet=None):
        e = {"src": src, "rel": rel, "dst": dst}
        if weight is not None:
            e["weight"] = weight
        if cond:
            e["cond"] = cond
        if sources:
            e["sources"] = list(sources)
        if snippet:
            e["snippet"] = snippet
        self.edges.append(e)
        self._out[src].append(e)
        self._in[dst].append(e)
        return e

    def add_snippet(self, snippet_id, text, source, location=None):
        self.snippets[snippet_id] = {"text": text, "source": source, "location": location}
        return snippet_id

    # ---- queries --------------------------------------------------------
    def node(self, node_id):
        return self.nodes.get(node_id)

    def nodes_of_type(self, node_type):
        return list(self._by_type.get(node_type, []))

    def edges_from(self, node_id, rel=None):
        return [e for e in self._out.get(node_id, []) if rel is None or e["rel"] == rel]

    def edges_to(self, node_id, rel=None):
        return [e for e in self._in.get(node_id, []) if rel is None or e["rel"] == rel]

    def patterns_for_value(self, feature, value):
        """[(pattern_id, weight, sources)] a given detected (feature,value) points to."""
        vid = "value:%s=%s" % (feature, value)
        out = [(e["dst"], e.get("weight", 1.0), e.get("sources", []))
               for e in self.edges_from(vid, "points_to")]
        return out

    def patterns_for_symptom(self, symptom_id):
        """[(pattern_id, weight, sources)] — the WS-B inverse lever (symptom/answer -> patterns)."""
        return [(e["dst"], e.get("weight", 1.0), e.get("sources", []))
                for e in self.edges_from(symptom_id, "evidence_for")]

    def questions_for_pattern(self, pattern_id):
        return [(e["dst"], e.get("weight", 1.0)) for e in self.edges_from(pattern_id, "probes")]

    def symptoms_for_pattern(self, pattern_id):
        return [e["dst"] for e in self.edges_from(pattern_id, "has_symptom")]

    def recs_for_pattern(self, pattern_id):
        return [e["dst"] for e in self.edges_from(pattern_id, "recommends")]

    def snippet(self, snippet_id):
        return self.snippets.get(snippet_id)

    def neighborhood(self, seeds, hops=2, rels=None, directed=True):
        """BFS from `seeds` (node ids) up to `hops` edges. Follows out-edges (plus in-edges when
        `directed=False`). Returns (reached_node_ids:set, induced_edges:list) — every edge whose
        endpoints are both reached (so inverse links like symptom->pattern come back even under a
        directed walk). This is the WS-C graph-RAG substrate: a connected 2-hop subgraph around the
        detected features instead of isolated facts."""
        reached = set(s for s in seeds if s in self.nodes)
        frontier = set(reached)
        for _ in range(max(0, hops)):
            nxt = set()
            for nid in frontier:
                for e in self._out.get(nid, []):
                    if rels and e["rel"] not in rels:
                        continue
                    if e["dst"] not in reached:
                        nxt.add(e["dst"])
                if not directed:
                    for e in self._in.get(nid, []):
                        if rels and e["rel"] not in rels:
                            continue
                        if e["src"] not in reached:
                            nxt.add(e["src"])
            nxt -= reached
            reached |= nxt
            frontier = nxt
            if not frontier:
                break
        induced = [e for e in self.edges
                   if e["src"] in reached and e["dst"] in reached and (not rels or e["rel"] in rels)]
        return reached, induced

    def stats(self):
        rels = defaultdict(int)
        for e in self.edges:
            rels[e["rel"]] += 1
        types = {t: len(v) for t, v in self._by_type.items()}
        return {"nodes": len(self.nodes), "node_types": dict(types),
                "edges": len(self.edges), "edge_rels": dict(rels),
                "rules": len(self.rules), "snippets": len(self.snippets),
                "layers": self.meta.get("layers", [])}


def slug(text, maxlen=60):
    """Stable, readable node-id slug from free text."""
    keep = []
    for ch in text.lower().strip():
        if ch.isalnum():
            keep.append(ch)
        elif ch in " -_/":
            keep.append("-")
    s = "".join(keep)
    while "--" in s:
        s = s.replace("--", "-")
    return s.strip("-")[:maxlen]
