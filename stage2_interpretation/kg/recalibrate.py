"""Recalibrate the rule-engine feature->pattern weights from the KG's empirical book-citation frequency.

The rule engine (`interpret.vote_patterns`) votes with the HAND-TUNED `points_to` weights in
`tcm_knowledge.json`. The 3 parsed books give an empirical signal — how often each feature->pattern
link is actually cited (kg_graph.json micro layer). This script bakes that signal back into a v2 KB so
the rule engine benefits from the book campaign, not just the graph/matcher.

Design (deliberately conservative, so books CALIBRATE rather than REPLACE hand knowledge):
  - Only v1's EXISTING (feature-value -> pattern) edges are re-weighted. v1's edge SET is TCM-vetted;
    the book extraction's noise lives mostly in spurious PAIRS (e.g. pale->yin-deficiency, which v1
    correctly omits), so restricting to v1's pairs filters it out.
  - For each such edge, blend v1's weight-share with the empirical citation-share (over v1's patterns
    for that feature-value):  share_v2 = (1-λ)·share_v1 + λ·share_emp,  λ default 0.35.
  - The total magnitude per feature-value is preserved (voting scale unchanged), so combination rules
    and thresholds still behave. Feature-values with no book citations are copied verbatim.
Everything else (patterns, symptoms, recs, combination_rules, symptom_patterns) is copied unchanged.

    python stage2_interpretation/kg/recalibrate.py --lam 0.35   # -> tcm_knowledge_v2.json
"""
import argparse
import collections
import copy
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
KB = os.path.join(HERE, "..", "knowledge_base", "tcm_knowledge.json")
GRAPH = os.path.join(HERE, "..", "knowledge_base", "kg_graph.json")
OUT = os.path.join(HERE, "..", "knowledge_base", "tcm_knowledge_v2.json")


def citation_freq(graph):
    """{(value_node_id, pattern_id): n book citations} from the micro layer's points_to edges."""
    freq = collections.Counter()
    for e in graph["edges"]:
        if e.get("cond", {}).get("layer") == "micro" and e["rel"] == "points_to":
            freq[(e["src"], e["dst"])] += 1
    return freq


def _reweight(feature, value, pts, freq, lam):
    """Blend v1 weights `pts` {pattern: w} for (feature,value) with the empirical citation shares,
    restricted to v1's patterns. Returns the new dict (magnitude preserved), or `pts` unchanged if no
    book signal. Also returns (touched, detail)."""
    vid = "value:%s=%s" % (feature, value)
    emp = {p: freq.get((vid, "pattern:%s" % p), 0) for p in pts}
    if sum(emp.values()) == 0:
        return pts, False, None
    base = sum(pts.values())
    s1 = sum(pts.values()) or 1.0
    se = sum(emp.values())
    new = {}
    for p, w in pts.items():
        share = (1 - lam) * (w / s1) + lam * (emp[p] / se)
        new[p] = round(base * share, 3)
    return new, True, {"vid": vid, "v1": pts, "emp": emp, "v2": new}


def recalibrate(kb, graph, lam):
    freq = citation_freq(graph)
    v2 = copy.deepcopy(kb)
    changes = []

    def handle(container, fk, fv):
        # graded_value / extra features carry points_to at the feature level (value = "present")
        if fv.get("kind") == "graded_value" and fv.get("points_to"):
            new, touched, d = _reweight(fk, "present", fv["points_to"], freq, lam)
            if touched:
                container[fk]["points_to"] = new; changes.append(d)
        for vk, vv in fv.get("values", {}).items():
            if vv.get("points_to"):
                new, touched, d = _reweight(fk, vk, vv["points_to"], freq, lam)
                if touched:
                    container[fk]["values"][vk]["points_to"] = new; changes.append(d)

    for fk, fv in kb.get("features", {}).items():
        handle(v2["features"], fk, fv)
    for fk, fv in kb.get("extra_features", {}).items():
        if fv.get("points_to"):
            new, touched, d = _reweight(fk, "present", fv["points_to"], freq, lam)
            if touched:
                v2["extra_features"][fk]["points_to"] = new; changes.append(d)

    v2.setdefault("_meta", {})["recalibrated"] = {
        "from": "tcm_knowledge.json", "source": "kg_graph.json micro-layer citation frequency",
        "lambda": lam, "n_edges_reweighted": len(changes),
        "note": "v2 re-weights v1's existing feature->pattern edges by empirical book-citation share; "
                "no new pairs added (v1 edge set is TCM-vetted). Toggle with TIH_KB_VERSION=v2."}
    return v2, changes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=0.35, help="empirical blend weight (0=v1, 1=book-only shares)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--show", action="store_true", help="print the per-edge before/after")
    args = ap.parse_args()

    kb = json.load(open(KB))
    graph = json.load(open(GRAPH))
    v2, changes = recalibrate(kb, graph, args.lam)
    json.dump(v2, open(args.out, "w"), ensure_ascii=False, indent=1)
    print("wrote %s (lambda=%s): re-weighted %d feature->pattern edges from book citations"
          % (args.out, args.lam, len(changes)))
    if args.show:
        for d in changes:
            print("  %-24s v1=%s  emp=%s  ->  v2=%s"
                  % (d["vid"].replace("value:", ""), d["v1"], d["emp"], d["v2"]))


if __name__ == "__main__":
    main()
