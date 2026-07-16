"""Build the macro-micro TCM knowledge graph (WS-A, docs/PLAN.md §3).

    python stage2_interpretation/kg/build_kg.py            # build seed layer -> kg_graph.json
    python stage2_interpretation/kg/build_kg.py --verify   # build + assert SUPERSET parity vs KB

The SEED layer re-expresses every fact in tcm_knowledge.json as typed nodes/edges. --verify then
checks nothing was lost: every pattern, feature, points_to weight, symptom, recommendation,
follow-up question and combination rule in the KB must be reachable in the graph. This is the
guarantee that turning on the graph changes NO current behaviour on day one.

MACRO (book sections) and MICRO (LLM-extracted book triplets) layers append to the same graph via
their own builders later; they reuse KnowledgeGraph.add_* and the same edge relations, so they merge
transparently. The compiled kg_graph.json is a rebuildable artifact (git-ignored); its sources
(tcm_knowledge.json now, triplet files later) are tracked.
"""
import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kg.graph import KnowledgeGraph, slug  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
KB = os.path.join(HERE, "..", "knowledge_base", "tcm_knowledge.json")
OUT = os.path.join(HERE, "..", "knowledge_base", "kg_graph.json")


def _pattern_node(g, pid, p):
    g.add_node("pattern:%s" % pid, "pattern",
               name={"tcm": p.get("tcm_name", pid), "plain": p.get("plain_name", "")},
               props={k: p[k] for k in ("explanation", "icd11", "modern_correlation") if k in p},
               sources=["WHO ICD-11 Ch.26", "CCMQ", "Maciocia"])


def seed_from_kb(g, kb):
    """Expand tcm_knowledge.json into typed nodes/edges. Idempotent-ish (fresh graph expected)."""
    src_all = kb.get("sources", [])[:2]

    # patterns ---------------------------------------------------------
    for pid, p in kb.get("patterns", {}).items():
        _pattern_node(g, pid, p)
        # symptoms
        for s in p.get("associated_symptoms", []):
            sid = "symptom:%s" % slug(s)
            g.add_node(sid, "symptom", name={"text": s})
            g.add_edge("pattern:%s" % pid, "has_symptom", sid)
            # inverse lever for WS-B: an observed symptom is soft evidence for its pattern
            g.add_edge(sid, "evidence_for", "pattern:%s" % pid, weight=0.3)
        # recommendations
        for cat in ("diet", "lifestyle"):
            for r in p.get("recommendations", {}).get(cat, []):
                rid = "rec:%s:%s" % (cat, slug(r))
                g.add_node(rid, "recommendation", name={"text": r}, props={"category": cat})
                g.add_edge("pattern:%s" % pid, "recommends", rid)
        # follow-up questions (probes) + their inverse evidence edge
        for q in p.get("followup_questions", []):
            qid = "question:%s" % slug(q["q"])
            g.add_node(qid, "question", name={"text": q["q"]})
            g.add_edge("pattern:%s" % pid, "probes", qid, weight=q.get("weight", 0.4))
            g.add_edge(qid, "evidence_for", "pattern:%s" % pid, weight=q.get("weight", 0.4))

    # core features ----------------------------------------------------
    for fk, fv in kb.get("features", {}).items():
        g.add_node("feature:%s" % fk, "feature",
                   name={"label": fv.get("label", fk), "tcm": fv.get("tcm_term", "")},
                   props={k: fv[k] for k in ("kind", "absent_value", "present_tcm", "present_plain",
                                             "absent_plain") if k in fv},
                   sources=src_all)
        if fv.get("kind") == "graded_value":
            # graded feature: points_to applies when PRESENT
            vid = "value:%s=present" % fk
            g.add_node(vid, "value", name={"feature": fk, "value": "present"})
            g.add_edge(vid, "is_value_of", "feature:%s" % fk)
            for pat, w in fv.get("points_to", {}).items():
                g.add_edge(vid, "points_to", "pattern:%s" % pat, weight=w, sources=src_all)
        for vk, vv in fv.get("values", {}).items():
            vid = "value:%s=%s" % (fk, vk)
            g.add_node(vid, "value",
                       name={"feature": fk, "value": vk, "tcm": vv.get("tcm_term", "")},
                       props={"plain_gloss": vv.get("plain_gloss", "")})
            g.add_edge(vid, "is_value_of", "feature:%s" % fk)
            for pat, w in vv.get("points_to", {}).items():
                g.add_edge(vid, "points_to", "pattern:%s" % pat, weight=w, sources=src_all)

    # extra features (all presence-type) -------------------------------
    for fk, fv in kb.get("extra_features", {}).items():
        g.add_node("feature:%s" % fk, "feature",
                   name={"label": fv.get("label", fk), "tcm": fv.get("tcm_term", "")},
                   props={k: fv[k] for k in ("present_tcm", "present_plain") if k in fv},
                   sources=src_all)
        vid = "value:%s=present" % fk
        g.add_node(vid, "value", name={"feature": fk, "value": "present"})
        g.add_edge(vid, "is_value_of", "feature:%s" % fk)
        for pat, w in fv.get("points_to", {}).items():
            g.add_edge(vid, "points_to", "pattern:%s" % pat, weight=w, sources=src_all)

    # regions -> organs ------------------------------------------------
    for zone, z in kb.get("regions", {}).items():
        if zone.startswith("_") or not isinstance(z, dict) or not z.get("organs"):
            continue
        rid = "region:%s" % zone
        g.add_node(rid, "region", name={"zone": zone}, props={"note": z.get("note", "")})
        for organ in [o.strip() for o in z["organs"].replace("/", ",").split(",") if o.strip()]:
            oid = "organ:%s" % slug(organ)
            g.add_node(oid, "organ", name={"text": organ})
            g.add_edge(rid, "maps_to", oid)

    # combination rules kept verbatim (hyperedges) --------------------
    g.rules = list(kb.get("combination_rules", []))


def verify_parity(g, kb):
    """Assert the graph is a strict superset of the KB. Returns list of problems (empty = ok)."""
    problems = []

    def has_edge(src, rel, dst, weight=None):
        for e in g.edges_from(src, rel):
            if e["dst"] == dst and (weight is None or abs(e.get("weight", 0) - weight) < 1e-9):
                return True
        return False

    for pid, p in kb.get("patterns", {}).items():
        if "pattern:%s" % pid not in g.nodes:
            problems.append("missing pattern node: %s" % pid)
            continue
        for s in p.get("associated_symptoms", []):
            if not has_edge("pattern:%s" % pid, "has_symptom", "symptom:%s" % slug(s)):
                problems.append("missing symptom edge: %s -> %s" % (pid, s[:40]))
        for cat in ("diet", "lifestyle"):
            for r in p.get("recommendations", {}).get(cat, []):
                if not has_edge("pattern:%s" % pid, "recommends", "rec:%s:%s" % (cat, slug(r))):
                    problems.append("missing rec edge: %s -> %s" % (pid, r[:40]))
        for q in p.get("followup_questions", []):
            if not has_edge("pattern:%s" % pid, "probes", "question:%s" % slug(q["q"]),
                            q.get("weight", 0.4)):
                problems.append("missing probe edge: %s -> %s" % (pid, q["q"][:40]))

    def check_points_to(feature, value, points_to):
        vid = "value:%s=%s" % (feature, value)
        for pat, w in points_to.items():
            if not has_edge(vid, "points_to", "pattern:%s" % pat, w):
                problems.append("missing/￮weight points_to: %s -> %s (%.2f)" % (vid, pat, w))

    for fk, fv in kb.get("features", {}).items():
        if "feature:%s" % fk not in g.nodes:
            problems.append("missing feature node: %s" % fk)
        if fv.get("kind") == "graded_value" and fv.get("points_to"):
            check_points_to(fk, "present", fv["points_to"])
        for vk, vv in fv.get("values", {}).items():
            check_points_to(fk, vk, vv.get("points_to", {}))
    for fk, fv in kb.get("extra_features", {}).items():
        if "feature:%s" % fk not in g.nodes:
            problems.append("missing extra-feature node: %s" % fk)
        check_points_to(fk, "present", fv.get("points_to", {}))

    kb_rule_ids = {r["id"] for r in kb.get("combination_rules", [])}
    g_rule_ids = {r["id"] for r in g.rules}
    for rid in kb_rule_ids - g_rule_ids:
        problems.append("missing combination rule: %s" % rid)

    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="assert superset parity vs the KB")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    kb = json.load(open(KB))
    g = KnowledgeGraph(meta={
        "purpose": "Macro-micro TCM tongue knowledge graph (seed layer from tcm_knowledge.json).",
        "built": datetime.date.today().isoformat(),
        "seeded_from": "tcm_knowledge.json",
        "layers": ["seed"],
        "framing": "Educational only — 'traditionally associated with…', never a diagnosis.",
    })
    seed_from_kb(g, kb)
    g.save(args.out)
    st = g.stats()
    print("wrote %s" % args.out)
    print("  nodes: %d %s" % (st["nodes"], st["node_types"]))
    print("  edges: %d %s" % (st["edges"], st["edge_rels"]))
    print("  rules: %d   snippets: %d" % (st["rules"], st["snippets"]))

    if args.verify:
        problems = verify_parity(g, kb)
        if problems:
            print("\nPARITY FAILED (%d problems):" % len(problems))
            for p in problems[:40]:
                print("  - %s" % p)
            sys.exit(1)
        print("\nPARITY OK — graph is a strict superset of tcm_knowledge.json")


if __name__ == "__main__":
    main()
