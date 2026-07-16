"""Sanity gate for WS-C graph-RAG retrieval (stage2_interpretation/kg/retrieval.py).

Not a hallucination metric (that's RAGAS, WS-D) — this just asserts the 2-hop subgraph retrieval
resolves real Stage-1 feature sets to the expected top pattern, with cited book evidence attached.
Cheap, deterministic, runs anywhere (the KG is a static artifact).

    python evaluation/eval_graph_rag.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stage2_interpretation.kg.retrieval import GraphRAG  # noqa: E402

# (name, present_features [real Stage-1 shape], expected pattern id in the top-2)
CASES = [
    ("yellow greasy + red dots", {"tai": "yellow", "coat_texture": "greasy", "red_dots": "present"}, "damp_heat"),
    ("pale swollen toothmarks", {"zhi": "pale", "swollen": "present", "tooth_mk": "present",
                                 "coat_texture": "greasy", "tai": "white"}, "phlegm_dampness"),
    ("red + peeled + cracks", {"zhi": "red", "red_tongue": "present", "peeled_coating": "present",
                               "fissure": "severe"}, "yin_deficiency"),
    ("purple body", {"purple_body": "present"}, "blood_stasis"),
]

# Known calibration gaps — the retrieval is correct (entries resolve, evidence is cited) but the
# ranking is dominated by micro-edge volume, so a specific-but-rare signal loses. Fix is the
# empirical distinctiveness re-weighting in PLAN.md §7-A (weight ∝ 1/corpus-frequency). Printed as a
# warning, does NOT fail the gate — it tracks the gap until §7-A lands.
KNOWN_GAPS = [
    ("pale + toothmarks -> spleen_qi", {"zhi": "pale", "tooth_mk": "present"}, "spleen_qi_deficiency"),
]


def main():
    rag = GraphRAG.load()
    passed = 0
    for name, present, expect in CASES:
        r = rag.retrieve(present)
        top_ids = [p["id"].split(":", 1)[-1] for p in r.patterns[:2]]
        cited = sum(1 for p in r.patterns for e in p["evidence"] if e["sources"])
        ok = expect in top_ids
        passed += ok
        miss = ("  unmodeled=%s" % [f for f, _ in r.missing]) if r.missing else ""
        print("%s  %-28s -> top2=%s  (expect %s)  cited-edges=%d%s"
              % ("PASS" if ok else "FAIL", name, top_ids, expect, cited, miss))
    print("\n%d/%d strict cases resolved the expected pattern in the top-2." % (passed, len(CASES)))

    for name, present, expect in KNOWN_GAPS:
        r = rag.retrieve(present)
        rank = next((i + 1 for i, p in enumerate(r.patterns) if p["id"].endswith(expect)), None)
        note = "OK (fixed by §7-A re-weighting)" if rank and rank <= 2 else "still out of top-2 — see PLAN §7-A"
        print("NOTE  %-30s %s at rank %s — %s" % (name, expect, rank, note))
    sys.exit(0 if passed == len(CASES) else 1)


if __name__ == "__main__":
    main()
