"""WS-B interactive evidence refinement (docs/PROJECT_HANDBOOK.md §WS-B) — the inference-time, data-free analog of
TCM-DiffRAG's train-time personalized KG.

Two-pass reading:
  pass 1  tongue features -> candidate patterns (rule+matcher ensemble) + the questions that best
          DISAMBIGUATE the top-2 candidates (information-gain selection, not a fixed per-pattern list).
  pass 2  the user's yes/no answers enter as evidence and re-score the SAME patterns over the KG's
          question/symptom -> pattern (`evidence_for`) edges — re-running reasoning, not a new model.

`select_questions` picks items by how much they separate the current top-2 (the plan's info-gain step);
`rescore` applies a transparent log-odds update per answer over the patterns each answer bears on, then
re-ranks. Pure and graph-driven (no LLM), so it is unit-testable and cheap. The single-pattern log-odds
`interpret.refine()` stays as the interim/fallback.
"""
import math

K = 1.4                       # log-odds step per unit weight (matches interpret.refine)


def _contested(pa, pb):
    """How close the top-2 are, 4·q·(1-q) on the A-vs-B split — 1.0 when tied, →0 when lopsided.
    Refinement is most valuable when the top-2 are contested, so this scales question value."""
    s = (pa or 0) + (pb or 0)
    if s <= 0:
        return 1.0
    q = pa / s
    return 4.0 * q * (1.0 - q)


def select_questions(graph, patterns, k=3):
    """Info-gain question selection over the top-2 candidates.

    Each question `probes` exactly one pattern with a published weight; a question that bears on one of
    the top-2 shifts the A-vs-B gap by ~its weight, so its discriminative value = weight × contested(A,B).
    We return up to k, guaranteeing coverage of BOTH top-2 when possible (so the user can confirm the
    lead OR raise the runner-up), then fill by value. Each item is tagged with its target pattern +
    weight so `rescore` can apply it. Falls back to [] if the graph lacks probes for the top patterns.
    """
    cand = [p for p in patterns if p.get("id") != "balanced"][:2]
    if not cand:
        return []
    conf = {p["id"]: float(p.get("confidence", 0.0)) for p in cand}
    pa = cand[0].get("confidence", 0.0)
    pb = cand[1].get("confidence", 0.0) if len(cand) > 1 else 0.0
    contest = _contested(pa, pb)

    items = []
    for p in cand:
        for qid, w in graph.questions_for_pattern("pattern:%s" % p["id"]):
            node = graph.node(qid)
            text = (node or {}).get("name", {}).get("text") if node else None
            if not text:
                continue
            items.append({"q": text, "weight": round(float(w), 3), "target_pattern": p["id"],
                          "target_name": p.get("plain_name") or p.get("tcm_name") or p["id"],
                          "value": round(float(w) * contest, 4)})
    if not items:
        return []
    items.sort(key=lambda it: -it["value"])

    # guarantee both top-2 are represented (disambiguation needs to be able to cut either way)
    picked, seen_targets, seen_q = [], set(), set()
    for it in items:                       # first pass: best unique question per target
        if it["target_pattern"] in seen_targets or it["q"] in seen_q:
            continue
        picked.append(it); seen_targets.add(it["target_pattern"]); seen_q.add(it["q"])
        if len(picked) >= min(k, len(cand)):
            break
    for it in items:                       # fill remaining slots by value
        if len(picked) >= k:
            break
        if it["q"] in seen_q:
            continue
        picked.append(it); seen_q.add(it["q"])
    return picked[:k]


def rescore(patterns, answers):
    """Pass-2 re-scoring: fold the user's answers into the pattern confidences and re-rank.

    `answers` = [{"target_pattern": id, "weight": w, "answer": bool}] (as emitted by `select_questions`,
    plus the yes/no). Yes pushes its target pattern's confidence up by K·w in log-odds, No pushes down —
    the same transparent update as `interpret.refine()`, but applied across the candidate set so the
    ranking itself can change (a strong 'yes' on the runner-up can overtake the lead). Returns
    (reranked_patterns, deltas) where deltas logs the confidence change per pattern.
    """
    logit = {}
    for p in patterns:
        c = min(max(float(p.get("confidence", 0.0)), 1e-3), 1 - 1e-3)
        logit[p["id"]] = math.log(c / (1 - c))
    for a in answers:
        pid = a.get("target_pattern")
        if pid not in logit:
            continue
        logit[pid] += (K * float(a.get("weight", 0.0))) * (1 if a.get("answer") else -1)

    out, deltas = [], {}
    for p in patterns:
        new = round(1 / (1 + math.exp(-logit[p["id"]])), 3)
        deltas[p["id"]] = round(new - float(p.get("confidence", 0.0)), 3)
        q = dict(p)
        q["confidence"] = new
        q["confidence_pct"] = round(new * 100)
        out.append(q)
    out.sort(key=lambda p: -(0 if p["id"] == "balanced" else p["confidence"]))
    return out, deltas
