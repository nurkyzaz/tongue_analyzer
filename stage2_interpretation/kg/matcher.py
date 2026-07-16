"""WS-C grounded cite-or-abstain matcher (docs/PLAN.md §3, §7-C).

Consumes the graph-RAG 2-hop retrieval (kg/retrieval.py) and asks an LLM to pick the patterns BEST
supported *by the retrieved, cited facts only* — never outside knowledge — with a strict JSON schema
(JSON-mode) and cite-or-abstain validation: any proposed pattern not grounded in the retrieval is
dropped, and if nothing is supported the matcher abstains. Temperature 0, deterministic.

Runs in SHADOW mode first: `shadow_compare` scores it against the rule engine's vote on the same
features (agreement, top-1 match, disagreements) so we promote on the numbers, not in advance. The
rule engine stays the production path; nothing here changes live output.

    # inspect the prompt without an LLM:
    python stage2_interpretation/kg/matcher.py --dry-run
    # on casper (LLM env set): run the matcher + shadow-compare on the built-in cases
    python stage2_interpretation/kg/matcher.py --shadow
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kg.retrieval import GraphRAG           # noqa: E402
from llm_client import LLMClient            # noqa: E402

SYSTEM = (
    "You are a careful traditional-Chinese-medicine tongue-diagnosis matcher. You may use ONLY the "
    "grounded facts given in the prompt — never outside knowledge, never a pattern not present in the "
    "facts. If the facts do not clearly support a pattern, do not name it. When the evidence is weak "
    "or conflicting, prefer fewer patterns or none. Output STRICT JSON only, no prose."
)


def build_prompt(present, retrieval, top_k=6):
    feats = ", ".join("%s=%s" % (f, v) for f, v in present.items())
    cards = retrieval.context_cards()
    ids = [p["id"].split(":", 1)[-1] for p in retrieval.patterns[:top_k]]
    return (
        "DETECTED TONGUE FEATURES: %s\n\n"
        "GROUNDED FACTS — cited associations from the TCM literature. Use ONLY these; each PATTERN "
        "line names an allowed pattern id in parentheses:\n%s\n\n"
        "Allowed pattern ids: %s\n\n"
        "TASK: choose the 1-3 patterns BEST supported by the facts above. Give each a confidence in "
        "[0,1] and cite the exact features + sources from the facts. If no pattern is well supported, "
        "return {\"patterns\": []} (abstain). Respond with JSON only:\n"
        "{\"patterns\": [{\"id\": \"<one of the allowed ids>\", \"confidence\": <0..1>, "
        "\"why\": \"<one sentence grounded in the facts>\", "
        "\"evidence\": [{\"feature\": \"<detected feature>\", \"source\": \"<book/source from a fact>\"}]}]}"
        % (feats, cards, ", ".join(ids))
    )


def _parse_json(txt):
    if not txt:
        return None
    m = re.search(r"\{.*\}", txt, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except Exception:
        return None


class GroundedMatcher:
    def __init__(self, graph_rag=None, llm=None):
        self.rag = graph_rag or GraphRAG.load()
        self.llm = llm or LLMClient()

    def match(self, present, top_k=6):
        """Return {matched, abstained, source, retrieval, raw}. `matched` is cite-or-abstain filtered:
        every entry is a pattern that (a) the retrieval actually reached and (b) the model grounded in
        a detected feature. Falls back to the graph-RAG ranking when no LLM is configured."""
        r = self.rag.retrieve(present)
        candidates = {p["id"].split(":", 1)[-1]: p for p in r.patterns}
        summary = {"entries": r.entries, "missing": r.missing,
                   "candidates": [(k, round(v["score"], 2)) for k, v in list(candidates.items())[:top_k]]}

        if not self.llm.enabled:                         # deterministic fallback = graph-RAG top-k
            matched = [self._from_candidate(c, c_dict, grounded=True)
                       for c, c_dict in list(candidates.items())[:3] if c_dict["score"] > 0]
            return {"matched": matched, "abstained": not matched, "source": "graph-rag-fallback",
                    "retrieval": summary, "raw": None}

        prompt = build_prompt(present, r, top_k=top_k)
        raw = self.llm.chat(SYSTEM, prompt, temperature=0.0, max_tokens=700,
                            response_format={"type": "json_object"})
        parsed = _parse_json(raw)
        matched, dropped = self._validate(parsed, candidates, present)
        return {"matched": matched, "dropped": dropped, "abstained": not matched,
                "source": "llm-cite-or-abstain", "retrieval": summary, "raw": raw}

    # ---- cite-or-abstain validation ------------------------------------
    def _validate(self, parsed, candidates, present):
        matched, dropped = [], []
        det = set(present.keys())
        for item in ((parsed or {}).get("patterns") or []):
            pid = (item.get("id") or "").strip().split(":", 1)[-1]
            if pid not in candidates:                    # not reachable in the retrieval -> hallucination
                dropped.append({"id": pid, "reason": "not in retrieved subgraph"})
                continue
            ev = [e for e in (item.get("evidence") or [])
                  if (e.get("feature") or "").split("=")[0] in det or _feat_in(e, det)]
            if not ev:                                   # named a real pattern but grounded in nothing
                dropped.append({"id": pid, "reason": "no grounded evidence"})
                continue
            try:
                conf = max(0.0, min(1.0, float(item.get("confidence", 0.5))))
            except Exception:
                conf = 0.5
            matched.append({"id": pid, "name": candidates[pid]["name"], "confidence": round(conf, 3),
                            "why": (item.get("why") or "").strip(),
                            "evidence": ev, "graph_score": round(candidates[pid]["score"], 3)})
        matched.sort(key=lambda x: -x["confidence"])
        return matched, dropped

    def _from_candidate(self, pid, c, grounded):
        cited = [{"feature": e["feature"], "source": (e["sources"][0] if e["sources"] else "rule KB")}
                 for e in c["evidence"][:3]]
        # squash graph score into a 0-1 confidence for a comparable number
        conf = round(min(1.0, c["score"] / 5.0), 3)
        return {"id": pid, "name": c["name"], "confidence": conf, "why": "",
                "evidence": cited, "graph_score": round(c["score"], 3)}


def _feat_in(ev, det):
    f = (ev.get("feature") or "").lower()
    return any(d.lower() in f or f in d.lower() for d in det)


def shadow_compare(present, matcher, rule_top):
    """Score the matcher against the rule engine's top patterns on one case (shadow mode)."""
    m = matcher.match(present)
    m_ids = [x["id"] for x in m["matched"]]
    r_ids = list(rule_top)
    top1 = bool(m_ids and r_ids and m_ids[0] == r_ids[0])
    inter = set(m_ids) & set(r_ids)
    union = set(m_ids) | set(r_ids)
    jacc = len(inter) / len(union) if union else 1.0
    return {"present": present, "matcher": m_ids, "rule": r_ids, "top1_agree": top1,
            "jaccard": round(jacc, 2), "abstained": m["abstained"], "source": m["source"],
            "dropped": m.get("dropped", [])}


# a few feature sets for shadow scoring (real Stage-1 shape)
SHADOW_CASES = [
    {"tai": "yellow", "coat_texture": "greasy", "red_dots": "present"},
    {"zhi": "pale", "swollen": "present", "tooth_mk": "present", "coat_texture": "greasy", "tai": "white"},
    {"zhi": "red", "red_tongue": "present", "peeled_coating": "present", "fissure": "severe"},
    {"purple_body": "present"},
    {"zhi": "pale", "tooth_mk": "present"},
]


def _rule_top(present, k=3):
    """Top-k rule-engine pattern ids for the same features (the production vote)."""
    import interpret
    kb = interpret.load_kb()
    # minimal stage1-like dict the rule path understands
    s1 = {"key_characteristics": {}, "extra_characteristics": {}, "zoned_analysis": {}}
    readings = []
    for f, v in present.items():
        readings.append({"key": f, "value": v, "points_to": {}})
    pats = interpret.vote_patterns(_synth_readings(present, kb), kb, present)
    return [p["id"] for p in pats[:k] if p["id"] != "balanced"]


def _synth_readings(present, kb):
    """Build rule-engine readings from raw features via the KB feature specs (mirrors feature_readings
    minus the model confidences), so the rule vote is comparable on the same inputs."""
    out = []
    for feat, val in present.items():
        spec = kb.get("features", {}).get(feat, {})
        pts = spec.get("points_to", {})
        vspec = spec.get("values", {}).get(val, {})
        pts = vspec.get("points_to", pts) if vspec else pts
        out.append({"key": feat, "value": val, "severity": 0.8, "rel": 0.8, "display": True,
                    "points_to": dict(pts)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print one matcher prompt, no LLM")
    ap.add_argument("--shadow", action="store_true", help="run matcher vs rule engine on the cases")
    args = ap.parse_args()

    matcher = GroundedMatcher()
    if args.dry_run:
        p = SHADOW_CASES[0]
        print(build_prompt(p, matcher.rag.retrieve(p)))
        return

    if args.shadow or True:
        agree = 0
        for present in SHADOW_CASES:
            rule_top = _rule_top(present)
            row = shadow_compare(present, matcher, rule_top)
            agree += row["top1_agree"]
            print("feats=%-64s matcher=%s rule=%s top1=%s jacc=%.2f%s"
                  % (json.dumps(present, ensure_ascii=False), row["matcher"], row["rule"],
                     row["top1_agree"], row["jaccard"],
                     "" if not row["dropped"] else "  dropped=%s" % row["dropped"]))
        print("\ntop-1 agreement with rule engine: %d/%d  (source=%s)"
              % (agree, len(SHADOW_CASES), matcher.match(SHADOW_CASES[0])["source"]))


if __name__ == "__main__":
    main()
