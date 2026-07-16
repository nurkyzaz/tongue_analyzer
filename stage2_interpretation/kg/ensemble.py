"""WS-C ensemble: rule prior + grounded cited evidence (docs/PLAN.md §WS-C step 3).

The shadow run (evaluation/eval_shadow_matcher.py, human40) showed the grounded matcher (kg/matcher.py)
hallucinates 0.0 but only top-1-agrees with the rule engine 0.50 of the time — and the disagreements
are within-family. Verdict from the plan: **ensemble, don't replace the rule ranker**. So we keep the
auditable severity-weighted rule vote as the PRIOR and let the matcher's cite-or-abstain confidence
nudge the ranking and, crucially, attach book citations + an honest confidence number to each pattern.

Design (deliberately rule-dominant, so it can't regress the production ranking):
  blended = (1 - alpha) * rule_conf + alpha * matcher_conf      # alpha < 0.5 => rule leads
A pattern the matcher grounds but the rule engine never surfaced has rule_conf = 0, so its blended
score is capped at `alpha` (<= a real rule lead) — it can appear as a secondary, cited hint but cannot
overturn a confident rule lead. When the rule engine leads with the balanced picture (no strong
pattern), the ensemble stays out of the way and returns the rule cards unchanged.

Pure and LLM-free: `ensemble_cards` takes the rule cards + a matcher result dict, so it is unit-testable
without Ollama (feed the graph-RAG fallback or a stub). interpret.py calls it behind a flag + the WS-D
faithfulness gate; nothing here is live until those pass.
"""
import os

DEFAULT_ALPHA = float(os.getenv("TIH_WSC_ALPHA", "0.35"))   # matcher weight; rule keeps the majority


def blend(rule_conf, matcher_conf, alpha=DEFAULT_ALPHA):
    return (1.0 - alpha) * rule_conf + alpha * matcher_conf


def _citations(m):
    """Book citations for a matched pattern, from the matcher's grounded evidence (source + feature)."""
    out, seen = [], set()
    for e in (m or {}).get("evidence", []):
        src = e.get("source") or (e.get("sources") or ["rule KB"])[0]
        feat = e.get("feature", "")
        key = (feat, src)
        if key in seen:
            continue
        seen.add(key)
        cite = {"feature": feat, "source": src}
        if e.get("snippet"):
            cite["snippet"] = e["snippet"]
        out.append(cite)
    return out[:4]


def ensemble_cards(rule_cards, matcher_out, make_card=None, alpha=DEFAULT_ALPHA, top_k=3):
    """Blend rule cards (the prior) with the grounded matcher result -> re-ranked, cited cards.

    rule_cards : output of interpret.vote_patterns (each has id + confidence + KB fields).
    matcher_out: GroundedMatcher.match(present) dict, or None (then rule cards pass through untouched).
    make_card  : optional (pid, conf) -> card, used to build a card for a matcher-only pattern.
    Returns (cards, meta) where meta records the blend for transparency / eval.
    """
    if not matcher_out or not matcher_out.get("matched"):
        return list(rule_cards), {"applied": False, "reason": "no matcher evidence"}

    # respect the rule engine's balanced-lead: if no strong pattern, don't fabricate a cited lean
    if rule_cards and rule_cards[0].get("id") == "balanced":
        return list(rule_cards), {"applied": False, "reason": "balanced lead"}

    rule = {c["id"]: c for c in rule_cards if c.get("id") != "balanced"}
    matched = {m["id"]: m for m in matcher_out["matched"] if m.get("id") != "balanced"}

    cards = []
    for pid in set(rule) | set(matched):
        rc = float(rule.get(pid, {}).get("confidence", 0.0))
        # abstention is NEUTRAL, not negative: a rule pattern the matcher didn't name keeps its prior
        # (mc defaults to rc), so the matcher only moves patterns it actually grounded.
        mc = float(matched[pid]["confidence"]) if pid in matched else rc
        b = blend(rc, mc, alpha)
        base = rule.get(pid)
        if base is None:
            base = make_card(pid, b) if make_card else {"id": pid,
                    "plain_name": matched[pid].get("name", pid), "tcm_name": pid}
        card = dict(base)
        card["confidence"] = round(b, 3)
        card["confidence_pct"] = round(b * 100)
        card["rule_confidence"] = round(rc, 3)
        card["matcher_confidence"] = round(mc, 3)
        card["citations"] = _citations(matched.get(pid))
        card["why"] = (matched.get(pid, {}).get("why") or "").strip()
        card["source"] = ("ensemble" if pid in rule and pid in matched
                          else "rule" if pid in rule else "matcher-added")
        cards.append(card)

    cards.sort(key=lambda c: -c["confidence"])
    meta = {"applied": True, "alpha": alpha, "matcher_source": matcher_out.get("source"),
            "dropped": matcher_out.get("dropped", []),
            "ranking": [(c["id"], c["confidence"], c["source"]) for c in cards[:top_k]]}
    return cards[:top_k], meta
