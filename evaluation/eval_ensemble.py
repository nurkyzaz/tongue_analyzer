"""WS-C ensemble eval — is the rule+matcher blend SAFE and does it ADD grounding? (PROJECT_HANDBOOK.md §WS-C step 3)

The shadow run already established the matcher hallucinates 0.0 but only top-1-agrees with the rule
engine 0.50 (within-family). This eval scores the *ensemble* (kg/ensemble.py) — rule prior + the
matcher's cited evidence — on the same real human40 features, to decide promotion on the numbers:

  - stability (safety): does the ensemble's top-1 still match the auditable RULE top-1? (rule-led ⇒ high)
  - grounding (value): what fraction of the surfaced patterns now carry a BOOK citation, and is the
    #1 pattern cited?
  - matcher-added: how often a grounded hint the rule missed appears (capped below the rule lead)
  - hallucination: dropped (ungrounded) patterns — must stay 0

Reuses the per-image `present` features cached in shadow_matcher_results.json (Stage-1 is frozen), so
it only needs Ollama. Run on casper:

    python evaluation/eval_ensemble.py                       # uses cached features + LLM
    python evaluation/eval_ensemble.py --alpha 0.5           # sweep the matcher weight
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "stage2_interpretation"))
import interpret                                           # noqa: E402
from kg.matcher import GroundedMatcher, _synth_readings    # noqa: E402
from kg.ensemble import ensemble_cards                     # noqa: E402

SHADOW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shadow_matcher_results.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shadow", default=SHADOW, help="cached per-image features")
    ap.add_argument("--alpha", type=float, default=None, help="override matcher weight (default 0.35)")
    ap.add_argument("--out", default="evaluation/ensemble_results.json")
    args = ap.parse_args()

    rows = json.load(open(args.shadow))["rows"]
    kb = interpret.load_kb()
    matcher = GroundedMatcher()
    assert matcher.llm.enabled, "set the LLM env (run on casper) — the matcher needs Ollama"
    alpha_kw = {"alpha": args.alpha} if args.alpha is not None else {}

    n = stable = lead_cited = added = halluc = 0
    cov_sum = 0.0
    out_rows = []
    for r in rows:
        present = r["present"]
        rule_cards = interpret.vote_patterns(_synth_readings(present, kb), kb, present)
        rule_top = next((c["id"] for c in rule_cards if c["id"] != "balanced"), None)
        mo = matcher.match(present)
        cards, meta = ensemble_cards(rule_cards, mo, make_card=lambda pid, c: interpret._card(kb, pid, c),
                                     **alpha_kw)
        ens_top = cards[0]["id"] if cards else None
        cited = [c for c in cards if c.get("citations")]
        n += 1
        stable += (ens_top == rule_top)
        lead_cited += bool(cards and cards[0].get("citations"))
        added += any(c.get("source") == "matcher-added" for c in cards)
        halluc += bool(mo.get("dropped"))
        cov_sum += (len(cited) / len(cards)) if cards else 0.0
        out_rows.append({"image": r["image"], "rule_top": rule_top, "ensemble_top": ens_top,
                         "stable": ens_top == rule_top, "lead_cited": bool(cards and cards[0].get("citations")),
                         "ranking": meta.get("ranking"), "dropped": len(mo.get("dropped", []))})
        print("%-9s rule=%-20s ens=%-20s %s lead_cited=%s" %
              (r["image"], rule_top, ens_top, "STABLE" if ens_top == rule_top else "shift ",
               bool(cards and cards[0].get("citations"))))

    summary = {"n": n, "alpha": args.alpha if args.alpha is not None else 0.35,
               "top1_stability_vs_rule": round(stable / n, 3),
               "lead_cited_rate": round(lead_cited / n, 3),
               "mean_citation_coverage": round(cov_sum / n, 3),
               "matcher_added_rate": round(added / n, 3),
               "hallucination_rate": round(halluc / n, 3),
               "model": matcher.llm.model}
    json.dump({"summary": summary, "rows": out_rows}, open(args.out, "w"), indent=1)
    print("\n== ENSEMBLE (alpha=%s) ==" % summary["alpha"])
    for k, v in summary.items():
        print("  %-26s %s" % (k, v))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
