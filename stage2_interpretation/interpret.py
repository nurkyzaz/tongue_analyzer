"""Stage 2 — educational interpretation grounded in the TCM knowledge base.

For each detected feature it reports the **degree** (from Stage-1 severity), the **TCM term**, and a
**plain-language gloss** of what practitioners in this tradition associate it with. It then votes
(weighted by severity) toward the patterns in the CCMQ 9-constitution model, and surfaces optional
**follow-up questions** (validated CCMQ items) that refine confidence. Framed throughout as one
tradition's perspective, never a diagnosis.

`interpret(stage1_output, metadata, llm)` -> structured dict (see bottom).
`refine(patterns, answers)` -> updated pattern confidences after follow-up answers.
"""
import json
import os

from llm_client import LLMClient

KB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base", "tcm_knowledge.json")
DISCLAIMER = ("Educational summary exploring the traditional-Chinese-medicine tongue-reading "
              "framework — not a medical diagnosis, and not validated by modern clinical evidence. "
              "For any health concern, please consult a qualified healthcare professional.")

GRADED = {"coating", "fissure", "tooth_mk"}   # features whose meaning scales with severity
MENTION_REL = 0.6                              # surface a graded feature only if its population rank >= this


STATS_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base", "reference_stats.json")


def load_kb(path=KB_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_stats(path=STATS_PATH):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"graded": {}, "categorical": {}}


def _rel(severity, feat, stats):
    """Distinctiveness = PERCENTILE RANK of this severity in the population (0..1). Handles skewed/
    biased features correctly: a coating value ~everyone has ranks ~0.5 (not distinctive), while any
    real crack (most tongues have none) ranks high. This is the data-grounded fix for both the
    'greasy on every tongue' and 'severe cracks read as balanced' bugs."""
    if severity < 0.12:                        # not present in absolute terms -> never distinctive
        return 0.0
    q = stats.get("graded", {}).get(feat, {}).get("q")
    if not q or q[-1] < 0.2:                    # feature ~never present in the population:
        return round(max(0.0, min(1.0, (severity - 0.3) / 0.7)), 4)   # rank by absolute severity
    # q holds values at percentiles 0,5,...,100 -> invert to get the rank of `severity`
    step = 100.0 / (len(q) - 1)
    for i in range(len(q) - 1):
        if severity <= q[i + 1]:
            lo, hi = q[i], q[i + 1]
            frac = 0.0 if hi <= lo else (severity - lo) / (hi - lo)
            return round((i + frac) * step / 100.0, 4)
    return 1.0


def _cat_weight(value, char, stats):
    """Informativeness of a categorical value = 1 - population frequency (rare value => informative)."""
    freq = stats.get("categorical", {}).get(char, {}).get(value, 0.3)
    return max(0.1, 1.0 - freq)


def _band(sev, kb):
    for b in kb["severity_bands"]:
        if sev <= b["max"]:
            return b
    return kb["severity_bands"][-1]


def feature_readings(chars, kb, stats):
    """Per-feature reading: degree band + dual-language + DISTINCTIVENESS-weighted pattern votes."""
    readings = []
    for ch, c in chars.items():
        spec = kb["features"].get(ch, {})
        sev = float(c.get("severity", 0.0))
        band = _band(sev, kb)
        if ch in GRADED:
            rel = _rel(sev, ch, stats)                 # percentile rank vs population
            present = rel >= MENTION_REL               # surface only if distinctively high
            band = _band(rel, kb)                      # degree reflects the population rank
            val_absent = (c["value"] == spec.get("absent_value"))
            reading = {
                "key": ch, "label": spec.get("label", ch), "value": c["value"],
                "severity": round(sev, 3), "rel": round(rel, 3), "band": band["word"], "mentioned": present,
                "tcm_term": spec.get("tcm_term", ""),
                "tcm": spec.get("present_tcm", "") if present else "",
                "plain": spec.get("present_plain", "") if present else
                         (spec.get("absent_plain", "") if val_absent else "within a typical range for this sign"),
                "points_to": {p: w * rel for p, w in spec.get("points_to", {}).items()} if present else {},
            }
        else:  # categorical color feature — weight by rarity of the value
            vinfo = spec.get("values", {}).get(c["value"], {})
            w_inf = _cat_weight(c["value"], ch, stats)
            reading = {
                "key": ch, "label": spec.get("label", ch), "value": c["value"],
                "severity": round(sev, 3), "rel": round(w_inf, 3), "band": "", "mentioned": True,
                "tcm_term": vinfo.get("tcm_term", ""), "tcm": vinfo.get("tcm_term", ""),
                "plain": vinfo.get("plain_gloss", ""),
                "points_to": {p: w * w_inf for p, w in vinfo.get("points_to", {}).items()},
            }
        readings.append(reading)
    return readings


# How much each extra feature is allowed to influence the PATTERN (from its val-AP; noisy ones barely).
EXTRA_RELIABILITY = {"peeled_coating": 0.7, "red_dots": 0.6, "thin": 0.55, "red_tongue": 0.55,
                     "black_coating": 0.35, "purple_body": 0.3, "swollen": 0.3, "slippery_coating": 0.2}


def extra_readings(extra_chars, kb, stats):
    """Readings for the Phase-4 multi-label features. These are noisier and over-predicted, so we
    surface one only when it's clearly ABOVE the population norm (distinctive), and weight its vote
    by that distinctiveness."""
    out, specs = [], kb.get("extra_features", {})
    for feat, c in (extra_chars or {}).items():
        sev = float(c.get("severity", 0.0))
        rel = _rel(sev, feat, stats)
        if rel < MENTION_REL:                   # only surface distinctively-present extras
            continue
        band = _band(rel, kb)                   # band reflects population rank, not raw prob
        spec = specs.get(feat, {})
        vote = rel * EXTRA_RELIABILITY.get(feat, 0.35)     # noisy detectors barely influence pattern
        out.append({"key": feat, "label": spec.get("label", feat), "value": "present",
                    "severity": round(sev, 3), "rel": round(rel, 3), "band": band["word"], "mentioned": True,
                    "tcm_term": spec.get("tcm_term", ""), "tcm": spec.get("present_tcm", ""),
                    "plain": spec.get("present_plain", ""),
                    "points_to": {p: w * vote for p, w in spec.get("points_to", {}).items()}})
    return out


def _card(kb, pid, conf):
    pat = kb["patterns"].get(pid, {})
    return {"id": pid, "tcm_name": pat.get("tcm_name", pid), "plain_name": pat.get("plain_name", ""),
            "explanation": pat.get("explanation", ""),
            "associated_symptoms": pat.get("associated_symptoms", []),
            "modern_correlation": pat.get("modern_correlation", ""),
            "recommendations": pat.get("recommendations", {}),
            "followup_questions": pat.get("followup_questions", []),
            "confidence": round(conf, 3)}


def vote_patterns(readings, kb, top_k=3):
    """Accumulate severity-weighted votes; map to ABSOLUTE confidence (saturating), so a weak top
    pattern reads as low confidence rather than a misleading 100%. Falls back to 'balanced' when
    nothing is notable."""
    scores = {}
    for r in readings:
        for p, w in r["points_to"].items():
            scores[p] = scores.get(p, 0.0) + w
    scores.pop("balanced", None)
    top_raw = max(scores.values()) if scores else 0.0
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
    # saturating map: raw ~1.5 -> ~0.65, raw 3 -> ~0.8; weak 0.3 -> ~0.17
    out = [_card(kb, pid, s / (s + 1.2)) for pid, s in ranked]
    if top_raw < 0.55:                       # no strong pattern -> lead with the balanced picture
        out = [_card(kb, "balanced", 1.0 - top_raw / 0.55)] + out
    return out


def _synthesis(readings, patterns):
    noted = [r for r in readings if r["mentioned"] and r["key"] in GRADED and r["band"] != "none"]
    if patterns and patterns[0]["id"] != "balanced":
        lead = patterns[0]
        txt = (f"Taken together, your signs most align with **{lead['tcm_name']}** "
               f"— in plain terms, *{lead['plain_name']}*. {lead['explanation']}")
        if len(patterns) > 1 and patterns[1]["id"] != "balanced":
            txt += f" There are also secondary hints of {patterns[1]['tcm_name']}."
        return txt
    return ("Your tongue looks close to the balanced picture — no strong traditional pattern stands "
            "out. The notes above describe each sign on its own.")


def _markdown(readings, patterns, combined, sources):
    L = ["**Your signs, one by one**"]
    for r in readings:
        deg = f"{r['band']} " if r["band"] and r["band"] != "none" else ""
        if r["key"] in GRADED and not r["mentioned"]:
            L.append(f"- **{r['label']}:** {r['plain']}")
        else:
            L.append(f"- **{deg}{r['tcm_term'] or r['value']}** — *TCM:* {r['tcm']}. *In plain terms:* {r['plain']}")
    L += ["", "**What they suggest together**", combined]
    if patterns and patterns[0]["id"] != "balanced":
        p = patterns[0]
        if p["associated_symptoms"]:
            L += ["", f"*In this tradition, {p['plain_name']} is often associated with:* "
                  + ", ".join(p["associated_symptoms"][:5]) + "."]
        rec = p.get("recommendations", {})
        recs = (rec.get("diet", []) + rec.get("lifestyle", []))[:4]
        if recs:
            L += ["", "**Traditional wellness notes**"] + [f"- {x}" for x in recs]
    L += ["", "**Grounding:** " + "; ".join(sources[:4]), "", "**Note**", DISCLAIMER]
    return "\n".join(L)


def _llm_narrative(readings, patterns, sources, llm):
    grounding = {
        "features": [{"sign": r["label"], "degree": r["band"], "tcm": r["tcm"], "plain": r["plain"]}
                     for r in readings],
        "patterns": [{"tcm_name": p["tcm_name"], "plain_name": p["plain_name"],
                      "explanation": p["explanation"], "symptoms": p["associated_symptoms"],
                      "recommendations": p["recommendations"]} for p in patterns[:2]],
        "sources": sources,
    }
    system = ("You are a careful TCM wellness educator. Using ONLY the grounding, write a warm, "
              "specific report: (1) each detected sign with its degree, its TCM term AND a plain-language "
              "gloss; (2) what they suggest together; (3) plain-language associations; (4) specific "
              "traditional wellness notes. Never invent signs/patterns/claims beyond the grounding, and "
              "never state a diagnosis. Frame as one tradition's perspective.")
    user = ("Grounding:\n" + json.dumps(grounding, ensure_ascii=False, indent=2) +
            "\n\nWrite it in Markdown, ending with a reminder to consult a professional.")
    text = llm.chat(system, user, max_tokens=1000)
    return (text + "\n\n**Note**\n" + DISCLAIMER) if text else None


# Shareable "tongue type" archetypes (one per lead pattern). Evocative but non-scary, educational.
ARCHETYPES = {
    "balanced":              ("The Well-Balanced", "✨", "harmony across the board"),
    "spleen_qi_deficiency":  ("The Gentle Engine", "\U0001F331", "runs best on warm, steady fuel"),
    "phlegm_dampness":       ("The Misty One", "\U0001F32B️", "carries a little fog & heaviness"),
    "damp_heat":             ("The Slow Fire", "\U0001F525", "runs a touch hot & humid"),
    "yin_deficiency":        ("The Overclocked", "\U0001F335", "gives a lot, runs a little dry"),
    "yang_deficiency":       ("The Cool Current", "❄️", "thrives with extra warmth"),
    "blood_deficiency":      ("The Understocked", "\U0001FA78", "running low on reserves"),
    "blood_stasis":          ("The Slow River", "\U0001F300", "circulation likes to keep moving"),
}


def build_card(readings, patterns):
    """A short, shareable result: a 'tongue type' archetype + a positive 0-100 balance score + the
    most DISTINCTIVE highlights (what stands out about this person, not what everyone has)."""
    dev = sum(float(r.get("rel", 0)) for r in readings
              if r["key"] not in ("tai", "zhi")) \
        + sum(0.5 * float(r.get("rel", 0)) for r in readings
              if r["key"] in ("tai", "zhi") and r["value"] not in ("regular", "white"))
    balance = int(max(40, min(99, round(100 - 22 * dev))))
    lead = patterns[0]["id"] if patterns else "balanced"
    name, emoji, blurb = ARCHETYPES.get(lead, ARCHETYPES["balanced"])
    hl = sorted([r for r in readings if r["key"] not in ("tai", "zhi") and float(r.get("rel", 0)) > 0.15],
                key=lambda r: -float(r.get("rel", 0)))[:3]
    return {
        "type_name": name, "emoji": emoji, "blurb": blurb,
        "balance_score": balance,
        "lead_plain": (patterns[0]["plain_name"] if patterns else "balanced picture"),
        "highlights": [{"label": r["label"], "band": r["band"],
                        "score": int(round(float(r.get("severity", 0)) * 100))} for r in hl],
    }


def interpret(stage1_output, metadata=None, llm: LLMClient = None):
    kb = load_kb()
    chars = stage1_output["key_characteristics"]
    quality = stage1_output.get("quality", {})
    if not quality.get("accepted", True):
        msg = ("The photo couldn't be read reliably (" + "; ".join(quality.get("reasons", ["low quality"]))
               + "). Please retake in good, even lighting with the tongue fully visible.")
        return {"report": msg, "features": [], "patterns": [], "combined": "", "sources": [],
                "disclaimer": DISCLAIMER}

    stats = load_stats()
    readings = (feature_readings(chars, kb, stats)
                + extra_readings(stage1_output.get("extra_characteristics", {}), kb, stats))
    patterns = vote_patterns(readings, kb)
    combined = _synthesis(readings, patterns)
    sources, overview = kb["sources"], kb["overview"]

    llm = llm or LLMClient()
    report = (_llm_narrative(readings, patterns, sources, llm) if llm.enabled else None) \
        or _markdown(readings, patterns, combined, sources)

    return {
        "overview": overview, "features": readings, "patterns": patterns,
        "card": build_card(readings, patterns),
        "regions": kb.get("regions", {}),
        "combined": combined, "sources": sources, "report": report, "disclaimer": DISCLAIMER,
        # follow-up flow: questions for the top non-balanced pattern
        "followup": ([{"pattern_id": patterns[0]["id"], "pattern": patterns[0]["tcm_name"],
                       "plain_name": patterns[0]["plain_name"], "base_confidence": patterns[0]["confidence"],
                       "questions": patterns[0]["followup_questions"]}]
                     if patterns and patterns[0]["id"] != "balanced" and patterns[0]["followup_questions"]
                     else []),
    }


def refine(base_confidence, answers):
    """Transparent log-odds update. `answers` = [{weight, answer(bool)}]. Yes pushes confidence up by
    the item's published weight, No pushes it down. Returns a refined 0-1 confidence."""
    import math
    p = min(max(base_confidence, 1e-3), 1 - 1e-3)
    logit = math.log(p / (1 - p))
    for a in answers:
        logit += (1.4 * a["weight"]) * (1 if a.get("answer") else -1)
    return round(1 / (1 + math.exp(-logit)), 3)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-json", required=True)
    args = ap.parse_args()
    with open(args.stage1_json) as f:
        print(interpret(json.load(f))["report"])
