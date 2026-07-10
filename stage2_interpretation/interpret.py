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


def load_kb(path=KB_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _band(sev, kb):
    for b in kb["severity_bands"]:
        if sev <= b["max"]:
            return b
    return kb["severity_bands"][-1]


def feature_readings(chars, kb):
    """Per-feature reading: degree band + dual-language + pattern contributions."""
    readings = []
    for ch, c in chars.items():
        spec = kb["features"].get(ch, {})
        sev = float(c.get("severity", 0.0))
        band = _band(sev, kb)
        if ch in GRADED:
            present = band["mention"]
            reading = {
                "key": ch, "label": spec.get("label", ch), "value": c["value"],
                "severity": round(sev, 3), "band": band["word"], "mentioned": present,
                "tcm_term": spec.get("tcm_term", ""),
                "tcm": spec.get("present_tcm", "") if present else "",
                "plain": spec.get("present_plain", "") if present else spec.get("absent_plain", ""),
                "points_to": {p: w * sev for p, w in spec.get("points_to", {}).items()} if present else {},
            }
        else:  # categorical color feature
            vinfo = spec.get("values", {}).get(c["value"], {})
            conf = float(c.get("confidence", 1.0))
            reading = {
                "key": ch, "label": spec.get("label", ch), "value": c["value"],
                "severity": round(sev, 3), "band": "", "mentioned": True,
                "tcm_term": vinfo.get("tcm_term", ""), "tcm": vinfo.get("tcm_term", ""),
                "plain": vinfo.get("plain_gloss", ""),
                "points_to": {p: w * conf for p, w in vinfo.get("points_to", {}).items()},
            }
        readings.append(reading)
    return readings


def extra_readings(extra_chars, kb):
    """Readings for the Phase-4 multi-label features (presence prob = severity). Only surfaced when
    present (>= faint band), so absent features don't clutter the report."""
    out, specs = [], kb.get("extra_features", {})
    for feat, c in (extra_chars or {}).items():
        sev = float(c.get("severity", 0.0))
        band = _band(sev, kb)
        if not band["mention"]:
            continue
        spec = specs.get(feat, {})
        out.append({"key": feat, "label": spec.get("label", feat), "value": "present",
                    "severity": round(sev, 3), "band": band["word"], "mentioned": True,
                    "tcm_term": spec.get("tcm_term", ""), "tcm": spec.get("present_tcm", ""),
                    "plain": spec.get("present_plain", ""),
                    "points_to": {p: w * sev for p, w in spec.get("points_to", {}).items()}})
    return out


def _card(kb, pid, conf):
    pat = kb["patterns"].get(pid, {})
    return {"id": pid, "tcm_name": pat.get("tcm_name", pid), "plain_name": pat.get("plain_name", ""),
            "explanation": pat.get("explanation", ""),
            "associated_symptoms": pat.get("associated_symptoms", []),
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


def interpret(stage1_output, metadata=None, llm: LLMClient = None):
    kb = load_kb()
    chars = stage1_output["key_characteristics"]
    quality = stage1_output.get("quality", {})
    if not quality.get("accepted", True):
        msg = ("The photo couldn't be read reliably (" + "; ".join(quality.get("reasons", ["low quality"]))
               + "). Please retake in good, even lighting with the tongue fully visible.")
        return {"report": msg, "features": [], "patterns": [], "combined": "", "sources": [],
                "disclaimer": DISCLAIMER}

    readings = feature_readings(chars, kb) + extra_readings(stage1_output.get("extra_characteristics", {}), kb)
    patterns = vote_patterns(readings, kb)
    combined = _synthesis(readings, patterns)
    sources, overview = kb["sources"], kb["overview"]

    llm = llm or LLMClient()
    report = (_llm_narrative(readings, patterns, sources, llm) if llm.enabled else None) \
        or _markdown(readings, patterns, combined, sources)

    return {
        "overview": overview, "features": readings, "patterns": patterns,
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
