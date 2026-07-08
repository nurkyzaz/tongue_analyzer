"""Stage 2 — turn Stage-1 structured output into a rich, grounded TCM wellness report.

Produces, for every request:
  - overview of tongue reading,
  - a per-characteristic breakdown (each of the 5 scores explained: what YOUR value means),
  - a combined interpretation synthesizing the 5 signs into TCM pattern(s),
  - traditional care notes, grounding sources, and a disclaimer.

Grounding = `knowledge_base/tcm_knowledge.json` (standard TCM tongue-diagnosis literature). If an
LLM backend is configured, the same grounding is handed to the model (RAG) to write the narrative;
otherwise a deterministic rich template is used. Framing is wellness/education, never diagnosis.
"""
import json
import os

from llm_client import LLMClient

KB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base", "tcm_knowledge.json")
DISCLAIMER = ("This is an automated wellness/educational summary based on traditional Chinese "
              "medicine tongue-reading literature — not a medical diagnosis, and not validated by "
              "modern clinical evidence. Please consult a qualified healthcare professional for any "
              "health concern.")


def load_kb(path=KB_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def per_characteristic(chars, kb):
    """For each of the 5 signs, attach the meaning of the specific predicted value."""
    out = []
    for ch, c in chars.items():
        spec = kb["characteristics"].get(ch, {})
        vinfo = spec.get("values", {}).get(c["value"], {})
        out.append({
            "key": ch,
            "label": spec.get("label", ch),
            "value": c["value"],
            "confidence": c["confidence"],
            "tcm_role": spec.get("tcm_role", ""),
            "headline": vinfo.get("headline", ""),
            "meaning": vinfo.get("meaning", ""),
            "tcm": vinfo.get("tcm", ""),
            "tendencies": vinfo.get("tendencies", []),
        })
    return out


def retrieve_patterns(chars, kb):
    fired = []
    for card in kb["combined_patterns"]:
        if all(chars.get(c, {}).get("value") in vals for c, vals in card["match"].items()):
            fired.append((len(card["match"]), card))
    fired.sort(key=lambda t: -t[0])
    return [c for _, c in fired]


def synthesize(per_char, patterns):
    """A plain-language paragraph on what the 5 signs together suggest."""
    if not patterns:
        return ("Individually your signs don't combine into a single strong traditional pattern. "
                "The per-sign notes above describe what each one means on its own.")
    lead = patterns[0]
    txt = f"Taken together, your signs most align with **{lead['name']}**. {lead['explanation']}"
    if len(patterns) > 1:
        others = ", ".join(p["name"] for p in patterns[1:3])
        txt += f" There are also secondary hints of {others}, which a practitioner would weigh alongside the main picture."
    return txt


def _markdown(overview, per_char, patterns, combined, sources):
    lines = ["**How tongue reading works**", overview, "", "**Your signs, one by one**"]
    for c in per_char:
        conf = f" ({round(c['confidence']*100)}% confidence)" if c.get("confidence") else ""
        lines.append(f"- **{c['label']}: {c['value']}**{conf} — {c['meaning']}")
    lines += ["", "**What they suggest together**", combined]
    if patterns:
        care = next((p.get("care") for p in patterns if p.get("care")), None)
        if care:
            lines += ["", "**Traditional wellness notes**", care]
    lines += ["", "**Grounding**", "Interpretations follow standard TCM references: "
              + "; ".join(s["ref"] for s in sources if s["type"].startswith("TCM")) + ".",
              "", "**Note**", DISCLAIMER]
    return "\n".join(lines)


def _llm_narrative(overview, per_char, patterns, sources, llm):
    grounding = {
        "overview": overview,
        "per_characteristic": [{"sign": c["label"], "your_value": c["value"],
                                "meaning": c["meaning"], "tcm": c["tcm"]} for c in per_char],
        "combined_patterns": [{"name": p["name"], "explanation": p["explanation"],
                               "care": p.get("care", "")} for p in patterns[:3]],
        "sources": [s["ref"] for s in sources],
    }
    system = ("You are a careful TCM wellness educator. Using ONLY the provided grounding, write an "
              "informative but friendly report with these sections: a short intro, 'Your signs one by "
              "one' (explain each value), 'What they suggest together' (synthesize into pattern(s)), "
              "and 'Traditional wellness notes'. Do NOT invent patterns, diagnoses, studies or facts "
              "beyond the grounding. Never claim medical certainty.")
    user = ("Grounding (JSON):\n" + json.dumps(grounding, ensure_ascii=False, indent=2) +
            "\n\nWrite the report in Markdown. End with a reminder to consult a professional.")
    text = llm.chat(system, user, max_tokens=900)
    return (text + "\n\n**Note**\n" + DISCLAIMER) if text else None


def interpret(stage1_output, metadata=None, llm: LLMClient = None):
    kb = load_kb()
    chars = stage1_output["key_characteristics"]
    quality = stage1_output.get("quality", {})
    if not quality.get("accepted", True):
        msg = ("The photo could not be analyzed reliably (" +
               "; ".join(quality.get("reasons", ["low quality"])) +
               "). Please retake in good, even lighting with the tongue fully visible.")
        return {"report": msg, "overview": "", "characteristics": [], "patterns": [],
                "combined": "", "sources": [], "disclaimer": DISCLAIMER}

    per_char = per_characteristic(chars, kb)
    patterns = retrieve_patterns(chars, kb)
    combined = synthesize(per_char, patterns)
    sources = kb["sources"]
    overview = kb["overview"]

    llm = llm or LLMClient()
    report = (_llm_narrative(overview, per_char, patterns, sources, llm) if llm.enabled else None) \
        or _markdown(overview, per_char, patterns, combined, sources)

    return {
        "overview": overview,
        "characteristics": per_char,
        "patterns": [{"name": p["name"], "explanation": p["explanation"], "care": p.get("care", "")}
                     for p in patterns[:3]],
        "combined": combined,
        "sources": [s["ref"] for s in sources],
        "report": report,
        "disclaimer": DISCLAIMER,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-json", required=True)
    args = ap.parse_args()
    with open(args.stage1_json) as f:
        print(interpret(json.load(f))["report"])
