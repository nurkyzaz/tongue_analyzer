"""Stage 2 — turn Stage-1 structured output into a grounded, plain-language wellness report.

Two modes, same grounding (retrieved TCM pattern cards):
  - template (default): deterministic report, no LLM. Always available.
  - LLM (if LLMClient.enabled): the retrieved cards + observations are passed as grounding and the
    model writes the narrative (RAG). The LLM only rephrases grounded facts — it is told not to add
    diagnoses beyond the retrieved cards.

This is intentionally conservative: wellness/education framing, explicit disclaimers, no medical claims.
"""
import json
import os

from llm_client import LLMClient

KB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base", "tcm_patterns.json")
DISCLAIMER = ("This is an automated wellness/informational summary based on traditional tongue-sign "
              "associations, not a medical diagnosis. Traditional Chinese Medicine patterns are not "
              "validated by modern clinical evidence. Consult a qualified healthcare professional for "
              "any health concern.")


def load_kb(path=KB_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def retrieve(chars, kb):
    """Return pattern cards whose match conditions are satisfied, most-specific first."""
    fired = []
    for card in kb["patterns"]:
        conds = card["match"]
        if all(chars.get(c, {}).get("value") in vals for c, vals in conds.items()):
            fired.append((len(conds), card))
    fired.sort(key=lambda t: -t[0])
    return [c for _, c in fired]


def observations(chars, kb):
    obs = kb["observations"]
    lines = []
    for ch, c in chars.items():
        phrase = obs.get(ch, {}).get(c["value"])
        if phrase:
            lines.append(phrase)
    return lines


def template_report(chars, kb, cards):
    obs = observations(chars, kb)
    parts = ["**What we observed**", "Your tongue shows " + ", ".join(obs) + ".", ""]
    if cards:
        parts.append("**Traditional associations**")
        for c in cards[:3]:
            parts.append(f"- *{c['name']}*: {c['explanation']}")
        sens = []
        for c in cards[:3]:
            sens += c.get("common_sensations", [])
        if sens:
            uniq = list(dict.fromkeys(sens))
            parts.append("")
            parts.append("These patterns are traditionally linked to sensations such as "
                         + ", ".join(uniq[:5]) + ".")
    else:
        parts.append("No strong traditional pattern associations were triggered by these features.")
    parts += ["", "**Note**", DISCLAIMER]
    return "\n".join(parts)


def llm_report(chars, kb, cards, llm: LLMClient):
    obs = observations(chars, kb)
    grounding = {
        "observations": obs,
        "retrieved_patterns": [{"name": c["name"], "explanation": c["explanation"],
                                "sensations": c.get("common_sensations", [])} for c in cards[:3]],
    }
    system = ("You are a careful wellness assistant summarizing traditional tongue-sign associations. "
              "Use ONLY the provided grounding. Do NOT invent diagnoses or patterns beyond those given. "
              "Write 2 short paragraphs in plain, friendly language. Do not claim medical certainty.")
    user = ("Grounding (JSON):\n" + json.dumps(grounding, ensure_ascii=False, indent=2) +
            "\n\nWrite the wellness summary. End with a one-line reminder to consult a professional.")
    text = llm.chat(system, user)
    return text + "\n\n**Note**\n" + DISCLAIMER if text else None


def interpret(stage1_output, metadata=None, llm: LLMClient = None):
    """stage1_output: dict (Stage1Output). Returns {report, patterns, observations, disclaimer}."""
    kb = load_kb()
    chars = stage1_output["key_characteristics"]
    quality = stage1_output.get("quality", {})
    if not quality.get("accepted", True):
        return {"report": "The photo could not be analyzed reliably (" +
                "; ".join(quality.get("reasons", ["low quality"])) + "). Please retake in good, "
                "even lighting with the tongue fully visible.", "patterns": [], "observations": [],
                "disclaimer": DISCLAIMER}

    cards = retrieve(chars, kb)
    llm = llm or LLMClient()
    report = (llm_report(chars, kb, cards, llm) if llm.enabled else None) or template_report(chars, kb, cards)
    return {"report": report,
            "patterns": [c["name"] for c in cards[:3]],
            "observations": observations(chars, kb),
            "disclaimer": DISCLAIMER}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-json", required=True, help="path to a Stage1Output JSON file")
    args = ap.parse_args()
    with open(args.stage1_json) as f:
        s1 = json.load(f)
    out = interpret(s1)
    print(out["report"])
