"""Build the RAG knowledge corpus (JSONL of retrievable, cited chunks) from our OWN authored, grounded
content — never copyrighted text. Sources: our tcm_knowledge.json (features/patterns/combination rules/
zoning, themselves grounded in Maciocia/ICD-11/CCMQ/SymMap) PLUS authored disambiguation cards that
capture the reasoning a human uses to tell similar pictures apart (the high-value RAG content).

    python stage2_interpretation/build_corpus.py   ->   knowledge_base/corpus.jsonl

Expand later with more legitimately-sourced chunks (open-access papers, ICD-11 / CCMQ descriptions).
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
KB = os.path.join(HERE, "knowledge_base", "tcm_knowledge.json")
OUT = os.path.join(HERE, "knowledge_base", "corpus.jsonl")

# --- authored disambiguation / nuance cards: how to tell similar tongue pictures apart, and the honest
#     limits. Grounded in the same references; these are the reasoning a good practitioner applies. ---
CARDS = [
    ("damp-heat vs phlegm-damp",
     "Both damp-heat and phlegm-dampness show a thick, greasy tongue coating. The deciding factor is HEAT: "
     "a YELLOW coating and/or a RED tongue body means damp-HEAT (heat + damp); a WHITE coating with a "
     "normal or pale body, especially if the tongue looks wet, means plain phlegm-dampness (cold-damp). "
     "Colour and moisture separate them.", "Sacred Lotus; Maciocia"),
    ("yang-deficiency vs phlegm-damp (both swollen)",
     "A swollen/enlarged tongue occurs in both. If it is also PALE and WET/glossy with a thin white coat, "
     "it points to Yang deficiency (the warmth to move fluids is low). If it carries a THICK GREASY coat, "
     "it points to phlegm-dampness (fluids already congealed). Moisture-plus-pale vs thick-greasy is the "
     "tell.", "Sacred Lotus; Maciocia"),
    ("why a pale tongue is ambiguous",
     "A pale tongue alone cannot be pinned to one pattern: it appears in Qi deficiency, Blood deficiency "
     "AND Yang deficiency. Co-signs disambiguate — pale + tooth-marks leans Qi deficiency; pale + thin "
     "body leans Blood deficiency; pale + swollen + wet leans Yang deficiency. Without pulse and symptoms "
     "the tongue is genuinely uncertain here, so hedge and ask follow-ups.", "Sacred Lotus; CONSTITUTION_BENCHMARK"),
    ("red tip meaning",
     "A tongue tip that is distinctly redder than the rest reflects heat in the upper body / Heart in this "
     "tradition, commonly linked to stress, poor sleep or restlessness. It is a localized sign and, on its "
     "own, a mild one — it nudges toward heat (yin-deficiency or damp-heat) rather than defining a pattern.",
     "Sacred Lotus; Maciocia"),
    ("cracks / fissures nuance",
     "Cracks usually indicate Yin or fluid depletion (a dry, under-nourished surface), especially when the "
     "body is red and the coat scanty. But a single deep central crack can be constitutional or reflect "
     "Stomach/Spleen, and some cracks are lifelong and harmless. Read cracks together with body colour and "
     "coating, not alone.", "Sacred Lotus; Maciocia"),
    ("greasy coating is texture, not colour",
     "'Greasy' describes coating TEXTURE — a sticky, curd-like film that hides the tongue surface — and is "
     "the core sign of Dampness/Phlegm. It is independent of coating COLOUR (white vs yellow) and of "
     "THICKNESS. Surface papillae can mimic a light film, so a faint greasy read is uncertain.",
     "Sacred Lotus; Maciocia"),
    ("wet vs dry surface",
     "A wet, glossy tongue suggests fluids that aren't being transformed (cold-damp / Yang-deficiency "
     "direction) and argues AGAINST heat, which dries the surface. A genuinely dry tongue points to Yin/"
     "fluid depletion or heat. Note: photo glare can mimic wetness, so treat moisture as a soft signal.",
     "Sacred Lotus"),
    ("sublingual veins (not captured here)",
     "Distended, dark under-tongue veins are the strongest single objective sign of Blood stasis, but they "
     "require lifting the tongue and a second photo, which this tool does not take — so a blood-stasis read "
     "here rests only on a purple/dusky body and is correspondingly less certain.", "Delphi consensus PMC8983216"),
    ("what the tongue cannot tell you",
     "Tongue reading is one of four traditional examinations (looking, listening, asking, pulse). The "
     "tongue is informative about coating/damp/heat but weak at separating the deficiency patterns, which "
     "need pulse and symptoms. Always frame results as a tendency to explore, never a diagnosis.",
     "Maciocia; WHO ICD-11 Ch.26"),
]


def chunks_from_kb(kb):
    out = []
    src_all = "; ".join(kb.get("sources", [])[:2])
    for fk, fv in kb.get("features", {}).items():
        label = fv.get("label", fk)
        if fv.get("kind") == "graded_value" and fv.get("present_tcm"):
            out.append((f"feature:{fk}", f"{label} — when present: {fv.get('present_tcm','')}. In plain terms: "
                        f"{fv.get('present_plain','')}.", src_all))
        for vk, vv in fv.get("values", {}).items():
            gloss = vv.get("plain_gloss", "")
            tcm = vv.get("tcm_term", "")
            pts = ", ".join(vv.get("points_to", {}).keys())
            if gloss or tcm:
                out.append((f"feature:{fk}={vk}", f"{label} '{vk}' ({tcm}): {gloss}"
                            + (f" Traditionally associated with: {pts}." if pts else ""), src_all))
    for feat, fv in kb.get("extra_features", {}).items():
        if fv.get("present_tcm") or fv.get("present_plain"):
            out.append((f"extra:{feat}", f"{fv.get('label', feat)}: {fv.get('present_tcm','')}. "
                        f"{fv.get('present_plain','')}", src_all))
    for pid, p in kb.get("patterns", {}).items():
        if pid == "balanced":
            continue
        txt = f"{p.get('tcm_name', pid)} ({p.get('plain_name','')}): {p.get('explanation','')}"
        if p.get("associated_symptoms"):
            txt += " Often noticed: " + ", ".join(p["associated_symptoms"]) + "."
        if p.get("modern_correlation"):
            txt += " Modern view: " + p["modern_correlation"] + "."
        recs = (p.get("recommendations", {}).get("diet", []) + p.get("recommendations", {}).get("lifestyle", []))
        if recs:
            txt += " Wellness notes: " + "; ".join(recs[:4]) + "."
        out.append((f"pattern:{pid}", txt, "WHO ICD-11 Ch.26; CCMQ; Maciocia"))
    for r in kb.get("combination_rules", []):
        if r.get("note"):
            out.append((f"rule:{r['id']}", "Combination reasoning: " + r["note"], r.get("cite", "")))
    reg = kb.get("regions", {})
    for zone in ("tip", "center", "sides", "root"):
        z = reg.get(zone, {})
        if z.get("organs"):
            out.append((f"zone:{zone}", f"Tongue {zone} reflects {z['organs']}: {z.get('note','')}", "Song Weijiang; Maciocia"))
    return out


def main():
    kb = json.load(open(KB))
    rows = chunks_from_kb(kb)
    rows += [(f"card:{t.replace(' ', '_')}", body, src) for t, body, src in CARDS]
    with open(OUT, "w") as f:
        for cid, text, src in rows:
            f.write(json.dumps({"id": cid, "text": text, "source": src}, ensure_ascii=False) + "\n")
    print(f"wrote {OUT}  ({len(rows)} chunks: {sum('card:' in r[0] for r in rows)} authored cards + "
          f"{len(rows)-sum('card:' in r[0] for r in rows)} from KB)")


if __name__ == "__main__":
    main()
