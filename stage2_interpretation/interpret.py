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
GRADED_DISPLAY = GRADED | {"coat_thickness", "coat_texture"}   # + display axes, for text synthesis
MENTION_REL = 0.6                              # surface a graded feature only if its population rank >= this

# How trustworthy each detector is (from the human-40 + practitioner benchmarks) — so the report can be
# HONEST: state reliable reads plainly, hedge the tentative ones. Not shown as jargon; mapped to words.
RELIABILITY = {
    "tai": "reliable", "fissure": "reliable", "red_dots": "reliable", "coat_thickness": "reliable",
    "zhi": "moderate", "tooth_mk": "moderate", "peeled_coating": "moderate", "red_tip": "moderate",
    "moisture": "moderate",
    "coat_texture": "tentative", "coating": "tentative", "red_tongue": "tentative",
    "purple_body": "tentative", "swollen": "tentative", "thin": "tentative",
    "black_coating": "tentative", "slippery_coating": "tentative",
}
CONF_WORD = {"reliable": "clear read", "moderate": "", "tentative": "less certain — worth confirming"}


# features whose `rel` IS a genuine population percentile-rank (from reference_stats). The coat axes,
# red_tip and moisture carry a probability/severity in `rel`, NOT a percentile — so no percentile hook.
_PERCENTILE_OK = {"fissure", "tooth_mk", "peeled_coating", "red_tongue", "purple_body", "swollen",
                  "thin", "red_dots", "black_coating", "slippery_coating"}


def _pctl_phrase(key, rel):
    """Turn a population percentile-rank (rel, 0-1) into a friendly distinctiveness hook, or '' if this
    feature isn't graded on a population curve."""
    if key not in _PERCENTILE_OK or rel is None:
        return ""
    p = round(rel * 100)
    if p >= 85:
        return f"more pronounced than ~{p}% of tongues"
    if p >= 70:
        return f"more pronounced than most (top ~{100 - p}%)"
    return ""


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


# The coating is displayed as two derived axes (thickness × texture); the conflated `coating` reading
# is kept ONLY to drive pattern voting (unchanged) and hidden from the UI via display=False.
COAT_AXES_DISPLAY = ("coat_thickness", "coat_texture")


def coat_axis_readings(chars):
    """Display-only readings for the two coating axes (they don't vote; `coating` still does)."""
    out = []
    cfg = {
        "coat_thickness": {
            "label": "Coating thickness", "abnormal": "thick", "tcm_term": "厚苔 (thick coating)",
            "present_plain": "The coating looks thicker than a thin, see-through film — in this tradition a thicker coating suggests more internal accumulation (often damp or phlegm).",
            "absent_plain": "The coating looks thin — a thin, see-through film is considered normal.",
        },
        "coat_texture": {
            "label": "Coating texture", "abnormal": "greasy", "tcm_term": "腻苔 (greasy coating)",
            "present_plain": "The coating looks greasy/slippery rather than dry and even — in this tradition a sign of dampness. (Surface papillae patterns can mimic this, so treat a low score with caution.)",
            "absent_plain": "The coating texture looks smooth and even, not greasy.",
        },
    }
    for ax in COAT_AXES_DISPLAY:
        c = chars.get(ax)
        if not c:
            continue
        cf = cfg[ax]
        present = c["value"] == cf["abnormal"]
        out.append({
            "key": ax, "label": cf["label"], "value": c["value"],
            "severity": round(float(c.get("severity", 0.0)), 3), "rel": round(float(c.get("severity", 0.0)), 3),
            "band": "", "mentioned": present, "display": True,
            "tcm_term": cf["tcm_term"] if present else "",
            "tcm": cf["tcm_term"] if present else "",
            "plain": cf["present_plain"] if present else cf["absent_plain"],
            "points_to": {},                     # display-only: coating carries the vote
        })
    return out


def zoned_readings(stage1_output):
    """Readings from the zoned colour analysis. Currently: RED TIP (Heart / upper-jiao heat) — a
    localized sign the whole-tongue colour head averages away. Only the reliable STRONG red tip votes
    (zoning flags red_tip.present at tip_redness_delta > 2.0; validated P=0.92 on human labels).
    Grounded: red tip = Heat in the Heart [Sacred Lotus; Maciocia]."""
    z = (stage1_output or {}).get("zoned_analysis", {})
    out = []
    rt = z.get("red_tip")
    if rt and rt.get("value") == "present":
        sev = float(rt.get("severity", 0.0))
        out.append({
            "key": "red_tip", "label": "Red tip", "value": "present",
            "severity": round(sev, 3), "rel": round(sev, 3), "band": "", "mentioned": True, "display": True,
            "tcm_term": "舌尖红 (red tip)", "tcm": "Heat in the Heart / upper jiao",
            "plain": "The tip of the tongue is noticeably redder than the rest — in this tradition linked to "
                     "heat in the upper body / Heart (often stress, poor sleep, or restlessness).",
            # localized heat sign -> small votes toward the heat patterns, weighted by strength
            "points_to": {"yin_deficiency": 0.4 * sev, "damp_heat": 0.3 * sev},
        })
    mo = z.get("moisture")
    if mo and mo.get("value") == "wet":
        sev = float(mo.get("severity", 0.0))
        out.append({
            "key": "moisture", "label": "Wet / glossy surface", "value": "wet",
            "severity": round(sev, 3), "rel": round(sev, 3), "band": "", "mentioned": True, "display": True,
            "tcm_term": "润/滑 (moist / slippery)", "tcm": "excess or untransformed fluids",
            "plain": "The tongue looks wet and glossy — in this tradition a sign of fluids not being fully "
                     "transformed (often linked to a cold/damp, low-warmth tendency).",
            # wet body -> fluids not moved: yang deficiency / dampness. (dry->yin is not measured; see zoning.py)
            "points_to": {"yang_deficiency": 0.4 * sev, "phlegm_dampness": 0.3 * sev},
        })
    return out


def feature_readings(chars, kb, stats):
    """Per-feature reading: degree band + dual-language + DISTINCTIVENESS-weighted pattern votes."""
    readings = []
    for ch, c in chars.items():
        if ch in COAT_AXES_DISPLAY:              # handled by coat_axis_readings (display-only)
            continue
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
        reading["display"] = ch != "coating"     # coating votes but is shown as the two axes instead
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


def _cond_ok(cond, present):
    """cond: {feature: value | [values]}. present: {feature_key: value}. All must match."""
    for k, v in cond.items():
        pv = present.get(k)
        if pv is None:
            return False
        if isinstance(v, list):
            if pv not in v:
                return False
        elif pv != v:
            return False
    return True


def present_features(stage1_output):
    """Flat {feature_key: value} of RAW detections for the combination rules — read from the model
    output directly (not the display-filtered readings), so a rule can key off a feature even when it
    wasn't distinctive enough to surface as its own card (e.g. swelling)."""
    p = {}
    for ch, c in (stage1_output.get("key_characteristics") or {}).items():
        p[ch] = c.get("value")                       # incl. coat_thickness / coat_texture in the real pipeline
    for feat, c in (stage1_output.get("extra_characteristics") or {}).items():
        if c.get("value") == "present":
            p[feat] = "present"
    z = stage1_output.get("zoned_analysis") or {}
    if (z.get("red_tip") or {}).get("value") == "present":
        p["red_tip"] = "present"
    if (z.get("moisture") or {}).get("value"):
        p["moisture"] = z["moisture"]["value"]
    return p


def apply_combination_rules(scores, present, kb):
    """Context layer on top of the additive vote. TCM signs are context-dependent — swelling means
    Yang-deficiency with a pale/wet tongue but Damp-Heat with a red/yellow one; a wet tongue argues
    AGAINST heat. Additive per-feature votes can't express this, so grounded co-occurrence rules
    (kb['combination_rules']) boost the contextually-correct pattern and suppress the wrong one. Each
    rule: {id, when:{feat:val}, any?:{feat:val}, boost:{pattern:delta}, cite, note}. `any` = at least
    one must match. Deltas are modest vs base weights; negatives clamp at 0."""
    fired = []
    for rule in kb.get("combination_rules", []):
        if not _cond_ok(rule.get("when", {}), present):
            continue
        anyc = rule.get("any")
        if anyc and not any(_cond_ok({k: v}, present) for k, v in anyc.items()):
            continue
        for p, d in rule.get("boost", {}).items():
            scores[p] = max(0.0, scores.get(p, 0.0) + d)
        fired.append(rule.get("id"))
    return fired


def vote_patterns(readings, kb, present=None, top_k=3):
    """Accumulate severity-weighted votes; map to ABSOLUTE confidence (saturating), so a weak top
    pattern reads as low confidence rather than a misleading 100%. Falls back to 'balanced' when
    nothing is notable. `present` = raw {feature: value} for the context (combination) rules."""
    scores = {}
    for r in readings:
        for p, w in r["points_to"].items():
            scores[p] = scores.get(p, 0.0) + w
    apply_combination_rules(scores, present or {}, kb)   # context layer (see docstring)
    scores.pop("balanced", None)
    top_raw = max(scores.values()) if scores else 0.0
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
    # saturating map: raw ~1.5 -> ~0.65, raw 3 -> ~0.8; weak 0.3 -> ~0.17
    out = [_card(kb, pid, s / (s + 1.2)) for pid, s in ranked]
    if top_raw < 0.55:                       # no strong pattern -> lead with the balanced picture
        out = [_card(kb, "balanced", 1.0 - top_raw / 0.55)] + out
    return out


def _synthesis(readings, patterns):
    if patterns and patterns[0]["id"] != "balanced":
        lead = patterns[0]
        txt = (f"Taken together, your signs lean toward **{lead['tcm_name']}** "
               f"— in plain terms, *{lead['plain_name']}*. {lead['explanation']}")
        syms = lead.get("associated_symptoms") or []
        if syms:
            txt += f" People with this tendency often notice {', '.join(s.lower() for s in syms[:4])}."
        if len(patterns) > 1 and patterns[1]["id"] != "balanced":
            txt += f" There are also lighter hints of {patterns[1]['plain_name']}."
        return txt
    return ("Your tongue looks close to the balanced picture — no strong traditional pattern stands "
            "out today. The notes below describe each sign on its own.")


def _headline(disp, patterns):
    """Lead insight: the single most distinctive sign + the overall lean, so the user gets the point in
    one line before the detail."""
    notable = [r for r in disp if r.get("mentioned")]
    top_pat = patterns[0] if patterns else None
    if not notable:
        return ("Your tongue looks close to a balanced picture today — no single sign stands out "
                "strongly. That's generally a good thing in this tradition.")
    top = max(notable, key=lambda r: float(r.get("rel", 0)))
    hook = _pctl_phrase(top["key"], float(top.get("rel", 0)))
    lead = f"The sign that stands out most is your **{top['label'].lower()}**" + (f" — {hook}." if hook else ".")
    if top_pat and top_pat["id"] != "balanced":
        lead += f" Overall, your tongue leans toward **{top_pat['plain_name']}**."
    return lead


def _confidence_note(disp):
    """Honest framing: name the reads that are less certain, and that this is a snapshot."""
    tent = sorted({r["label"].lower() for r in disp
                   if r.get("mentioned") and RELIABILITY.get(r["key"]) == "tentative"})
    note = ("This is a snapshot — a tongue can shift day to day, and this reflects one traditional "
            "framework, not a medical test.")
    if tent:
        note += (" A couple of reads are less certain and worth confirming: " + ", ".join(tent) +
                 ". The optional questions below help pin them down.")
    return note


def _markdown(readings, patterns, combined, sources, headline="", confidence_note=""):
    L = ([f"**At a glance:** {headline}", ""] if headline else [])
    L += ["**Your signs, one by one**"]
    normal = []
    for r in readings:
        deg = f"{r['band']} " if r["band"] and r["band"] != "none" else ""
        if r["key"] in GRADED_DISPLAY and not r["mentioned"]:
            normal.append(r["label"].lower())          # summarise the un-notable ones in one line
            continue
        dist = f" _(more distinctive — {r['pctl_phrase']})_" if r.get("pctl_phrase") else ""
        tcm = f" *TCM:* {r['tcm']}." if r.get("tcm") else ""
        conf = f" _[{r['confidence']}]_" if r.get("confidence") else ""
        L.append(f"- **{deg}{r['tcm_term'] or r['value']}**{dist} —{tcm} *In plain terms:* {r['plain']}{conf}")
    if normal:
        L.append(f"- _Within a typical range:_ {', '.join(normal)}.")
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
    if confidence_note:
        L += ["", "**How sure is this?** " + confidence_note]
    L += ["", "**Grounding:** " + "; ".join(sources[:4]), "", "**Note**", DISCLAIMER]
    return "\n".join(L)


def fired_rule_notes(present, kb):
    """The combination rules that fire for THIS tongue, as grounded context for the LLM — this is the
    'why these signs together mean X' reasoning, already cited."""
    out = []
    for rule in kb.get("combination_rules", []):
        if not _cond_ok(rule.get("when", {}), present):
            continue
        anyc = rule.get("any")
        if anyc and not any(_cond_ok({k: v}, present) for k, v in anyc.items()):
            continue
        if rule.get("note"):
            out.append(rule["note"])
    return out


def _llm_narrative(readings, patterns, sources, llm, present=None, kb=None):
    """RAG-style GROUNDED narrative: the rule engine already decided the patterns + confidence; the LLM
    only re-expresses the SUPPLIED grounding (detected signs + their reliability/distinctiveness + the
    combination reasoning + the pattern knowledge) into a nuanced, warm read. It is forbidden from adding
    signs/patterns/claims or diagnosing. Backbone stays deterministic + testable; this is the language."""
    notable = [r for r in readings if r.get("mentioned")]
    grounding = {
        "detected_signs": [{"sign": r["label"], "value": r.get("tcm_term") or r.get("value"),
                            "degree": r.get("band") or "", "distinctiveness": r.get("pctl_phrase") or "",
                            "reliability": r.get("confidence") or "reliable", "plain_meaning": r["plain"]}
                           for r in notable],
        "why_together": fired_rule_notes(present or {}, kb or {}),   # the cited combination reasoning
        # when 'balanced' leads, the tongue is near-normal -> DON'T feed weak secondary patterns (the LLM
        # would over-narrate them); leave leaning_pattern empty so it honestly says "balanced".
        "leaning_pattern": [] if (patterns and patterns[0]["id"] == "balanced") else
                            [{"name": p["plain_name"], "tcm_name": p["tcm_name"],
                              "confidence_pct": round(p["confidence"] * 100),
                              "explanation": p["explanation"], "often_noticed": p.get("associated_symptoms", []),
                              "modern_view": p.get("modern_correlation", ""),
                              "wellness_notes": p.get("recommendations", {})} for p in patterns[:2]
                             if p["id"] != "balanced"],
        "sources": sources,
    }
    system = (
        "You are a careful traditional-Chinese-medicine WELLNESS EDUCATOR writing for a layperson about "
        "their tongue photo. You will be given GROUNDING (the signs a vision model detected, how reliable "
        "and distinctive each is, the cited reasoning for how they combine, and the leaning pattern the "
        "system computed). RULES YOU MUST FOLLOW:\n"
        "1. Use ONLY facts in the grounding. NEVER add a sign, pattern, symptom, cause, or number that is "
        "not there. If unsure, say less.\n"
        "2. This is NOT a medical diagnosis. Never name a disease or tell them what they 'have'. Frame "
        "everything as 'in this tradition, ... is associated with ...'.\n"
        "3. HEDGE on any sign whose reliability says 'less certain', and invite them to confirm with the "
        "follow-up questions.\n"
        "4. Be warm, specific and concise — connect the signs to everyday experience the grounding lists, "
        "and weave them into ONE story rather than a flat list.\n"
        "5. The headline and read must reflect the leaning_pattern given. If leaning_pattern is EMPTY, the "
        "tongue looks balanced — say that plainly and do NOT invent a pattern. Draw wellness suggestions "
        "only from the wellness_notes provided. 'distinctiveness' is how this sign compares to other "
        "people's tongues, not a location on the tongue.")
    user = ("GROUNDING (the only facts you may use):\n" + json.dumps(grounding, ensure_ascii=False, indent=2) +
            "\n\nWrite ~150-230 words of Markdown: a one-line headline reflecting the leaning pattern (or "
            "'a balanced picture' if none), then the woven read, then a short 'in this tradition you might "
            "try' note drawn only from wellness_notes. Do NOT restate the rules or JSON. End with one line "
            "reminding this is educational, not a diagnosis.")
    text = llm.chat(system, user, max_tokens=700)
    return (text.strip() + "\n\n**Note**\n" + DISCLAIMER) if text else None


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
                + coat_axis_readings(chars)
                + zoned_readings(stage1_output)
                + extra_readings(stage1_output.get("extra_characteristics", {}), kb, stats))
    patterns = vote_patterns(readings, kb, present_features(stage1_output))   # + context rules
    disp = [r for r in readings if r.get("display", True)]   # display hides the conflated coating
    for r in disp:                                           # per-sign distinctiveness + honesty tags
        rel = r.get("rel")
        r["pct"] = round(float(rel) * 100) if (rel is not None and r["key"] not in ("tai", "zhi")) else None
        r["pctl_phrase"] = _pctl_phrase(r["key"], float(rel) if rel is not None else None)
        r["confidence"] = CONF_WORD.get(RELIABILITY.get(r["key"], "moderate"), "")
    headline = _headline(disp, patterns)
    confidence_note = _confidence_note(disp)
    combined = _synthesis(disp, patterns)
    sources, overview = kb["sources"], kb["overview"]

    llm = llm or LLMClient()
    report = (_llm_narrative(disp, patterns, sources, llm, present_features(stage1_output), kb)
              if llm.enabled else None) \
        or _markdown(disp, patterns, combined, sources, headline, confidence_note)

    return {
        "overview": overview, "headline": headline, "confidence_note": confidence_note,
        "features": disp, "patterns": patterns,
        "card": build_card(disp, patterns),
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
