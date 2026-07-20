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

_RETRIEVER = None
_MATCHER = None

# WS-C ensemble: layer the grounded cite-or-abstain matcher (cited book evidence + honest confidence)
# on top of the rule-engine prior. PROMOTED default-ON (2026-07-16) — the α=0.2 sweep cleared the WS-D
# faithfulness gate at ~the rule-only baseline (0.929 vs 0.936) while keeping the book grounding
# (lead-cited 0.90) and 0 hallucination. Set TIH_WSC_ENSEMBLE=0 to fall back to the pure rule ranker.
WSC_ENSEMBLE = os.getenv("TIH_WSC_ENSEMBLE", "1") == "1"


def _matcher():
    """Lazy singleton grounded matcher; None if the graph/LLM aren't available so callers degrade."""
    global _MATCHER
    if _MATCHER is None:
        try:
            from kg.matcher import GroundedMatcher
            _MATCHER = GroundedMatcher()
        except Exception:
            _MATCHER = False
    return _MATCHER or None


_GRAPH = None


def _graph():
    """Lazy singleton knowledge graph (for WS-B info-gain question selection); None if unavailable."""
    global _GRAPH
    if _GRAPH is None:
        try:
            from kg.graph import KnowledgeGraph
            g = json.load(open(os.path.join(os.path.dirname(__file__), "knowledge_base", "kg_graph.json")))
            _GRAPH = KnowledgeGraph(nodes=g["nodes"], edges=g["edges"], rules=g.get("rules"),
                                    snippets=g.get("snippets"), meta=g.get("_meta"))
        except Exception:
            _GRAPH = False
    return _GRAPH or None


def _followup_block(patterns):
    """WS-B pass-1: the questions that best DISAMBIGUATE the top-2 candidates (information gain over the
    KG's `probes` edges), plus the two candidates being separated. Falls back to the lead pattern's
    fixed KB question list if the graph is unavailable."""
    top = [p for p in patterns if p.get("id") != "balanced"]
    if not top:
        return []
    lead = top[0]
    questions = []
    g = _graph()
    if g is not None:
        try:
            from kg.refine_engine import select_questions
            questions = select_questions(g, patterns, k=3)
        except Exception:
            questions = []
    if not questions:                       # degraded: the lead pattern's own fixed questions
        questions = [dict(q, target_pattern=lead["id"], target_name=lead.get("plain_name", ""))
                     for q in lead.get("followup_questions", [])]
    if not questions:
        return []
    return [{"pattern_id": lead["id"], "pattern": lead.get("tcm_name", ""),
             "plain_name": lead.get("plain_name", ""), "base_confidence": lead.get("confidence", 0.0),
             "questions": questions,
             "candidates": [{"id": p["id"], "plain_name": p.get("plain_name", ""),
                             "confidence": p.get("confidence", 0.0)} for p in top[:2]]}]


def _retriever():
    """Lazy singleton RAG retriever; None if unavailable (no index / import fails) so callers degrade."""
    global _RETRIEVER
    if _RETRIEVER is None:
        try:
            from rag import Retriever
            _RETRIEVER = Retriever()
        except Exception:
            _RETRIEVER = False
    return _RETRIEVER or None

KB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base", "tcm_knowledge.json")
# TIH_KB_VERSION=v2 loads the book-recalibrated KB (kg/recalibrate.py); default v1 = hand-tuned. Instant
# rollback by unsetting. Only tcm_knowledge_v2.json's feature->pattern weights differ from v1.
KB_VERSION = os.getenv("TIH_KB_VERSION", "v1")
if KB_VERSION == "v2":
    _v2 = os.path.join(os.path.dirname(__file__), "knowledge_base", "tcm_knowledge_v2.json")
    if os.path.exists(_v2):
        KB_PATH = _v2
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
        kb = json.load(f)
    # feature->symptom map lives in a side file (merge it in); tolerate absence
    fs = os.path.join(os.path.dirname(path), "feature_symptoms.json")
    if "feature_symptoms" not in kb and os.path.exists(fs):
        with open(fs, encoding="utf-8") as f:
            kb["feature_symptoms"] = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    return kb


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
            "confidence": round(conf, 3),
            "confidence_pct": round(conf * 100)}   # WS-C step 4: honest raw number, not just a word


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


def _apply_wsc_ensemble(patterns, present, kb):
    """WS-C: blend the grounded matcher (cited book evidence) onto the rule prior. Fully degrading —
    any failure (no graph, LLM down, exception) returns the rule cards untouched, so serving never
    breaks. The matcher itself falls back to the graph-RAG ranking when no LLM is configured, so the
    ensemble still attaches book citations even offline."""
    try:
        m = _matcher()
        if not m:
            return patterns, {"applied": False, "reason": "matcher unavailable"}
        from kg.ensemble import ensemble_cards
        matcher_out = m.match(present)
        cards, meta = ensemble_cards(patterns, matcher_out,
                                     make_card=lambda pid, c: _card(kb, pid, c))
        meta["matcher_dropped"] = len(matcher_out.get("dropped", []))
        return cards, meta
    except Exception as e:
        return patterns, {"applied": False, "reason": "error: %s" % type(e).__name__}


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


# ---- salient findings (S1/S2): the present, notable signs in plain words, ranked by prominence ----
_PHRASE = {
    ("zhi", "dark"): "a red or dark body", ("zhi", "light"): "a pale body",
    ("tai", "yellow"): "a yellow coating", ("tai", "light_yellow"): "a slightly yellow coating",
    ("coat_texture", "greasy"): "a greasy coating", ("coat_thickness", "thick"): "a thick coating",
    ("red_tip", "present"): "a redder tip", ("moisture", "wet"): "a wet, glossy surface",
}
_PHRASE_KEY = {
    "fissure": "cracks", "tooth_mk": "tooth-marked (scalloped) edges", "red_dots": "red spots",
    "swollen": "a swollen body", "peeled_coating": "a peeled/patchy coating",
    "purple_body": "a purple/dusky body", "thin": "a thin body", "red_tongue": "a red body",
    "slippery_coating": "a wet/slippery coating", "black_coating": "a grey/dark coating",
}
# concrete, plain, finding-specific actions (over-ride the generic pattern recs)
_ACTION = {
    ("zhi", "light"): "favour iron-rich, blood-nourishing foods (red meat, dark leafy greens, beetroot) and get enough sleep",
    ("zhi", "dark"): "favour cooling foods (cucumber, mung bean, pear) and ease off spicy/fried food, alcohol and late nights",
    ("coat_texture", "greasy"): "keep meals lighter, warm and well-cooked; cut back on greasy, sweet, cold and raw food",
    ("coat_thickness", "thick"): "eat lighter and don't overeat; add gentle daily movement to help clear the heaviness",
    ("tai", "yellow"): "favour cooling, light foods and cut back on alcohol, fried food and late nights",
    ("fissure", "present"): "drink more water and prioritise rest — this tradition links cracks to dryness/depletion",
    ("tooth_mk", "present"): "favour warm, well-cooked, easy-to-digest meals and avoid overeating",
    ("red_tip", "present"): "wind down before bed and manage stress — the tip links to the Heart/upper body",
    ("moisture", "wet"): "favour warming, well-cooked foods and go easy on cold/raw food and iced drinks",
}


def _is_notable(r):
    if r["key"] in ("tai", "zhi"):
        return r["value"] not in ("white", "regular")     # non-default colour = worth surfacing
    return bool(r.get("mentioned"))


def _finding_phrase(r, coat_crack_caveat=False):
    k, v = r["key"], r["value"]
    if (k, v) in _PHRASE:
        return _PHRASE[(k, v)]
    if k == "fissure":
        return "cracks (possibly in the thick coating)" if coat_crack_caveat else "cracks"
    return _PHRASE_KEY.get(k)


# rank signs by prominence WEIGHTED by how much we trust the detector, so reliable signs (body colour,
# cracks) lead and the marginal ones (red_tip, moisture) sink rather than heading the description.
_REL_W = {"reliable": 1.0, "moderate": 0.6, "tentative": 0.45}
# clinical priority tier — the primary signs always lead the description; the marginal zoned signals
# (red_tip, moisture — known to be weak) come last, so they never head the findings.
_TIER = {"zhi": 0, "coat_texture": 0, "coat_thickness": 0, "tai": 0, "fissure": 0, "tooth_mk": 0,
         "red_tip": 2, "moisture": 2}   # everything else (extras) = tier 1


def build_findings(disp, present):
    """Ranked list of the notable PRESENT signs: {key,value,phrase,prominence,confidence}."""
    # cracks on a thick greasy coat may be coating cracks, not body fissures (t08 feedback)
    coat_crack = (present.get("coat_thickness") == "thick" and present.get("coat_texture") == "greasy")
    out = []
    for r in disp:
        r["notable"] = _is_notable(r)
        if not r["notable"]:
            continue
        ph = _finding_phrase(r, coat_crack and r["key"] == "fissure")
        if not ph:
            continue
        w = _REL_W.get(RELIABILITY.get(r["key"], "moderate"), 0.6)
        prom = float(r.get("rel", 0) or 0) * w
        out.append({"key": r["key"], "value": r["value"], "phrase": ph, "tier": _TIER.get(r["key"], 1),
                    "prominence": round(prom, 3), "confidence": r.get("confidence", "")})
    out.sort(key=lambda x: (x["tier"], -x["prominence"]))   # primary signs lead, marginal ones last
    return out


def findings_text(findings):
    if not findings:
        return "Your tongue looks close to a balanced picture — no single sign stands out today."
    ph = [f["phrase"] for f in findings[:4]]
    body = ph[0] if len(ph) == 1 else ", ".join(ph[:-1]) + " and " + ph[-1]
    return "Your tongue shows " + body + "."


def build_recommendation(findings, patterns, kb):
    """The Recommendation card: ranked findings -> likely condition(s) -> specific actions."""
    conditions = [{"name": p["plain_name"], "confidence": round(p["confidence"], 2)}
                  for p in patterns[:2] if p["id"] != "balanced"]
    actions, seen = [], set()
    for f in findings:
        if f["key"] == "fissure" and "possibly" in f["phrase"]:
            continue                                    # coating-crack caveat -> skip the dryness action
        key = (f["key"], "present" if f["value"] not in ("dark", "light", "yellow", "light_yellow",
                                                          "greasy", "thick", "wet") else f["value"])
        a = _ACTION.get((f["key"], f["value"])) or _ACTION.get(key)
        if a and a not in seen:
            actions.append(a); seen.add(a)
        if len(actions) >= 3:
            break
    if not actions and patterns and patterns[0]["id"] != "balanced":     # fall back to grounded pattern recs
        rec = patterns[0].get("recommendations", {})
        actions = (rec.get("diet", []) + rec.get("lifestyle", []))[:2]
    return {"findings": [f["phrase"] for f in findings[:3]], "conditions": conditions, "actions": actions}


def _symptom_key(f):
    """Normalise a finding to its feature_symptoms key."""
    k, v = f["key"], f["value"]
    if k in ("zhi", "tai", "coat_texture", "coat_thickness"):
        return f"{k}={v}"
    if k == "moisture":
        return "moisture=wet"
    return f"{k}=present"          # graded (fissure/tooth_mk) + extras


def build_symptoms(findings, patterns, kb):
    """DIRECT feature -> symptom mapping: each present sign contributes the symptoms it traditionally
    suggests, ATTRIBUTED to that sign (personalised & transparent, reflects THIS tongue). Supplemented
    from the lead pattern only if the feature layer is sparse. Keeps the pattern/condition output too."""
    fs = kb.get("feature_symptoms", {})
    by_sign, items, seen = [], [], set()
    for f in findings[:4]:            # top signs already ranked (primary signs lead)
        if f["key"] == "fissure" and "possibly" in f["phrase"]:
            continue                  # coating-crack caveat -> don't attribute dryness symptoms
        syms = fs.get(_symptom_key(f))
        if not syms:
            continue
        by_sign.append({"sign": f["phrase"], "symptoms": syms})
        for s in syms:
            if s.lower() not in seen:
                items.append(s); seen.add(s.lower())
    if len(items) < 4 and patterns and patterns[0]["id"] != "balanced":   # supplement if sparse
        for s in patterns[0].get("associated_symptoms", []):
            if s.lower() not in seen:
                items.append(s); seen.add(s.lower())
            if len(items) >= 6:
                break
    return {"by_sign": by_sign, "items": items[:10]}


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


def fired_rules_structured(present, kb):
    """The combination rules that FIRE for this tongue, with their note, citation and pattern boosts —
    the 'why these signs together mean X' reasoning, already cited. Basis for both the LLM grounding
    and the app's tap-to-see-why on each linkage card."""
    out = []
    for rule in kb.get("combination_rules", []):
        if not _cond_ok(rule.get("when", {}), present):
            continue
        anyc = rule.get("any")
        if anyc and not any(_cond_ok({k: v}, present) for k, v in anyc.items()):
            continue
        out.append({"id": rule.get("id"), "note": rule.get("note", ""),
                    "cite": rule.get("cite", ""), "boost": rule.get("boost", {})})
    return out


def fired_rule_notes(present, kb):
    """Back-compat: just the note strings (used as LLM grounding)."""
    return [r["note"] for r in fired_rules_structured(present, kb) if r["note"]]


def reasoning_for_pattern(pid, fired):
    """The cited combination-rule notes that positively SUPPORT this pattern — the 'why' the app shows
    when a linkage card is tapped."""
    return [{"note": r["note"], "cite": r["cite"]}
            for r in fired if r["note"] and r["boost"].get(pid, 0) > 0]


def _llm_narrative(readings, patterns, sources, llm, present=None, kb=None):
    """RAG-style GROUNDED narrative: the rule engine already decided the patterns + confidence; the LLM
    only re-expresses the SUPPLIED grounding (detected signs + their reliability/distinctiveness + the
    combination reasoning + the pattern knowledge) into a nuanced, warm read. It is forbidden from adding
    signs/patterns/claims or diagnosing. Backbone stays deterministic + testable; this is the language."""
    notable = [r for r in readings if r.get("mentioned")]
    # --- true RAG: retrieve cited knowledge for THIS tongue (disambiguation/nuance the structured facts
    #     don't carry), so the LLM reasons over combinations from sourced material ---
    reference_notes = []
    rtv = _retriever()
    if rtv is not None:
        lead = patterns[0]["plain_name"] if (patterns and patterns[0]["id"] != "balanced") else ""
        query = "; ".join(r.get("tcm_term") or r.get("value") or r["label"] for r in notable) \
            + (f"; leaning {lead}" if lead else "")
        reference_notes = [{"note": h["text"], "source": h["source"]}
                           for h in rtv.retrieve(query, k=5)]
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
        "reference_notes": reference_notes,     # retrieved, cited knowledge (RAG)
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
        "people's tongues, not a location on the tongue.\n"
        "6. Use 'reference_notes' (retrieved reference knowledge) to reason about how these particular "
        "signs combine and to tell apart similar-looking patterns — but still assert only what the "
        "grounding supports, and prefer the computed leaning_pattern.")
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
    patterns = vote_patterns(readings, kb, present_features(stage1_output))   # rule prior (+ context rules)
    ensemble_meta = None
    if WSC_ENSEMBLE:                                          # WS-C: blend grounded cited evidence on top
        patterns, ensemble_meta = _apply_wsc_ensemble(patterns, present_features(stage1_output), kb)
    disp = [r for r in readings if r.get("display", True)]   # display hides the conflated coating
    for r in disp:                                           # per-sign distinctiveness + honesty tags
        rel = r.get("rel")
        r["pct"] = round(float(rel) * 100) if (rel is not None and r["key"] not in ("tai", "zhi")) else None
        r["pctl_phrase"] = _pctl_phrase(r["key"], float(rel) if rel is not None else None)
        r["confidence"] = CONF_WORD.get(RELIABILITY.get(r["key"], "moderate"), "")
    headline = _headline(disp, patterns)
    confidence_note = _confidence_note(disp)
    combined = _synthesis(disp, patterns)
    present = present_features(stage1_output)
    fired = fired_rules_structured(present, kb)        # "show the reasoning": cited combination rules
    for p in patterns:                                 # attach each card's own why (rules supporting it)
        p["reasoning"] = reasoning_for_pattern(p["id"], fired)
    findings = build_findings(disp, present)          # sets r["notable"] on disp
    found_text = findings_text(findings)
    recommendation = build_recommendation(findings, patterns, kb)
    symptoms = build_symptoms(findings, patterns, kb)
    sources, overview = kb["sources"], kb["overview"]

    llm = llm or LLMClient()
    report = (_llm_narrative(disp, patterns, sources, llm, present_features(stage1_output), kb)
              if llm.enabled else None) \
        or _markdown(disp, patterns, combined, sources, headline, confidence_note)

    return {
        "overview": overview, "headline": headline, "confidence_note": confidence_note,
        "findings_text": found_text, "findings": findings, "recommendation": recommendation,
        "symptoms": symptoms,
        "features": disp, "patterns": patterns,
        "ensemble": ensemble_meta,          # WS-C transparency: how the blend ranked (None if off)
        "reasoning": [{"note": r["note"], "cite": r["cite"]} for r in fired if r["note"]],
        "regions": kb.get("regions", {}),
        "combined": combined, "sources": sources, "report": report, "disclaimer": DISCLAIMER,
        # WS-B pass-1 follow-up flow: info-gain questions that disambiguate the top-2 candidates
        "followup": _followup_block(patterns),
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
