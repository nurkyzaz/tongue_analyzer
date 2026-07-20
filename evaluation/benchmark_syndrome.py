"""Benchmark our KNOWLEDGE-BASE reasoning against expert syndrome differentiation (TCMEval-SDT).

Public data: TCMEval-SDT (Nature Sci Data 2025, CC-BY 4.0) — 200 train cases, 83% with an explicit
tongue description (舌…) + a gold expert syndrome (multiple-choice). We:
  1. parse the Chinese tongue text of each case -> our feature values,
  2. run our KB feature->pattern mapping,
  3. reduce both our pattern and the gold syndrome to a coarse TCM axis (Heat/Cold/Damp/Qi-def/
     Blood-def/Yin-def/Stasis),
  4. measure CONSISTENCY (does our tongue-derived axis match the expert's?).

Honest framing: the tongue alone does NOT determine a full syndrome (experts also use pulse +
symptoms), so we report *directional consistency*, not exact syndrome accuracy. This validates
whether our KB reasons in the right direction on real expert cases — a check on the KB, not the
image model.

    python evaluation/benchmark_syndrome.py --data data/external/TCMEval/evaluation/TCMEval-SDT/data/Train_TCM_Data_v1.json
"""
import argparse, json, os, re, sys, collections

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stage2_interpretation"))
from interpret import load_kb

# --- Tongue-clause scoping ---------------------------------------------------
# TCM cases read "…舌{body}{coating}，{symptoms}，脉{pulse}". We must isolate the tongue clause so
# stray colour characters in later symptom text (e.g. 小便赤 dark urine, 面色苍白 pale face, 大便黄)
# don't leak into the tongue reading. Tongue vocabulary is a bounded charset, so we scan forward from
# the first 舌 and stop at the first character that is not a tongue term or separator — which is
# exactly where the tongue clause ends and symptoms/pulse begin. Validated on all 167 tongue cases.
TONGUE_CHARS = set("舌苔质体尖边根中心"          # structure (note: 面/大/口… deliberately excluded)
                   "红绛紫淡白黄灰黑青赤"        # colours
                   "薄厚腻滑润干燥糙浊垢洁少无剥"  # coating texture / quantity
                   "胖瘦嫩老"                    # shape
                   "齿痕印裂芒刺点斑瘀暗"         # marks
                   "微稍略深浅有起")             # modifiers
SEP_CHARS = set("、，, ·　")


def tongue_clause(text):
    i = text.find("舌")
    if i < 0:
        return ""
    out = []
    for ch in text[i:]:
        if ch in TONGUE_CHARS or ch in SEP_CHARS:
            out.append(ch)
        else:
            break
    return "".join(out)


# --- coating lexicon -> our feature values (order: most specific first) ---
COAT_THICK = [("厚腻", ("coating", "greasy_thick")), ("厚", ("coating", "greasy_thick")),
              ("腻", ("coating", "greasy")), ("薄", ("coating", "non_greasy"))]
# extra (multi-label) pathological features; body colour + red/purple are handled in parse_tongue
EXTRA = [("裂", ("fissure", "severe")), ("齿痕", ("tooth_mk", "severe")), ("齿印", ("tooth_mk", "severe")),
         ("胖", ("swollen", 1)), ("瘦", ("thin", 1)), ("芒刺", ("red_dots", 1)), ("红点", ("red_dots", 1)),
         ("剥", ("peeled_coating", 1)), ("无苔", ("peeled_coating", 1)),
         ("少", ("peeled_coating", 1))]   # 少苔/少津 (scanty coating) — same Yin-depletion direction as peeled

# --- coarse axes ---
AXES = ["Heat", "Cold", "Damp", "QiDef", "BloodDef", "YinDef", "Stasis"]
PATTERN_AXIS = {
    "damp_heat": {"Damp", "Heat"}, "phlegm_dampness": {"Damp"}, "spleen_qi_deficiency": {"QiDef"},
    "blood_deficiency": {"BloodDef"}, "yin_deficiency": {"YinDef", "Heat"},
    "yang_deficiency": {"Cold"}, "blood_stasis": {"Stasis"}, "balanced": set(),
    # 气滞 / 特禀 have no clean thermal/deficiency axis and the tongue rarely determines them -> no axis
    "qi_stagnation": set(), "special_diathesis": set(),
}
# gold syndrome (Chinese) keyword -> axis
GOLD_AXIS = [("阳虚", "Cold"), ("寒", "Cold"), ("湿热", "Damp"), ("痰", "Damp"), ("湿", "Damp"),
             ("血瘀", "Stasis"), ("瘀", "Stasis"), ("阴虚", "YinDef"), ("阴亏", "YinDef"),
             ("血虚", "BloodDef"), ("血燥", "BloodDef"), ("气虚", "QiDef"), ("脾虚", "QiDef"),
             ("脾胃", "QiDef"), ("热", "Heat"), ("火", "Heat"), ("暑", "Heat")]


def parse_tongue(text):
    t = tongue_clause(text)
    chars, extra = {}, {}

    # --- body colour (zhi) + the pathological red/purple detectors it triggers ---
    # Fidelity note: the product runs BOTH a 3-class body-colour head (light/regular/dark) AND
    # dedicated red_tongue / purple_body detectors. A red tongue must fire red_tongue (Heat), else the
    # coarse "dark" bucket — which the KB weights toward blood_stasis — misreads 血热 red tongues as
    # stasis. Likewise 淡红 is the NORMAL colour, not "pale", so it must not trigger blood-deficiency.
    zhi = None
    if "紫" in t or "暗" in t:                     # purple / dusky -> stasis
        zhi = "dark"
        extra["purple_body"] = 1
    t_nored = t.replace("淡红", "").replace("淡紅", "")   # strip the normal light-red before testing red
    if "绛" in t or "赤" in t or "红" in t_nored or "紅" in t_nored:
        extra["red_tongue"] = 1                   # genuine red -> Heat
        if zhi is None:
            zhi = "dark"
    if zhi is None:
        if "淡红" in t or "淡紅" in t:
            zhi = "regular"                        # 淡红 = normal, healthy tongue colour
        elif "淡" in t:
            zhi = "light"                          # pale -> blood/qi/yang deficiency
    if zhi:
        chars["zhi"] = zhi

    # --- coating colour (tai): 黄 -> yellow, 白 -> white (exclude 淡白 pale-body from the white test) ---
    if "黄" in t:
        chars["tai"] = "yellow"
    elif "白" in t.replace("淡白", ""):
        chars["tai"] = "white"

    # --- coating thickness / greasiness + extra multi-label features ---
    for lex, (feat, val) in COAT_THICK:
        if lex in t:
            chars.setdefault("coating", val)
    for lex, (feat, val) in EXTRA:
        if lex in t:
            if feat in ("fissure", "tooth_mk"):
                chars[feat] = val                 # core graded feature
            else:
                extra[feat] = 1                   # extra multi-label feature
    return chars, extra


def kb_pattern(chars, extra, kb):
    scores = collections.Counter()
    for ch, val in chars.items():
        spec = kb["features"].get(ch, {})
        pts = {}
        if ch in ("coating", "fissure", "tooth_mk"):
            if val not in ("non_greasy", "none"):
                pts = spec.get("points_to", {})
        else:
            pts = spec.get("values", {}).get(val, {}).get("points_to", {})
        for p, w in pts.items():
            scores[p] += w
    for feat in extra:
        pts = kb.get("extra_features", {}).get(feat, {}).get("points_to", {})
        for p, w in pts.items():
            scores[p] += w * 0.6
    scores.pop("balanced", None)
    return scores.most_common(1)[0][0] if scores else "balanced"


def _present(chars, extra):
    """Build the matcher's {feature: value} view from the parsed tongue clause."""
    present = dict(chars)
    for feat in extra:
        present[feat] = "present"
    return present


def ensemble_pattern(chars, extra, kb, matcher, alpha):
    """Top pattern via the WS-C ensemble (rule prior + cited matcher, weight alpha) on the SAME parsed
    features — so rule vs matcher-forward is apples-to-apples against the gold syndrome. alpha=1.0 ~
    matcher-forward (rule fills only the patterns the matcher abstains on)."""
    import interpret
    from kg.matcher import _synth_readings
    from kg.ensemble import ensemble_cards
    present = _present(chars, extra)
    rule_cards = interpret.vote_patterns(_synth_readings(present, kb), kb, present)
    mo = matcher.match(present)
    cards, _ = ensemble_cards(rule_cards, mo, make_card=lambda pid, c: interpret._card(kb, pid, c),
                              alpha=alpha)
    cards = [c for c in cards if c["id"] != "balanced"]
    return cards[0]["id"] if cards else "balanced"


def gold_axes(options, answer):
    axes = set()
    for opt in answer.replace("；", ";").split(";"):
        opt = opt.strip()
        m = re.search(rf"{re.escape(opt)}:([^;]+)", options)
        txt = m.group(1) if m else opt
        for kw, ax in GOLD_AXIS:
            if kw in txt:
                axes.add(ax)
    return axes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--engine", choices=["rule", "ensemble"], default="rule",
                    help="rule = KB vote (the 69.7% baseline); ensemble = WS-C rule+matcher blend")
    ap.add_argument("--alpha", type=float, default=0.2, help="ensemble matcher weight (1.0 = matcher-forward)")
    args = ap.parse_args()
    kb = load_kb()
    matcher = None
    if args.engine == "ensemble":
        from kg.matcher import GroundedMatcher
        matcher = GroundedMatcher()
        assert matcher.llm.enabled, "ensemble engine needs the LLM env (run on casper)"
    cases = json.load(open(args.data, encoding="utf-8"))
    n = hit = 0
    by_gold = collections.Counter(); by_hit = collections.Counter()
    for c in cases:
        data = c.get("Clinical Data", "")
        ans = c.get("Answers of TCM Syndrome", "").strip()
        opts = c.get("Options of TCM Syndrome", "")
        if "舌" not in data or not ans:
            continue
        chars, extra = parse_tongue(data)
        if not chars and not extra:
            continue
        pat = (ensemble_pattern(chars, extra, kb, matcher, args.alpha)
               if args.engine == "ensemble" else kb_pattern(chars, extra, kb))
        our_ax = PATTERN_AXIS.get(pat, set())
        gax = gold_axes(opts, ans)
        if not gax:
            continue
        n += 1
        consistent = bool(our_ax & gax)
        hit += consistent
        for a in gax:
            by_gold[a] += 1; by_hit[a] += consistent
    tag = "engine=%s" % args.engine + (" alpha=%s" % args.alpha if args.engine == "ensemble" else "")
    print(f"KB-reasoning vs expert syndrome (TCMEval-SDT, tongue-bearing cases): n={n}  [{tag}]")
    print(f"Directional consistency (our tongue-axis ∈ expert syndrome axes): {hit}/{n} = {hit/max(n,1):.1%}")
    print("Per expert-axis recall:")
    for a in AXES:
        if by_gold[a]:
            print(f"  {a:9s} {by_hit[a]}/{by_gold[a]} = {by_hit[a]/by_gold[a]:.0%}")


if __name__ == "__main__":
    main()
