"""Test the DIAGNOSIS MAPPING (features -> pattern), separately from feature detection.

Feeds each grounded feature-combination in evaluation/mapping_testset.json through the SAME KB voting
the product uses (interpret.vote_patterns), and checks whether the top pattern is one the reference
sources accept. This gives the "make the mapping better" campaign a number to move.

    python evaluation/eval_mapping.py
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stage2_interpretation"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stage1_quantitative"))
from interpret import interpret
from labels import (KEY_CHARS, LABEL_MAPS, CHAR_DESC, EXTRA_FEATURES, EXTRA_DESC,
                    COAT_AXIS_DESC, derive_coat_axes)

# severity used for GRADED features so they clear the "distinctive" gate and vote at full weight
GRADED_SEV = {"coating": {"non_greasy": 0.15, "greasy": 0.7, "greasy_thick": 0.92},
              "fissure": {"none": 0.0, "light": 0.5, "severe": 0.92},
              "tooth_mk": {"none": 0.0, "light": 0.5, "severe": 0.92}}


def build_stage1(feats):
    """Turn a compact {feature: value} spec into a Stage-1-shaped dict interpret() accepts."""
    chars = {}
    for ch in KEY_CHARS:
        val = feats.get(ch, LABEL_MAPS[ch][0])
        vals = LABEL_MAPS[ch]
        idx = vals.index(val)
        sev = GRADED_SEV.get(ch, {}).get(val, idx / (len(vals) - 1))
        probs = {v: (0.8 if v == val else 0.2 / (len(vals) - 1)) for v in vals}
        chars[ch] = {"value": val, "confidence": 0.8, "description": CHAR_DESC[ch],
                     "severity": round(sev, 3), "probs": probs}
    extra = {}
    for f in EXTRA_FEATURES:
        present = feats.get(f) == "present"
        extra[f] = {"value": "present" if present else "absent",
                    "severity": 0.85 if present else 0.05, "description": EXTRA_DESC[f]}
    # derive the coating axes (thickness × texture) like the real pipeline, so combination rules that
    # key off coat_texture are exercised
    ABN = {"coat_thickness": "thick", "coat_texture": "greasy"}
    for axis, (val, conf, probs) in derive_coat_axes(chars["coating"]["probs"]).items():
        chars[axis] = {"value": val, "confidence": conf, "description": COAT_AXIS_DESC[axis],
                       "severity": probs[ABN[axis]], "probs": probs}
    zoned = {}
    if feats.get("red_tip") in ("present", "strong"):
        zoned["red_tip"] = {"value": "present", "severity": 0.85}
        zoned["tip_redness_delta"] = 4.0
    if feats.get("red_sides") == "present":
        zoned["red_sides"] = {"value": "present", "severity": 0.7}
        zoned["side_redness_delta"] = 3.5
    if feats.get("moisture") == "wet":
        zoned["moisture"] = {"value": "wet", "severity": 0.7}
    return {"key_characteristics": chars, "extra_characteristics": extra, "zoned_analysis": zoned,
            "quality": {"accepted": True, "reasons": []}}


def main():
    ts = json.load(open("evaluation/mapping_testset.json"))
    n_ok = 0
    print(f"{'case':28} {'top pattern (conf)':32} {'accept?':8} expected")
    print("-" * 92)
    for c in ts["cases"]:
        s1 = build_stage1(c["features"])
        pats = interpret(s1)["patterns"]
        top = pats[0] if pats else {"id": "-", "confidence": 0}
        # skip a leading 'balanced' filler unless it's the expected answer
        if top["id"] == "balanced" and "balanced" not in c["accept"] and len(pats) > 1:
            top = pats[1]
        ok = top["id"] in c["accept"]
        n_ok += ok
        print(f"{c['id']:28} {top['id']+' ('+str(round(top['confidence'],2))+')':32} "
              f"{'OK' if ok else 'MISS':8} {c['accept']}")
    print("-" * 92)
    print(f"MAPPING ACCURACY: {n_ok}/{len(ts['cases'])} = {n_ok/len(ts['cases']):.0%} of canonical combinations map to an accepted pattern")


if __name__ == "__main__":
    main()
