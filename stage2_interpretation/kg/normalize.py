"""Deterministic normalization of extracted triplets -> our canonical vocabulary (WS-A micro layer).

The LLM extractor returns feature/pattern strings that are faithful to the book but not always in our
keys — especially Gerlach's Latin guiding-criteria terms (calor/algor/humor/pituita/xue). Those are
STABLE terminology, so we map them deterministically here rather than relying on model power. Each
triplet is resolved to one of three statuses:

  mapped     -> a canonical (feature[/value], pattern) our pipeline already knows -> becomes a graph edge
  candidate  -> a real TCM concept our FEATURE DETECTOR cannot observe (sublingual veins, white gums,
                jing) or a pattern outside our set -> kept for ontology review, NOT merged as an edge
  junk       -> not a real sign/pattern (other:<non-physiological>, free-text blobs) -> dropped

This keeps the graph clean (only observable, canonical edges) while preserving everything else for the
WHO-ontology-spine expansion step. Maps are data, edit freely.
"""
import re

# --- pattern aliases: Gerlach guiding-criteria / pathogenic terms -> our pattern keys -------------
# (imperfect where our constitution set has no exact node — flagged in NOTE comments.)
PATTERN_ALIASES = {
    "algor": "yang_deficiency",              # cold (han)
    "bagang algor": "yang_deficiency",
    "cold": "yang_deficiency",
    "humor": "phlegm_dampness",              # humidity (shi) / dampness
    "humidity": "phlegm_dampness",
    "shi": "phlegm_dampness",
    "pituita": "phlegm_dampness",            # phlegm (tan)
    "phlegm": "phlegm_dampness",
    "tan": "phlegm_dampness",
    "xue stasis": "blood_stasis",
    "xue (blood) stasis": "blood_stasis",
    "blood stasis": "blood_stasis",
    "xue deficiency": "blood_deficiency",
    "xue (blood) deficiency": "blood_deficiency",
    "blood deficiency": "blood_deficiency",
    "fluid deficiency": "yin_deficiency",    # body-fluid (jinye) depletion -> nearest is Yin def
    "fluid": "yin_deficiency",
    "yin deficiency": "yin_deficiency",
    "calor": "damp_heat",                    # NOTE: heat (re) -> our only heat-bearing node is damp_heat
    "reactive calor": "damp_heat",
    "heat": "damp_heat",
    "calor (heat, re) and xue (blood) deficiency": "blood_deficiency",  # dominant term
}

# --- feature aliases: free-text sign -> (canonical feature key, canonical value) ------------------
FEATURE_ALIASES = {
    "pale tongue body": ("zhi", "light"),
    "pale": ("zhi", "light"),
    "red tongue body": ("red_tongue", "present"),
    "red_tongue": ("red_tongue", "present"),
    "over-reddened": ("red_tongue", "present"),
    "blue tongue body": ("purple_body", "present"),
    "blue": ("purple_body", "present"),
    "livid discolorations": ("purple_body", "present"),
    "lividity": ("purple_body", "present"),
    "large tongue body": ("swollen", "present"),
    "edematous swelling": ("swollen", "present"),
    "plump swelling": ("swollen", "present"),
    "swollen": ("swollen", "present"),
    "small tongue body": ("thin", "present"),
    "thin": ("thin", "present"),
    "fissures": ("fissure", "present"),
    "fissures / cracks": ("fissure", "present"),
    "cracks": ("fissure", "present"),
    "dry surface": ("fissure", "present"),   # Gerlach groups dry+fissures; our nearest is fissure
    "white sticky coating": ("coating", "present"),
    "white sticky coatings": ("coating", "present"),
    "white, sticky coating": ("coating", "present"),
    "yellow sticky coating": ("coating", "present"),
    "yellow sticky coatings": ("coating", "present"),
    "yellow, sticky coating": ("coating", "present"),
    "yellow": ("tai", "yellow"),
    "sticky coating": ("coating", "present"),
    "gray/brown/black coating": ("black_coating", "present"),
    "gray, brown or black coating": ("black_coating", "present"),
    "black coating": ("black_coating", "present"),
    "peeled_coating": ("peeled_coating", "present"),
    "slippery_coating": ("slippery_coating", "present"),
    "tooth_mk": ("tooth_mk", "present"),
    "tooth impressions": ("tooth_mk", "present"),
    "red_dots": ("red_dots", "present"),
}

# real TCM signs our single-photo detector cannot observe -> candidate (keep for review, don't merge)
UNOBSERVABLE = {"sublingual veins", "white gums", "hyaline spots", "jing", "dry_tongue",
                "rough_surface", "moist surface", "wadi pattern", "tiger pattern", "dry leather pattern"}


def _clean(s):
    s = s or ""
    s = re.sub(r"^\s*other\s*:\s*", "", s, flags=re.I)     # strip a stray "other:" the model prepends
    s = s.strip().strip("<>")                               # "other:<yin_deficiency>" -> "yin_deficiency"
    s = re.sub(r"\s*\(.*?\)\s*", " ", s)                    # drop parentheticals: "red_tongue (red tongue body)"
    return re.sub(r"\s+", " ", s).strip().lower().strip(".,:;")


# markers the model emits for "this isn't a real sign/pattern" -> drop
JUNK_MARKERS = ("non-physiolog", "pathological embedding", "non physiolog", "n/a", "none")


def _is_junk(c):
    return (not c) or any(m in c for m in JUNK_MARKERS)


def resolve_pattern(raw, canonical_patterns):
    c = _clean(raw)                              # _clean already strips a stray "other:" prefix
    if _is_junk(c):
        return None, "junk"
    if c in canonical_patterns:
        return c, "mapped"
    if c in PATTERN_ALIASES:
        return PATTERN_ALIASES[c], "mapped"
    # substring fallback for compound phrases ("... and xue (blood) deficiency")
    for k, v in PATTERN_ALIASES.items():
        if k in c:
            return v, "mapped"
    return c, "candidate"


def resolve_feature(raw, value, canonical_features):
    c = _clean(raw)
    if _is_junk(c):
        return None, None, "junk"
    if c in canonical_features:
        return c, (value or "present"), "mapped"
    if c in FEATURE_ALIASES:
        fk, fv = FEATURE_ALIASES[c]
        return fk, fv, "mapped"
    for u in UNOBSERVABLE:
        if u in c:
            return c, value, "candidate"
    if len(c) > 45:                              # free-text blob, not a sign
        return None, None, "junk"
    return c, value, "candidate"


def normalize_triplet(t, canonical_features, canonical_patterns):
    """-> dict with resolved feature/value/pattern + overall status (mapped|candidate|junk)."""
    fk, fv, fstat = resolve_feature(t["feature"], t.get("value"), canonical_features)
    pk, pstat = resolve_pattern(t["pattern"], canonical_patterns)
    if fstat == "junk" or pstat == "junk":
        status = "junk"
    elif fstat == "candidate" or pstat == "candidate":
        status = "candidate"
    else:
        status = "mapped"
    return {"feature": fk, "value": fv, "pattern": pk, "polarity": t.get("polarity", "supports"),
            "context": t.get("context"), "snippet": t.get("snippet"),
            "raw_feature": t["feature"], "raw_pattern": t["pattern"], "status": status}


def normalize_records(records, canonical_features, canonical_patterns):
    out = {"mapped": [], "candidate": [], "junk": []}
    for r in records:
        for t in r["triplets"]:
            nt = normalize_triplet(t, canonical_features, canonical_patterns)
            nt["section"] = r["section"]
            out[nt["status"]].append(nt)
    return out
