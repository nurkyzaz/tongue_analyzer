"""Shared label vocabulary for the 5 key tongue characteristics (SSC-Net's clinical set).

Class order is fixed here so training and inference agree. `none` is index 0 where it exists
(absent = negative class). Ordinal characteristics (fissure, tooth_mk) go none < light < severe.
"""

KEY_CHARS = ["coating", "tai", "zhi", "fissure", "tooth_mk"]

LABEL_MAPS = {
    "coating":  ["non_greasy", "greasy", "greasy_thick"],   # coating thickness / greasiness
    "tai":      ["white", "light_yellow", "yellow"],         # coating color
    "zhi":      ["light", "regular", "dark"],                # body color
    "fissure":  ["none", "light", "severe"],                 # tongue cracks
    "tooth_mk": ["none", "light", "severe"],                 # tooth marks
}

# Human-readable descriptions for the Stage-1 JSON / report.
CHAR_DESC = {
    "coating":  "coating thickness (greasiness)",
    "tai":      "coating color",
    "zhi":      "tongue body color",
    "fissure":  "fissures / cracks",
    "tooth_mk": "tooth marks",
}

CLASS_TO_IDX = {c: {v: i for i, v in enumerate(vals)} for c, vals in LABEL_MAPS.items()}
NUM_CLASSES = {c: len(v) for c, v in LABEL_MAPS.items()}

# --- coating split into two independent axes -----------------------------------------------------
# The single `coating` label conflates two things TCM grades separately: THICKNESS (薄/厚 thin/thick)
# and TEXTURE (腻/滑 greasy/slippery vs smooth). The 3-way label is compositional, so we derive two
# faithful axes from the coating class probabilities (no new ground truth needed, no retrain):
#   thickness: thick <- greasy_thick        ;  thin  <- non_greasy + greasy
#   texture:   greasy <- greasy + greasy_thick;  smooth <- non_greasy
# This lets the demo report a confident thickness reading even when greasiness is ambiguous (the
# user's exact complaint: surface patterns make greasiness hard to judge).
COAT_AXES = {
    "coat_thickness": ["thin", "thick"],
    "coat_texture":   ["smooth", "greasy"],
}
COAT_AXIS_DESC = {
    "coat_thickness": "coating thickness (thin → thick)",
    "coat_texture":   "coating texture (smooth → greasy)",
}


def derive_coat_axes(coating_probs):
    """coating_probs: {non_greasy, greasy, greasy_thick} -> {axis: (value, prob_of_value, {cls:prob})}.
    prob_of_value is the probability mass on the CHOSEN class (>=0.5), usable as confidence/severity."""
    ng = coating_probs.get("non_greasy", 0.0)
    g = coating_probs.get("greasy", 0.0)
    gt = coating_probs.get("greasy_thick", 0.0)
    p_thick, p_greasy = gt, g + gt
    thick = p_thick > 0.5
    greasy = p_greasy > 0.5
    return {
        "coat_thickness": (("thick" if thick else "thin"),
                           (p_thick if thick else 1 - p_thick),
                           {"thin": 1 - p_thick, "thick": p_thick}),
        "coat_texture":   (("greasy" if greasy else "smooth"),
                           (p_greasy if greasy else 1 - p_greasy),
                           {"smooth": 1 - p_greasy, "greasy": p_greasy}),
    }

# Continuous severity regression targets (0-1), from TonguExpert phenotype measurements.
# These give a graded "degree" per feature (the sensitivity fix), each tied to a characteristic.
SEVERITY_KEYS = ["fissure_sev", "toothmk_sev", "coating_cov"]
SEVERITY_OF_CHAR = {"fissure": "fissure_sev", "tooth_mk": "toothmk_sev", "coating": "coating_cov"}

# Phase 4: extra multi-label features from the TCM-Tongue dataset (order matches the label CSV).
EXTRA_FEATURES = ["peeled_coating", "red_tongue", "purple_body", "swollen",
                  "thin", "red_dots", "black_coating", "slippery_coating"]
EXTRA_DESC = {
    "peeled_coating": "peeled / mirror coating",
    "red_tongue": "red tongue body",
    "purple_body": "purple / dusky body",
    "swollen": "swollen / enlarged body",
    "thin": "thin body",
    "red_dots": "red dots / prickles",
    "black_coating": "grey-black coating",
    "slippery_coating": "wet / slippery coating",
}
