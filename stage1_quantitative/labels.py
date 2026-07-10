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
