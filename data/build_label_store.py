"""Unified label store — merge every label we hold, across datasets, into ONE long-format table so a
model can be tested conveniently against any source.

Sources merged (each row is one (image, feature, value) with provenance):
  - human       : the user's hand labels (human40 core + extra, human10 core). GOLD.
  - expert      : TonguExpert manual grading (manifest *_manual cols; sparse; no coating).
  - auto        : TonguExpert auto labels (manifest core cols; noisy; this is the TRAIN signal).
  - practitioner: TCM-Tongue (shezhen) YOLO category annotations -> our schema.

Key design choices:
  * Keyed on the UNDERLYING image path (repo-relative), so the same physical image gets rows from
    several sources. The human-40 ids (t00..) are linked to their real dataset path via meta.json, so a
    human label and a practitioner label for the same tongue land on the same key (great for cross-check).
  * `split` is carried through for TCM (train/val/test) and TonguExpert, so eval can avoid testing a
    model on the data it trained on (extra-features model trained on TCM train; core model on TE auto).
  * TCM presence features: a category box present => value "present". Absence is treated as "absent"
    for the binary attributes (the set annotates all visible attributes per image). Fissure/tooth-mark
    from TCM are presence-only ("present"/"absent"), NOT graded none/light/severe.

    python data/build_label_store.py   ->   data/processed/label_store.csv  (+ coverage summary)
"""
import os, json, glob
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "data", "processed", "label_store.csv")

# ---- TCM-Tongue (shezhen) YOLO class -> (feature, value). Fixes the old botaishe->peeled bug:
#      class 1 botaishe = 薄苔 THIN coating, not peeled. ----
TCM_ROOT = "data/external/tcm_tongue/shezhenv3-txt"
TCM_MAP = {
    1:  ("thin_coating",     "present"),   # botaishe    薄苔  (thin, not peeled!)
    2:  ("red_tongue",       "present"),   # hongshe     红舌
    3:  ("purple_body",      "present"),   # zishe       紫舌
    4:  ("swollen",          "present"),   # pangdashe   胖大舌
    5:  ("thin_body",        "present"),   # shoushe     瘦舌
    6:  ("red_dots",         "present"),   # hongdianshe 红点舌
    7:  ("fissure",          "present"),   # liewenshe   裂纹舌  (presence only)
    8:  ("tooth_mk",         "present"),   # chihenshe   齿痕舌  (presence only)
    9:  ("tai",              "white"),     # baitaishe   白苔
    10: ("tai",              "yellow"),    # huangtaishe 黄苔
    11: ("black_coating",    "present"),   # heitaishe   黑苔
    12: ("slippery_coating", "present"),   # huataishe   滑苔
    13: ("zone_kidney",      "concave"),   14: ("zone_kidney",     "convex"),
    15: ("zone_liver",       "concave"),   16: ("zone_liver",      "convex"),
    17: ("zone_spleen",      "concave"),
    18: ("zone_heartlung",   "convex"),    19: ("zone_heartlung",  "concave"),
}
# binary presence features for which "no box" => absent (dataset annotates all visible attributes)
TCM_BINARY = {"thin_coating","red_tongue","purple_body","swollen","thin_body","red_dots",
              "fissure","tooth_mk","black_coating","slippery_coating"}


def rows_tcm():
    out = []
    for split in ("train", "val", "test"):
        idir = os.path.join(TCM_ROOT, split, "images")
        ldir = os.path.join(TCM_ROOT, split, "labels")
        for img in sorted(glob.glob(os.path.join(idir, "*.jpg"))):
            stem = os.path.splitext(os.path.basename(img))[0]
            lf = os.path.join(ldir, stem + ".txt")
            present_feats, cats = {}, set()
            if os.path.exists(lf):
                for line in open(lf):
                    p = line.split()
                    if p:
                        cats.add(int(float(p[0])))
            for cid in cats:
                if cid in TCM_MAP:
                    feat, val = TCM_MAP[cid]
                    present_feats[feat] = val
            # emit present features + explicit "absent" for unseen binary attributes
            emitted = set(present_feats)
            for feat, val in present_feats.items():
                out.append((img, "TCM", split, feat, val, "practitioner"))
            for feat in TCM_BINARY - emitted:
                out.append((img, "TCM", split, feat, "absent", "practitioner"))
    return out


def rows_tonguexpert():
    out = []
    m = pd.read_csv("data/processed/manifest.csv")
    core = ["coating", "tai", "zhi", "fissure", "tooth_mk"]
    man = ["tai", "zhi", "fissure", "tooth_mk"]         # manual has no coating
    for _, r in m.iterrows():
        path = "data/raw/" + str(r.raw_path)
        split = r.get("split", "")
        for ch in core:                                  # auto labels (train signal)
            if pd.notna(r.get(ch)):
                out.append((path, "TE", split, ch, str(r[ch]), "auto"))
        if int(r.get("has_manual", 0)) == 1:             # sparse expert grading
            for ch in man:
                v = r.get(ch + "_manual")
                if pd.notna(v):
                    out.append((path, "TE", split, ch, str(v), "expert"))
    return out


def rows_human():
    """Both human-labeled sets (human40 + human40b). Keyed on each image's REAL dataset path (from that
    set's meta.json), so the two sets don't collide even though both reuse t00.. ids."""
    out = []
    for setname in ("human40", "human40b"):
        meta_f = f"data/eval/{setname}/meta.json"
        meta = {e["id"]: e for e in json.load(open(meta_f))} if os.path.exists(meta_f) else {}
        for fn in (f"evaluation/{setname}_labels.json", f"evaluation/{setname}_extra_labels.json"):
            if not os.path.exists(fn):
                continue
            for iid, feats in json.load(open(fn)).items():
                if iid.startswith("_"):
                    continue
                path = meta[iid]["path"] if iid in meta else f"data/eval/{setname}/{iid}.jpg"
                for feat, val in feats.items():
                    out.append((path, meta.get(iid, {}).get("src", "?"), "eval", feat, str(val), "human"))
    return out


def build():
    rows = rows_human() + rows_tonguexpert() + rows_tcm()
    df = pd.DataFrame(rows, columns=["image_path", "dataset", "split", "feature", "value", "source"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT}  ({len(df)} label rows, {df.image_path.nunique()} unique images)\n")
    print("coverage: rows per (feature, source)")
    piv = df.groupby(["feature", "source"]).size().unstack(fill_value=0)
    print(piv.to_string())
    print("\nimages with a human label that ALSO have a professional label (mergeable cross-check):")
    hp = df[df.source == "human"].image_path.unique()
    pro = df[df.source.isin(["expert", "practitioner"])].image_path.unique()
    print(f"  {len(set(hp) & set(pro))} images")


if __name__ == "__main__":
    build()
