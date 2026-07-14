"""Build a LABELED TONGUE GALLERY for testing + visual reference: takes the human-labeled human-40 set
and produces (1) a folder of the images renamed by their labels, and (2) a self-contained HTML reference
that groups clear examples under each feature value — so you can see at a glance what 'greasy_thick' vs
'non_greasy', 'pale' vs 'dark', etc. actually look like, and use it as a known-answer test set.

    python evaluation/build_gallery.py
    -> data/eval/gallery/<labels>__tNN.jpg   +   data/eval/gallery/manifest.json
    -> evaluation/gallery.html  (open in a browser — no server needed)
"""
import base64, io, json, os, shutil
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT_DIR = os.path.join(REPO, "data", "eval", "gallery")
HTML = os.path.join(HERE, "gallery.html")

# both human-labeled sets, namespaced (they reuse t00.. ids)
SETS = [("human40", "a"), ("human40b", "b")]
CORE, EXTRA, SRC = {}, {}, {}
for _set, _pfx in SETS:
    cf = os.path.join(HERE, f"{_set}_labels.json")
    if not os.path.exists(cf):
        continue
    for iid, lab in json.load(open(cf)).items():
        key = f"{_pfx}-{iid}"
        CORE[key] = lab
        SRC[key] = os.path.join(REPO, "data", "eval", _set, iid + ".jpg")
    ef = os.path.join(HERE, f"{_set}_extra_labels.json")
    if os.path.exists(ef):
        for iid, lab in json.load(open(ef)).items():
            EXTRA[f"{_pfx}-{iid}"] = lab

# feature -> (heading, ordered values) for the grouped reference. Core signs + the two useful extras.
FEATURES = [
    ("coating", "Coating — greasiness/thickness", ["non_greasy", "greasy", "greasy_thick"]),
    ("tai", "Coating colour", ["white", "light_yellow", "yellow"]),
    ("zhi", "Body colour", ["light", "regular", "dark"]),
    ("fissure", "Fissures / cracks", ["none", "light", "severe"]),
    ("tooth_mk", "Tooth marks", ["none", "light", "severe"]),
    ("red_tip", "Red tip", ["none", "mild", "strong"]),
    ("red_dots", "Red dots / prickles", ["none", "few", "many"]),
]
SHORT = {"coating": "coat", "tai": "tai", "zhi": "body", "fissure": "fis", "tooth_mk": "tooth"}


def thumb_b64(path, maxdim=460, q=80):
    im = Image.open(path).convert("RGB")
    s = maxdim / max(im.size)
    if s < 1:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)
    ids = sorted(CORE)
    manifest, imgs = {}, {}
    for iid in ids:
        src = SRC[iid]
        if not os.path.exists(src):
            continue
        lab = dict(CORE[iid]); lab.update(EXTRA.get(iid, {}))
        manifest[iid] = lab
        imgs[iid] = thumb_b64(src)
        tag = "__".join(f"{SHORT.get(k, k)}-{lab[k]}" for k in ("coating", "tai", "zhi", "fissure", "tooth_mk") if lab.get(k))
        shutil.copy(src, os.path.join(OUT_DIR, f"{tag}__{iid}.jpg"))
    json.dump(manifest, open(os.path.join(OUT_DIR, "manifest.json"), "w"), indent=1)

    # --- self-contained HTML reference ---
    def caption(iid):
        l = manifest[iid]
        return " · ".join(f"{SHORT.get(k, k)}:{l[k]}" for k in ("coating", "tai", "zhi", "fissure", "tooth_mk") if l.get(k))
    sections = []
    for key, heading, values in FEATURES:
        groups = []
        for v in values:
            members = [i for i in ids if manifest.get(i, {}).get(key) == v]
            if not members:
                continue
            cards = "".join(
                f'<figure><img src="{imgs[i]}" title="{caption(i)}"><figcaption>{i} · <span>{caption(i)}</span></figcaption></figure>'
                for i in members)
            groups.append(f'<div class="grp"><h3>{v} <small>({len(members)})</small></h3><div class="row">{cards}</div></div>')
        sections.append(f'<section><h2>{heading}</h2>{"".join(groups)}</section>')
    html = _PAGE.replace("__BODY__", "".join(sections)).replace("__N__", str(len(manifest)))
    open(HTML, "w").write(html)
    print(f"gallery: {len(manifest)} labelled images -> {OUT_DIR}/  (+ manifest.json)")
    print(f"reference: {HTML}  ({len(html)/1e6:.1f} MB, open in a browser)")


_PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Labeled tongue gallery</title><style>
 :root{--bg:#f6f5f2;--card:#fff;--ink:#1c1a17;--mut:#7a736a;--line:#e2ddd4;--accent:#b5443a}
 @media(prefers-color-scheme:dark){:root{--bg:#17150f;--card:#211e18;--ink:#efe9df;--mut:#9a9184;--line:#332e25;--accent:#e0685c}}
 *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}
 header{padding:18px 20px;border-bottom:1px solid var(--line)}h1{margin:0;font-size:19px}
 .sub{color:var(--mut);font-size:13px;margin-top:4px}
 section{padding:14px 20px;border-bottom:1px solid var(--line)}h2{font-size:16px;margin:6px 0 12px}
 .grp{margin:0 0 14px}h3{font-size:14px;margin:0 0 7px;color:var(--accent);text-transform:capitalize}h3 small{color:var(--mut);font-weight:400}
 .row{display:flex;flex-wrap:wrap;gap:10px}
 figure{margin:0;width:150px}figure img{width:150px;height:150px;object-fit:cover;border-radius:9px;border:1px solid var(--line);display:block;background:#000}
 figcaption{font-size:10.5px;color:var(--mut);margin-top:3px;line-height:1.3}figcaption span{display:block}
</style></head><body>
<header><h1>👅 Labeled tongue gallery</h1><div class="sub">__N__ human-labeled tongues, grouped by each sign's value — a visual reference and a known-answer test set. Hover an image for its full labels.</div></header>
__BODY__
</body></html>"""


if __name__ == "__main__":
    main()
