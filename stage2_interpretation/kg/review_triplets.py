"""Build a self-contained HTML review page for LLM-extracted book triplets (WS-A micro layer QA).

    python stage2_interpretation/kg/review_triplets.py   ->  kg_triplets_review.html  (open by double-click)

Shows every extracted triplet grouped by book section, next to its verbatim snippet (the evidence),
so a human can judge FAITHFULNESS (does the snippet support the claim?) at a glance. Colour-codes:
  green  = feature AND pattern both map to our canonical vocabulary
  amber  = maps to a non-canonical string (needs the normalization alias map — expected for Gerlach's
           Latin terms like "fluid deficiency", "pituita")
  red    = likely junk (other:<...>, or a free-text blob that isn't a real sign/pattern)
The page is fully local (embeds nothing external); the compiled HTML is git-ignored (rebuildable).
"""
import json
import os
import re
import html

HERE = os.path.dirname(os.path.abspath(__file__))
KB = os.path.join(HERE, "..", "knowledge_base", "tcm_knowledge.json")
TRIPLETS = os.path.join(HERE, "..", "knowledge_base", "book_triplets.json")
SECTIONS = os.path.join(HERE, "..", "knowledge_base", "book_sections.json")
OUT = os.path.join(HERE, "kg_triplets_review.html")


def canon(kb):
    feats = set(kb.get("features", {})) | set(kb.get("extra_features", {}))
    pats = set(kb.get("patterns", {}))
    return feats, pats


def classify(t, feats, pats):
    f, p = t["feature"], t["pattern"]
    # strip a trailing "(label)" the model sometimes appends: "red_tongue (red tongue body)"
    fkey = re.sub(r"\s*\(.*\)\s*$", "", f).strip()
    f_ok = fkey in feats
    p_ok = p in pats
    if f.startswith("other:") or p.startswith("other:") or len(f) > 45 or len(p) > 45:
        return "junk"
    if f_ok and p_ok:
        return "ok"
    return "norm"


def main():
    kb = json.load(open(KB))
    feats, pats = canon(kb)
    data = json.load(open(TRIPLETS))
    sec_titles = {}
    if os.path.exists(SECTIONS):
        for bk in json.load(open(SECTIONS)).values():
            for s in bk["sections"]:
                sec_titles[(bk["book_id"], s["num"])] = s["title"]

    records = [r for r in data["records"] if r["triplets"]]
    all_t = [t for r in records for t in r["triplets"]]
    counts = {"ok": 0, "norm": 0, "junk": 0}
    for t in all_t:
        counts[classify(t, feats, pats)] += 1
    n = max(len(all_t), 1)

    # distinct non-canonical strings (motivates the alias map)
    bad_feat, bad_pat = {}, {}
    for t in all_t:
        cls = classify(t, feats, pats)
        if cls != "ok":
            fkey = re.sub(r"\s*\(.*\)\s*$", "", t["feature"]).strip()
            if fkey not in feats:
                bad_feat[fkey] = bad_feat.get(fkey, 0) + 1
            if t["pattern"] not in pats:
                bad_pat[t["pattern"]] = bad_pat.get(t["pattern"], 0) + 1

    def esc(s):
        return html.escape(str(s))

    rows = []
    for r in records:
        cite = "Gerlach §%s — %s" % (r["section"], sec_titles.get((data["book_id"], r["section"]), r["title"]))
        rows.append('<h3>%s <span class="cnt">%d</span></h3>' % (esc(cite), len(r["triplets"])))
        rows.append('<table>')
        for t in r["triplets"]:
            cls = classify(t, feats, pats)
            pol = "→" if t["polarity"] == "supports" else "⊣ (argues against)"
            rows.append(
                '<tr class="%s">'
                '<td class="f"><b>%s</b>%s</td>'
                '<td class="rel">%s</td>'
                '<td class="p">%s</td>'
                '<td class="ctx">%s</td>'
                '<td class="snip">%s</td></tr>' % (
                    cls, esc(t["feature"]),
                    ("" if (t.get("value") in (None, "present")) else ' <span class="v">= %s</span>' % esc(t["value"])),
                    pol, esc(t["pattern"]),
                    ('<span class="ctxt">%s</span>' % esc(t["context"]) if t.get("context") else ""),
                    esc(t["snippet"])))
        rows.append('</table>')

    def tbl(d):
        return "".join('<li><code>%s</code> <span class="cnt">%d</span></li>' % (esc(k), v)
                       for k, v in sorted(d.items(), key=lambda kv: -kv[1]))

    page = """<!doctype html><meta charset="utf-8"><title>KG triplet review — chapter 2</title>
<style>
 body{{font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#222}}
 h1{{margin-bottom:.2rem}} .sub{{color:#666;margin-top:0}}
 .stat{{display:inline-block;padding:.4rem .8rem;border-radius:8px;margin:.2rem .4rem .2rem 0;font-weight:600}}
 .s-ok{{background:#e5f5e9;color:#1a7f37}} .s-norm{{background:#fff4e0;color:#a15c00}} .s-junk{{background:#fde8e8;color:#b42318}}
 .check{{background:#f4f6fb;border-left:4px solid #4a6fd4;padding:.8rem 1rem;border-radius:6px;margin:1rem 0}}
 h3{{margin:1.6rem 0 .3rem;font-size:1rem;color:#33415c;border-bottom:1px solid #eee;padding-bottom:.2rem}}
 .cnt{{color:#999;font-weight:400;font-size:.85em}}
 table{{border-collapse:collapse;width:100%;margin-bottom:.4rem}}
 td{{padding:.35rem .5rem;vertical-align:top;border-bottom:1px solid #f0f0f0;font-size:.92rem}}
 td.f{{width:20%}} td.rel{{width:6%;color:#888;text-align:center}} td.p{{width:16%}} td.ctx{{width:12%;color:#a15c00;font-size:.85rem}}
 td.snip{{color:#555;font-style:italic}}
 tr.ok td.f b,tr.ok td.p{{color:#1a7f37}} tr.norm td.f b,tr.norm td.p{{color:#a15c00}}
 tr.junk{{background:#fff6f6}} tr.junk td.f b,tr.junk td.p{{color:#b42318}}
 .v{{color:#888;font-weight:400}} .cols{{display:flex;gap:2rem;flex-wrap:wrap}} .cols>div{{flex:1;min-width:280px}}
 code{{background:#f4f4f4;padding:.05rem .3rem;border-radius:3px}} ul{{padding-left:1.1rem}}
</style>
<h1>Knowledge-graph triplet review</h1>
<p class="sub">Gerlach chapter 2 · extracted by gemma3:4B (local, casper GPU 0) · cite-or-abstain enforced</p>
<div>
 <span class="stat s-ok">{ok} canonical ({okp:.0f}%)</span>
 <span class="stat s-norm">{norm} need normalization ({normp:.0f}%)</span>
 <span class="stat s-junk">{junk} likely junk ({junkp:.0f}%)</span>
 <span class="stat" style="background:#eee">{total} total · {nsec} sections</span>
</div>
<div class="check"><b>What to check (≈10 min):</b><ol style="margin:.4rem 0">
 <li><b>Faithfulness</b> — for the <span style="color:#1a7f37">green</span> and <span style="color:#a15c00">amber</span> rows, does the italic <i>snippet</i> actually back the <b>feature → pattern</b> claim? (This is the real question: is the model reading the book, or inventing links?)</li>
 <li><b>Junk</b> — are the <span style="color:#b42318">red</span> rows correctly garbage (a pain description as a "feature", <code>other:&lt;non-physiological&gt;</code>)? A few false-reds/greens are fine.</li>
 <li><b>Amber = expected, not error</b> — these map to Gerlach's Latin (<code>fluid deficiency</code>, <code>pituita</code>). See the two lists at the bottom: those become a small deterministic alias map (their terms → our keys) I'll apply on merge. Skim them for anything nonsensical.</li>
 <li><b>Gut call</b> — is this good enough <i>with</i> a normalization + light-review layer, or should we get the vLLM key for a stronger model? Tell me the ratio of good:bad you saw.</li>
</ol></div>
{rows}
<h2>Non-canonical strings → alias map to build</h2>
<div class="cols">
 <div><h3>Patterns needing an alias</h3><ul>{badpat}</ul></div>
 <div><h3>Features needing an alias</h3><ul>{badfeat}</ul></div>
</div>
""".format(ok=counts["ok"], norm=counts["norm"], junk=counts["junk"], total=len(all_t), nsec=len(records),
           okp=100 * counts["ok"] / n, normp=100 * counts["norm"] / n, junkp=100 * counts["junk"] / n,
           rows="\n".join(rows), badpat=tbl(bad_pat) or "<li>none</li>", badfeat=tbl(bad_feat) or "<li>none</li>")

    with open(OUT, "w") as f:
        f.write(page)
    print("wrote %s" % OUT)
    print("  %d triplets: %d canonical / %d need-normalization / %d junk  across %d sections"
          % (len(all_t), counts["ok"], counts["norm"], counts["junk"], len(records)))


if __name__ == "__main__":
    main()
