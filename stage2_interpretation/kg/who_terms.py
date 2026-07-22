"""WHO IST 2022 ontology spine — extract canonical bilingual terminology (WS-A, docs/PROJECT_HANDBOOK.md §3).

The WHO *International Standard Terminologies on Traditional Chinese Medicine* (2022) is a numbered,
bilingual glossary: each entry is `<code>  <English term>  [<synonyms>]  <definition>  <简体中文>
<pīnyīn>  [<中文别名>]`. We use it NOT as a triplet source but as an **ontology spine**: a canonical
code + English + Chinese + pinyin for each of our pattern / tongue-sign nodes, so the KG can emit
bilingual output and future books merge onto stable identifiers instead of ad-hoc English strings.

This module extracts the entry HEADERS only (code, English head, 简体, pinyin) — terminology, which is
CC BY-NC-SA 3.0 IGO (attributable) — and deliberately drops the definition prose. The curated
node->code mapping lives in `who_spine.json` (small, hand-verified); this extractor just builds a
lookup index `who_terms.json` (git-ignored reference) to verify and maintain it.

    # convert the PDF once (locally or on casper):
    pdftotext -layout tongue_lit/9789240042322-eng.pdf who.txt
    python stage2_interpretation/kg/who_terms.py --src who.txt          # -> who_terms.json
    python stage2_interpretation/kg/who_terms.py --src who.txt --find "pale tongue"
"""
import argparse
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "knowledge_base", "who_terms.json")

CJK = re.compile(r"[一-鿿]")
# an entry header: leading 3-4 digit code, then text, and a Chinese term somewhere on the line
HEADER_RE = re.compile(r"^\s*(\d{3,4})\s+(\S.*)$")
# pinyin: latin letters with tone diacritics / plain, spaces, no CJK
PINYIN_RE = re.compile(r"^[a-zàáǎàāēéěèêīíǐìōóǒòūúǔùǖǘǚǜüA-Z' \-]+$")


def parse_terms(text):
    """-> {code: {"en": english head, "zh": simplified term, "pinyin": romanization}}."""
    terms = {}
    for line in text.splitlines():
        m = HEADER_RE.match(line)
        if not m or not CJK.search(line):
            continue
        code, rest = m.group(1), m.group(2)
        fields = re.split(r" {2,}", rest.strip())
        # first field carrying CJK = the simplified Chinese term
        zh_idx = next((i for i, f in enumerate(fields) if CJK.search(f)), None)
        if zh_idx is None:
            continue
        zh = fields[zh_idx].strip()
        # pinyin = the next latin field (if any)
        pinyin = ""
        if zh_idx + 1 < len(fields):
            nxt = fields[zh_idx + 1].strip()
            if nxt and not CJK.search(nxt) and PINYIN_RE.match(nxt):
                pinyin = nxt
        # english head = the first pre-Chinese text field (the term, before the definition column)
        en = fields[1].strip() if len(fields) > 1 and zh_idx > 0 else ""
        # a later duplicate code shouldn't clobber the first (definitions can restate a code)
        terms.setdefault(code, {"en": en, "zh": zh, "pinyin": pinyin})
    return terms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="pdftotext -layout output of the WHO IST PDF")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--find", default=None, help="print entries whose en/zh contains this (debug)")
    args = ap.parse_args()

    text = open(args.src, encoding="utf-8", errors="replace").read()
    terms = parse_terms(text)

    if args.find:
        q = args.find.lower()
        for code, t in terms.items():
            if q in t["en"].lower() or q in t["zh"]:
                print("%-6s %-40s %-12s %s" % (code, t["en"][:40], t["zh"], t["pinyin"]))
        return

    json.dump(terms, open(args.out, "w"), ensure_ascii=False, indent=1)
    print("extracted %d WHO term headers -> %s" % (len(terms), args.out))
    sample = list(terms.items())[:5]
    for code, t in sample:
        print("  %-6s %-34s %-10s %s" % (code, t["en"][:34], t["zh"], t["pinyin"]))


if __name__ == "__main__":
    main()
