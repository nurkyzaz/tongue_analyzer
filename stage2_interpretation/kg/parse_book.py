"""Parse a licensed TCM book's section hierarchy (WS-A macro layer, docs/PLAN.md §3).

Extracts the numbered section skeleton (e.g. `2.2.2 Pale tongue body`) from a book text file and
records, per section: number, title, level, parent, and the CHARACTER OFFSETS of its body span in
the source file. It deliberately does NOT store the book text (copyrighted) — only structural
metadata + offsets, so `section_text()` can slice it on demand from the local file for the offline
micro-extraction step. The emitted `book_sections.json` is safe to track.

    python stage2_interpretation/kg/parse_book.py \
        --book tongue_lit/874856627-TCM-Tongue-Diagnosis-Explained.txt --id gerlach

Gerlach (2025) is the designated backbone: modern, feature-organized, clean numbering. Other books
can be parsed with the same tool if their headings follow `N(.N){1,3} Title`.
"""
import argparse
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "..", "knowledge_base", "book_sections.json")

# A bare body heading: leading section number, a title, nothing else. ToC lines are excluded by the
# dot-leader guard; wrapped ToC titles (no leaders) are resolved by keeping the LAST occurrence of
# each number (the body heading always follows the table of contents).
HEADING_RE = re.compile(r"^[ \t]*(\d+(?:\.\d+){1,3})[ \t]+(\S.*\S)[ \t]*$")
LEADER_RE = re.compile(r"\.[ \t]+\.")           # ". ." dot-leader run => table of contents
PAGEND_RE = re.compile(r"\s\d{1,3}$")            # trailing page number => table of contents


def parse_sections(text):
    """-> list of dicts sorted by document order, with body-span char offsets."""
    # locate every candidate heading with its char offset
    cands = []  # (num, title, char_start_of_line, char_end_of_line)
    pos = 0
    for line in text.splitlines(keepends=True):
        start = pos
        pos += len(line)
        stripped = line.rstrip("\n")
        m = HEADING_RE.match(stripped)
        if not m:
            continue
        if LEADER_RE.search(stripped) or PAGEND_RE.search(stripped):
            continue  # ToC line
        num, title = m.group(1), m.group(2).strip()
        if len(title) < 3 or len(title) > 80 or not title[0].isalpha():
            continue
        cands.append((num, title, start, start + len(line)))

    # keep the LAST occurrence of each section number (body heading, not the ToC entry)
    last = {}
    for num, title, s, e in cands:
        last[num] = (num, title, s, e)
    heads = sorted(last.values(), key=lambda t: t[2])

    sections = []
    for i, (num, title, hstart, hend) in enumerate(heads):
        body_end = heads[i + 1][2] if i + 1 < len(heads) else len(text)
        parts = num.split(".")
        sections.append({
            "num": num,
            "title": title,
            "level": len(parts),
            "parent": ".".join(parts[:-1]) if len(parts) > 1 else None,
            "chapter": parts[0],
            "body_start": hend,       # first char after the heading line
            "body_end": body_end,     # start of the next heading
            "n_chars": body_end - hend,
        })
    return sections


def section_text(book_path, section):
    """Slice a section's body text from the local (copyrighted) book file — for offline use only."""
    with open(book_path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    return text[section["body_start"]:section["body_end"]].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", required=True, help="path to the book .txt")
    ap.add_argument("--id", required=True, help="short book id, e.g. gerlach")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--title", default=None, help="human-readable book title for citations")
    args = ap.parse_args()

    with open(args.book, encoding="utf-8", errors="replace") as f:
        text = f.read()
    sections = parse_sections(text)

    # merge into any existing multi-book file, keyed by book id
    data = {}
    if os.path.exists(args.out):
        data = json.load(open(args.out))
    data[args.id] = {
        "book_id": args.id,
        "title": args.title or os.path.basename(args.book),
        "source_file": args.book,
        "n_sections": len(sections),
        "sections": sections,
    }
    with open(args.out, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)

    by_level = {}
    for s in sections:
        by_level[s["level"]] = by_level.get(s["level"], 0) + 1
    print("parsed %d sections from %s" % (len(sections), args.book))
    print("  by level:", dict(sorted(by_level.items())))
    print("  wrote %s (book id=%s)" % (args.out, args.id))
    print("  sample:")
    for s in sections[:3] + sections[len(sections) // 2: len(sections) // 2 + 2]:
        print("    %-8s L%d  %-45s  (%d chars body)" % (s["num"], s["level"], s["title"][:45], s["n_chars"]))


if __name__ == "__main__":
    main()
