"""Micro layer: LLM-extract cited feature->pattern triplets from the licensed books (WS-A, PLAN §3).

Runs OFFLINE (casper GPU + an OpenAI-compatible LLM endpoint — same env as the rest of Stage 2).
For each book section it asks the model to extract ONLY relationships explicitly stated in that
section, mapped into OUR canonical feature/pattern vocabulary, each with:
  - a verbatim snippet (cite-or-abstain: triplets whose snippet is not found in the section are dropped)
  - polarity (supports / argues_against — captures Gerlach's disambiguations)
  - optional context (co-conditions that gate the reading)

Output is a tracked, reviewable `book_triplets.json` (our structured extractions + short attributed
snippets). `build_kg.py` then folds it in as the MICRO layer (edges cited to Gerlach §N.N.N + a
snippet store). Nothing here stores long copyrighted passages — only short attributed quotes.

    # on casper, with the LLM env set (TIH_LLM_BACKEND=openai TIH_LLM_BASE_URL=... TIH_LLM_MODEL=...):
    python stage2_interpretation/kg/micro_extract.py --book-id gerlach --chapters 2,3,4
    python stage2_interpretation/kg/micro_extract.py --dry-run --book-id gerlach   # print 1 prompt, no LLM
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kg.parse_book import section_text  # noqa: E402
from llm_client import LLMClient  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
KB = os.path.join(HERE, "..", "knowledge_base", "tcm_knowledge.json")
SECTIONS = os.path.join(HERE, "..", "knowledge_base", "book_sections.json")
OUT = os.path.join(HERE, "..", "knowledge_base", "book_triplets.json")

SYSTEM = (
    "You are a careful TCM knowledge engineer. You extract tongue-diagnosis relationships that are "
    "EXPLICITLY stated in the provided book section — never inferred, never from prior knowledge. "
    "If the section states no tongue-sign -> pattern relationship, return an empty list. Output "
    "strict JSON only."
)


def _vocab(kb):
    feats = {}
    for fk, fv in kb.get("features", {}).items():
        vals = list(fv.get("values", {}).keys()) or ["present"]
        feats[fk] = {"label": fv.get("label", fk), "values": vals}
    for fk, fv in kb.get("extra_features", {}).items():
        feats[fk] = {"label": fv.get("label", fk), "values": ["present"]}
    pats = {pid: p.get("plain_name", pid) for pid, p in kb.get("patterns", {}).items()}
    return feats, pats


def build_prompt(kb, section, text):
    feats, pats = _vocab(kb)
    feat_lines = "\n".join("  - %s (%s): values %s" % (k, v["label"], v["values"]) for k, v in feats.items())
    pat_lines = "\n".join("  - %s (%s)" % (k, v) for k, v in pats.items())
    return (
        "CANONICAL TONGUE FEATURES (map each sign to the closest key; if none fits use "
        "\"other:<short label>\"):\n%s\n\n"
        "CANONICAL PATTERNS (map each pattern/guiding-criterion to the closest key; Gerlach's Latin "
        "terms map roughly: calor/heat->damp_heat or yin_deficiency, algor/cold->yang_deficiency, "
        "humor/humidity & pituita/phlegm->phlegm_dampness, xue-stasis->blood_stasis; if none fits use "
        "\"other:<short label>\"):\n%s\n\n"
        "BOOK SECTION  [Gerlach §%s — %s]:\n\"\"\"\n%s\n\"\"\"\n\n"
        "Extract every tongue-sign -> pattern relationship EXPLICITLY stated in this section. Return "
        "JSON: {\"triplets\": [{\"feature\": \"<key or other:...>\", \"value\": \"<value or present>\", "
        "\"pattern\": \"<key or other:...>\", \"polarity\": \"supports|argues_against\", "
        "\"context\": \"<gating co-signs or null>\", \"snippet\": \"<verbatim phrase copied from the "
        "section, <=200 chars>\"}]}. The snippet MUST be copied verbatim from the section text. If "
        "nothing is explicitly stated, return {\"triplets\": []}."
        % (feat_lines, pat_lines, section["num"], section["title"], text)
    )


def _parse_json(txt):
    if not txt:
        return None
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _norm(s):
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def validate_triplets(raw, text):
    """cite-or-abstain: keep only well-formed triplets whose snippet is verbatim in the section."""
    hay = _norm(text)
    out = []
    for t in (raw or {}).get("triplets", []):
        if not isinstance(t, dict):
            continue
        snip = (t.get("snippet") or "").strip()
        feat = (t.get("feature") or "").strip()
        pat = (t.get("pattern") or "").strip()
        if not snip or not feat or not pat:
            continue
        if _norm(snip) not in hay:      # snippet not actually in the text -> reject (hallucination guard)
            continue
        out.append({
            "feature": feat, "value": (t.get("value") or "present").strip(),
            "pattern": pat, "polarity": t.get("polarity", "supports"),
            "context": t.get("context") or None, "snippet": snip[:200],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book-id", default="gerlach")
    ap.add_argument("--chapters", default="2,3,4", help="comma list of chapters to mine")
    ap.add_argument("--min-chars", type=int, default=200, help="skip near-empty sections")
    ap.add_argument("--limit", type=int, default=0, help="cap sections (0 = all)")
    ap.add_argument("--dry-run", action="store_true", help="print one prompt and exit (no LLM)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    kb = json.load(open(KB))
    books = json.load(open(SECTIONS))
    bk = books[args.book_id]
    book_path = bk["source_file"]
    chapters = set(args.chapters.split(","))
    secs = [s for s in bk["sections"]
            if s["chapter"] in chapters and s["n_chars"] >= args.min_chars]
    if args.limit:
        secs = secs[:args.limit]

    if args.dry_run:
        s = secs[0]
        print(build_prompt(kb, s, section_text(book_path, s)))
        print("\n[dry-run] %d sections would be mined from chapters %s" % (len(secs), sorted(chapters)))
        return

    llm = LLMClient()
    if not llm.enabled:
        sys.exit("No LLM backend configured. Set TIH_LLM_BACKEND=openai + TIH_LLM_BASE_URL/MODEL "
                 "(run on casper). Use --dry-run to inspect prompts without an LLM.")

    # resume support: keep already-mined sections
    done = {}
    if os.path.exists(args.out):
        prev = json.load(open(args.out))
        done = {r["section"]: r for r in prev.get("records", []) if r.get("book_id") == args.book_id}

    records = list(done.values())
    n_new = n_trip = 0
    for i, s in enumerate(secs):
        key = "%s:%s" % (args.book_id, s["num"])
        if key in done:
            continue
        text = section_text(book_path, s)
        raw = _parse_json(llm.chat(SYSTEM, build_prompt(kb, s, text), temperature=0.0, max_tokens=1200))
        trips = validate_triplets(raw, text)
        records.append({"book_id": args.book_id, "section": s["num"], "title": s["title"],
                        "triplets": trips})
        n_new += 1
        n_trip += len(trips)
        print("  [%d/%d] §%-7s %-40s -> %d triplets" % (i + 1, len(secs), s["num"], s["title"][:40], len(trips)))
        if n_new % 10 == 0:  # checkpoint
            json.dump({"book_id": args.book_id, "records": records}, open(args.out, "w"),
                      ensure_ascii=False, indent=1)

    json.dump({"book_id": args.book_id, "records": records}, open(args.out, "w"),
              ensure_ascii=False, indent=1)
    print("\nwrote %s: %d new sections, %d validated triplets (total records %d)"
          % (args.out, n_new, n_trip, len(records)))


if __name__ == "__main__":
    main()
