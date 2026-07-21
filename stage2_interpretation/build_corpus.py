"""Build the RAG knowledge corpus (JSONL of retrievable, cited chunks) from our OWN authored, grounded
content — never copyrighted text. Sources: tcm_knowledge.json (features/patterns/combination rules/
zoning) PLUS authored cards (knowledge_cards.json) that capture the reasoning a human uses to tell
similar pictures apart (the high-value RAG content).

Every chunk is stamped with provenance resolved from the SOURCE REGISTRY (knowledge_base/sources.json):
`source_keys`, `license`, and `usage` — so downstream we always know what may be surfaced to end users
vs. what is internal-only, and adding a licensed source is a registry edit, not a code change.

    python stage2_interpretation/build_corpus.py            # -> knowledge_base/corpus.jsonl
    python stage2_interpretation/build_corpus.py --stats     # counts by type / source / license / lang
    python stage2_interpretation/build_corpus.py --validate  # lint cards + provenance, exit 1 on problems

To ADD a legitimately-sourced source: register it in sources.json (with its license + usage), then add
authored cards to knowledge_cards.json citing it. `--validate` flags any card whose source is unregistered
or whose source is still permission-pending.
"""
import argparse
import collections
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
KB = os.path.join(HERE, "knowledge_base", "tcm_knowledge.json")
CARDS_FILE = os.path.join(HERE, "knowledge_base", "knowledge_cards.json")
SOURCES_FILE = os.path.join(HERE, "knowledge_base", "sources.json")
OUT = os.path.join(HERE, "knowledge_base", "corpus.jsonl")

# most-restrictive-wins ranking, for when a chunk cites several sources (governs snippet surfacing)
_LICENSE_RANK = {"copyrighted": 5, "ISO (proprietary)": 4, "published-instrument": 4, "academic": 3,
                 "CC-BY-NC-SA-3.0-IGO": 3, "CC-BY-4.0": 2, "CC-BY (verify)": 2, "CC-BY": 2,
                 "web": 1, "own": 0}


def load_registry():
    reg = json.load(open(SOURCES_FILE))["sources"]
    alias_to_key = []
    for key, s in reg.items():
        for a in s.get("aliases", [key]):
            alias_to_key.append((a.lower(), key))
    alias_to_key.sort(key=lambda t: -len(t[0]))          # match longer aliases first
    return reg, alias_to_key


def resolve_source(source_str, reg, alias_to_key):
    """Map a free-text `source` string to registry keys, and pick the most-restrictive license/usage."""
    s = (source_str or "").lower()
    keys = []
    for alias, key in alias_to_key:
        if alias in s and key not in keys:
            keys.append(key)
    if not keys:
        keys = ["authored"] if not s.strip() else []
    lic, usage, best = "own", "authored-summary", -1
    for k in keys:
        L = reg.get(k, {}).get("license", "own")
        if _LICENSE_RANK.get(L, 1) > best:
            best, lic, usage = _LICENSE_RANK.get(L, 1), L, reg.get(k, {}).get("usage", "authored-summary")
    return keys, lic, usage


def _pattern_vocab(kb):
    """canonical pattern id + its display names, for tagging cards by which patterns they discuss."""
    vocab = {}
    for pid, p in kb.get("patterns", {}).items():
        names = [pid, pid.replace("_", " "), (p.get("plain_name") or ""), (p.get("tcm_name") or "")]
        vocab[pid] = [n.lower() for n in names if n]
    return vocab


def _tags_for(cid, text, pvocab):
    """Structured tags: the canonical patterns a chunk is about (id prefix + name mentions)."""
    body = (cid + " " + text).lower()
    return sorted(pid for pid, names in pvocab.items() if any(n in body for n in names))


def _type_of(cid):
    prefix = cid.split(":", 1)[0]
    return {"feature": "feature", "extra": "feature", "pattern": "pattern", "rule": "rule",
            "zone": "zone", "constitution": "constitution", "disambig": "disambiguation",
            "combo": "combination", "food": "food", "symptom": "symptom", "sign": "sign",
            "modern": "modern", "limit": "limitation", "nuance": "nuance"}.get(prefix, "card")


def chunks_from_kb(kb):
    out = []
    src_all = "; ".join(kb.get("sources", [])[:2])
    for fk, fv in kb.get("features", {}).items():
        label = fv.get("label", fk)
        if fv.get("kind") == "graded_value" and fv.get("present_tcm"):
            out.append((f"feature:{fk}", f"{label} — when present: {fv.get('present_tcm','')}. In plain terms: "
                        f"{fv.get('present_plain','')}.", src_all))
        for vk, vv in fv.get("values", {}).items():
            gloss, tcm = vv.get("plain_gloss", ""), vv.get("tcm_term", "")
            pts = ", ".join(vv.get("points_to", {}).keys())
            if gloss or tcm:
                out.append((f"feature:{fk}={vk}", f"{label} '{vk}' ({tcm}): {gloss}"
                            + (f" Traditionally associated with: {pts}." if pts else ""), src_all))
    for feat, fv in kb.get("extra_features", {}).items():
        if fv.get("present_tcm") or fv.get("present_plain"):
            out.append((f"extra:{feat}", f"{fv.get('label', feat)}: {fv.get('present_tcm','')}. "
                        f"{fv.get('present_plain','')}", src_all))
    for pid, p in kb.get("patterns", {}).items():
        if pid == "balanced":
            continue
        txt = f"{p.get('tcm_name', pid)} ({p.get('plain_name','')}): {p.get('explanation','')}"
        if p.get("associated_symptoms"):
            txt += " Often noticed: " + ", ".join(p["associated_symptoms"]) + "."
        if p.get("modern_correlation"):
            txt += " Modern view: " + p["modern_correlation"] + "."
        recs = (p.get("recommendations", {}).get("diet", []) + p.get("recommendations", {}).get("lifestyle", []))
        if recs:
            txt += " Wellness notes: " + "; ".join(recs[:4]) + "."
        out.append((f"pattern:{pid}", txt, "WHO ICD-11 Ch.26; CCMQ; Maciocia"))
    for r in kb.get("combination_rules", []):
        if r.get("note"):
            out.append((f"rule:{r['id']}", "Combination reasoning: " + r["note"], r.get("cite", "")))
    for zone in ("tip", "center", "sides", "root"):
        z = kb.get("regions", {}).get(zone, {})
        if z.get("organs"):
            out.append((f"zone:{zone}", f"Tongue {zone} reflects {z['organs']}: {z.get('note','')}", "Maciocia"))
    return out


def load_cards():
    """[(id, text, source, lang)] from knowledge_cards.json (lang optional, defaults en)."""
    if not os.path.exists(CARDS_FILE):
        return []
    return [(c["id"], c["text"], c.get("source", ""), c.get("lang", "en"))
            for c in json.load(open(CARDS_FILE))]


def build_rows(kb, reg, alias_to_key):
    pvocab = _pattern_vocab(kb)
    rows = [(cid, text, src, "en") for cid, text, src in chunks_from_kb(kb)] + load_cards()
    out = []
    for cid, text, src, lang in rows:
        keys, lic, usage = resolve_source(src, reg, alias_to_key)
        out.append({"id": cid, "text": text, "source": src, "source_keys": keys,
                    "license": lic, "usage": usage, "lang": lang,
                    "type": _type_of(cid), "tags": _tags_for(cid, text, pvocab)})
    return out


def validate(rows):
    problems, warnings = [], []
    seen_ids, seen_text = set(), {}
    for r in rows:
        if r["id"] in seen_ids:
            problems.append(f"duplicate id: {r['id']}")
        seen_ids.add(r["id"])
        norm = re.sub(r"\s+", " ", r["text"].lower())[:80]
        if norm in seen_text:
            warnings.append(f"near-duplicate text: {r['id']} ~ {seen_text[norm]}")
        seen_text[norm] = r["id"]
        n = len(r["text"])
        if n < 40:
            warnings.append(f"very short chunk ({n} chars): {r['id']}")
        if n > 900:
            warnings.append(f"long chunk ({n} chars, consider splitting): {r['id']}")
        if r["source"] and not r["source_keys"]:
            problems.append(f"UNREGISTERED source '{r['source']}' on {r['id']} — add it to sources.json")
        if r["usage"] == "permission-pending":
            problems.append(f"card cites a PERMISSION-PENDING source ({r['source']}) on {r['id']} — "
                            "confirm the grant or remove")
    return problems, warnings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    kb = json.load(open(KB))
    reg, alias_to_key = load_registry()
    rows = build_rows(kb, reg, alias_to_key)

    if args.validate:
        problems, warnings = validate(rows)
        for w in warnings:
            print("  warn:", w)
        if problems:
            print("\nVALIDATION FAILED (%d):" % len(problems))
            for p in problems:
                print("  -", p)
            raise SystemExit(1)
        print("validation OK — %d chunks, %d warnings" % (len(rows), len(warnings)))
        return

    if args.stats:
        for dim in ("type", "license", "usage", "lang"):
            print("%-8s %s" % (dim, dict(collections.Counter(r[dim] for r in rows).most_common())))
        return

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_cards = sum(1 for r in rows if r["type"] not in ("feature", "pattern", "rule", "zone"))
    print("wrote %s (%d chunks: %d authored cards + %d from KB)" % (args.out, len(rows), n_cards, len(rows) - n_cards))


if __name__ == "__main__":
    main()
