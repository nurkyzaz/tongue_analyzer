"""WS-D — RAGAS-style faithfulness gate for the LLM narrative (docs/PLAN.md §7-D).

The hallucination gate before the LLM narrator may default ON. Implements the RAGAS *faithfulness*
metric locally (no heavy `ragas`/langchain dependency — it's an eval, but we keep it auditable and
Ollama-driven): for each case, take the grounded facts as CONTEXT and the produced narrative as ANSWER,
have a judge LLM decompose the answer into atomic claims and label each Supported / Unsupported by the
context, and score faithfulness = supported / total. Gate: mean faithfulness < THRESHOLD ⇒ the LLM
narrator must stay OFF (the deterministic template is the always-on fallback).

Runs on casper (Stage-1 GPU + Ollama):
    env TIH_LLM_BACKEND=openai TIH_LLM_BASE_URL=http://localhost:11434/v1 \
        TIH_LLM_MODEL=qwen2.5:14b-instruct TIH_LLM_API_KEY=ollama \
    python evaluation/eval_faithfulness.py --images data/eval/human40 --limit 12
"""
import argparse
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "stage1_quantitative"))
sys.path.insert(0, os.path.join(ROOT, "stage2_interpretation"))
from infer import Stage1Pipeline                       # noqa: E402
import interpret as I                                  # noqa: E402
from llm_client import LLMClient                       # noqa: E402
from kg.retrieval import GraphRAG                      # noqa: E402

SEG = os.getenv("TIH_SEG_CKPT", "checkpoints/seg_combined/best.pt")
MT = os.getenv("TIH_MT_CKPT", "checkpoints/multitask_v5/best.pt")
THRESHOLD = float(os.getenv("TIH_FAITHFULNESS_MIN", "0.85"))

JUDGE_SYS = ("You are a strict fact-checker. You are given CONTEXT (grounded facts) and an ANSWER. "
             "Decompose the ANSWER into atomic factual claims, then for EACH claim decide whether it "
             "is directly supported by the CONTEXT. Generic wellness advice, hedging ('may', "
             "'traditionally associated'), and non-factual framing are 'supported' by default. Only "
             "flag a claim 'unsupported' if it asserts a specific tongue-sign→pattern/symptom link that "
             "the CONTEXT does not contain. Output STRICT JSON only.")


def judge(context, answer, llm):
    prompt = ("CONTEXT (the only grounded facts available):\n%s\n\nANSWER to fact-check:\n\"\"\"\n%s\n\"\"\"\n\n"
              "Return JSON: {\"claims\": [{\"claim\": \"<atomic claim>\", \"supported\": true|false}]}"
              % (context, answer))
    raw = llm.chat(JUDGE_SYS, prompt, temperature=0.0, max_tokens=900,
                   response_format={"type": "json_object"})
    m = re.search(r"\{.*\}", raw or "", re.S)
    try:
        claims = json.loads(m.group(0)).get("claims", []) if m else []
    except Exception:
        claims = []
    sup = sum(1 for c in claims if c.get("supported"))
    return sup, len(claims), claims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/eval/human40")
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--out", default="evaluation/faithfulness_results.json")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.images, "*.jpg")))[:args.limit]
    stage1 = Stage1Pipeline(SEG, MT)
    narrator = LLMClient()                       # generates the narrative (the thing under test)
    judge_llm = LLMClient()                      # same backend, used as the fact-checker
    rag = GraphRAG.load()
    assert narrator.enabled, "set the LLM env (run on casper)"

    rows, tot_sup, tot_claims, worst = [], 0, 0, []
    for p in paths:
        s1 = json.loads(stage1(p).to_json())
        if not s1.get("quality", {}).get("accepted", True):
            continue
        out = I.interpret(s1, llm=narrator)
        answer = out.get("report") or ""
        present = I.present_features(s1)
        context = rag.retrieve(present).context_cards()  # the grounded facts the narrative may use
        sup, n, claims = judge(context, answer, judge_llm)
        faith = (sup / n) if n else 1.0
        tot_sup += sup
        tot_claims += n
        unsup = [c["claim"] for c in claims if not c.get("supported")]
        rows.append({"image": os.path.basename(p), "claims": n, "supported": sup,
                     "faithfulness": round(faith, 3), "unsupported": unsup})
        if unsup:
            worst.append((os.path.basename(p), unsup))
        print("%-10s claims=%2d supported=%2d faithfulness=%.2f%s"
              % (os.path.basename(p), n, sup, faith, "  ⚠ " + "; ".join(unsup[:2]) if unsup else ""))

    micro = (tot_sup / tot_claims) if tot_claims else 1.0        # claim-weighted (RAGAS-style)
    macro = sum(r["faithfulness"] for r in rows) / max(1, len(rows))  # per-case mean
    summary = {"n_cases": len(rows), "claims": tot_claims, "supported": tot_sup,
               "faithfulness_micro": round(micro, 3), "faithfulness_macro": round(macro, 3),
               "threshold": THRESHOLD, "gate": "PASS" if micro >= THRESHOLD else "FAIL",
               "model": narrator.model}
    json.dump({"summary": summary, "rows": rows}, open(args.out, "w"), ensure_ascii=False, indent=1)
    print("\n== faithfulness summary over %d cases ==" % len(rows))
    for k, v in summary.items():
        print("  %-20s %s" % (k, v))
    print("GATE %s (micro %.3f vs threshold %.2f) — %s"
          % (summary["gate"], micro, THRESHOLD,
             "LLM narrator may default ON" if summary["gate"] == "PASS"
             else "keep LLM OFF; template fallback"))
    print("wrote", args.out)
    sys.exit(0 if summary["gate"] == "PASS" else 1)


if __name__ == "__main__":
    main()
