"""WS-C shadow run on REAL Stage-1 outputs (docs/PLAN.md §7-C).

For each real tongue photo: run Stage-1 -> features, then score the grounded cite-or-abstain matcher
(kg/matcher.py) against the production rule-engine vote on the SAME features. Logs top-1 agreement,
Jaccard overlap, abstain rate, and hallucination (dropped) rate. The rule engine stays production;
this just tells us whether to promote the matcher (ensemble / full) — on the numbers.

Runs on casper (Stage-1 GPU + Ollama for the matcher):
    env TIH_LLM_BACKEND=openai TIH_LLM_BASE_URL=http://localhost:11434/v1 \
        TIH_LLM_MODEL=qwen2.5:14b-instruct TIH_LLM_API_KEY=ollama \
    python evaluation/eval_shadow_matcher.py --images data/eval/human40 --limit 40
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "stage1_quantitative"))
sys.path.insert(0, os.path.join(ROOT, "stage2_interpretation"))
from infer import Stage1Pipeline                       # noqa: E402
import interpret as I                                  # noqa: E402
from llm_client import LLMClient                       # noqa: E402
from kg.matcher import GroundedMatcher                 # noqa: E402

SEG = os.getenv("TIH_SEG_CKPT", "checkpoints/seg_combined/best.pt")
MT = os.getenv("TIH_MT_CKPT", "checkpoints/multitask_v5/best.pt")


def rule_top(stage1_output, k=3):
    """Production rule-engine top-k pattern ids (LLM disabled — patterns are LLM-independent)."""
    out = I.interpret(stage1_output, llm=LLMClient(backend="none"))
    return [p["id"] for p in out.get("patterns", [])[:k] if p["id"] != "balanced"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/eval/human40")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--out", default="evaluation/shadow_matcher_results.json")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.images, "*.jpg")) +
                   glob.glob(os.path.join(args.images, "*.png")))[:args.limit]
    stage1 = Stage1Pipeline(SEG, MT)
    matcher = GroundedMatcher()
    assert matcher.llm.enabled, "set the LLM env (run on casper) — matcher needs Ollama"

    rows, agree, jsum, abst, halluc, both = [], 0, 0.0, 0, 0, 0
    for p in paths:
        s1 = json.loads(stage1(p).to_json())
        if not s1.get("quality", {}).get("accepted", True):
            continue
        present = I.present_features(s1)
        r_ids = rule_top(s1)
        m = matcher.match(present)
        m_ids = [x["id"] for x in m["matched"]]
        top1 = bool(m_ids and r_ids and m_ids[0] == r_ids[0])
        inter, union = set(m_ids) & set(r_ids), set(m_ids) | set(r_ids)
        jac = len(inter) / len(union) if union else 1.0
        dropped = m.get("dropped", [])
        agree += top1
        jsum += jac
        abst += m["abstained"]
        halluc += bool(dropped)
        both += 1
        rows.append({"image": os.path.basename(p), "present": present, "matcher": m_ids,
                     "rule": r_ids, "top1": top1, "jaccard": round(jac, 2),
                     "abstained": m["abstained"], "dropped": dropped})
        print("%-10s matcher=%-42s rule=%-42s top1=%s jac=%.2f%s"
              % (os.path.basename(p), ",".join(m_ids) or "(abstain)", ",".join(r_ids) or "(none)",
                 top1, jac, "  HALLUC=%s" % dropped if dropped else ""))

    n = max(1, both)
    summary = {"n": both, "top1_agreement": round(agree / n, 3), "mean_jaccard": round(jsum / n, 3),
               "abstain_rate": round(abst / n, 3), "hallucination_rate": round(halluc / n, 3),
               "seg": SEG, "mt": MT, "model": matcher.llm.model}
    json.dump({"summary": summary, "rows": rows}, open(args.out, "w"), ensure_ascii=False, indent=1)
    print("\n== shadow summary over %d images ==" % both)
    for k, v in summary.items():
        print("  %-20s %s" % (k, v))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
