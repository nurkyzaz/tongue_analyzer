"""Full TongueInsight pipeline orchestrator: image (+ metadata) -> quantitative JSON + report.

    python pipeline.py --image photo.jpg \
        --seg checkpoints/seg/best.pt --mt checkpoints/multitask_v2/best.pt

This is the single entry point the API/mobile backend calls. Stage 2 runs LLM-free by default
(template report); set TIH_LLM_BACKEND=openai + endpoint env vars to use an LLM.
"""
import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stage1_quantitative"))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stage2_interpretation"))
from infer import Stage1Pipeline          # noqa: E402
from interpret import interpret           # noqa: E402
from llm_client import LLMClient          # noqa: E402


class FullPipeline:
    def __init__(self, seg_ckpt, mt_ckpt, size=384):
        self.stage1 = Stage1Pipeline(seg_ckpt, mt_ckpt, size=size)
        self.llm = LLMClient()

    def _assemble(self, s1, s2):
        s1_dict = json.loads(s1.to_json())
        return {
            "quantitative": {
                "key_characteristics": s1_dict["key_characteristics"],
                "quality": s1_dict["quality"],
                "summary": s1.summary_lines(),
            },
            "interpretation": s2,
            "model_version": s1_dict["model_version"],
            "llm_backend": self.llm.backend,
        }

    def analyze(self, image_path, metadata=None, sid=None):
        s1 = self.stage1(image_path, sid=sid)
        s2 = interpret(json.loads(s1.to_json()), metadata=metadata, llm=self.llm)
        return self._assemble(s1, s2)

    def analyze_array(self, img_rgb, metadata=None):
        """Analyze an in-memory RGB image; also returns the mask + letterboxed display image
        (both HxW at model size) so the caller can render a framing/mask overlay."""
        s1, mask, disp = self.stage1(img_rgb, return_mask=True)
        s2 = interpret(json.loads(s1.to_json()), metadata=metadata, llm=self.llm)
        return self._assemble(s1, s2), mask, disp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--seg", default="checkpoints/seg/best.pt")
    ap.add_argument("--mt", default="checkpoints/multitask_v2/best.pt")
    ap.add_argument("--size", type=int, default=384)
    args = ap.parse_args()
    result = FullPipeline(args.seg, args.mt, args.size).analyze(args.image)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
