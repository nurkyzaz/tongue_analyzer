"""WS3b — does color calibration (Shades-of-Gray white balance) help the colour features?

We don't have real diverse-phone photos, so we SIMULATE phone colour casts: apply a warm/yellow and a
cool/blue illuminant to each human-40 image and measure coating-colour (tai) and body-colour (zhi)
accuracy vs the human labels, with calibration OFF vs ON. Calibration is worth enabling if it RECOVERS
the accuracy that the cast destroys, without hurting the un-cast (original) images.

    CUDA_VISIBLE_DEVICES=0 envs/tih/bin/python evaluation/eval_color_calib.py
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stage1_quantitative"))
import cv2
from infer import Stage1Pipeline

CASTS = {
    "none": (1.00, 1.00, 1.00),
    "warm": (1.18, 1.02, 0.82),   # indoor tungsten / warm phone -> reds up, blues down
    "cool": (0.85, 1.00, 1.20),   # cool daylight / screen light -> blues up
}


def apply_cast(img_rgb, g):
    x = img_rgb.astype(np.float32) * np.array(g, np.float32).reshape(1, 1, 3)
    return np.clip(x, 0, 255).astype(np.uint8)


def main():
    gold = json.load(open("evaluation/human40_labels.json"))
    seg = "checkpoints/seg_combined/best.pt"
    mt = "checkpoints/multitask_v5/best.pt"
    pipe_off = Stage1Pipeline(seg, mt, color_calib=False)
    pipe_on = Stage1Pipeline(seg, mt, color_calib=True)   # reads TIH_CC_STRENGTH at call time
    STRENGTHS = [0.25, 0.35, 0.45, 0.6]
    configs = [("off", None)] + [(f"on@{s}", s) for s in STRENGTHS]

    acc = {(c, cfg): {"tai": [0, 0], "zhi": [0, 0]} for c in CASTS for cfg, _ in configs}
    imgs = [(iid, f"data/eval/human40/{iid}.jpg") for iid in sorted(gold)]
    imgs = [(iid, p) for iid, p in imgs if os.path.exists(p)]
    for iid, p in imgs:
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        g = gold[iid]
        for cast, coef in CASTS.items():
            ci = apply_cast(img, coef)
            for cfg, strength in configs:
                if strength is None:
                    s = json.loads(pipe_off(ci).to_json())["key_characteristics"]
                else:
                    os.environ["TIH_CC_STRENGTH"] = str(strength)
                    s = json.loads(pipe_on(ci).to_json())["key_characteristics"]
                for ch in ("tai", "zhi"):
                    gv = g.get(ch)
                    if gv is None:
                        continue
                    acc[(cast, cfg)][ch][0] += (s[ch]["value"] == gv)
                    acc[(cast, cfg)][ch][1] += 1

    def pct(cell):
        c, n = cell
        return f"{c/max(n,1):.0%}"
    for ch in ("tai", "zhi"):
        print(f"\n=== {ch} ({'coating colour' if ch=='tai' else 'body colour'}) accuracy ===")
        print(f"{'config':9} | " + "  ".join(f"{c:>6}" for c in CASTS))
        print("-" * 40)
        for cfg, _ in configs:
            print(f"{cfg:9} | " + "  ".join(f"{pct(acc[(c,cfg)][ch]):>6}" for c in CASTS))
    print("\nbest config = one where none stays ~= off AND warm/cool recover most")


if __name__ == "__main__":
    main()
