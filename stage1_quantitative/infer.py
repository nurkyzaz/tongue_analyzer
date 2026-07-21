"""Stage-1 end-to-end inference: photo -> segmentation -> mask-guided characteristics -> JSON.

    python stage1_quantitative/infer.py --image path.jpg \
        --seg checkpoints/seg/best.pt --mt checkpoints/multitask_v2/best.pt

Runs the U-Net++ segmenter to get the tongue mask, then the multi-task head (mask-guided) to
predict the 5 key characteristics with confidences, and emits a Stage1Output JSON. A simple quality
gate rejects images whose predicted tongue coverage is implausibly small/large.
"""
import argparse
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from labels import (KEY_CHARS, LABEL_MAPS, CHAR_DESC, SEVERITY_KEYS, SEVERITY_OF_CHAR,
                    EXTRA_FEATURES, EXTRA_DESC, COAT_AXIS_DESC, derive_coat_axes)
from feature_extraction.model import MultiTaskTongueNet
from schema import Stage1Output

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def _preprocess(img_rgb, size):
    tf = A.Compose([A.LongestMaxSize(max_size=size),
                    A.PadIfNeeded(size, size, border_mode=cv2.BORDER_CONSTANT, value=0),
                    A.Normalize(MEAN, STD), ToTensorV2()])
    return tf(image=img_rgb)["image"].unsqueeze(0)


def color_calibrate(img_rgb, strength=None, p=6):
    """Shades-of-Gray white balance: estimate the illuminant colour and neutralise it, so phone-photo
    colour casts don't bias the colour features (a warm/yellow cast otherwise over-reads Heat/Damp-Heat).
    `strength` blends toward the correction (0 = off, 1 = full). A face/tongue scene averages warm/pink
    (not grey), so FULL strength over-corrects and washes red out of the body colour — a partial strength
    (env TIH_CC_STRENGTH, default 0.35) rescues colour casts while barely touching clean images."""
    if strength is None:
        strength = float(os.getenv("TIH_CC_STRENGTH", "0.35"))
    x = img_rgb.astype(np.float32)
    ill = np.power(np.mean(np.power(x, p), axis=(0, 1)), 1.0 / p)   # per-channel illuminant
    ill = ill / (ill.mean() + 1e-6)
    ill = 1.0 + strength * (ill - 1.0)
    return np.clip(x / ill.reshape(1, 1, 3), 0, 255).astype(np.uint8)


def _letterbox(img_rgb, size):
    """Same letterbox geometry as _preprocess (no normalization) so masks align with the image."""
    tf = A.Compose([A.LongestMaxSize(max_size=size),
                    A.PadIfNeeded(size, size, border_mode=cv2.BORDER_CONSTANT, value=0)])
    return tf(image=img_rgb)["image"]


class Stage1Pipeline:
    def __init__(self, seg_ckpt, mt_ckpt, device=None, size=384, extra_ckpt=None, color_calib=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.size = size
        self.color_calib = os.getenv("TIH_COLOR_CALIB", "0") == "1" if color_calib is None else color_calib
        seg_state = torch.load(seg_ckpt, map_location=self.device, weights_only=False)
        enc = seg_state["args"].get("encoder", "resnet34")
        self.seg = smp.UnetPlusPlus(encoder_name=enc, encoder_weights=None, in_channels=3, classes=1)
        self.seg.load_state_dict(seg_state["model"])
        self.seg.to(self.device).eval()

        mt_state = torch.load(mt_ckpt, map_location=self.device, weights_only=False)
        self.mt = MultiTaskTongueNet(mt_state["args"].get("encoder", "resnet34"), pretrained=False)
        # strict=False so older checkpoints without the severity head still load
        self.mt.load_state_dict(mt_state["model"], strict=False)
        self.mt.to(self.device).eval()
        self.sev_trained = "sev_mae" in mt_state         # regression head is meaningful

        # Phase 4: optional extra-features model (8 new multi-label features). Loaded if present.
        self.extra = None
        extra_ckpt = extra_ckpt or "checkpoints/extra_features/best.pt"
        if extra_ckpt and os.path.exists(extra_ckpt):
            from feature_extraction.extra_model import ExtraFeaturesNet
            es = torch.load(extra_ckpt, map_location=self.device, weights_only=False)
            self.extra = ExtraFeaturesNet(es["args"].get("encoder", "resnet34"), pretrained=False)
            self.extra.load_state_dict(es["model"])
            self.extra.to(self.device).eval()

    # tip_redness_delta (a* of tip minus whole tongue) above this => report a red tip. Chosen on the
    # human-40 set: flags 13/38, matching visual inspection (red-tip tongues t05/t12/t29/t35…).
    RED_TIP_THRESH = 2.0
    # side_redness_delta (a* of the lateral middle-third edges minus whole) above this => red SIDES
    # (Liver/GB zone). Set slightly stricter than the tip: it's a geometry heuristic NOT yet validated on
    # a labelled set (like the "dry" gap in zoning.py) — kept conservative so it only fires on clearly
    # redder edges, and it feeds only a small, tentative vote. TODO: calibrate on human labels.
    RED_SIDE_THRESH = 2.5

    def _zoned(self, img, mask):
        from zoning import analyze
        m = mask[0, 0].cpu().numpy().astype(np.uint8)
        disp = _letterbox(img, self.size)
        z = analyze(disp, m)
        if not z.get("ok"):
            return {}
        tip = z.get("tip_redness_delta")
        red_tip = tip is not None and tip > self.RED_TIP_THRESH
        side = z.get("side_redness_delta")
        red_sides = side is not None and side > self.RED_SIDE_THRESH
        wet = z.get("moisture") == "wet"
        return {
            "tip_redness_delta": tip,           # a*(tip) - a*(whole); + = tip redder (heat sign)
            "side_redness_delta": side,         # a*(sides) - a*(whole); + = lateral (Liver/GB) edges redder
            "redness_gradient": z.get("redness_gradient"),
            "red_tip": {
                "value": "present" if red_tip else "absent",
                "severity": round(min(max((tip or 0) / 8.0, 0.0), 1.0), 4),
                "description": "tip redder than the tongue body (upper-jiao / heart heat sign)",
            },
            "red_sides": {
                "value": "present" if red_sides else "absent",
                "severity": round(min(max((side or 0) / 8.0, 0.0), 1.0), 4),
                "description": "lateral edges redder than the body (Liver/Gallbladder zone sign)",
            },
            "gloss": z.get("gloss"),
            "moisture": {
                "value": z.get("moisture", "normal"),   # "wet" | "normal" ("dry" not inferred from gloss)
                "severity": round(min(max((z.get("gloss") or 0) / 0.04, 0.0), 1.0), 4),
                "description": "wet/glossy surface (excess or untransformed fluids)" if wet
                               else "no strong wet sheen detected",
            },
            "zones": z.get("zones", {}),
        }

    @torch.no_grad()
    def __call__(self, image_path, sid=None, return_mask=False):
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB) \
            if isinstance(image_path, str) else image_path
        if self.color_calib:
            img = color_calibrate(img)
        x = _preprocess(img, self.size).to(self.device)

        mask_prob = self.seg(x).sigmoid()
        mask = (mask_prob > 0.5).float()
        coverage = mask.mean().item()
        accepted, reasons = True, []
        if coverage < 0.05:
            accepted, reasons = False, reasons + ["tongue not detected / too small"]
        if coverage > 0.85:
            accepted, reasons = False, reasons + ["mask implausibly large (framing/quality issue)"]

        out = self.mt(x, mask)
        sev_reg = out["severity"][0].cpu().numpy() if self.sev_trained else None
        chars = {}
        for ch in KEY_CHARS:
            prob = F.softmax(out[ch], dim=1)[0]
            idx = int(prob.argmax())
            n = len(LABEL_MAPS[ch])
            # expected-ordinal severity in [0,1]: works with any checkpoint, extracts graded degree
            sev_ord = float((prob * torch.arange(n, device=prob.device)).sum() / (n - 1))
            # blend with the trained regression head when available and this char maps to one
            sev = sev_ord
            if sev_reg is not None and ch in SEVERITY_OF_CHAR:
                sev_meas = float(sev_reg[SEVERITY_KEYS.index(SEVERITY_OF_CHAR[ch])])
                sev = 0.5 * sev_ord + 0.5 * sev_meas
            chars[ch] = {"value": LABEL_MAPS[ch][idx],
                         "confidence": round(float(prob[idx]), 4),
                         "description": CHAR_DESC[ch],
                         "severity": round(sev, 4),
                         "probs": {LABEL_MAPS[ch][j]: round(float(prob[j]), 4) for j in range(n)}}

        # Split the coating prediction into two independent, interpretable axes (thickness × texture).
        # severity = probability of the ABNORMAL class (thick / greasy) so it reads as a degree.
        coat_probs = chars["coating"]["probs"]
        ABNORMAL = {"coat_thickness": "thick", "coat_texture": "greasy"}
        for axis, (val, conf, probs) in derive_coat_axes(coat_probs).items():
            chars[axis] = {"value": val, "confidence": round(float(conf), 4),
                           "description": COAT_AXIS_DESC[axis], "severity": round(float(probs[ABNORMAL[axis]]), 4),
                           "probs": {k: round(float(v), 4) for k, v in probs.items()}}

        # Phase 4: extra multi-label features (presence prob doubles as severity)
        extra = {}
        if self.extra is not None:
            ep = torch.sigmoid(self.extra(x, mask))[0].cpu().numpy()
            for k, feat in enumerate(EXTRA_FEATURES):
                extra[feat] = {"value": "present" if ep[k] > 0.5 else "absent",
                               "severity": round(float(ep[k]), 4),
                               "description": EXTRA_DESC[feat]}

        # Zoned colour analysis (no training): measures colour by region so a red tip / pale coated
        # centre isn't averaged away. Surfaces the "red tip" heat sign the whole-tongue argmax misses.
        zoned = self._zoned(img, mask)

        out_obj = Stage1Output(
            sid=sid,
            key_characteristics=chars,
            extra_characteristics=extra,
            zoned_analysis=zoned,
            quality={"mask_coverage": round(coverage, 4), "accepted": accepted, "reasons": reasons},
        )
        if return_mask:
            # mask + letterboxed display image (both at self.size) for visualization/framing
            m = mask[0, 0].cpu().numpy().astype(np.uint8)
            disp = _letterbox(img, self.size)
            return out_obj, m, disp
        return out_obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--seg", default="checkpoints/seg/best.pt")
    ap.add_argument("--mt", default="checkpoints/multitask_v2/best.pt")
    ap.add_argument("--size", type=int, default=384)
    args = ap.parse_args()
    pipe = Stage1Pipeline(args.seg, args.mt, size=args.size)
    result = pipe(args.image)
    print(result.to_json())
    print("\n--- summary ---")
    print("\n".join(result.summary_lines()))


if __name__ == "__main__":
    main()
