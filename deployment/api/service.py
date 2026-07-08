"""Service layer wrapping the full pipeline for the API/demo: adds mask overlay + framing feedback."""
import os
import sys
import base64
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from pipeline import FullPipeline

# Target framing oval (fractions of frame): where the user should place the tongue.
OVAL_CX, OVAL_CY, OVAL_RX, OVAL_RY = 0.5, 0.5, 0.30, 0.38


def framing_feedback(mask):
    """Guidance so the user positions the tongue inside the guide oval."""
    H, W = mask.shape
    cov = float(mask.mean())
    ys, xs = np.where(mask > 0)
    if cov < 0.04 or xs.size == 0:
        return {"status": "error",
                "message": "No tongue detected — stick your tongue out and fill the oval.",
                "coverage": round(cov, 3)}
    cx, cy = xs.mean() / W, ys.mean() / H
    offset = float(((cx - OVAL_CX) ** 2 + (cy - OVAL_CY) ** 2) ** 0.5)
    border = (xs.min() <= 1 or xs.max() >= W - 2 or ys.min() <= 1 or ys.max() >= H - 2)
    msgs, status = [], "ok"
    if offset > 0.12:
        status = "warn"; msgs.append("center your tongue in the oval")
    if cov < 0.12:
        status = "warn"; msgs.append("move closer / extend your tongue more")
    if border and cov > 0.45:
        status = "warn"; msgs.append("move back slightly")
    msg = "✓ Great framing — hold still" if status == "ok" else "Adjust: " + ", ".join(msgs)
    return {"status": status, "message": msg, "coverage": round(cov, 3), "offset": round(offset, 3)}


def make_overlay(disp_rgb, mask):
    """Draw the target guide oval + the detected tongue mask (fill + contour) on the image."""
    ov = disp_rgb.copy()
    H, W = mask.shape
    green = np.zeros_like(ov); green[..., 1] = 255
    m3 = mask[..., None] > 0
    ov = np.where(m3, (0.45 * ov + 0.55 * green).astype(np.uint8), ov)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov, cnts, -1, (0, 255, 0), 2)
    cv2.ellipse(ov, (int(W * OVAL_CX), int(H * OVAL_CY)),
                (int(W * OVAL_RX), int(H * OVAL_RY)), 0, 0, 360, (255, 210, 0), 2)
    return ov


def _png_b64(rgb):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return "data:image/png;base64," + base64.b64encode(buf).decode()


class TongueService:
    def __init__(self, seg_ckpt, mt_ckpt, size=384):
        self.pipe = FullPipeline(seg_ckpt, mt_ckpt, size=size)

    def analyze(self, img_rgb, metadata=None):
        result, mask, disp = self.pipe.analyze_array(img_rgb, metadata=metadata)
        result["framing"] = framing_feedback(mask)
        result["overlay_png"] = _png_b64(make_overlay(disp, mask))
        return result
