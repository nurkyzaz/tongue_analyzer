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

# Coverage tiers (fraction of frame the tongue mask fills). A well-framed tongue fills the guide
# oval ≈ π·RX·RY ≈ 0.36 of the frame. So these are a graded gate, NOT one "good enough" bar:
COV_PRESENT = 0.05   # below this there is essentially no tongue -> hard error, no reading
                     # (matches Stage-1 acceptance floor in stage1_quantitative/infer.py)
COV_RELIABLE = 0.12  # below this a tongue is present but too small/far for a trustworthy colour
                     # /coating read -> reading is emitted but flagged low-reliability ("move closer")
_SEV = {"ok": 0, "warn": 1, "error": 2}


def _worse(a, b):
    return a if _SEV[a] >= _SEV[b] else b


def capture_quality(img_rgb, mask):
    """ML-free capture-quality gate run *before* we trust a reading. Lighting/blur are the #1
    phone-accuracy killers in the tongue-imaging literature, so a clearly bad photo is refused
    rather than silently misread. Cheap (a Laplacian + a histogram); adds no model weight.

    Returns {status: ok|warn|error, checks: {...}, message}. `error` means "don't produce a reading".
    Measured on the tongue's bounding box so letterbox bars / background don't skew the stats."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    ys, xs = np.where(mask > 0)
    if xs.size:
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        roi = gray[y0:y1, x0:x1]
        tongue = gray[ys, xs].astype(np.float32)      # tongue pixels only (exposure/clipping)
    else:
        roi, tongue = gray, gray.reshape(-1).astype(np.float32)

    status, msgs, checks = "ok", [], {}

    # --- blur: variance of the Laplacian over the tongue's bounding box (focus/motion) ---
    lap_var = float(cv2.Laplacian(roi, cv2.CV_64F).var())
    checks["sharpness"] = round(lap_var, 1)
    if lap_var < 45:
        status = _worse(status, "error"); msgs.append("image is blurry — hold steady & let it focus")
    elif lap_var < 110:
        status = _worse(status, "warn"); msgs.append("a little soft — hold steadier")

    # --- exposure: mean luma of the tongue + clipped-shadow/highlight fractions ---
    mean_luma = float(tongue.mean())
    clip_lo = float((tongue <= 8).mean())
    clip_hi = float((tongue >= 250).mean())
    checks.update(mean_luma=round(mean_luma, 1), clip_lo=round(clip_lo, 3), clip_hi=round(clip_hi, 3))
    if mean_luma < 55 or clip_lo > 0.35:
        status = _worse(status, "error"); msgs.append("too dark — move to brighter, even light")
    elif mean_luma > 220 or clip_hi > 0.25:
        status = _worse(status, "error"); msgs.append("over-exposed / glare — avoid direct light & flash")
    elif mean_luma < 80 or mean_luma > 200 or clip_hi > 0.08:
        status = _worse(status, "warn"); msgs.append("uneven light — face a soft, neutral light source")

    msg = "✓ Lighting & focus look good" if status == "ok" else "Adjust: " + "; ".join(msgs)
    return {"status": status, "checks": checks, "message": msg}


def framing_feedback(mask):
    """Guidance so the user positions the tongue inside the guide oval."""
    H, W = mask.shape
    cov = float(mask.mean())
    ys, xs = np.where(mask > 0)
    if cov < COV_PRESENT or xs.size == 0:
        return {"status": "error",
                "message": "No tongue detected — stick your tongue out and fill the oval.",
                "coverage": round(cov, 3), "reliable": False}
    cx, cy = xs.mean() / W, ys.mean() / H
    offset = float(((cx - OVAL_CX) ** 2 + (cy - OVAL_CY) ** 2) ** 0.5)
    border = (xs.min() <= 1 or xs.max() >= W - 2 or ys.min() <= 1 or ys.max() >= H - 2)
    msgs, status = [], "ok"
    if offset > 0.12:
        status = "warn"; msgs.append("center your tongue in the oval")
    if cov < COV_RELIABLE:
        status = "warn"; msgs.append("move closer / extend your tongue more")
    if border and cov > 0.45:
        status = "warn"; msgs.append("move back slightly")
    msg = "✓ Great framing — hold still" if status == "ok" else "Adjust: " + ", ".join(msgs)
    # `reliable`: the tongue is big enough (past the 12% line) to trust the colour/coating read.
    return {"status": status, "message": msg, "coverage": round(cov, 3),
            "offset": round(offset, 3), "reliable": cov >= COV_RELIABLE}


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
        framing = framing_feedback(mask)
        quality = capture_quality(disp, mask)
        # A hard capture failure (blur / bad exposure) means the reading isn't trustworthy — surface it
        # on the framing status the client already gates on, so a bad photo is refused end-to-end.
        if quality["status"] == "error" and framing["status"] != "error":
            framing = {**framing, "status": "error", "message": quality["message"]}
        result["framing"] = framing
        result["quality_gate"] = quality
        result["overlay_png"] = _png_b64(make_overlay(disp, mask))
        return result
