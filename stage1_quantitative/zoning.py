"""Zoned colour analysis of a segmented tongue — measures colour *by region* instead of one
whole-tongue average.

Motivation (user feedback, 2026-07-13): on many real tongues the body colour is not uniform. A greasy
tongue is often white/pale in the CENTRE (that's the coating hiding the body) while the TIP stays far
redder (the exposed body, and a classic TCM heat sign). A single argmax over the whole tongue conflates
these, so the model reads such a tongue as "regular" and the pale centre / red tip both get lost.

This module works purely on the seg mask + pixels (no training). It uses PCA on the mask to find the
tongue's long axis (rotation-robust — handles tilted photos), splits it into tip / middle / root thirds,
and separately measures a central core vs. a lateral ring. For each zone it reports CIELAB stats:
  L* (lightness — high & desaturated in the centre = thick coating covering the body),
  a* (green<->red — the redness axis; high a* at the tip = red tip / heat sign).

It emits interpretable, unit-free signals:
  tip_redness_delta  = a*(tip)    - a*(whole)      > 0  => tip redder than the rest (heat sign)
  center_coating     = L*(center) - L*(edge ring)  > 0  => pale/white centre vs. redder rim (coating)
  redness_gradient   = a*(tip)    - a*(root)               tip-to-root redness slope

The tip is disambiguated from the root by width: the tongue tapers, so the narrower end of the long
axis is the tip. This is orientation-free, so it survives the rotated images (e.g. t24).
"""
import cv2
import numpy as np


def _lab(img_rgb):
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)


def _zone_stats(lab, sel):
    """Mean L*, a*, b* over a boolean pixel selection (OpenCV Lab: L in 0..255, a/b in 0..255 centred 128)."""
    if sel.sum() < 20:
        return None
    px = lab[sel]
    L = px[:, 0].mean() * 100.0 / 255.0          # -> perceptual 0..100
    a = px[:, 1].mean() - 128.0                   # -> signed, + = red
    b = px[:, 2].mean() - 128.0                   # -> signed, + = yellow
    return {"L": float(L), "a": float(a), "b": float(b), "n": int(sel.sum())}


def analyze(img_rgb, mask):
    """img_rgb: HxWx3 uint8 (letterboxed display image). mask: HxW uint8/bool, tongue=1.
    Returns a dict of zoned colour stats + derived signals, or {'ok': False} if the mask is too small."""
    mask = mask.astype(bool)
    ys, xs = np.where(mask)
    if len(xs) < 200:
        return {"ok": False}
    lab = _lab(img_rgb)

    # --- long axis via PCA on mask pixel coords (rotation-robust) ---
    pts = np.stack([xs, ys], 1).astype(np.float32)
    c = pts.mean(0)
    u, s, vt = np.linalg.svd(pts - c, full_matrices=False)
    axis = vt[0]                                   # principal (long) direction
    perp = vt[1]
    t = (pts - c) @ axis                           # coord along long axis
    w = (pts - c) @ perp                           # coord across

    # which end is the tip? the tongue tapers -> narrower cross-section. Compare mask width in the
    # far-positive vs far-negative 25% of the axis; the narrower end is the tip.
    lo, hi = np.quantile(t, 0.25), np.quantile(t, 0.75)
    end_pos = np.abs(w[t > hi]).mean() if (t > hi).any() else 1e9
    end_neg = np.abs(w[t < lo]).mean() if (t < lo).any() else 1e9
    sign = 1.0 if end_pos < end_neg else -1.0      # +1 if positive end is the tip
    tt = t * sign                                  # tt increases toward the tip
    tn = (tt - tt.min()) / (tt.max() - tt.min() + 1e-6)   # 0 (root) .. 1 (tip)

    # --- build per-pixel maps back onto the image grid ---
    axis_pos = np.full(mask.shape, np.nan, np.float32)
    axis_pos[ys, xs] = tn

    tip_sel    = mask & (axis_pos >= 0.66)
    mid_sel    = mask & (axis_pos >= 0.33) & (axis_pos < 0.66)
    root_sel   = mask & (axis_pos < 0.33)

    # centre core vs lateral edge ring, via distance transform (independent of the tip/root split)
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    dmax = dist.max() + 1e-6
    center_sel = mask & (dist >= 0.55 * dmax)
    edge_sel   = mask & (dist > 0.10 * dmax) & (dist <= 0.35 * dmax)

    zones = {
        "whole":  _zone_stats(lab, mask),
        "tip":    _zone_stats(lab, tip_sel),
        "middle": _zone_stats(lab, mid_sel),
        "root":   _zone_stats(lab, root_sel),
        "center": _zone_stats(lab, center_sel),
        "edge":   _zone_stats(lab, edge_sel),
    }
    out = {"ok": True, "zones": zones}
    W, T, R = zones["whole"], zones["tip"], zones["root"]
    C, E, M = zones["center"], zones["edge"], zones["middle"]
    if W and T:
        out["tip_redness_delta"] = round(T["a"] - W["a"], 2)
    if T and R:
        out["redness_gradient"] = round(T["a"] - R["a"], 2)
    if C and E:
        out["center_coating"] = round(C["L"] - E["L"], 2)     # centre lighter than rim => coating
        out["center_desat"]   = round((abs(E["a"]) + abs(E["b"])) - (abs(C["a"]) + abs(C["b"])), 2)

    # --- moisture via SPECULAR GLOSS -----------------------------------------------------------------
    # A wet tongue mirrors the light source as small, very-bright, desaturated glints; a dry tongue is
    # matte. We measure the fraction of tongue pixels that are specular highlights: high lightness AND
    # low chroma (a near-neutral bright spot is a reflection, not the tongue's own colour/coating).
    # We assert ONLY "wet" (the reliable extreme — validated on human-40: the high-gloss tongues are
    # genuinely moist). We deliberately do NOT infer "dry" from LOW gloss: a matte photo is usually just
    # diffuse lighting, not a dry tongue (26/38 would falsely read "dry"). True dryness needs controlled
    # capture or a texture model — left as an honest gap. Cutoff calibrated on the human-40 distribution.
    Lp = lab[mask][:, 0]                                  # 0..255
    chroma = np.sqrt((lab[mask][:, 1] - 128.0) ** 2 + (lab[mask][:, 2] - 128.0) ** 2)
    l_hi = max(210.0, np.percentile(Lp, 96))             # bright relative to THIS tongue
    spec_frac = float(((Lp >= l_hi) & (chroma <= 14.0)).mean())
    out["gloss"] = round(spec_frac, 4)                   # fraction of specular pixels
    out["moisture"] = "wet" if spec_frac >= 0.020 else "normal"   # "dry" intentionally not asserted
    return out


if __name__ == "__main__":
    import argparse, json, sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from infer import Stage1Pipeline
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--seg", default="checkpoints/seg_combined/best.pt")
    ap.add_argument("--mt", default="checkpoints/multitask_v5/best.pt")
    args = ap.parse_args()
    pipe = Stage1Pipeline(args.seg, args.mt)
    _, m, disp = pipe(args.image, return_mask=True)
    print(json.dumps(analyze(disp, m), indent=2))
