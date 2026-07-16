"""Benchmark Memory-SAM (SAM2 + retrieval-to-prompt) on the SM-Tongue test split.

DINOv3 weights are gated by Meta, so this uses a drop-in DINOv2-L/14 feature extractor (same
`get_intermediate_layers` interface, not gated) via a monkeypatch — the cloned memory-sam repo is
left untouched. Builds the memory bank from SM-Tongue's 20 reference examples, then segments each
test image and reports Dice/IoU against the ground-truth masks.

    python evaluation/eval_memory_sam.py --repo repos/memory-sam \
        --sam repos/memory-sam/checkpoints/sam2.1_hiera_large.pt \
        --mem-manifest data/external/sm_tongue/SM-Tongue-2155-anonymized/memory-sm_tongue/manifest.csv \
        --mem-root data/external/sm_tongue/SM-Tongue-2155-anonymized/memory-sm_tongue \
        --seg-manifest data/processed/seg_manifest.csv --limit 215
"""
import argparse
import os
import sys
import csv
import tempfile
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image


class DINOv2FeatureExtractor:
    """Drop-in replacement for memory-sam's DINOv3FeatureExtractor (non-gated DINOv2-L/14)."""
    def __init__(self, model_name="dinov2_vitl14", repo_dir=None, weights_dir=None,
                 image_size=512, patch_size=14, device="cuda"):
        self.image_size = int(image_size)
        self.patch_size = 14
        self.device = device
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.n_layers = {"dinov2_vits14": 12, "dinov2_vitb14": 12,
                         "dinov2_vitl14": 24, "dinov2_vitg14": 40}.get(model_name, 24)
        self.model = torch.hub.load("facebookresearch/dinov2", model_name).to(device).eval()

    def prepare_image(self, image):
        pil = Image.fromarray(image)
        w, h = pil.size
        h_p = int(self.image_size / self.patch_size)
        w_p = int((w * self.image_size) / (h * self.patch_size))
        t = TF.to_tensor(TF.resize(pil, (h_p * self.patch_size, w_p * self.patch_size)))
        return TF.normalize(t, self.mean, self.std), (h_p, w_p)

    def _layers(self):
        return [11, 17, 23] if self.n_layers >= 24 else \
            [self.n_layers - 3, self.n_layers - 2, self.n_layers - 1]

    def extract_patch_features(self, image):
        t, grid = self.prepare_image(image)
        with torch.no_grad():
            feats = self.model.get_intermediate_layers(
                t.unsqueeze(0).to(self.device), n=self._layers(), reshape=True, norm=True)
            fused = torch.cat(feats, dim=1).squeeze(0).float().cpu()
        d = fused.shape[0]
        p = fused.reshape(d, -1).T.numpy().astype(np.float32)
        p /= np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-8)
        return p, grid

    def extract_global_features(self, image):
        t, _ = self.prepare_image(image)
        with torch.no_grad():
            out = self.model.forward_features(t.unsqueeze(0).to(self.device))
            desc = F.normalize(out["x_norm_clstoken"][0], p=2, dim=0)
        return desc.cpu().numpy().astype(np.float32)


def dice_iou(pred, gt):
    p = (pred > 0).astype(np.uint8); g = (gt > 127).astype(np.uint8)
    tp = (p & g).sum(); fp = (p & (1 - g)).sum(); fn = ((1 - p) & g).sum()
    inter = tp; uni = (p | g).sum()
    return 2 * tp / (2 * tp + fp + fn + 1e-6), inter / (uni + 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="repos/memory-sam")
    ap.add_argument("--sam", default="repos/memory-sam/checkpoints/sam2.1_hiera_large.pt")
    ap.add_argument("--mem-manifest", required=True)
    ap.add_argument("--mem-root", required=True)
    ap.add_argument("--seg-manifest", default="data/processed/seg_manifest.csv")
    ap.add_argument("--limit", type=int, default=215)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Pre-import the pip-installed sam2 so its repo-shadowing guard passes before memory-sam's
    # ensure_third_party_imports() prepends the repo path (which would shadow the package).
    import sam2.build_sam  # noqa: F401
    sys.path.insert(0, os.path.abspath(args.repo))
    import mt_sam.predictor as P
    P.DINOv3FeatureExtractor = DINOv2FeatureExtractor          # monkeypatch: DINOv2 fallback
    from mt_sam.predictor import MTSAMPredictor, MTSAMConfig

    memdir = tempfile.mkdtemp(prefix="mtsam_mem_")
    cfg = MTSAMConfig(sam_checkpoint=os.path.abspath(args.sam),
                      sam_config="configs/sam2.1/sam2.1_hiera_l",
                      dinov3_model="dinov2_vitl14", memory_dir=memdir, device=device)
    predictor = MTSAMPredictor(cfg)

    # Build memory bank from SM-Tongue's reference examples
    with open(args.mem_manifest) as f:
        refs = list(csv.DictReader(f))
    for r in refs:
        img = np.array(Image.open(os.path.join(args.mem_root, r["image"])).convert("RGB"))
        msk = np.array(Image.open(os.path.join(args.mem_root, r["mask"])).convert("L"))
        predictor.add_reference(img, msk)
    print(f"memory bank: {len(refs)} references")

    # Evaluate on SM-Tongue test split
    df = pd.read_csv(args.seg_manifest)
    test = df[(df.split == "test") & (df.source == "sm_tongue")].reset_index(drop=True)
    if args.limit:
        test = test.iloc[:args.limit]
    dices, ious, lat = [], [], []
    for _, row in test.iterrows():
        img = np.array(Image.open(row.img_path).convert("RGB"))
        gt = np.array(Image.open(row.mask_path).convert("L"))
        out = predictor.predict_array(img)
        pred = cv2.resize(out["mask"].astype(np.uint8), (gt.shape[1], gt.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
        d, i = dice_iou(pred, gt)
        dices.append(d); ious.append(i); lat.append(out["latency_ms"])
    print(f"n={len(dices)}  Memory-SAM(DINOv2)  mean Dice={np.mean(dices):.4f}  "
          f"mean IoU={np.mean(ious):.4f}  Dice<0.9: {np.mean(np.array(dices)<0.9)*100:.1f}%  "
          f"latency={np.mean(lat):.0f}ms")


if __name__ == "__main__":
    main()
