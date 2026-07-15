"""Self-supervised (SimCLR-style) DOMAIN-ADAPTIVE pretraining of the ResNet-34 encoder on UNLABELED
tongue images (TonguExpert + SM real photos + TCM), to close the clinical->phone domain gap
(docs/LABEL_IMPROVEMENT_PLAN.md Path C). The encoder is the SAME timm resnet34 the multitask model uses,
so the pretrained weights drop straight in via `train.py --init`.

CRITICAL: augmentations are COLOUR-PRESERVING (no hue/saturation jitter, no grayscale) — standard SimCLR
makes features colour-INVARIANT, which would wreck our colour-sensitive tai/zhi task. We keep geometry +
blur + mild brightness only, so the encoder still learns tongue structure/texture that transfers to real
photos while preserving colour information for the downstream heads.

    CUDA_VISIBLE_DEVICES=0 python stage1_quantitative/feature_extraction/ssl_pretrain.py \
        --epochs 40 --batch-size 256 --img-size 224 --out checkpoints/ssl_resnet34
"""
import argparse, glob, os, time
import cv2, torch, torch.nn as nn, torch.nn.functional as F, timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def image_pool():
    imgs = glob.glob("data/raw/TongueImage/Raw/*.jpg")
    imgs += [p for p in glob.glob("data/external/sm_tongue/**/*.png", recursive=True)
             if "mask" not in p.lower() and "overlay" not in p.lower()]
    imgs += glob.glob("data/external/tcm_tongue/shezhenv3-txt/train/images/*.jpg")
    return imgs


def aug(size):
    return A.Compose([
        A.RandomResizedCrop(size=(size, size), scale=(0.5, 1.0), ratio=(0.8, 1.25)),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.6),      # lighting robustness (brightness, not hue)
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.ImageCompression(quality_lower=50, quality_upper=95, p=0.3),
        A.Normalize(MEAN, STD), ToTensorV2()])            # NO hue/sat/grayscale -> colour preserved


class TwoView(Dataset):
    def __init__(self, paths, size):
        self.paths, self.tf = paths, aug(size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.imread(self.paths[i])
        if img is None:
            img = cv2.imread(self.paths[(i + 1) % len(self.paths)])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.tf(image=img)["image"], self.tf(image=img)["image"]


class SSLNet(nn.Module):
    def __init__(self, encoder="resnet34", dim=128):
        super().__init__()
        self.encoder = timm.create_model(encoder, pretrained=True, features_only=True)
        c = self.encoder.feature_info.channels()[-1]
        self.proj = nn.Sequential(nn.Linear(c, c), nn.BatchNorm1d(c), nn.ReLU(inplace=True), nn.Linear(c, dim))

    def forward(self, x):
        f = self.encoder(x)[-1].mean(dim=(2, 3))          # global-avg-pool last feature map
        return F.normalize(self.proj(f), dim=1)


def nt_xent(z1, z2, temp=0.2):
    B = z1.shape[0]
    z = torch.cat([z1, z2], 0)
    sim = (z @ z.T) / temp
    sim.fill_diagonal_(-1e4)                               # Half-safe mask (logits are ~[-5,5])
    tgt = torch.cat([torch.arange(B) + B, torch.arange(B)]).to(z.device)
    return F.cross_entropy(sim, tgt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="checkpoints/ssl_resnet34")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    paths = image_pool()
    ds = TwoView(paths, args.img_size)
    dl = DataLoader(ds, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    print(f"SSL pretrain: {len(paths)} unlabeled tongues, {len(dl)} steps/epoch, size={args.img_size}", flush=True)

    model = SSLNet(args.encoder).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()

    for ep in range(args.epochs):
        model.train(); run = 0.0; t = time.time()
        for a, b in dl:
            a, b = a.to(device, non_blocking=True), b.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                loss = nt_xent(model(a), model(b))
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run += loss.item()
        sched.step()
        print(f"ep {ep+1:02d}/{args.epochs} loss={run/len(dl):.3f} ({time.time()-t:.0f}s)", flush=True)
        # save the encoder with an 'encoder.' prefix so train.py --init loads it into MultiTaskTongueNet
        enc = {"encoder." + k: v for k, v in model.encoder.state_dict().items()}
        torch.save({"model": enc, "args": {"encoder": args.encoder}}, os.path.join(args.out, "best.pt"))
    print("done ->", os.path.join(args.out, "best.pt"), flush=True)


if __name__ == "__main__":
    main()
