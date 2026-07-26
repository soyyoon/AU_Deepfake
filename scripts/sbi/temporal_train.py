#!/usr/bin/env python
"""Temporal 브랜치 학습: real 클립 -> 온더플라이 temporal fake -> CNN+GRU 클립 분류.

reenactment/lip-sync의 시간 불일치를 학습(SBI 공간 블렌딩의 보완). sbi env.
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 conda run -n sbi python scripts/sbi/temporal_train.py \
      --clips data_sbi/clips --size 160 --epochs 20 --out outputs/temporal
"""
import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from temporal_dataset import TemporalDataset, build_clip_manifest   # noqa: E402


class TemporalNet(nn.Module):
    def __init__(self, backbone="efficientnet_b0", hidden=256):
        super().__init__()
        import timm
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0, global_pool="avg")
        d = self.backbone.num_features
        self.gru = nn.GRU(d, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.LayerNorm(2 * hidden), nn.Dropout(0.2), nn.Linear(2 * hidden, 1))

    def forward(self, x):                      # x: [B,T,3,H,W]
        B, T, C, H, W = x.shape
        f = self.backbone(x.reshape(B * T, C, H, W)).reshape(B, T, -1)
        out, _ = self.gru(f)
        return self.head(out.mean(1)).squeeze(1)


def roc_auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    p, n = (y == 1).sum(), (y == 0).sum()
    if p == 0 or n == 0:
        return float("nan")
    r = np.argsort(np.argsort(s)) + 1
    return float((r[y == 1].sum() - p * (p + 1) / 2) / (p * n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="data_sbi/clips")
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--out", default="outputs/temporal")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.out).mkdir(parents=True, exist_ok=True)

    items = build_clip_manifest(args.clips)
    vids = sorted({v for _, v in items})
    random.Random(args.seed).shuffle(vids)
    val_vids = set(vids[:max(1, int(len(vids) * args.val_frac))])
    tr = [it for it in items if it[1] not in val_vids]
    va = [it for it in items if it[1] in val_vids]
    print(f"clips: {len(items)} | videos: {len(vids)} | train {len(tr)} / val {len(va)}", flush=True)

    tr_dl = DataLoader(TemporalDataset(tr, args.size, True), args.batch, shuffle=True,
                       num_workers=args.workers, pin_memory=True, drop_last=True)
    va_dl = DataLoader(TemporalDataset(va, args.size, True), args.batch, shuffle=False,
                       num_workers=args.workers, pin_memory=True)

    model = TemporalNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")
    crit = nn.BCEWithLogitsLoss()

    best = -1.0
    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); tot = 0.0
        for x, y in tr_dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tot += loss.item()
        sched.step()
        model.eval(); ys, ss = [], []
        with torch.no_grad():
            for x, y in va_dl:
                with torch.amp.autocast("cuda"):
                    p = torch.sigmoid(model(x.to(dev)))
                ss.append(p.float().cpu().numpy()); ys.append(y.numpy())
        auc = roc_auc(np.concatenate(ys), np.concatenate(ss))
        print(f"[ep {ep:02d}/{args.epochs}] loss={tot/len(tr_dl):.4f} val_temporal_auc={auc:.4f} "
              f"{time.time()-t0:.0f}s", flush=True)
        ckpt = {"model": model.state_dict(), "cfg": vars(args), "epoch": ep, "val_auc": auc}
        torch.save(ckpt, Path(args.out) / "last.pt")
        if auc > best:
            best = auc; torch.save(ckpt, Path(args.out) / "best.pt")
    print(f"done. best={best:.4f} -> {args.out}/best.pt")


if __name__ == "__main__":
    main()
