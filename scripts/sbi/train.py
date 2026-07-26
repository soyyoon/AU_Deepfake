#!/usr/bin/env python
"""SBI 학습: real 크롭 -> 온더플라이 self-blend -> EfficientNet-b4 이진 분류.

sbi env(cuDNN 정상). 프레임 단위 학습. 비디오-레벨 평가는 별도(eval_sbi).
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 conda run -n sbi python scripts/sbi/train.py \
      --root data_sbi/real --model tf_efficientnet_b4 --size 256 --epochs 20 --out outputs/sbi
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
from dataset import SBIDataset, build_manifest    # noqa: E402


def roc_auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    p, n = (y == 1).sum(), (y == 0).sum()
    if p == 0 or n == 0:
        return float("nan")
    r = np.argsort(np.argsort(s)) + 1
    return float((r[y == 1].sum() - p * (p + 1) / 2) / (p * n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data_sbi/real")
    ap.add_argument("--model", default="tf_efficientnet_b4")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--out", default="outputs/sbi")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import timm
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.out).mkdir(parents=True, exist_ok=True)

    items = build_manifest(args.root)
    vids = sorted({v for _, _, v in items})
    random.Random(args.seed).shuffle(vids)
    n_val = max(1, int(len(vids) * args.val_frac))
    val_vids = set(vids[:n_val])
    tr = [it for it in items if it[2] not in val_vids]
    va = [it for it in items if it[2] in val_vids]
    print(f"crops: {len(items)} | videos: {len(vids)} | train {len(tr)} / val {len(va)}", flush=True)

    tr_ds = SBIDataset(tr, args.size, train=True)
    va_ds = SBIDataset(va, args.size, train=True)   # val도 랜덤 50/50 fake로 AUC 모니터
    tr_dl = DataLoader(tr_ds, args.batch, shuffle=True, num_workers=args.workers,
                       pin_memory=True, drop_last=True)
    va_dl = DataLoader(va_ds, args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = timm.create_model(args.model, pretrained=True, num_classes=1).to(dev)
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
                logit = model(x).squeeze(1)
                loss = crit(logit, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += loss.item()
        sched.step()

        model.eval(); ys, ss = [], []
        with torch.no_grad():
            for x, y in va_dl:
                with torch.amp.autocast("cuda"):
                    p = torch.sigmoid(model(x.to(dev)).squeeze(1))
                ss.append(p.float().cpu().numpy()); ys.append(y.numpy())
        auc = roc_auc(np.concatenate(ys), np.concatenate(ss))
        print(f"[ep {ep:02d}/{args.epochs}] loss={tot/len(tr_dl):.4f} val_SBI_auc={auc:.4f} "
              f"lr={opt.param_groups[0]['lr']:.2e} {time.time()-t0:.0f}s", flush=True)

        ckpt = {"model": model.state_dict(), "cfg": vars(args), "epoch": ep, "val_sbi_auc": auc}
        torch.save(ckpt, Path(args.out) / "last.pt")
        if auc > best:
            best = auc
            torch.save(ckpt, Path(args.out) / "best.pt")
    print(f"done. best val_SBI_auc={best:.4f} -> {args.out}/best.pt")


if __name__ == "__main__":
    main()
