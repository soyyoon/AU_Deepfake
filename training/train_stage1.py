"""
Stage 1 — Xception artifact detector.

Usage:
    cd training && python train_stage1.py --config configs/stage1.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# resolve imports when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasets.frame_dataset import FrameDataset, build_transforms
from utils.common import (
    CheckpointSaver, compute_metrics, compute_pos_weight, load_config, set_seed
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def build_model(cfg: dict) -> nn.Module:
    backbone = timm.create_model(
        cfg["model"]["backbone"],
        pretrained=cfg["model"]["pretrained"],
        num_classes=0,
        global_pool="avg",
    )
    feat = backbone.num_features
    head = nn.Sequential(
        nn.Linear(feat, 256),
        nn.GELU(),
        nn.Dropout(cfg["model"]["dropout"]),
        nn.Linear(256, 1),
    )
    return nn.Sequential(backbone, head)


def make_loader(df: pd.DataFrame, cfg: dict, train: bool) -> DataLoader:
    ds = FrameDataset(
        df=df,
        processed_root=cfg["data"]["processed_root"],
        frames_per_video=cfg["data"]["frames_per_video"],
        frames_sampled_per_epoch=cfg["data"]["frames_sampled_per_epoch"],
        transform=build_transforms(cfg["data"]["image_size"], train=train),
    )
    return DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=train,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        drop_last=train,
    )


@torch.no_grad()
def evaluate(model, loader, device, threshold: float):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logit = model(x).squeeze(-1)
        probs.append(torch.sigmoid(logit).cpu().numpy())
        labels.append(y.numpy())
    return compute_metrics(np.concatenate(probs), np.concatenate(labels), threshold)


def train_one_epoch(model, loader, opt, loss_fn, scaler, device, epoch: int):
    model.train()
    total, count = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logit = model(x).squeeze(-1)
                loss = loss_fn(logit, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            logit = model(x).squeeze(-1)
            loss = loss_fn(logit, y)
            loss.backward()
            opt.step()

        total += loss.item() * x.size(0)
        count += x.size(0)
    avg = total / max(count, 1)
    log.info("epoch %d  train_loss=%.4f", epoch, avg)
    return avg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(cfg["data"]["metadata_csv"])
    train_df = df[df.split == "train"].reset_index(drop=True)
    val_df = df[df.split == "val"].reset_index(drop=True)
    log.info("videos: train=%d  val=%d", len(train_df), len(val_df))

    train_loader = make_loader(train_df, cfg, train=True)
    val_loader = make_loader(val_df, cfg, train=False)

    model = build_model(cfg).to(device)
    log.info("params=%.2fM", sum(p.numel() for p in model.parameters()) / 1e6)

    pw = cfg["train"]["pos_weight"]
    if pw is None:
        pw = compute_pos_weight(train_df["label"].values)
        log.info("auto pos_weight=%.3f", pw)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=device))

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["train"]["epochs"])
    scaler = torch.cuda.amp.GradScaler() if cfg["train"]["amp"] and device == "cuda" else None

    saver = CheckpointSaver(cfg["output"]["ckpt_dir"], cfg["output"]["best_metric"])
    no_improve = 0

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        # resample frames each epoch
        train_loader.dataset._regen_index()

        train_one_epoch(model, train_loader, opt, loss_fn, scaler, device, epoch)
        m = evaluate(model, val_loader, device, cfg["eval"]["threshold"])
        log.info(
            "epoch %d  val auc=%.4f  ap=%.4f  f1=%.4f  acc=%.4f",
            epoch, m.auc, m.ap, m.f1, m.acc,
        )
        sched.step()

        if saver.save_if_best(model, epoch, m):
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg["train"]["early_stop_patience"]:
                log.info("early stop at epoch %d", epoch)
                break

    log.info("done. best %s=%.4f", saver.metric_name, saver.best)


if __name__ == "__main__":
    main()
