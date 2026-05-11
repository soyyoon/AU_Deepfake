"""
Stage 2 — AU dual-head temporal model.

Loss:
    total = BCE(seq_logit, video_label)
          + lambda_frame * mean_t BCE(frame_logit_t, video_label)

The frame loss broadcasts the video label to all 64 frames. This is
weak supervision but it forces the frame head to be predictive on its
own — which is what gets used when the Chrome extension hits us with
a single image (1 frame).

Usage:
    cd training && python train_stage2.py --config configs/stage2.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasets.au_dataset import AUSequenceDataset
from models.au_temporal import AUDualHeadNet
from utils.common import (
    CheckpointSaver, compute_metrics, compute_pos_weight, load_config, set_seed
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def make_loader(df: pd.DataFrame, cfg: dict, train: bool) -> DataLoader:
    ds = AUSequenceDataset(
        df=df,
        processed_root=cfg["data"]["processed_root"],
        seq_len=cfg["data"]["seq_len"],
        num_aus=cfg["data"]["num_aus"],
        augment=train,
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
def evaluate(model, loader, device, threshold: float, head: str = "seq"):
    model.eval()
    probs, labels = [], []
    for seq, y in loader:
        seq = seq.to(device, non_blocking=True)
        out = model(seq)
        if head == "seq":
            p = torch.sigmoid(out["seq_logit"])
        else:
            # frame head -> mean over time as video-level prediction
            p = torch.sigmoid(out["frame_logit"]).mean(dim=1)
        probs.append(p.cpu().numpy())
        labels.append(y.numpy())
    return compute_metrics(np.concatenate(probs), np.concatenate(labels), threshold)


def train_one_epoch(model, loader, opt, scaler, device, lambda_frame: float, pos_weight, epoch: int):
    model.train()
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    total, count = 0.0, 0
    for seq, y in loader:
        seq = seq.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)               # (B,)
        y_frame = y.unsqueeze(1).expand(-1, seq.size(1))   # (B, T)

        opt.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                out = model(seq)
                loss_seq = bce(out["seq_logit"], y)
                loss_frame = bce(out["frame_logit"], y_frame)
                loss = loss_seq + lambda_frame * loss_frame
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            out = model(seq)
            loss_seq = bce(out["seq_logit"], y)
            loss_frame = bce(out["frame_logit"], y_frame)
            loss = loss_seq + lambda_frame * loss_frame
            loss.backward()
            opt.step()

        total += loss.item() * seq.size(0)
        count += seq.size(0)
    log.info("epoch %d  train_loss=%.4f", epoch, total / max(count, 1))


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

    model = AUDualHeadNet(
        num_aus=cfg["data"]["num_aus"],
        trunk_dim=cfg["model"]["trunk_dim"],
        num_layers=cfg["model"]["num_layers"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    log.info("params=%.2fM", sum(p.numel() for p in model.parameters()) / 1e6)

    pw = cfg["train"]["pos_weight"]
    if pw is None:
        pw = compute_pos_weight(train_df["label"].values)
        log.info("auto pos_weight=%.3f", pw)
    pos_weight = torch.tensor(pw, device=device)

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
        train_one_epoch(
            model, train_loader, opt, scaler, device,
            cfg["train"]["lambda_frame"], pos_weight, epoch,
        )

        m_seq = evaluate(model, val_loader, device, cfg["eval"]["threshold"], head="seq")
        m_frm = evaluate(model, val_loader, device, cfg["eval"]["threshold"], head="frame")
        log.info(
            "epoch %d  seq[auc=%.4f f1=%.4f]  frame[auc=%.4f f1=%.4f]",
            epoch, m_seq.auc, m_seq.f1, m_frm.auc, m_frm.f1,
        )
        sched.step()

        # checkpoint by sequence head AUC (the stronger signal)
        if saver.save_if_best(model, epoch, m_seq, extra={"frame_metrics": m_frm.to_dict()}):
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg["train"]["early_stop_patience"]:
                log.info("early stop at epoch %d", epoch)
                break

    log.info("done. best %s=%.4f", saver.metric_name, saver.best)


if __name__ == "__main__":
    main()
