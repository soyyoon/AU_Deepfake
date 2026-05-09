"""
End-to-end cascade evaluation.

Loads both stage checkpoints and reports:
  - Stage 1 alone (baseline)
  - Stage 2 (frame head) alone (baseline)
  - Stage 2 (sequence head) alone (baseline)
  - Cascade: Stage1 -> if uncertain band -> Stage2 (frame head)
            with combination weights w1*p1 + w2*p2

Useful for tuning the (low, high) uncertainty band and the weights.

Usage:
    python eval.py \\
      --stage1-ckpt checkpoints/stage1/best.pt \\
      --stage2-ckpt checkpoints/stage2/best.pt \\
      --metadata-csv /kaggle/working/processed/dataset_metadata.csv \\
      --processed-root /kaggle/working/processed \\
      --split test
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasets.frame_dataset import FrameDataset, build_transforms
from datasets.au_dataset import AUSequenceDataset
from models.au_temporal import AUDualHeadNet
from utils.common import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_stage1(ckpt_path: str, device: str) -> nn.Module:
    backbone = timm.create_model("xception", pretrained=False, num_classes=0, global_pool="avg")
    head = nn.Sequential(
        nn.Linear(backbone.num_features, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 1)
    )
    model = nn.Sequential(backbone, head).to(device).eval()
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    return model


def load_stage2(ckpt_path: str, device: str) -> AUDualHeadNet:
    model = AUDualHeadNet().to(device).eval()
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    return model


@torch.no_grad()
def video_level_p_stage1(model, df, root, image_size, device, batch_size=32) -> np.ndarray:
    """Stage 1 video-level prob = mean of per-frame probs (all 64 frames)."""
    ds = FrameDataset(
        df=df,
        processed_root=root,
        frames_per_video=64,
        frames_sampled_per_epoch=64,  # use all frames
        transform=build_transforms(image_size, train=False),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    probs = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        p = torch.sigmoid(model(x).squeeze(-1)).cpu().numpy()
        probs.append(p)
    probs = np.concatenate(probs)              # (N_videos * 64,)
    return probs.reshape(len(df), 64).mean(axis=1)  # (N_videos,)


@torch.no_grad()
def video_level_p_stage2(model: AUDualHeadNet, df, root, device, batch_size=64):
    ds = AUSequenceDataset(df=df, processed_root=root, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    seq_probs, frame_probs = [], []
    for seq, _ in loader:
        seq = seq.to(device, non_blocking=True)
        out = model(seq)
        seq_probs.append(torch.sigmoid(out["seq_logit"]).cpu().numpy())
        frame_probs.append(torch.sigmoid(out["frame_logit"]).mean(dim=1).cpu().numpy())
    return np.concatenate(seq_probs), np.concatenate(frame_probs)


def cascade_combine(p1, p2_frame, low, high, w1, w2):
    p = p1.copy()
    mask = (p1 >= low) & (p1 <= high)
    p[mask] = w1 * p1[mask] + w2 * p2_frame[mask]
    return p, mask.mean()  # also return fraction routed to stage2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-ckpt", required=True)
    ap.add_argument("--stage2-ckpt", required=True)
    ap.add_argument("--metadata-csv", required=True)
    ap.add_argument("--processed-root", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--image-size", type=int, default=299)
    ap.add_argument("--low", type=float, default=0.4)
    ap.add_argument("--high", type=float, default=0.6)
    ap.add_argument("--w1", type=float, default=0.4)
    ap.add_argument("--w2", type=float, default=0.6)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(args.metadata_csv)
    df = df[df.split == args.split].reset_index(drop=True)
    y = df["label"].values
    log.info("split=%s  n=%d  pos=%d", args.split, len(df), int((y == 1).sum()))

    # --- run both stages ---
    log.info("running stage1 ...")
    stage1 = load_stage1(args.stage1_ckpt, device)
    p1 = video_level_p_stage1(stage1, df, args.processed_root, args.image_size, device)

    log.info("running stage2 ...")
    stage2 = load_stage2(args.stage2_ckpt, device)
    p2_seq, p2_frame = video_level_p_stage2(stage2, df, args.processed_root, device)

    # --- baselines ---
    log.info("--- baselines ---")
    for name, p in [("stage1", p1), ("stage2_seq", p2_seq), ("stage2_frame", p2_frame)]:
        m = compute_metrics(p, y)
        log.info("%-13s auc=%.4f  ap=%.4f  f1=%.4f  acc=%.4f",
                 name, m.auc, m.ap, m.f1, m.acc)

    # --- cascade ---
    p_cascade, routed = cascade_combine(p1, p2_frame, args.low, args.high, args.w1, args.w2)
    m = compute_metrics(p_cascade, y)
    log.info("--- cascade ---")
    log.info("low=%.2f high=%.2f w1=%.2f w2=%.2f  -> %.1f%% routed to stage2",
             args.low, args.high, args.w1, args.w2, routed * 100)
    log.info("cascade       auc=%.4f  ap=%.4f  f1=%.4f  acc=%.4f",
             m.auc, m.ap, m.f1, m.acc)


if __name__ == "__main__":
    main()
