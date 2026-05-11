"""
Pre-flight sanity check. Run this BEFORE training.

Verifies:
  - dataset_metadata.csv exists and has required columns
  - split column has train/val/test
  - At least one video per split is loadable
  - frames/ directory has 64 PNGs per video
  - au_sequence.npy is shape (64, 17), values in [0, 1]
  - frame and AU dataset classes both work end-to-end

Usage:
    python scripts/sanity_check.py \\
      --metadata-csv /kaggle/working/processed/dataset_metadata.csv \\
      --processed-root /kaggle/working/processed
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def fail(msg):
    print(f"\033[31m[FAIL]\033[0m {msg}")
    sys.exit(1)


def ok(msg):
    print(f"\033[32m[ OK ]\033[0m {msg}")


def warn(msg):
    print(f"\033[33m[WARN]\033[0m {msg}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata-csv", required=True)
    p.add_argument("--processed-root", required=True)
    args = p.parse_args()

    csv = Path(args.metadata_csv)
    root = Path(args.processed_root)

    # 1. csv exists ----------------------------------------------------
    if not csv.exists():
        fail(f"metadata csv not found: {csv}")
    df = pd.read_csv(csv)
    ok(f"metadata csv loaded: {len(df)} rows")

    required_cols = {"video_id", "label", "source", "split"}
    missing = required_cols - set(df.columns)
    if missing:
        fail(f"missing columns: {missing}")
    ok(f"columns OK: {list(df.columns)}")

    # 2. split distribution -------------------------------------------
    print("\nSplit distribution:")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0))
    for split in ["train", "val", "test"]:
        n = (df.split == split).sum()
        if n == 0:
            warn(f"split='{split}' has 0 videos")
        else:
            ok(f"split='{split}': {n} videos")

    # 3. label distribution -------------------------------------------
    n_real = (df.label == 0).sum()
    n_fake = (df.label == 1).sum()
    ratio = n_fake / max(n_real, 1)
    ok(f"labels: real={n_real}, fake={n_fake}, fake/real={ratio:.2f}")
    if ratio > 5 or ratio < 0.2:
        warn("severe class imbalance — pos_weight will help but consider sampling")

    # 4. probe one video per split ------------------------------------
    print()
    for split in ["train", "val", "test"]:
        sub = df[df.split == split]
        if len(sub) == 0:
            continue
        row = sub.iloc[0]
        vid = row["video_id"]
        vdir = root / vid

        # frames dir
        frames_dir = vdir / "frames"
        if not frames_dir.exists():
            fail(f"[{split}] {vid}: frames/ missing")
        pngs = sorted(frames_dir.glob("*.png"))
        if len(pngs) != 64:
            warn(f"[{split}] {vid}: expected 64 PNGs, got {len(pngs)}")
        else:
            ok(f"[{split}] {vid}: 64 frames OK")

        # AU file
        au_path = vdir / "au_sequence.npy"
        if not au_path.exists():
            fail(f"[{split}] {vid}: au_sequence.npy missing")
        au = np.load(au_path)
        if au.shape != (64, 17):
            fail(f"[{split}] {vid}: AU shape {au.shape}, expected (64, 17)")
        if au.min() < 0 or au.max() > 1.0:
            fail(f"[{split}] {vid}: AU range [{au.min()}, {au.max()}], expected [0, 1]")
        ok(f"[{split}] {vid}: AU shape={au.shape} range=[{au.min():.3f}, {au.max():.3f}]")

    # 5. exercise dataset classes -------------------------------------
    print()
    from datasets.frame_dataset import FrameDataset, build_transforms
    from datasets.au_dataset import AUSequenceDataset

    train_df = df[df.split == "train"].head(2)
    if len(train_df) == 0:
        fail("no train data to test datasets")

    fd = FrameDataset(
        df=train_df, processed_root=root,
        frames_per_video=64, frames_sampled_per_epoch=2,
        transform=build_transforms(image_size=299, train=False),
    )
    img, lbl = fd[0]
    ok(f"FrameDataset: img tensor {tuple(img.shape)}, label={lbl.item()}")

    ad = AUSequenceDataset(df=train_df, processed_root=root)
    seq, lbl = ad[0]
    ok(f"AUSequenceDataset: seq tensor {tuple(seq.shape)}, label={lbl.item()}")

    print("\n\033[32mAll sanity checks passed. Ready to train.\033[0m")


if __name__ == "__main__":
    main()
