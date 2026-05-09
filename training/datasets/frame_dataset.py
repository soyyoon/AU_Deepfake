"""
Frame-level Dataset for Stage 1 (Xception) training.

Reads the dataset_metadata.csv produced by the preprocessing notebook and
yields (image, label) pairs. Each video has 64 saved frames in
{processed_root}/{video_id}/frames/0000.png ... 0063.png.

To keep training fast and balanced, we sample a fixed number of frames
per video per epoch (`frames_sampled_per_epoch`).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class FrameDataset(Dataset):
    """
    One sample = one frame.
    Length = len(videos) * frames_sampled_per_epoch (resampled each epoch).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        processed_root: Path | str,
        frames_per_video: int = 64,
        frames_sampled_per_epoch: int = 8,
        transform=None,
    ):
        self.df = df.reset_index(drop=True)
        self.root = Path(processed_root)
        self.frames_per_video = frames_per_video
        self.frames_sampled = frames_sampled_per_epoch
        self.transform = transform
        self._regen_index()

    def _regen_index(self):
        """Resample the (video_idx, frame_idx) pairs for a new epoch."""
        idx = []
        for v_idx in range(len(self.df)):
            picks = random.sample(
                range(self.frames_per_video),
                k=min(self.frames_sampled, self.frames_per_video),
            )
            for f in picks:
                idx.append((v_idx, f))
        self._index = idx

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i):
        v_idx, frame_idx = self._index[i]
        row = self.df.iloc[v_idx]
        path = self.root / row["video_id"] / "frames" / f"{frame_idx:04d}.png"

        img = cv2.imread(str(path))
        if img is None:
            # rare — fall back to another frame
            return self.__getitem__((i + 1) % len(self))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(image=img)["image"] if hasattr(
                self.transform, "__call__") and "image" in (
                    getattr(self.transform, "__call__", None).__code__.co_varnames
                    if hasattr(self.transform, "__call__") else []) else self.transform(img)
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        label = torch.tensor(row["label"], dtype=torch.float32)
        return img, label


def build_transforms(image_size: int, train: bool):
    """
    Lightweight torchvision transforms (no extra deps).
    For stronger augmentations, swap with albumentations.
    """
    from torchvision import transforms as T
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if train:
        return T.Compose([
            T.ToPILImage(),
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    return T.Compose([
        T.ToPILImage(),
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
