"""
Video-level AU sequence Dataset for Stage 2.

Each sample = (au_sequence (T, 17), label).
Reads {processed_root}/{video_id}/au_sequence.npy, already shape (64, 17),
already normalized to [0, 1] in preprocessing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class AUSequenceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        processed_root: Path | str,
        seq_len: int = 64,
        num_aus: int = 17,
        augment: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.root = Path(processed_root)
        self.seq_len = seq_len
        self.num_aus = num_aus
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        au_path = self.root / row["video_id"] / "au_sequence.npy"

        seq = np.load(au_path).astype(np.float32)  # (T, 17)
        # safety: enforce expected shape
        if seq.shape[0] != self.seq_len:
            if seq.shape[0] > self.seq_len:
                seq = seq[: self.seq_len]
            else:
                pad = np.tile(seq[-1:], (self.seq_len - seq.shape[0], 1))
                seq = np.concatenate([seq, pad], axis=0)
        if seq.shape[1] != self.num_aus:
            raise ValueError(
                f"AU dim mismatch: got {seq.shape[1]}, expected {self.num_aus}"
            )

        if self.augment:
            seq = self._augment(seq)

        seq_t = torch.from_numpy(seq)                     # (64, 17)
        label = torch.tensor(row["label"], dtype=torch.float32)
        return seq_t, label

    def _augment(self, seq: np.ndarray) -> np.ndarray:
        # mild Gaussian noise on AU intensities
        if np.random.rand() < 0.5:
            seq = seq + np.random.normal(0, 0.01, seq.shape).astype(np.float32)
            seq = np.clip(seq, 0.0, 1.0)
        # temporal jitter: shift by ±2 frames
        if np.random.rand() < 0.3:
            shift = np.random.randint(-2, 3)
            if shift != 0:
                seq = np.roll(seq, shift, axis=0)
        return seq
