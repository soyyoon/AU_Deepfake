"""AUSequenceDataset: loads data/<video_id>/<feature_file> with per-channel
z-score normalization computed from the TRAIN split only.

`feature_file` selects the representation:
  au_sequence.npy  -> AU-only      (v1 [64,17] OpenFace, or v2 [64,20] Py-Feat)
  features.npy     -> v2 rich [64,30] (AUs + head-pose + emotions)
"""
import csv
import os

import numpy as np
import torch
from torch.utils.data import Dataset


def _npy_path(data_dir, video_id, feature_file="au_sequence.npy"):
    return os.path.join(data_dir, video_id, feature_file)


def read_splits(splits_csv):
    rows = list(csv.DictReader(open(splits_csv)))
    for r in rows:
        r["label"] = int(r["label"])
    return rows


def compute_norm_stats(rows, data_dir, seq_len, n_channels, feature_file="au_sequence.npy"):
    """Per-channel mean/std over all TRAIN frames."""
    train = [r for r in rows if r["split"] == "train"]
    chunks = []
    for r in train:
        a = np.load(_npy_path(data_dir, r["video_id"], feature_file)).astype(np.float64)
        chunks.append(a.reshape(-1, n_channels))
    allf = np.concatenate(chunks, axis=0)
    mean = allf.mean(axis=0)
    std = allf.std(axis=0)
    std[std < 1e-6] = 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


class AUSequenceDataset(Dataset):
    def __init__(self, rows, data_dir, split, mean, std,
                 seq_len=64, n_channels=17, preload=True,
                 feature_file="au_sequence.npy"):
        self.items = [r for r in rows if r["split"] == split]
        self.data_dir = data_dir
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.preload = preload
        self.feature_file = feature_file
        self._cache = None
        if preload:
            self._cache = [self._load_raw(r["video_id"]) for r in self.items]

    def _load_raw(self, video_id):
        a = np.load(_npy_path(self.data_dir, video_id, self.feature_file)).astype(np.float32)
        if a.shape != (self.seq_len, self.n_channels):
            # pad/truncate defensively
            out = np.zeros((self.seq_len, self.n_channels), dtype=np.float32)
            t = min(self.seq_len, a.shape[0])
            out[:t] = a[:t, : self.n_channels]
            a = out
        return a

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        raw = self._cache[i] if self.preload else self._load_raw(self.items[i]["video_id"])
        x = (raw - self.mean) / self.std
        y = float(self.items[i]["label"])
        # torch.as_tensor (not from_numpy): this env's numpy<->torch ABI trips
        # from_numpy's strict type check but as_tensor handles it correctly.
        return (torch.as_tensor(x, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32),
                self.items[i]["source"])
