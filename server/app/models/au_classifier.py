"""
Stage 2 frame-level classifier.

OpenFace 2.0 outputs 17 AU intensities. The training pipeline uses this
specific column order (note: AU24 included, AU45 NOT included):

    AU01, AU02, AU04, AU05, AU06, AU07,
    AU09, AU10, AU12, AU14, AU15, AU17,
    AU20, AU23, AU24, AU25, AU26

Values are normalized by /5.0 in preprocessing -> [0, 1].

This module exposes the SIMPLE per-frame MLP. The dual-head temporal
model lives in `training/models/au_temporal.py` and is what the server
actually loads at inference time, but only the frame-head branch.
"""
import torch
import torch.nn as nn

NUM_AUS = 17

AU_COLS = [
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r",
    "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r",
    "AU20_r", "AU23_r", "AU24_r", "AU25_r", "AU26_r",
]


class AUFrameClassifier(nn.Module):
    """Frame-level baseline. Used as a sanity check / lightweight fallback."""

    def __init__(self, in_dim: int = NUM_AUS, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 17) AU intensities, normalized to [0, 1]
        return self.net(x).squeeze(-1)  # (B,)
