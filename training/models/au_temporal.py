"""
Stage 2 dual-head temporal model.

Trunk:
    1D-CNN over the time axis with dilated convolutions
    to capture micro-expression dynamics.

Heads:
    - sequence head: attention pool over time -> MLP -> 1 logit
                     (used for video-level training + video inference)
    - frame head:    per-frame MLP -> 1 logit per timestep
                     (used as auxiliary loss during training,
                      and as the SOLE inference path for single images)

Why dual head?
    DFD/CelebDF-v2 give us video-level labels, but the Chrome extension
    sees single images. Training a frame head with the video label as
    a weak per-frame supervision lets us deploy to image inference
    without retraining. The sequence head is still useful for video
    pages (YouTube) and as the stronger training signal.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedConvBlock(nn.Module):
    def __init__(self, ch: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Conv1d(
            ch, ch, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.norm = nn.BatchNorm1d(ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                 # x: (B, C, T)
        return self.drop(F.gelu(self.norm(self.conv(x)) + x))


class AttentionPool(nn.Module):
    """Learns per-frame importance weights then pools."""
    def __init__(self, ch: int):
        super().__init__()
        self.score = nn.Linear(ch, 1)

    def forward(self, x):                 # x: (B, T, C)
        w = torch.softmax(self.score(x), dim=1)   # (B, T, 1)
        return (x * w).sum(dim=1)                  # (B, C)


class AUDualHeadNet(nn.Module):
    def __init__(
        self,
        num_aus: int = 17,
        trunk_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(num_aus, trunk_dim)
        self.input_norm = nn.LayerNorm(trunk_dim)

        # dilations 1, 2, 4, 8, ... grow receptive field exponentially
        dilations = [2 ** i for i in range(num_layers)]
        self.blocks = nn.ModuleList([
            DilatedConvBlock(trunk_dim, d, dropout) for d in dilations
        ])

        # heads
        self.pool = AttentionPool(trunk_dim)
        self.seq_head = nn.Sequential(
            nn.Linear(trunk_dim, trunk_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(trunk_dim // 2, 1),
        )
        self.frame_head = nn.Sequential(
            nn.Linear(trunk_dim, trunk_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(trunk_dim // 2, 1),
        )

    def encode(self, au_seq: torch.Tensor) -> torch.Tensor:
        """
        au_seq: (B, T, num_aus)  -> H: (B, T, trunk_dim)
        """
        h = self.input_norm(self.input_proj(au_seq))     # (B, T, C)
        h = h.transpose(1, 2)                            # (B, C, T) for conv1d
        for blk in self.blocks:
            h = blk(h)
        return h.transpose(1, 2)                         # (B, T, C)

    def forward(self, au_seq: torch.Tensor) -> dict:
        """
        Returns dict with both head logits.
            seq_logit:   (B,)        -- video-level
            frame_logit: (B, T)      -- per-frame
        """
        h = self.encode(au_seq)                          # (B, T, C)
        pooled = self.pool(h)                            # (B, C)
        seq_logit = self.seq_head(pooled).squeeze(-1)    # (B,)
        frame_logit = self.frame_head(h).squeeze(-1)     # (B, T)
        return {"seq_logit": seq_logit, "frame_logit": frame_logit, "embeds": h}

    @torch.no_grad()
    def predict_frame(self, au_frame: torch.Tensor) -> torch.Tensor:
        """
        Inference path for single-image (1 frame) input from the Chrome
        extension.

        au_frame: (B, num_aus) — single-frame AU vector.
        Returns:  (B,) P(fake).

        We replicate the frame to a length-1 sequence so the trunk runs
        identically; padding=dilation in our conv blocks means short
        sequences are still valid.
        """
        if au_frame.dim() == 2:
            au_seq = au_frame.unsqueeze(1)               # (B, 1, num_aus)
        else:
            au_seq = au_frame
        h = self.encode(au_seq)                          # (B, 1, C)
        logit = self.frame_head(h).squeeze(-1).squeeze(-1)  # (B,)
        return torch.sigmoid(logit)
