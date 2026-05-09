"""
Stage 2 inference runner — STUB.

This module is currently a placeholder. The previous MediaPipe-blendshape
implementation is OUT OF SYNC with the training pipeline, which uses
OpenFace 2.0 to extract 17 AU intensities (with AU24, /5.0 normalized).

NEXT STEP — pick one of these inference tools and rewrite this file:

  Option A — OpenFace Docker (matches training exactly)
    + Identical 17-AU output format
    + ~200-500ms per face (ok for cascade since stage 2 is rare)
    - Requires OpenFace built into server image (multi-stage Dockerfile)

  Option B — py-feat
    + pip install, lightweight
    - Outputs 20 AUs in different order; requires retraining or remap

For now this stub returns None so the orchestrator gracefully falls
back to Stage 1 output only. Server still boots in dummy mode.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)


class AUDetector:
    def __init__(self, weights_path: Optional[Path] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.dummy = True
        logger.warning(
            "Stage2 (AU) is currently a STUB. Implement OpenFace or "
            "py-feat extraction here and load training/checkpoints/stage2/best.pt"
        )

    @torch.no_grad()
    def predict(self, face: Image.Image) -> Optional[float]:
        # Returning None signals the orchestrator to skip Stage 2
        # and fall back to Stage 1 output.
        return None
