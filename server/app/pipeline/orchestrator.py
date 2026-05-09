"""
Two-stage pipeline orchestrator.

Flow:
    image -> face crop -> Stage1 (artifact)
        if P(fake) outside [low, high]: return Stage1 result
        else: -> Stage2 (AU)
              combine: p_final = w1 * p1 + w2 * p2
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
from PIL import Image

from app.config import settings
from app.pipeline.preprocess import FaceCropper
from app.pipeline.stage1_artifact import ArtifactDetector
from app.pipeline.stage2_au import AUDetector

logger = logging.getLogger(__name__)


@dataclass
class StageTiming:
    stage: str
    ms: float


@dataclass
class PipelineResult:
    label: str                              # "real" | "fake" | "no_face"
    confidence: float                       # 0..1, distance from 0.5
    fake_prob: float                        # 0..1
    stage_used: str                         # "stage1" | "stage1+stage2" | "skip_no_face"
    stage1_prob: Optional[float] = None
    stage2_prob: Optional[float] = None
    face_detected: bool = False
    timings: list[StageTiming] = field(default_factory=list)


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


class Pipeline:
    def __init__(self) -> None:
        device = settings.device if torch.cuda.is_available() else "cpu"
        self.device = device
        logger.info("Pipeline initializing on device=%s", device)

        self.cropper = FaceCropper(device=device)
        self.stage1 = ArtifactDetector(device=device)
        self.stage2 = AUDetector(device=device)

        self.low = settings.stage1_uncertainty_low
        self.high = settings.stage1_uncertainty_high
        self.threshold = settings.decision_threshold
        self.w1 = settings.stage1_weight
        self.w2 = settings.stage2_weight

    def run(self, image: Image.Image) -> PipelineResult:
        timings: list[StageTiming] = []

        # 1. Face crop ------------------------------------------------------
        t0 = _now_ms()
        face = self.cropper.crop(image)
        timings.append(StageTiming("face_crop", _now_ms() - t0))

        if face is None:
            return PipelineResult(
                label="no_face",
                confidence=1.0,
                fake_prob=0.0,
                stage_used="skip_no_face",
                face_detected=False,
                timings=timings,
            )

        # 2. Stage 1 -------------------------------------------------------
        t0 = _now_ms()
        p1 = self.stage1.predict(face)
        timings.append(StageTiming("stage1", _now_ms() - t0))

        # 3. Stage 2 only if uncertain -------------------------------------
        p2: Optional[float] = None
        if self.low <= p1 <= self.high:
            t0 = _now_ms()
            p2 = self.stage2.predict(face)
            timings.append(StageTiming("stage2", _now_ms() - t0))

        # 4. Combine -------------------------------------------------------
        if p2 is None:
            p_final = p1
            stage_used = "stage1"
        else:
            p_final = self.w1 * p1 + self.w2 * p2
            stage_used = "stage1+stage2"

        label = "fake" if p_final >= self.threshold else "real"
        confidence = abs(p_final - 0.5) * 2  # 0 at boundary, 1 at extremes

        return PipelineResult(
            label=label,
            confidence=confidence,
            fake_prob=p_final,
            stage_used=stage_used,
            stage1_prob=p1,
            stage2_prob=p2,
            face_detected=True,
            timings=timings,
        )
