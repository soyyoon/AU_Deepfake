"""
Face detection and cropping with MTCNN (facenet-pytorch).
Returns the largest detected face as a PIL.Image, or None.
"""
import logging
from typing import Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class FaceCropper:
    def __init__(self, device: str = "cpu", min_size: int = 64):
        # Lazy import so server still boots if facenet-pytorch missing.
        from facenet_pytorch import MTCNN  # type: ignore

        self.mtcnn = MTCNN(
            keep_all=True,
            device=device,
            post_process=False,  # we'll handle normalization ourselves
            min_face_size=min_size,
        )
        self.device = device

    @torch.no_grad()
    def crop(self, image: Image.Image, margin: float = 0.2) -> Optional[Image.Image]:
        """Detect faces, return the largest crop with margin, or None."""
        boxes, probs = self.mtcnn.detect(image)
        if boxes is None or len(boxes) == 0:
            return None

        # filter low-confidence detections
        keep = [(b, p) for b, p in zip(boxes, probs) if p is not None and p > 0.9]
        if not keep:
            return None

        # largest by area
        boxes_kept = [b for b, _ in keep]
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes_kept]
        b = boxes_kept[int(np.argmax(areas))]

        x1, y1, x2, y2 = b
        w, h = x2 - x1, y2 - y1
        # apply margin
        x1 = max(0, x1 - margin * w)
        y1 = max(0, y1 - margin * h)
        x2 = min(image.width, x2 + margin * w)
        y2 = min(image.height, y2 + margin * h)
        return image.crop((x1, y1, x2, y2))
