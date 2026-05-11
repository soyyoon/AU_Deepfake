"""
Stage 1 runner: face crop -> normalized tensor -> Xception -> P(fake).
"""
import logging
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torchvision import transforms

from app.models.artifact_net import ArtifactNet
from app.config import settings

logger = logging.getLogger(__name__)


class ArtifactDetector:
    """Runs Xception-based artifact detection."""

    def __init__(self, weights_path: Optional[Path] = None, device: str = "cpu"):
        self.device = torch.device(device)

        weights_path = weights_path or settings.stage1_weights
        self.dummy = False

        # Build model. Try to load weights; fall back to dummy if missing.
        try:
            self.model = ArtifactNet(pretrained=False).to(self.device).eval()
            if weights_path.exists():
                state = torch.load(weights_path, map_location=self.device)
                # accept either raw state_dict or {"model": state_dict}
                if isinstance(state, dict) and "model" in state:
                    state = state["model"]
                self.model.load_state_dict(state)
                logger.info("Stage1 weights loaded from %s", weights_path)
            else:
                if settings.dummy_mode_if_missing_weights:
                    self.dummy = True
                    logger.warning(
                        "Stage1 weights not found at %s; running in DUMMY mode",
                        weights_path,
                    )
                else:
                    raise FileNotFoundError(weights_path)
        except Exception as e:
            if settings.dummy_mode_if_missing_weights:
                self.dummy = True
                logger.warning("Stage1 init failed (%s); DUMMY mode", e)
            else:
                raise

        self.tf = transforms.Compose([
            transforms.Resize((settings.image_size, settings.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])

    @torch.no_grad()
    def predict(self, face: Image.Image) -> float:
        """Returns P(fake) in [0, 1]."""
        if self.dummy:
            # deterministic-ish pseudo-prob from image content
            import hashlib
            h = int(hashlib.md5(face.tobytes()[:1024]).hexdigest(), 16)
            return ((h % 1000) / 1000.0) * 0.6 + 0.2  # in [0.2, 0.8]

        x = self.tf(face).unsqueeze(0).to(self.device)
        logit = self.model(x)
        return torch.sigmoid(logit).item()
