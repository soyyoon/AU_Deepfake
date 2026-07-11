"""Inference service for the browser-extension MVP."""

from __future__ import annotations

import base64
import io
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/deepfake-signal-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/deepfake-signal-cache")

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


DATA_URL_RE = re.compile(r"^data:(?P<mime>[-\w.]+/[-\w.+]+);base64,(?P<payload>.*)$")
DEFAULT_CHECKPOINT_PATH = Path(__file__).resolve().parent / "models" / "best_model.pt"
DEFAULT_FACE_MODEL_PATH = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
DEEPFAKE_THRESHOLD = 0.95
UNCERTAIN_THRESHOLD = 0.75


@dataclass(frozen=True)
class FrameRequest:
    image_data_url: str
    media: dict[str, Any]
    source: str
    page_url: str | None = None


@dataclass(frozen=True)
class SequenceRequest:
    frames: list[str]
    media: dict[str, Any]
    page_url: str | None = None


def decode_data_url(data_url: str) -> tuple[str, bytes]:
    match = DATA_URL_RE.match(data_url)
    if not match:
        raise ValueError("image must be a base64 data URL")

    payload = match.group("payload")
    try:
        return match.group("mime"), base64.b64decode(payload, validate=True)
    except ValueError as exc:
        raise ValueError("image payload is not valid base64") from exc


class HighPassPreprocess(nn.Module):
    def __init__(self, strength: float = 0.15) -> None:
        super().__init__()
        kernel = torch.tensor(
            [
                [0.0, -1.0, 0.0],
                [-1.0, 4.0, -1.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        kernel = kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        self.register_buffer("kernel", kernel)
        self.strength = strength

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hp = F.conv2d(x, self.kernel, padding=1, groups=3)
        return x + self.strength * hp


class ConvNeXtArtifactDetector(nn.Module):
    def __init__(
        self,
        *,
        model_name: str = "convnext_tiny",
        dropout: float = 0.2,
        use_highpass: bool = True,
    ) -> None:
        super().__init__()
        self.use_highpass = use_highpass
        self.pre = HighPassPreprocess(strength=0.15) if use_highpass else nn.Identity()
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        batch_size, frame_count, channels, height, width = x.shape
        x = x.reshape(batch_size * frame_count, channels, height, width)
        x = self.pre(x)
        feat = self.backbone(x)
        feat = feat.reshape(batch_size, frame_count, -1).mean(dim=1)
        return self.head(feat).squeeze(1)


class FaceCropper:
    """Small local face cropper to keep non-face thumbnails out of the model."""

    def __init__(self) -> None:
        self.backend = os.environ.get("DEEPFAKE_FACE_BACKEND", "yunet")
        self.min_confidence = float(os.environ.get("DEEPFAKE_FACE_CONFIDENCE", "0.78"))
        self.yunet_detector = None
        self._detector_lock = threading.Lock()
        cascade_dir = Path(cv2.data.haarcascades)
        self.cascades = [
            cv2.CascadeClassifier(str(cascade_dir / "haarcascade_frontalface_default.xml")),
            cv2.CascadeClassifier(str(cascade_dir / "haarcascade_profileface.xml")),
        ]
        if self.backend == "yunet":
            face_model_path = Path(os.environ.get("DEEPFAKE_FACE_MODEL", DEFAULT_FACE_MODEL_PATH))
            if not face_model_path.exists():
                raise FileNotFoundError(f"YuNet face model not found: {face_model_path}")
            self.yunet_detector = cv2.FaceDetectorYN.create(
                str(face_model_path),
                "",
                (320, 320),
                self.min_confidence,
                0.3,
                5000,
            )
        elif all(cascade.empty() for cascade in self.cascades):
            raise RuntimeError("OpenCV face cascades are not available")

    def crop_faces(self, image: Image.Image, *, max_faces: int = 8) -> list[tuple[Image.Image, dict[str, Any]]]:
        rgb = np.array(image)
        height, width = rgb.shape[:2]
        min_side = min(width, height)
        min_face = max(24, min_side // 10)

        if self.yunet_detector is not None:
            detections = self._detect_with_yunet(rgb, width, height, min_face)
        else:
            detections = self._detect_with_haar(rgb, min_face)

        detections = self._dedupe_faces(detections)[:max_faces]
        if not detections:
            return []

        cropped_faces = []
        for face_index, detection in enumerate(detections):
            x = detection["left"]
            y = detection["top"]
            face_width = detection["width"]
            face_height = detection["height"]
            pad_x = int(face_width * 0.45)
            pad_y = int(face_height * 0.55)
            left = max(0, x - pad_x)
            top = max(0, y - pad_y)
            right = min(width, x + face_width + pad_x)
            bottom = min(height, y + face_height + pad_y)

            cropped_faces.append(
                (
                    image.crop((left, top, right, bottom)),
                    {
                        "detected": True,
                        "count": len(detections),
                        "image_width": width,
                        "image_height": height,
                        "face_index": face_index,
                        "detector": detection["detector"],
                        "confidence": detection["confidence"],
                        "face_rect": {
                            "left": x,
                            "top": y,
                            "width": face_width,
                            "height": face_height,
                        },
                        "crop_rect": {
                            "left": left,
                            "top": top,
                            "right": right,
                            "bottom": bottom,
                        },
                    },
                )
            )

        return cropped_faces

    def _detect_with_yunet(
        self,
        rgb: np.ndarray,
        width: int,
        height: int,
        min_face: int,
    ) -> list[dict[str, Any]]:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        with self._detector_lock:
            self.yunet_detector.setInputSize((width, height))
            _, faces = self.yunet_detector.detect(bgr)
        detections: list[dict[str, Any]] = []

        if faces is None:
            return detections

        for face in faces:
            score = float(face[-1])
            left = max(0, int(round(face[0])))
            top = max(0, int(round(face[1])))
            right = min(width, int(round(face[0] + face[2])))
            bottom = min(height, int(round(face[1] + face[3])))
            face_width = max(0, right - left)
            face_height = max(0, bottom - top)

            if face_width < min_face or face_height < min_face:
                continue

            detections.append(
                {
                    "left": left,
                    "top": top,
                    "width": face_width,
                    "height": face_height,
                    "confidence": None if score is None else round(score, 4),
                    "detector": "yunet",
                }
            )

        return detections

    def _detect_with_haar(self, rgb: np.ndarray, min_face: int) -> list[dict[str, Any]]:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        detections: list[dict[str, Any]] = []

        for cascade in self.cascades:
            if cascade.empty():
                continue
            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=1.06,
                minNeighbors=6,
                minSize=(min_face, min_face),
            )
            detections.extend(
                {
                    "left": int(x),
                    "top": int(y),
                    "width": int(face_width),
                    "height": int(face_height),
                    "confidence": None,
                    "detector": "haar",
                }
                for x, y, face_width, face_height in faces
            )

        return detections

    def _dedupe_faces(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sorted_faces = sorted(
            detections,
            key=lambda face: face["width"] * face["height"] * float(face.get("confidence") or 0.5),
            reverse=True,
        )
        unique_faces: list[dict[str, Any]] = []

        for face in sorted_faces:
            if all(self._overlap_ratio(face, existing) < 0.45 for existing in unique_faces):
                unique_faces.append(face)

        return unique_faces

    @staticmethod
    def _overlap_ratio(first: dict[str, Any], second: dict[str, Any]) -> float:
        first_left = first["left"]
        first_top = first["top"]
        first_width = first["width"]
        first_height = first["height"]
        second_left = second["left"]
        second_top = second["top"]
        second_width = second["width"]
        second_height = second["height"]
        first_right = first_left + first_width
        first_bottom = first_top + first_height
        second_right = second_left + second_width
        second_bottom = second_top + second_height
        intersection_width = max(0, min(first_right, second_right) - max(first_left, second_left))
        intersection_height = max(0, min(first_bottom, second_bottom) - max(first_top, second_top))
        intersection = intersection_width * intersection_height
        smaller_area = min(first_width * first_height, second_width * second_height)
        return intersection / smaller_area if smaller_area else 0.0


class ArtifactDetector:
    """ConvNeXt-T artifact detector loaded from the trained checkpoint."""

    def __init__(self, checkpoint_path: Path | str = DEFAULT_CHECKPOINT_PATH) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"model checkpoint not found: {self.checkpoint_path}")

        self.device = torch.device(os.environ.get("DEEPFAKE_DEVICE", "cpu"))
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        cfg = checkpoint.get("cfg", {})
        model_cfg = cfg.get("model", {})
        input_cfg = cfg.get("input", {})

        self.image_size = int(input_cfg.get("image_size", 224))
        self.model = ConvNeXtArtifactDetector(
            model_name=str(model_cfg.get("model_name", "convnext_tiny")),
            dropout=float(model_cfg.get("dropout", 0.2)),
            use_highpass=bool(model_cfg.get("use_highpass", True)),
        )
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD),
            ]
        )
        self.epoch = checkpoint.get("epoch")
        self.val_metrics = checkpoint.get("val_metrics", {})
        self.require_face = os.environ.get("DEEPFAKE_REQUIRE_FACE", "1") != "0"
        self.max_faces_per_frame = int(os.environ.get("DEEPFAKE_MAX_FACES", "8"))
        self.face_cropper = FaceCropper()

    def predict(self, image_bytes: bytes, media: dict[str, Any]) -> dict[str, Any]:
        return self.predict_frames([image_bytes], media)

    def predict_frames(self, frame_bytes: list[bytes], media: dict[str, Any]) -> dict[str, Any]:
        scored_faces: list[dict[str, Any]] = []
        no_face_frames: list[dict[str, Any]] = []
        face_samples: list[dict[str, Any]] = []
        skipped_frames = 0

        for frame_index, image_bytes in enumerate(frame_bytes):
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            face_crops = self.face_cropper.crop_faces(image, max_faces=self.max_faces_per_frame)

            if self.require_face and not face_crops:
                skipped_frames += 1
                no_face_frames.append(
                    {
                        "detected": False,
                        "count": 0,
                        "image_width": image.width,
                        "image_height": image.height,
                        "frame_index": frame_index,
                    }
                )
                continue

            if not face_crops:
                score = self._score_image(image)
                scored_faces.append(
                    {
                        "detected": False,
                        "count": 0,
                        "image_width": image.width,
                        "image_height": image.height,
                        "frame_index": frame_index,
                        "face_index": None,
                        "score": round(score, 4),
                        "label": self._score_label(score),
                    }
                )
                continue

            for cropped, face in face_crops:
                score = self._score_image(cropped)
                scored_face = {
                    **face,
                    "frame_index": frame_index,
                    "score": round(score, 4),
                    "label": self._score_label(score),
                }
                scored_faces.append(scored_face)
                face_samples.append(scored_face)

        if not scored_faces:
            return {
                "score": None,
                "label": "no_face",
                "evidence": [
                    "no face detected in the submitted crop",
                    "artifact model skipped to reduce false positives on thumbnails, graphics, and scenery",
                ],
                "metadata": {
                    "face_detected": False,
                    "face_required": self.require_face,
                    "analyzed_frames": 0,
                    "skipped_frames": skipped_frames,
                    "frame_count": len(frame_bytes),
                    "analyzed_faces": 0,
                    "face_samples": no_face_frames[:3],
                    "primary_face": None,
                    "display_faces": [],
                },
            }

        primary_face = max(scored_faces, key=lambda face: float(face.get("score") or 0.0))
        primary_frame_index = primary_face.get("frame_index")
        display_faces = [
            face for face in scored_faces
            if face.get("detected") and face.get("frame_index") == primary_frame_index
        ]
        score = float(primary_face["score"])
        return {
            "score": round(score, 4),
            "evidence": [
                "ConvNeXt-T artifact detector",
                "all detected face crops were scored independently",
                "frame is flagged from the highest face fake score",
                f"checkpoint={self.checkpoint_path.name}",
                f"epoch={self.epoch}",
                f"device={self.device}",
                f"input={self.image_size}x{self.image_size}, faces={len(scored_faces)}",
            ],
            "metadata": {
                "face_detected": any(sample["detected"] for sample in scored_faces),
                "face_required": self.require_face,
                "analyzed_frames": len(set(face["frame_index"] for face in scored_faces)),
                "skipped_frames": skipped_frames,
                "frame_count": len(frame_bytes),
                "analyzed_faces": len(scored_faces),
                "face_samples": scored_faces[:12],
                "primary_face": primary_face,
                "display_faces": display_faces[: self.max_faces_per_frame],
                "score_method": "max_face_fake_probability" if len(scored_faces) > 1 else "single_face_fake_probability",
            },
        }

    def _score_image(self, image: Image.Image) -> float:
        frame = self.transform(image)
        batch = frame.unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logit = self.model(batch)
            return float(torch.sigmoid(logit).item())

    @staticmethod
    def _score_label(score: float) -> str:
        if score >= DEEPFAKE_THRESHOLD:
            return "high_suspicion"
        if score >= UNCERTAIN_THRESHOLD:
            return "uncertain"
        return "low_suspicion"


class DemoAUSequenceDetector:
    """Placeholder for a temporal AU sequence detector.

    It intentionally returns no score so a demo hash cannot influence real model output.
    """

    def predict(self, frames: list[str], media: dict[str, Any]) -> dict[str, Any]:
        return {
            "score": None,
            "evidence": [
                "AU sequence detector is not configured yet",
                "no demo sequence score was fused into this result",
            ],
        }


class DeepfakeFusionService:
    def __init__(self) -> None:
        self.artifact_detector = ArtifactDetector()
        self.au_detector = DemoAUSequenceDetector()

    def analyze_frame(self, request: FrameRequest) -> dict[str, Any]:
        mime, image_bytes = decode_data_url(request.image_data_url)
        artifact = self.artifact_detector.predict(image_bytes, request.media)
        return self._build_response(
            artifact_score=artifact["score"],
            au_score=None,
            label_override=artifact.get("label"),
            evidence=artifact["evidence"],
            metadata={
                "mime": mime,
                "bytes": len(image_bytes),
                "source": request.source,
                "artifact": artifact.get("metadata", {}),
            },
        )

    def analyze_sequence(self, request: SequenceRequest) -> dict[str, Any]:
        if not request.frames:
            raise ValueError("frames must not be empty")

        decoded_frames = [decode_data_url(frame) for frame in request.frames]
        first_mime, first_bytes = decoded_frames[0]
        artifact = self.artifact_detector.predict_frames(
            [image_bytes for _, image_bytes in decoded_frames],
            request.media,
        )
        au = self.au_detector.predict(request.frames, request.media)
        return self._build_response(
            artifact_score=artifact["score"],
            au_score=au["score"],
            label_override=artifact.get("label"),
            evidence=artifact["evidence"] + au["evidence"],
            metadata={
                "mime": first_mime,
                "frame_count": len(request.frames),
                "first_frame_bytes": len(first_bytes),
                "source": "sequence",
                "artifact": artifact.get("metadata", {}),
            },
        )

    def _build_response(
        self,
        *,
        artifact_score: float | None,
        au_score: float | None,
        label_override: str | None,
        evidence: list[str],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        if artifact_score is None:
            return {
                "artifact_score": None,
                "au_score": None if au_score is None else round(au_score, 4),
                "final_score": None,
                "uncertainty": 1.0,
                "label": label_override or "unsupported_input",
                "evidence": evidence,
                "metadata": metadata,
            }

        if au_score is None:
            final_score = artifact_score
            uncertainty = 0.45
        else:
            final_score = artifact_score * 0.6 + au_score * 0.4
            uncertainty = abs(artifact_score - au_score)

        return {
            "artifact_score": round(artifact_score, 4),
            "au_score": None if au_score is None else round(au_score, 4),
            "final_score": round(final_score, 4),
            "uncertainty": round(uncertainty, 4),
            "label": self._label(final_score, uncertainty, has_au=au_score is not None),
            "evidence": evidence,
            "metadata": metadata,
        }

    @staticmethod
    def _label(score: float, uncertainty: float, *, has_au: bool) -> str:
        if has_au and uncertainty >= 0.45:
            return "uncertain"
        if score >= DEEPFAKE_THRESHOLD:
            return "high_suspicion"
        if score >= UNCERTAIN_THRESHOLD:
            return "uncertain"
        return "low_suspicion"
