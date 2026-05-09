"""
Pydantic schemas for the detect endpoint.
The client sends EITHER an image_url OR a base64-encoded image (image_b64).
"""
from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator


class DetectRequest(BaseModel):
    image_url: Optional[str] = None
    image_b64: Optional[str] = None  # raw base64 (no data: prefix)
    return_debug: bool = False

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "DetectRequest":
        if bool(self.image_url) == bool(self.image_b64):
            raise ValueError("Provide exactly one of image_url or image_b64.")
        return self


class StageTiming(BaseModel):
    stage: str
    ms: float


class DebugInfo(BaseModel):
    face_detected: bool
    stage1_prob: Optional[float] = None
    stage2_prob: Optional[float] = None
    timings: list[StageTiming] = Field(default_factory=list)


class DetectResponse(BaseModel):
    label: Literal["real", "fake", "no_face"]
    confidence: float = Field(ge=0.0, le=1.0)
    fake_prob: float = Field(ge=0.0, le=1.0)
    stage_used: Literal["stage1", "stage1+stage2", "skip_no_face"]
    debug: Optional[DebugInfo] = None
