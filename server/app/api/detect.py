"""
POST /api/v1/detect

Request body: {"image_url": "..."} OR {"image_b64": "..."}
"""
import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import DebugInfo, DetectRequest, DetectResponse, StageTiming
from app.utils.image_io import (
    ImageLoadError,
    decode_image_b64,
    fetch_image_from_url,
    image_hash,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["detect"])


@router.post("/detect", response_model=DetectResponse)
async def detect(req: DetectRequest, request: Request) -> DetectResponse:
    pipeline = request.app.state.pipeline
    cache = request.app.state.cache

    # 1. Load image ---------------------------------------------------------
    try:
        if req.image_url:
            image = await fetch_image_from_url(req.image_url)
        else:
            image = decode_image_b64(req.image_b64)  # type: ignore[arg-type]
    except ImageLoadError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2. Cache lookup -------------------------------------------------------
    key = image_hash(image)
    cached = await cache.get_json(key)
    if cached is not None and not req.return_debug:
        return DetectResponse.model_validate(cached)

    # 3. Run pipeline (CPU/GPU heavy -> thread) ----------------------------
    result = await asyncio.to_thread(pipeline.run, image)

    # 4. Build response -----------------------------------------------------
    debug = None
    if req.return_debug:
        debug = DebugInfo(
            face_detected=result.face_detected,
            stage1_prob=result.stage1_prob,
            stage2_prob=result.stage2_prob,
            timings=[StageTiming(stage=t.stage, ms=t.ms) for t in result.timings],
        )

    resp = DetectResponse(
        label=result.label,
        confidence=result.confidence,
        fake_prob=result.fake_prob,
        stage_used=result.stage_used,
        debug=debug,
    )

    # 5. Cache (without debug payload) -------------------------------------
    await cache.set_json(key, resp.model_dump(exclude={"debug"}))
    return resp


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}
