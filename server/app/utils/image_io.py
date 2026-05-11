"""
Convert image_url or image_b64 -> PIL.Image.
"""
import base64
import io
import logging

import httpx
from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)


class ImageLoadError(Exception):
    pass


async def fetch_image_from_url(url: str) -> Image.Image:
    try:
        async with httpx.AsyncClient(timeout=settings.image_fetch_timeout) as client:
            r = await client.get(url, follow_redirects=True)
        if r.status_code != 200:
            raise ImageLoadError(f"HTTP {r.status_code} fetching {url}")
        if len(r.content) > settings.max_image_bytes:
            raise ImageLoadError("Image too large")
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except (httpx.RequestError, OSError) as e:
        raise ImageLoadError(f"Failed to load image: {e}") from e


def decode_image_b64(b64: str) -> Image.Image:
    try:
        # tolerate data:image/...;base64, prefix
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        raw = base64.b64decode(b64, validate=True)
        if len(raw) > settings.max_image_bytes:
            raise ImageLoadError("Image too large")
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except (ValueError, OSError) as e:
        raise ImageLoadError(f"Invalid base64 image: {e}") from e


def image_hash(img: Image.Image) -> str:
    """Stable cache key. Uses bytes of the resized image."""
    import hashlib
    small = img.resize((32, 32))
    return hashlib.sha1(small.tobytes()).hexdigest()
