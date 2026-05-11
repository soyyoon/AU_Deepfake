"""
End-to-end smoke test using the dummy mode (no model weights required).

Run with:
    cd server && pytest tests/test_smoke.py -v
"""
import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _make_b64_image(size: int = 320) -> str:
    img = Image.new("RGB", (size, size), color=(123, 117, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_detect_no_face(client):
    """A flat color image has no face -> label='no_face'."""
    r = client.post(
        "/api/v1/detect",
        json={"image_b64": _make_b64_image(), "return_debug": True},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "no_face"
    assert body["stage_used"] == "skip_no_face"
    assert body["debug"]["face_detected"] is False


def test_invalid_request(client):
    r = client.post("/api/v1/detect", json={})  # neither url nor b64
    assert r.status_code == 422
