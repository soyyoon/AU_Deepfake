"""Tiny JSON API for the deepfake detector Chrome extension MVP."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from inference import DeepfakeFusionService, FrameRequest, SequenceRequest


HOST = "127.0.0.1"
PORT = 8000
MAX_BODY_BYTES = 12 * 1024 * 1024

service = DeepfakeFusionService()


class ApiHandler(BaseHTTPRequestHandler):
    server_version = "DeepfakeMVP/0.1"

    def do_OPTIONS(self) -> None:
        self._send_empty(HTTPStatus.NO_CONTENT)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json({"ok": True, "service": "deepfake-detector-mvp"})
            return
        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            if self.path == "/analyze/frame":
                result = service.analyze_frame(
                    FrameRequest(
                        image_data_url=str(payload.get("image", "")),
                        media=dict(payload.get("media") or {}),
                        source=str(payload.get("source") or "visible_tab"),
                        page_url=payload.get("pageUrl"),
                    )
                )
                self._send_json(result)
                return

            if self.path == "/analyze/sequence":
                frames = payload.get("frames")
                if not isinstance(frames, list):
                    raise ValueError("frames must be a list")
                result = service.analyze_sequence(
                    SequenceRequest(
                        frames=[str(frame) for frame in frames],
                        media=dict(payload.get("media") or {}),
                        page_url=payload.get("pageUrl"),
                    )
                )
                self._send_json(result)
                return

            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # pragma: no cover - defensive for MVP server.
            self._send_json({"error": "internal server error", "detail": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format: str, *args: Any) -> None:
        print("%s - %s" % (self.address_string(), format % args))

    def _read_json(self) -> dict[str, Any]:
        raw_length = self.headers.get("content-length")
        if raw_length is None:
            raise ValueError("missing content-length")

        length = int(raw_length)
        if length > MAX_BODY_BYTES:
            raise ValueError("request body is too large")

        body = self.rfile.read(length)
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("request body must be JSON") from exc

        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _send_empty(self, status: HTTPStatus) -> None:
        self.send_response(status)
        self._send_cors_headers()
        self.end_headers()

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_cors_headers(self) -> None:
        self.send_header("access-control-allow-origin", "*")
        self.send_header("access-control-allow-methods", "GET, POST, OPTIONS")
        self.send_header("access-control-allow-headers", "content-type")


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), ApiHandler)
    print(f"Deepfake detector MVP API listening on http://{HOST}:{PORT}")
    print("Health check: http://127.0.0.1:8000/health")
    server.serve_forever()


if __name__ == "__main__":
    main()
