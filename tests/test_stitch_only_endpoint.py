from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

import backend.app as backend_app


def _jpeg_bytes(channel: int) -> bytes:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[:, :, channel] = 255
    success, encoded = cv2.imencode(".jpg", image)
    if not success:
        raise AssertionError("OpenCV could not prepare a test JPEG")
    return encoded.tobytes()


def _multipart_payload(images: list[bytes], metadata: dict) -> tuple[str, bytes]:
    boundary = "----StitchOnlyContractBoundary"
    parts: list[bytes] = []
    for index, content in enumerate(images, start=1):
        parts.extend([
            f"--{boundary}\r\n".encode(),
            (
                'Content-Disposition: form-data; name="photos"; '
                f'filename="capture_{index}.jpg"\r\n'
            ).encode(),
            b"Content-Type: image/jpeg\r\n\r\n",
            content,
            b"\r\n",
        ])
    parts.extend([
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="capture_metadata"\r\n\r\n',
        json.dumps(metadata).encode(),
        b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ])
    return boundary, b"".join(parts)


async def _post(body: bytes, boundary: str) -> tuple[int, bytes]:
    messages: list[dict] = []
    request_sent = False

    async def receive() -> dict:
        nonlocal request_sent
        if request_sent:
            return {"type": "http.disconnect"}
        request_sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: dict) -> None:
        messages.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/v1/stitch",
        "raw_path": b"/api/v1/stitch",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", f"multipart/form-data; boundary={boundary}".encode()),
            (b"content-length", str(len(body)).encode()),
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "state": {},
    }
    await backend_app.app(scope, receive, send)
    status = next(message["status"] for message in messages if message["type"] == "http.response.start")
    response = b"".join(
        message.get("body", b"") for message in messages if message["type"] == "http.response.body"
    )
    return status, response


class StitchOnlyEndpointTests(unittest.TestCase):
    def test_real_three_photo_endpoint_stitches_without_ai_assets(self) -> None:
        rng = np.random.default_rng(19)
        base = rng.integers(0, 256, size=(420, 1400, 3), dtype=np.uint8)
        for x in range(30, 1370, 70):
            cv2.circle(base, (x, 70 + (x % 260)), 15, (255, 255, 255), 3)
            cv2.putText(base, str(x), (x - 15, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        images = []
        for crop in (base[:, 0:700], base[:, 350:1050], base[:, 700:1400]):
            success, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            self.assertTrue(success)
            images.append(encoded.tobytes())
        boundary, request = _multipart_payload(images, {"capture_order": [0, 1, 2]})

        with tempfile.TemporaryDirectory() as temporary_directory:
            session_root = Path(temporary_directory)
            with (
                patch.object(backend_app, "SESSION_ROOT", session_root),
                patch.object(backend_app, "get_recognizer", side_effect=AssertionError("recognition must not run")),
            ):
                status, response_bytes = asyncio.run(_post(request, boundary))
            self.assertEqual(status, 200, response_bytes)
            response = json.loads(response_bytes)
            self.assertEqual(response["status"], "success", response)
            self.assertEqual(response["input_count"], 3)
            panorama_path = session_root / response["session_id"] / "panorama.jpg"
            self.assertTrue(panorama_path.is_file())
            panorama = cv2.imread(str(panorama_path))
            self.assertIsNotNone(panorama)
            self.assertGreater(panorama.shape[1], 1200)

    def test_three_ordered_photos_do_not_load_recognition(self) -> None:
        images = [_jpeg_bytes(0), _jpeg_bytes(1), _jpeg_bytes(2)]
        metadata = {"capture_order": [0, 1, 2], "platform": "android"}
        boundary, request = _multipart_payload(images, metadata)
        panorama = np.zeros((20, 40, 3), dtype=np.uint8)
        mask = np.full((20, 40), 255, dtype=np.uint8)
        stitch_success = SimpleNamespace(
            success=True,
            status="suitable_for_stitching",
            reason_code=None,
            message="mocked stitch success",
            report={"mocked": True},
            panorama=panorama,
            valid_mask=mask,
            match_preview=None,
            alignment_preview=None,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            session_root = Path(temporary_directory)
            with (
                patch.object(backend_app, "SESSION_ROOT", session_root),
                patch.object(backend_app, "stitch_image_sequence", return_value=stitch_success),
                patch.object(backend_app, "get_recognizer", side_effect=AssertionError("recognition must not run")),
            ):
                status, response_bytes = asyncio.run(_post(request, boundary))

            self.assertEqual(status, 200, response_bytes)
            response = json.loads(response_bytes)
            self.assertEqual(response["mode"], "stitch_only")
            self.assertEqual(response["input_count"], 3)
            self.assertNotIn("recognition", response)
            session_dir = session_root / response["session_id"]
            for index, expected in enumerate(images):
                self.assertEqual((session_dir / f"capture_{index:02d}.jpg").read_bytes(), expected)
            saved = json.loads((session_dir / "capture_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["capture_order"], ["capture_00.jpg", "capture_01.jpg", "capture_02.jpg"])

    def test_rejects_more_than_configured_photo_limit(self) -> None:
        images = [_jpeg_bytes(index % 3) for index in range(7)]
        boundary, request = _multipart_payload(images, {"capture_order": list(range(7))})
        status, response_bytes = asyncio.run(_post(request, boundary))
        self.assertEqual(status, 400, response_bytes)
        self.assertIn("At most 6 photos", json.loads(response_bytes)["detail"])


if __name__ == "__main__":
    unittest.main()
