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
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, :, channel] = 255
    success, encoded = cv2.imencode(".jpg", image)
    if not success:
        raise AssertionError("OpenCV could not prepare a test JPEG")
    return encoded.tobytes()


def _multipart_payload(photos: list[tuple[str, bytes]], metadata: dict) -> tuple[str, bytes]:
    boundary = "----ShelfCaptureUploadContractBoundary"
    parts: list[bytes] = []

    def add_photo(filename: str, content: bytes) -> None:
        parts.extend(
            [
                f"--{boundary}\r\n".encode(),
                (
                    'Content-Disposition: form-data; name="photos"; '
                    f'filename="{filename}"\r\n'
                ).encode(),
                b"Content-Type: image/jpeg\r\n\r\n",
                content,
                b"\r\n",
            ]
        )

    for filename, content in photos:
        add_photo(filename, content)
    parts.extend(
        [
            f"--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="capture_metadata"\r\n\r\n',
            json.dumps(metadata).encode(),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return boundary, b"".join(parts)


async def _post_multipart(body: bytes, boundary: str) -> tuple[int, bytes]:
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
        "path": "/api/v1/process",
        "raw_path": b"/api/v1/process",
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
    response_body = b"".join(
        message.get("body", b"") for message in messages if message["type"] == "http.response.body"
    )
    return status, response_body


class BackendUploadContractTests(unittest.TestCase):
    def test_process_applies_configured_decoded_megapixel_limit(self) -> None:
        boundary, request_body = _multipart_payload(
            [("first.jpg", _jpeg_bytes(0)), ("second.jpg", _jpeg_bytes(1))],
            {"capture_order": [0, 1]},
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            with (
                patch.object(backend_app, "SESSION_ROOT", Path(temporary_directory)),
                patch.dict(
                    backend_app.BACKEND_CONFIG,
                    {"max_input_megapixels": 0.00001},
                ),
                patch.object(backend_app, "stitch_two_images") as stitch,
            ):
                status, response_body = asyncio.run(
                    _post_multipart(request_body, boundary)
                )
        self.assertEqual(status, 413, response_body)
        self.assertIn("megapixels", json.loads(response_body)["detail"])
        stitch.assert_not_called()

    def test_repeated_photo_fields_keep_order_and_parse_json_metadata(self) -> None:
        first = _jpeg_bytes(0)
        second = _jpeg_bytes(1)
        metadata = {
            "capture_order": [0, 1],
            "coverage_complete": False,
            "platform": "android",
            "photos": [{"uri": "file:///first.jpg"}, {"uri": "file:///second.jpg"}],
            "guidance_config": {"maxMotionDelta": 0.18, "stabilityWindowMs": 350},
        }
        stitch_failure = SimpleNamespace(
            success=False,
            report={"mocked": True},
            match_preview=None,
            reason_code="mocked_failure",
            message="Mocked: stitching was not run",
        )
        boundary, request_body = _multipart_payload(
            [("capture_1.jpg", first), ("capture_2.jpg", second)],
            metadata,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            session_root = Path(temporary_directory)
            with (
                patch.object(backend_app, "SESSION_ROOT", session_root),
                patch.object(backend_app, "stitch_two_images", return_value=stitch_failure),
            ):
                status, response_body = asyncio.run(_post_multipart(request_body, boundary))

            self.assertEqual(status, 200, response_body)
            body = json.loads(response_body)
            self.assertEqual(body["status"], "retake_required")

            session_dir = session_root / body["session_id"]
            self.assertEqual((session_dir / "capture_00.jpg").read_bytes(), first)
            self.assertEqual((session_dir / "capture_01.jpg").read_bytes(), second)
            saved_metadata = json.loads(
                (session_dir / "capture_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved_metadata["capture_order"], ["capture_00.jpg", "capture_01.jpg"])
            self.assertEqual(saved_metadata["client_metadata"], metadata)
            self.assertIsNone(saved_metadata["expected_item_codes"])

    def test_three_photos_use_ordered_sequence_stitching(self) -> None:
        photos = [
            ("capture_1.jpg", _jpeg_bytes(0)),
            ("capture_2.jpg", _jpeg_bytes(1)),
            ("capture_3.jpg", _jpeg_bytes(2)),
        ]
        metadata = {
            "capture_order": [0, 1, 2],
            "sweep_direction": "left",
            "coverage_complete": False,
        }
        stitch_failure = SimpleNamespace(
            success=False,
            report={"mocked": True, "mode": "ordered_sequence"},
            match_preview=None,
            reason_code="mocked_sequence_failure",
            message="Mocked: sequence stitching was not run",
        )
        boundary, request_body = _multipart_payload(photos, metadata)

        with tempfile.TemporaryDirectory() as temporary_directory:
            session_root = Path(temporary_directory)
            with (
                patch.object(backend_app, "SESSION_ROOT", session_root),
                patch.object(backend_app, "stitch_two_images") as pair_stitch,
                patch.object(
                    backend_app,
                    "stitch_image_sequence",
                    return_value=stitch_failure,
                ) as sequence_stitch,
            ):
                status, response_body = asyncio.run(_post_multipart(request_body, boundary))

            self.assertEqual(status, 200, response_body)
            body = json.loads(response_body)
            self.assertEqual(body["status"], "retake_required")
            self.assertEqual(body["input_count"], 3)
            pair_stitch.assert_not_called()
            sequence_stitch.assert_called_once()
            called_paths = sequence_stitch.call_args.args[0]
            self.assertEqual([path.name for path in called_paths], [
                "capture_00.jpg", "capture_01.jpg", "capture_02.jpg",
            ])
            self.assertEqual(sequence_stitch.call_args.kwargs["capture_metadata"], metadata)


if __name__ == "__main__":
    unittest.main()
