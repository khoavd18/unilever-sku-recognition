"""LAN-friendly API for ordered shelf captures, stitching, recognition, and OSA observations."""

from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any

import cv2
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError

from src.osa.results import apply_count_corrections, build_osa_summary
from src.recognition.pipeline import PanoramaRecognizer, RecognitionConfig
from src.stitching.two_image import stitch_two_images
from src.stitching import StitchConfig, stitch_image_sequence


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(os.environ.get("PANORAMA_CONFIG", ROOT / "configs" / "panorama" / "backend.yaml")).resolve()
CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
BACKEND_CONFIG = CONFIG["backend"]
SESSION_ROOT = (ROOT / BACKEND_CONFIG["session_root"]).resolve()
SESSION_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Shelf Panorama Recognition API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(BACKEND_CONFIG.get("cors_origins", ["*"])),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_recognizer: PanoramaRecognizer | None = None
_recognizer_error: str | None = None
_recognizer_lock = threading.Lock()


class CountCorrection(BaseModel):
    item_code: str = Field(min_length=1, max_length=128)
    visible_count: int = Field(ge=0, le=100000)


class CorrectionRequest(BaseModel):
    actor: str = Field(min_length=1, max_length=128)
    corrections: list[CountCorrection] = Field(min_length=1, max_length=500)


def _recognition_paths() -> dict[str, Path]:
    config = RecognitionConfig(**CONFIG["recognition"])
    return {
        "yolo_weights": (ROOT / config.yolo_weights).resolve(),
        "embedding_checkpoint": (ROOT / config.embedding_checkpoint).resolve(),
        "catalog": (ROOT / config.catalog).resolve(),
        "gallery_dir": (ROOT / config.gallery_dir).resolve(),
        "mapping": (ROOT / config.mapping).resolve(),
    }


def _resolved_recognition_config() -> RecognitionConfig:
    value = dict(CONFIG["recognition"])
    for key, path in _recognition_paths().items():
        value[key] = str(path)
    return RecognitionConfig(**value)


def get_recognizer() -> PanoramaRecognizer:
    global _recognizer, _recognizer_error
    if _recognizer is not None:
        return _recognizer
    with _recognizer_lock:
        if _recognizer is None:
            try:
                missing = [name for name, path in _recognition_paths().items() if not path.exists()]
                if missing:
                    raise FileNotFoundError(f"Missing recognition artifacts: {missing}")
                _recognizer = PanoramaRecognizer(_resolved_recognition_config())
                _recognizer_error = None
            except Exception as exc:
                _recognizer_error = f"{type(exc).__name__}: {exc}"
                raise
    return _recognizer


def _artifact_url(session_id: str, filename: str) -> str:
    return f"/api/v1/sessions/{session_id}/artifacts/{filename}"


def _session_dir(session_id: str) -> Path:
    if not session_id or any(char not in "0123456789abcdef-" for char in session_id.casefold()):
        raise HTTPException(404, "Unknown session")
    directory = (SESSION_ROOT / session_id).resolve()
    if directory.parent != SESSION_ROOT or not directory.is_dir():
        raise HTTPException(404, "Unknown session")
    return directory


def _write_json_new(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite session artifact: {path}")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


@app.get("/api/v1/health")
def health() -> dict[str, Any]:
    paths = _recognition_paths()
    availability = {name: path.exists() for name, path in paths.items()}
    return {
        "status": "ready" if all(availability.values()) else "degraded",
        "stitching_available": True,
        "recognition_artifacts": availability,
        "recognition_loaded": _recognizer is not None,
        "recognition_load_error": _recognizer_error,
        "recognition_detector": str(paths["yolo_weights"]),
        "expected_detector_classes": _resolved_recognition_config().detector_class_names,
        "config": str(CONFIG_PATH),
        "note": "A degraded server can validate uploads and stitch; recognition needs all listed artifacts.",
    }


async def _save_upload(
    upload: UploadFile,
    destination: Path,
    max_bytes: int,
    allowed: set[str],
    max_megapixels: float | None = None,
) -> None:
    suffix = Path(upload.filename or "").suffix.casefold()
    if suffix not in allowed:
        raise HTTPException(415, f"Unsupported image extension: {suffix or '(none)'}")
    content = await upload.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise HTTPException(413, f"Upload exceeds {max_bytes} bytes")
    if not content:
        raise HTTPException(400, "Empty upload")
    destination.write_bytes(content)
    try:
        with Image.open(destination) as image_header:
            width, height = image_header.size
            image_header.verify()
    except (UnidentifiedImageError, OSError):
        destination.unlink(missing_ok=True)
        raise HTTPException(400, "Upload is not a decodable image")
    if max_megapixels is not None and width * height > max_megapixels * 1_000_000:
        # Close Pillow's file handle before unlinking; Windows does not allow an
        # open image to be removed and would otherwise mask this as a decode error.
        destination.unlink(missing_ok=True)
        raise HTTPException(413, f"Decoded image exceeds {max_megapixels:g} megapixels")
    image = cv2.imread(str(destination), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(400, "Upload is not a decodable image")


def _parse_metadata(value: str) -> dict[str, Any]:
    try:
        metadata = json.loads(value)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, "capture_metadata must be a JSON object") from exc
    if not isinstance(metadata, dict):
        raise HTTPException(400, "capture_metadata must be a JSON object")
    return metadata


@app.post("/api/v1/stitch")
async def stitch_capture(
    photos: list[UploadFile] = File(...),
    capture_metadata: str = Form("{}"),
) -> dict[str, Any]:
    max_uploads = int(BACKEND_CONFIG.get("max_stitch_uploads", 6))
    if len(photos) < 2:
        raise HTTPException(400, "Stitching requires at least two ordered photos")
    if len(photos) > max_uploads:
        raise HTTPException(400, f"At most {max_uploads} photos are allowed for stitching")
    metadata = _parse_metadata(capture_metadata)

    session_id = str(uuid.uuid4())
    session_dir = SESSION_ROOT / session_id
    session_dir.mkdir(parents=False)
    allowed = {str(value).casefold() for value in BACKEND_CONFIG.get("allowed_extensions", [])}
    saved_paths: list[Path] = []
    try:
        for index, upload in enumerate(photos):
            suffix = Path(upload.filename or ".jpg").suffix.casefold()
            destination = session_dir / f"capture_{index:02d}{suffix}"
            await _save_upload(
                upload,
                destination,
                int(BACKEND_CONFIG["max_upload_bytes"]),
                allowed,
                float(BACKEND_CONFIG.get("max_input_megapixels", 24)),
            )
            saved_paths.append(destination)
        _write_json_new(session_dir / "capture_metadata.json", {
            "mode": "stitch_only",
            "capture_order": [path.name for path in saved_paths],
            "client_metadata": metadata,
        })
        stitch_config = StitchConfig(
            max_canvas_megapixels=float(BACKEND_CONFIG.get("max_stitch_canvas_megapixels", 60))
        )
        stitched = stitch_image_sequence(
            saved_paths,
            config=stitch_config,
            capture_metadata=metadata,
        )
        _write_json_new(session_dir / "stitch_report.json", stitched.report)
        if stitched.match_preview is not None:
            cv2.imwrite(str(session_dir / "match_preview.jpg"), stitched.match_preview)
        if not stitched.success:
            response = {
                "session_id": session_id,
                "mode": "stitch_only",
                "status": "retake_required",
                "input_count": len(saved_paths),
                "stitching": {
                    "success": False,
                    "reason_code": stitched.reason_code,
                    "message": stitched.message,
                    "report_url": _artifact_url(session_id, "stitch_report.json"),
                    "match_preview_url": _artifact_url(session_id, "match_preview.jpg")
                    if stitched.match_preview is not None else None,
                },
            }
            _write_json_new(session_dir / "result.json", response)
            return response

        assert stitched.panorama is not None and stitched.valid_mask is not None
        cv2.imwrite(str(session_dir / "panorama.jpg"), stitched.panorama)
        cv2.imwrite(str(session_dir / "valid_mask.png"), stitched.valid_mask)
        if stitched.alignment_preview is not None:
            cv2.imwrite(str(session_dir / "alignment_preview.jpg"), stitched.alignment_preview)
        response = {
            "session_id": session_id,
            "mode": "stitch_only",
            "status": "success",
            "input_count": len(saved_paths),
            "stitching": {
                "success": True,
                "status": stitched.status,
                "message": stitched.message,
                "panorama_url": _artifact_url(session_id, "panorama.jpg"),
                "panorama_width": int(stitched.panorama.shape[1]),
                "panorama_height": int(stitched.panorama.shape[0]),
                "valid_mask_url": _artifact_url(session_id, "valid_mask.png"),
                "report_url": _artifact_url(session_id, "stitch_report.json"),
                "match_preview_url": _artifact_url(session_id, "match_preview.jpg")
                if stitched.match_preview is not None else None,
                "alignment_preview_url": _artifact_url(session_id, "alignment_preview.jpg")
                if stitched.alignment_preview is not None else None,
            },
        }
        _write_json_new(session_dir / "result.json", response)
        return response
    except HTTPException:
        raise
    except Exception as exc:
        _write_json_new(session_dir / "processing_error.json", {
            "type": type(exc).__name__, "message": str(exc), "mode": "stitch_only"
        })
        raise HTTPException(500, f"Stitching failed: {type(exc).__name__}") from exc


@app.post("/api/v1/process")
async def process_capture(
    photos: list[UploadFile] = File(...),
    capture_metadata: str = Form("{}"),
    expected_item_codes: str | None = Form(None),
) -> dict[str, Any]:
    max_uploads = int(BACKEND_CONFIG.get("max_uploads", 6))
    if len(photos) < 2:
        raise HTTPException(400, "Recognition requires at least two adjacent photos in capture order")
    if len(photos) > max_uploads:
        raise HTTPException(400, f"At most {max_uploads} photos are allowed")
    try:
        metadata = json.loads(capture_metadata)
        if not isinstance(metadata, dict):
            raise ValueError
        expected = None if expected_item_codes is None else json.loads(expected_item_codes)
        if expected is not None and not isinstance(expected, list):
            raise ValueError
        if expected is not None and any(
            not isinstance(item_code, str) or not item_code.strip()
            for item_code in expected
        ):
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(400, "capture_metadata must be a JSON object and expected_item_codes a JSON list")

    session_id = str(uuid.uuid4())
    session_dir = SESSION_ROOT / session_id
    session_dir.mkdir(parents=False)
    allowed = {str(value).casefold() for value in BACKEND_CONFIG.get("allowed_extensions", [])}
    saved_paths = []
    try:
        for index, upload in enumerate(photos):
            suffix = Path(upload.filename or ".jpg").suffix.casefold()
            destination = session_dir / f"capture_{index:02d}{suffix}"
            await _save_upload(
                upload,
                destination,
                int(BACKEND_CONFIG["max_upload_bytes"]),
                allowed,
                float(BACKEND_CONFIG.get("max_input_megapixels", 24)),
            )
            saved_paths.append(destination)
        _write_json_new(session_dir / "capture_metadata.json", {
            "capture_order": [path.name for path in saved_paths],
            "client_metadata": metadata,
            "expected_item_codes": expected,
        })
        if len(saved_paths) == 2:
            stitched = stitch_two_images(saved_paths[0], saved_paths[1], capture_metadata=metadata)
        else:
            stitch_config = StitchConfig(
                max_canvas_megapixels=float(BACKEND_CONFIG.get("max_stitch_canvas_megapixels", 60))
            )
            stitched = stitch_image_sequence(
                saved_paths,
                config=stitch_config,
                capture_metadata=metadata,
            )
        _write_json_new(session_dir / "stitch_report.json", stitched.report)
        if stitched.match_preview is not None:
            cv2.imwrite(str(session_dir / "match_preview.jpg"), stitched.match_preview)
        if not stitched.success:
            response = {
                "session_id": session_id,
                "status": "retake_required",
                "input_count": len(saved_paths),
                "stitching": {
                    "success": False, "reason_code": stitched.reason_code, "message": stitched.message,
                    "report_url": _artifact_url(session_id, "stitch_report.json"),
                    "match_preview_url": _artifact_url(session_id, "match_preview.jpg") if stitched.match_preview is not None else None,
                },
                "recognition": None,
                "osa": build_osa_summary([], expected, "insufficient"),
            }
            _write_json_new(session_dir / "result.json", response)
            return response

        cv2.imwrite(str(session_dir / "panorama.jpg"), stitched.panorama)
        cv2.imwrite(str(session_dir / "valid_mask.png"), stitched.valid_mask)
        if stitched.alignment_preview is not None:
            cv2.imwrite(str(session_dir / "alignment_preview.jpg"), stitched.alignment_preview)
        recognition: dict[str, Any] | None = None
        recognition_error: str | None = None
        try:
            recognition = get_recognizer().recognize(stitched.panorama, stitched.valid_mask)
        except Exception as exc:
            recognition_error = f"{type(exc).__name__}: {exc}"
        detections = recognition["detections"] if recognition else []
        coverage_status = "complete" if metadata.get("coverage_complete") is True else "unknown"
        osa = build_osa_summary(detections, expected, coverage_status)
        response = {
            "session_id": session_id,
            "status": "success" if recognition else "partial_recognition_unavailable",
            "input_count": len(saved_paths),
            "stitching": {
                "success": True, "status": stitched.status, "message": stitched.message,
                "panorama_url": _artifact_url(session_id, "panorama.jpg"),
                "panorama_width": int(stitched.panorama.shape[1]),
                "panorama_height": int(stitched.panorama.shape[0]),
                "valid_mask_url": _artifact_url(session_id, "valid_mask.png"),
                "report_url": _artifact_url(session_id, "stitch_report.json"),
                "match_preview_url": _artifact_url(session_id, "match_preview.jpg"),
                "alignment_preview_url": _artifact_url(session_id, "alignment_preview.jpg"),
            },
            "recognition": recognition,
            "recognition_error": recognition_error,
            "osa": osa,
        }
        _write_json_new(session_dir / "result.json", response)
        return response
    except HTTPException:
        raise
    except Exception as exc:
        _write_json_new(session_dir / "processing_error.json", {"type": type(exc).__name__, "message": str(exc)})
        raise HTTPException(500, f"Session processing failed: {type(exc).__name__}") from exc


@app.get("/api/v1/sessions/{session_id}/artifacts/{filename}")
def get_artifact(session_id: str, filename: str) -> FileResponse:
    directory = _session_dir(session_id)
    target = (directory / filename).resolve()
    if target.parent != directory or not target.is_file():
        raise HTTPException(404, "Artifact not found")
    return FileResponse(target)


@app.post("/api/v1/sessions/{session_id}/corrections")
def correct_counts(session_id: str, request: CorrectionRequest) -> dict[str, Any]:
    directory = _session_dir(session_id)
    result_path = directory / "result.json"
    if not result_path.is_file():
        raise HTTPException(409, "Session has no completed result")
    original = json.loads(result_path.read_text(encoding="utf-8"))
    existing = sorted(directory.glob("corrections.v*.json"))
    if existing:
        base = json.loads(existing[-1].read_text(encoding="utf-8"))
        summary = base["osa"]
    else:
        summary = original["osa"]
    corrected = apply_count_corrections(
        summary, [item.model_dump() for item in request.corrections], request.actor
    )
    version = len(existing) + 1
    artifact = {
        "session_id": session_id,
        "correction_version": version,
        "original_model_result": "result.json",
        "osa": corrected,
    }
    path = directory / f"corrections.v{version:03d}.json"
    _write_json_new(path, artifact)
    artifact["artifact_url"] = _artifact_url(session_id, path.name)
    return artifact
