"""
TruthLens — Inference Routing Layer
Author: Parth Maheshwari
Sprint 2: Backend integration of classification and video deepfake services.

Single entry point that dispatches an uploaded file to the correct
microservice (image classification vs video deepfake detection) based on
media type, and returns a unified response schema.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

from backend.ingestion.preprocessing import preprocess_image_file, preprocess_video_file
from backend.inference.classification_service import classification_service
from backend.inference.video_deepfake_service import video_deepfake_service

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}


@dataclass
class InferenceResponse:
    media_type: str
    label: str
    confidence: float
    model_version: str
    details: dict


def detect_media_type(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in IMAGE_EXT:
        return "image"
    if ext in VIDEO_EXT:
        return "video"
    raise ValueError(f"Unsupported file type: {ext}")


def route_inference(file_path: str | Path, filename: str) -> InferenceResponse:
    """Preprocess + classify a file through the branch matching its media type."""
    media_type = detect_media_type(filename)

    if media_type == "image":
        tensor = preprocess_image_file(file_path)
        result = classification_service.predict(tensor)
        return InferenceResponse(
            media_type="image",
            label=result.label,
            confidence=result.confidence,
            model_version=result.model_version,
            details={},
        )

    tensor = preprocess_video_file(file_path)
    result = video_deepfake_service.predict(tensor)
    return InferenceResponse(
        media_type="video",
        label=result.label,
        confidence=result.confidence,
        model_version=result.model_version,
        details={"frame_scores": result.frame_scores},
    )