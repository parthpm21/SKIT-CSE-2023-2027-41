"""
TruthLens — Video Deepfake Detection Microservice
Author: Parth Maheshwari
Sprint 2: Backend integration of classification and video deepfake services.

Wraps the Xception/EfficientNet ensemble (owned/trained under Pratham's
"video deepfake detection pipeline" story) behind a stable service contract:
frame-tensor stack in, per-video verdict out.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class VideoVerdict:
    label: str          # "real" | "fake"
    confidence: float
    frame_scores: list[float]
    model_version: str


class VideoDeepfakeService:
    """Serves real/manipulated verdicts for a (T, C, H, W) frame tensor."""

    def __init__(self, weights_path: Optional[str | Path] = None):
        self.model_version = "xception-efficientnet-ensemble-v0"
        self._model = None
        self._weights_loaded = False
        if weights_path:
            self.load_weights(weights_path)

    def _lazy_load_backbone(self):
        if self._model is not None:
            return
        try:
            import torch
            import torchvision.models as models
            model = models.efficientnet_b0(weights=None)
            model.eval()
            self._model = model
            self._torch = torch
        except ImportError:
            self._model = "unavailable"

    def load_weights(self, weights_path: str | Path) -> None:
        """Drop-in point for Pratham's trained ensemble checkpoint."""
        self._lazy_load_backbone()
        import torch
        state = torch.load(weights_path, map_location="cpu")
        self._model.load_state_dict(state)
        self._weights_loaded = True

    def predict(self, frame_tensor: np.ndarray) -> VideoVerdict:
        """
        frame_tensor: (T, C, H, W) normalized array from
        backend.ingestion.preprocessing.preprocess_video_file
        """
        self._lazy_load_backbone()

        if self._model == "unavailable" or not self._weights_loaded:
            # Placeholder per-frame heuristic so the endpoint/routing layer
            # is fully testable ahead of the trained ensemble landing.
            frame_scores = [float(np.tanh(frame.mean())) for frame in frame_tensor]
        else:
            with self._torch.no_grad():
                batch = self._torch.from_numpy(frame_tensor)
                logits = self._model(batch)
                frame_scores = self._torch.sigmoid(logits[:, 0]).tolist()

        avg_score = float(np.mean(frame_scores)) if frame_scores else 0.0
        label = "fake" if avg_score > 0 else "real"
        version = self.model_version if self._weights_loaded else self.model_version + "-placeholder"
        return VideoVerdict(label, abs(avg_score), frame_scores, version)


video_deepfake_service = VideoDeepfakeService()