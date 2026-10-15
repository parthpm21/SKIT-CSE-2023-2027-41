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


from backend.inference.video_backbone import dual_stream_backbone, DualStreamVideoBackbone
from backend.inference.face_aligner import video_face_pipeline, VideoFacePipeline


@dataclass
class VideoVerdict:
    label: str          # "real" | "fake"
    confidence: float
    frame_scores: list[float]
    model_version: str


class VideoDeepfakeService:
    """Serves real/manipulated verdicts for a (T, C, H, W) frame tensor."""

    def __init__(
        self,
        weights_path: Optional[str | Path] = None,
        backbone: Optional[DualStreamVideoBackbone] = None,
        face_pipeline: Optional[VideoFacePipeline] = None,
    ):
        self.model_version = "xception-efficientnet-ensemble-v1"
        self._model = None
        self._weights_loaded = False
        self.backbone = backbone or dual_stream_backbone
        self.face_pipeline = face_pipeline or video_face_pipeline
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
        """Drop-in point for trained ensemble checkpoint."""
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

        if self._model != "unavailable" and self._weights_loaded:
            with self._torch.no_grad():
                batch = self._torch.from_numpy(frame_tensor)
                logits = self._model(batch)
                frame_scores = self._torch.sigmoid(logits[:, 0]).tolist()
            avg_score = float(np.mean(frame_scores)) if frame_scores else 0.0
            label = "fake" if avg_score >= 0.5 else "real"
            confidence = avg_score if label == "fake" else 1.0 - avg_score
            return VideoVerdict(label, confidence, frame_scores, self.model_version)

        # Execute Pratham's Xception + EfficientNet Dual-Stream Backbone
        if frame_tensor.shape[0] == 0:
            return VideoVerdict("real", 1.0, [], self.model_version)

        # Align faces across frames if spatial dimensions match
        aligned_seq = self.face_pipeline.process_frames(frame_tensor)
        inference_res = self.backbone.predict_video_sequence(aligned_seq.aligned_frames)

        return VideoVerdict(
            label=inference_res.label,
            confidence=inference_res.confidence,
            frame_scores=inference_res.frame_scores,
            model_version=self.model_version,
        )


video_deepfake_service = VideoDeepfakeService()