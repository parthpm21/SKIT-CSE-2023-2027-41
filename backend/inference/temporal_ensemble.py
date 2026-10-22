"""
TruthLens — Temporal Ensemble Aggregator & Sequence Voting Model
Author: Pratham Yadav
Sprint 3: Development of the video deepfake detection pipeline (User Story 3 - Part 4)

This module provides temporal aggregation and sequence voting for video deepfake detection.
It fuses frame-level spatial feature embeddings (Xception + EfficientNet), SRM noise-fingerprint
temporal variance, and inter-frame feature drift via temporal attention pooling to produce
calibrated video authenticity verdicts and frame-level anomaly timelines.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from backend.inference.face_aligner import (
    AlignedVideoSequence,
    VideoFacePipeline,
    video_face_pipeline,
)
from backend.inference.video_backbone import (
    DualStreamVideoBackbone,
    FrameInferenceResult,
    dual_stream_backbone,
)
from backend.inference.noise_fingerprint import (
    NoiseFingerprintExtractor,
    NoiseFingerprintResult,
    noise_fingerprint_extractor,
)


@dataclass
class VideoEnsembleVerdict:
    """Unified verdict from the temporal video deepfake detection pipeline."""
    label: str                       # "real" | "fake"
    confidence: float                # [0.0, 1.0]
    frame_scores: List[float]        # Per-frame manipulation score sequence [0, 1]
    peak_score: float                # Maximum anomaly score observed across all frames
    mean_score: float                # Sequence mean anomaly score
    temporal_drift_score: float      # Inter-frame feature instability / flicker metric
    noise_inconsistency_score: float # Disruption in sensor PRNU noise across frames
    anomalous_frames_count: int      # Number of frames classified above detection threshold
    model_version: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "frame_scores": [round(float(s), 4) for s in self.frame_scores],
            "peak_score": round(float(self.peak_score), 4),
            "mean_score": round(float(self.mean_score), 4),
            "temporal_drift_score": round(float(self.temporal_drift_score), 4),
            "noise_inconsistency_score": round(float(self.noise_inconsistency_score), 4),
            "anomalous_frames_count": self.anomalous_frames_count,
            "total_frames": len(self.frame_scores),
            "model_version": self.model_version,
            "details": self.details,
        }


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + 1e-8)


class TemporalAttentionAggregator:
    """
    Computes temporal attention weights and inter-frame feature drift.
    Deepfakes introduce temporal discontinuities, landmark jitter, and feature drift.
    """

    def __init__(self, embed_dim: int = 512, seed: int = 42) -> None:
        self.embed_dim = embed_dim
        rng = np.random.RandomState(seed)
        scale = 1.0 / math.sqrt(embed_dim)
        self.q_weight = rng.uniform(-scale, scale, (embed_dim, embed_dim)).astype(np.float32)
        self.k_weight = rng.uniform(-scale, scale, (embed_dim, embed_dim)).astype(np.float32)
        self.scale = 1.0 / math.sqrt(embed_dim)

    def compute_temporal_drift(self, embeddings: np.ndarray) -> Tuple[float, List[float]]:
        """
        Calculates cosine distance drift between consecutive frames:
        Drift(t) = 1.0 - cosine_similarity(e_t, e_{t-1})
        """
        T = embeddings.shape[0]
        if T <= 1:
            return 0.0, [0.0] * T

        # Normalize embeddings
        norm = np.linalg.norm(embeddings, axis=-1, keepdims=True) + 1e-8
        unit_embeds = embeddings / norm

        drifts = [0.0]  # Frame 0 has 0 relative drift
        for t in range(1, T):
            cos_sim = float(np.dot(unit_embeds[t], unit_embeds[t - 1]))
            drift = max(0.0, 1.0 - cos_sim)
            drifts.append(drift)

        avg_drift = float(np.mean(drifts[1:]))
        # Logistic normalization: natural face transitions have drift < 0.15
        normalized_drift = float(1.0 / (1.0 + math.exp(-15.0 * (avg_drift - 0.20))))
        return normalized_drift, drifts

    def aggregate_attention(self, embeddings: np.ndarray, raw_scores: List[float]) -> Tuple[List[float], np.ndarray]:
        """
        Computes temporal attention weights across the frame sequence and refines scores.
        """
        T = embeddings.shape[0]
        if T <= 1:
            return raw_scores, np.ones(T, dtype=np.float32)

        # Q and K projections: (T, D)
        q = np.matmul(embeddings, self.q_weight)
        k = np.matmul(embeddings, self.k_weight)

        # Dot-product temporal attention matrix: (T, T)
        scores_matrix = np.matmul(q, k.T) * self.scale
        attn_matrix = _softmax(scores_matrix, axis=-1)

        # Frame importance weights (mean attention received by each frame)
        frame_weights = np.mean(attn_matrix, axis=0)  # (T,)
        frame_weights = frame_weights / (np.sum(frame_weights) + 1e-8)

        # Refined frame scores blending spatial predictions with temporal context
        refined_scores = []
        for t in range(T):
            # Contextual score is attention-weighted sum of all frame scores
            context_score = float(np.sum(attn_matrix[t] * np.array(raw_scores)))
            # Blend 70% immediate spatial + 30% temporal context
            blended = 0.7 * raw_scores[t] + 0.3 * context_score
            refined_scores.append(float(np.clip(blended, 0.0, 1.0)))

        return refined_scores, frame_weights


class VideoEnsemblePipeline:
    """
    End-to-end video deepfake detection orchestration pipeline.
    Coordinates:
    1. Facial landmark alignment (VideoFacePipeline)
    2. Spatial feature extraction (DualStreamVideoBackbone)
    3. High-frequency noise residuals (NoiseFingerprintExtractor)
    4. Temporal attention and sequence voting (TemporalAttentionAggregator)
    """

    def __init__(
        self,
        face_pipeline: Optional[VideoFacePipeline] = None,
        backbone: Optional[DualStreamVideoBackbone] = None,
        noise_extractor: Optional[NoiseFingerprintExtractor] = None,
        spatial_weight: float = 0.50,
        noise_weight: float = 0.25,
        temporal_weight: float = 0.25,
    ) -> None:
        self.face_pipeline = face_pipeline or video_face_pipeline
        self.backbone = backbone or dual_stream_backbone
        self.noise_extractor = noise_extractor or noise_fingerprint_extractor
        self.spatial_weight = spatial_weight
        self.noise_weight = noise_weight
        self.temporal_weight = temporal_weight
        self.temporal_aggregator = TemporalAttentionAggregator(embed_dim=self.backbone.config.embed_dim)
        self.model_version = "video-deepfake-temporal-ensemble-v2"

    def predict(self, frame_tensor: np.ndarray) -> VideoEnsembleVerdict:
        """
        Executes multi-stage video deepfake detection over an input frame tensor (T, C, H, W).
        """
        T = frame_tensor.shape[0]
        if T == 0:
            return VideoEnsembleVerdict(
                label="real",
                confidence=1.0,
                frame_scores=[],
                peak_score=0.0,
                mean_score=0.0,
                temporal_drift_score=0.0,
                noise_inconsistency_score=0.0,
                anomalous_frames_count=0,
                model_version=self.model_version,
            )

        # 1. Facial Landmark Alignment Pipeline
        aligned_seq: AlignedVideoSequence = self.face_pipeline.process_frames(frame_tensor)
        aligned_frames = aligned_seq.aligned_frames  # (T, 3, 224, 224)

        # 2. Dual-Stream Spatial Feature Extraction (Xception + EfficientNet)
        spatial_result: FrameInferenceResult = self.backbone.predict_video_sequence(aligned_frames)
        raw_scores = spatial_result.frame_scores
        embeddings = spatial_result.frame_embeddings

        # 3. High-Frequency SRM Noise Fingerprint Analysis
        noise_result: NoiseFingerprintResult = self.noise_extractor.analyze_sequence(aligned_frames)

        # 4. Temporal Attention & Inter-frame Feature Drift
        norm_drift, per_frame_drift = self.temporal_aggregator.compute_temporal_drift(embeddings)
        refined_scores, temporal_weights = self.temporal_aggregator.aggregate_attention(embeddings, raw_scores)

        # 5. Multi-Signal Ensemble Scoring
        # Combine spatial scores, noise temporal inconsistency, and feature drift
        spatial_mean = float(np.mean(refined_scores)) if refined_scores else 0.0
        noise_inconsistency = noise_result.temporal_inconsistency_score

        # Top-K anomalous keyframes
        sorted_scores = sorted(refined_scores, reverse=True)
        top_k = max(1, len(refined_scores) // 4)
        top_k_mean = float(np.mean(sorted_scores[:top_k]))

        # Final sequence ensemble score
        ensemble_score = (
            self.spatial_weight * (0.6 * top_k_mean + 0.4 * spatial_mean)
            + self.noise_weight * noise_inconsistency
            + self.temporal_weight * norm_drift
        )
        ensemble_score = float(np.clip(ensemble_score, 0.0, 1.0))

        is_fake = ensemble_score >= 0.50
        label = "fake" if is_fake else "real"
        confidence = ensemble_score if is_fake else (1.0 - ensemble_score)

        anomalous_count = sum(1 for s in refined_scores if s >= 0.50)
        peak_score = float(np.max(refined_scores)) if refined_scores else 0.0

        details = {
            "face_tracking": aligned_seq.to_dict(),
            "noise_fingerprint": noise_result.to_dict(),
            "temporal_drift": {
                "aggregate_drift": round(norm_drift, 4),
                "per_frame_drift": [round(float(d), 4) for d in per_frame_drift],
            },
            "ensemble_weights": {
                "spatial_weight": self.spatial_weight,
                "noise_weight": self.noise_weight,
                "temporal_weight": self.temporal_weight,
            },
            "top_anomalous_score": round(top_k_mean, 4),
        }

        return VideoEnsembleVerdict(
            label=label,
            confidence=confidence,
            frame_scores=refined_scores,
            peak_score=peak_score,
            mean_score=spatial_mean,
            temporal_drift_score=norm_drift,
            noise_inconsistency_score=noise_inconsistency,
            anomalous_frames_count=anomalous_count,
            model_version=self.model_version,
            details=details,
        )


# Global singleton instance
video_ensemble_pipeline = VideoEnsemblePipeline()
