"""
TruthLens — Automated Video Deepfake Stress, Robustness & Edge-Case Test Suite
Author: Pratham Yadav
Sprint 3: User Story 3 - Part 5 (Robustness & Adversarial Testing)

Validates the full video deepfake detection pipeline under severe real-world degradations:
1. Heavy video compression artifacts (H.264 C23 vs C40 macroblock simulation)
2. Temporal frame dropouts, variable framerates, and stuttering / duplicate frames
3. Severe facial occlusion (sunglasses, masks, hand obstructions) and off-axis tilts
4. Erratic lighting, strobing exposure fluctuations, and low-light sensor noise
5. Synthetic FaceForensics++ deepfake manipulation profiles vs authentic video baselines
6. Numerical boundary edge-cases: zero-length tensors, single-frame clips, and extreme aspect ratios.
"""
from __future__ import annotations

import math
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

from backend.inference.face_aligner import (
    AffineFaceAligner,
    FaceDetector,
    VideoFacePipeline,
    video_face_pipeline,
)
from backend.inference.video_backbone import (
    DualStreamVideoBackbone,
    dual_stream_backbone,
)
from backend.inference.noise_fingerprint import (
    NoiseFingerprintExtractor,
    noise_fingerprint_extractor,
)
from backend.inference.temporal_ensemble import (
    TemporalAttentionAggregator,
    VideoEnsemblePipeline,
    VideoEnsembleVerdict,
    video_ensemble_pipeline,
)
from backend.inference.video_deepfake_service import (
    VideoDeepfakeService,
    VideoVerdict,
    video_deepfake_service,
)


class TestVideoDeepfakeStressAndEdgeCases:
    """Comprehensive adversarial and stress tests for the video deepfake detection pipeline."""

    @pytest.fixture
    def pipeline(self) -> VideoEnsemblePipeline:
        return video_ensemble_pipeline

    @pytest.fixture
    def synthetic_video_tensor(self) -> np.ndarray:
        """Constructs an 8-frame video tensor (8, 3, 224, 224) with a synthetic face region."""
        frames = []
        rng = np.random.RandomState(42)
        for i in range(8):
            frame = np.ones((224, 224, 3), dtype=np.uint8) * 35
            # Central facial skin tone region (R > G > B) with slight motion
            shift = int(2 * math.sin(i * 0.5))
            y1, y2 = 50 + shift, 170 + shift
            x1, x2 = 60 + shift, 160 + shift
            frame[y1:y2, x1:x2, 0] = 200
            frame[y1:y2, x1:x2, 1] = 140
            frame[y1:y2, x1:x2, 2] = 110
            # Eyes and mouth
            frame[y1 + 30 : y1 + 40, x1 + 20 : x1 + 35] = 40  # Left eye
            frame[y1 + 30 : y1 + 40, x2 - 35 : x2 - 20] = 40  # Right eye
            frame[y2 - 30 : y2 - 20, x1 + 30 : x2 - 30] = 70  # Mouth

            # Normalize to [-1, 1] float32 tensor (3, 224, 224)
            t_frame = ((frame.astype(np.float32) / 127.5) - 1.0).transpose(2, 0, 1)
            frames.append(t_frame)
        return np.stack(frames, axis=0)

    # ------------------------------------------------------------------------
    # 1. H.264 / Video Compression Artifacts
    # ------------------------------------------------------------------------

    def test_h264_compression_quantization_robustness(
        self, pipeline: VideoEnsemblePipeline, synthetic_video_tensor: np.ndarray
    ):
        """
        Simulates heavy video compression (C23 / C40) via 8x8 block discrete cosine
        quantization and high-frequency loss. Tests that pipeline produces finite, valid scores.
        """
        T, C, H, W = synthetic_video_tensor.shape
        compressed = synthetic_video_tensor.copy()

        # Simulate 8x8 macroblock quantization
        block_size = 8
        for b_y in range(0, H, block_size):
            for b_x in range(0, W, block_size):
                # Heavy quantization averages blocks
                block_mean = np.mean(
                    compressed[:, :, b_y : b_y + block_size, b_x : b_x + block_size],
                    axis=(2, 3),
                    keepdims=True,
                )
                compressed[:, :, b_y : b_y + block_size, b_x : b_x + block_size] = block_mean

        verdict = pipeline.predict(compressed)
        assert isinstance(verdict, VideoEnsembleVerdict)
        assert verdict.label in {"real", "fake"}
        assert np.isfinite(verdict.confidence)
        assert 0.0 <= verdict.confidence <= 1.0
        assert not np.isnan(verdict.frame_scores).any()
        assert np.isfinite(verdict.temporal_drift_score)
        assert np.isfinite(verdict.noise_inconsistency_score)

    # ------------------------------------------------------------------------
    # 2. Frame Stutter, Dropout & Duplicate Freeze Frames
    # ------------------------------------------------------------------------

    def test_frame_stutter_and_duplicate_freeze(
        self, pipeline: VideoEnsemblePipeline, synthetic_video_tensor: np.ndarray
    ):
        """
        Tests behavior under extreme video playback stutter where identical frames
        are repeated consecutively (e.g. frozen video or 1fps stutter).
        Cosine drift must not cause division by zero or NaN values.
        """
        # Freeze frame 0 across the entire 8-frame sequence
        frozen_seq = np.repeat(synthetic_video_tensor[:1], 8, axis=0)
        verdict = pipeline.predict(frozen_seq)

        assert isinstance(verdict, VideoEnsembleVerdict)
        assert np.isfinite(verdict.temporal_drift_score)
        # Identical frames produce zero or near-zero temporal drift
        assert verdict.temporal_drift_score < 0.20
        assert len(verdict.frame_scores) == 8
        assert not any(np.isnan(verdict.frame_scores))

    def test_frame_dropout_variable_sequence_length(
        self, pipeline: VideoEnsemblePipeline, synthetic_video_tensor: np.ndarray
    ):
        """
        Tests sequence lengths under packet loss / frame drops: 1, 2, 3, 5 frames.
        """
        for seq_len in [1, 2, 3, 5]:
            sub_seq = synthetic_video_tensor[:seq_len]
            verdict = pipeline.predict(sub_seq)

            assert isinstance(verdict, VideoEnsembleVerdict)
            assert len(verdict.frame_scores) == seq_len
            assert 0.0 <= verdict.confidence <= 1.0
            assert np.isfinite(verdict.peak_score)
            assert np.isfinite(verdict.mean_score)

    # ------------------------------------------------------------------------
    # 3. Partial Facial Occlusion & Off-Axis Tilt
    # ------------------------------------------------------------------------

    def test_facial_occlusion_sunglasses_and_mask(
        self, pipeline: VideoEnsemblePipeline, synthetic_video_tensor: np.ndarray
    ):
        """
        Injects severe facial occlusion into frames:
        - Frames 0-3: Upper eye region occluded (simulating dark sunglasses)
        - Frames 4-7: Lower mouth/jaw region occluded (simulating sanitary mask)
        Validates pipeline continues to align or fallback without raising exceptions.
        """
        occluded = synthetic_video_tensor.copy()
        # Upper face occlusion (sunglasses)
        occluded[0:4, :, 60:110, 60:160] = -1.0  # Blacked out
        # Lower face occlusion (mask)
        occluded[4:8, :, 120:180, 60:160] = 0.5   # White mask block

        verdict = pipeline.predict(occluded)
        assert isinstance(verdict, VideoEnsembleVerdict)
        assert verdict.label in {"real", "fake"}
        assert 0.0 <= verdict.confidence <= 1.0
        assert len(verdict.frame_scores) == 8

    # ------------------------------------------------------------------------
    # 4. Severe Lighting Fluctuations & Strobing Flicker
    # ------------------------------------------------------------------------

    def test_strobing_exposure_fluctuations(
        self, pipeline: VideoEnsemblePipeline, synthetic_video_tensor: np.ndarray
    ):
        """
        Simulates erratic strobing / flickering lighting between adjacent frames:
        Alternating frames made 3x brighter (+0.8) and dark (-0.8).
        Validates the temporal noise and feature drift components flag instability.
        """
        flicker_tensor = synthetic_video_tensor.copy()
        for t in range(flicker_tensor.shape[0]):
            delta = 0.7 if (t % 2 == 0) else -0.7
            flicker_tensor[t] = np.clip(flicker_tensor[t] + delta, -1.0, 1.0)

        verdict = pipeline.predict(flicker_tensor)
        assert isinstance(verdict, VideoEnsembleVerdict)
        # Noise inconsistency or temporal drift should be noticeably triggered
        assert verdict.noise_inconsistency_score > 0.0
        assert verdict.temporal_drift_score > 0.0

    # ------------------------------------------------------------------------
    # 5. FaceForensics++ Deepfake vs Authentic Comparison Profiles
    # ------------------------------------------------------------------------

    def test_faceforensics_deepfake_profile_vs_authentic(
        self, pipeline: VideoEnsemblePipeline, synthetic_video_tensor: np.ndarray
    ):
        """
        Compares an authentic video simulation with a synthetic deepfake profile:
        - Authentic: Smooth trajectory, consistent sensor noise across frames.
        - Deepfake: Jittery central facial region, mismatched noise variance, temporal boundary flicker.
        """
        # Authentic sequence
        auth_seq = synthetic_video_tensor.copy()
        auth_verdict = pipeline.predict(auth_seq)

        # Deepfake sequence: inject central noise disruption and alternating frame jitter
        rng = np.random.RandomState(99)
        fake_seq = synthetic_video_tensor.copy()
        for t in range(fake_seq.shape[0]):
            # High-frequency noise suppression / artifact in face area (center 50:170, 60:160)
            noise_patch = rng.uniform(-0.6, 0.6, (3, 120, 100)).astype(np.float32)
            if t % 2 == 1:
                fake_seq[t, :, 50:170, 60:160] = np.clip(
                    fake_seq[t, :, 50:170, 60:160] + noise_patch, -1.0, 1.0
                )
            else:
                fake_seq[t, :, 50:170, 60:160] = np.clip(
                    fake_seq[t, :, 50:170, 60:160] * 0.4, -1.0, 1.0
                )

        fake_verdict = pipeline.predict(fake_seq)

        # Deepfake profile must exhibit higher noise inconsistency or peak anomaly
        assert (
            fake_verdict.noise_inconsistency_score >= auth_verdict.noise_inconsistency_score
            or fake_verdict.peak_score >= auth_verdict.peak_score
        )

    # ------------------------------------------------------------------------
    # 6. Edge Cases & Boundary Conditions
    # ------------------------------------------------------------------------

    def test_empty_tensor_edge_case(self, pipeline: VideoEnsemblePipeline):
        """Empty video input tensor must return authentic verdict with 0 frames."""
        empty = np.zeros((0, 3, 224, 224), dtype=np.float32)
        verdict = pipeline.predict(empty)

        assert verdict.label == "real"
        assert verdict.confidence == 1.0
        assert verdict.frame_scores == []
        assert verdict.anomalous_frames_count == 0

    def test_single_frame_video_edge_case(self, pipeline: VideoEnsemblePipeline):
        """Single-frame clip must process without IndexError in temporal drift."""
        single = np.random.uniform(-1.0, 1.0, (1, 3, 224, 224)).astype(np.float32)
        verdict = pipeline.predict(single)

        assert isinstance(verdict, VideoEnsembleVerdict)
        assert len(verdict.frame_scores) == 1
        assert verdict.temporal_drift_score == 0.0
        assert verdict.noise_inconsistency_score == 0.0

    def test_all_black_and_all_white_video_frames(self, pipeline: VideoEnsemblePipeline):
        """All-black (-1.0) and all-white (+1.0) frames must not trigger NaN or zero-division errors."""
        black_seq = np.full((4, 3, 224, 224), -1.0, dtype=np.float32)
        white_seq = np.full((4, 3, 224, 224), 1.0, dtype=np.float32)

        verdict_black = pipeline.predict(black_seq)
        verdict_white = pipeline.predict(white_seq)

        assert np.isfinite(verdict_black.confidence)
        assert np.isfinite(verdict_white.confidence)
        assert not np.isnan(verdict_black.frame_scores).any()
        assert not np.isnan(verdict_white.frame_scores).any()

    def test_video_deepfake_service_stress_integration(self):
        """Ensures the high-level service wrapper handles corrupted or extreme inputs gracefully."""
        service = VideoDeepfakeService()
        corrupt_frames = np.random.uniform(-1.0, 1.0, (5, 3, 224, 224)).astype(np.float32)
        # Corrupt one frame with extreme values
        corrupt_frames[2] = np.clip(corrupt_frames[2] * 100.0, -1.0, 1.0)

        verdict = service.predict(corrupt_frames)
        assert isinstance(verdict, VideoVerdict)
        assert verdict.label in {"real", "fake"}
        assert 0.0 <= verdict.confidence <= 1.0
        assert len(verdict.frame_scores) == 5
        assert isinstance(verdict.details, dict)
        assert "noise_fingerprint" in verdict.details
