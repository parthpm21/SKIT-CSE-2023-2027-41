"""
TruthLens — Automated Tests for Noise Fingerprinting & Temporal Attention Ensemble
Author: Pratham Yadav
Sprint 3: User Story 3 - Parts 3 & 4 Validation

Validates:
- Spatial Rich Model (SRM) high-pass noise-fingerprint extraction and kernel zero-sum constraints.
- Central facial vs background noise discrepancy ratios.
- Inter-frame noise residual inconsistency metrics across video frames.
- Scaled dot-product temporal attention pooling and cosine feature drift.
- Full video ensemble pipeline and service-layer verdict integration.
"""
import numpy as np
import pytest

from backend.inference.noise_fingerprint import (
    NoiseFingerprintExtractor,
    NoiseFingerprintResult,
    SRM_KERNEL_1ST_ORDER,
    SRM_KERNEL_2ND_ORDER,
    SRM_KERNEL_SQUARE_3X3,
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
)


# ============================================================================
# Part 3: Noise-Fingerprint Residual Extractor Tests
# ============================================================================

class TestNoiseFingerprintExtractor:
    """Validates SRM filter banks and sensor noise consistency extraction."""

    @pytest.fixture
    def extractor(self) -> NoiseFingerprintExtractor:
        return NoiseFingerprintExtractor(
            inconsistency_threshold=0.45,
            discrepancy_threshold=0.35,
        )

    def test_srm_filters_sum_to_zero(self, extractor: NoiseFingerprintExtractor):
        """All SRM kernels must sum to 0.0 to eliminate base image luminance."""
        for i, kernel in enumerate(extractor.kernels):
            kernel_sum = float(np.sum(kernel))
            assert abs(kernel_sum) < 1e-5, f"SRM filter #{i} does not sum to zero (sum={kernel_sum})"

    def test_extract_residual_flat_image(self, extractor: NoiseFingerprintExtractor):
        """Flat uniform image must produce zero residual energy and zero discrepancy."""
        flat_image = np.full((3, 128, 128), 0.5, dtype=np.float32)
        norm_map, energy, discrepancy = extractor.extract_frame_noise(flat_image)
        assert norm_map.shape == (128, 128)
        assert energy < 1e-4

    def test_extract_residual_noisy_image(self, extractor: NoiseFingerprintExtractor):
        """Image with additive noise must produce non-zero high-frequency energy."""
        rng = np.random.RandomState(42)
        noisy = rng.uniform(-1.0, 1.0, (3, 128, 128)).astype(np.float32)

        norm_map, energy, discrepancy = extractor.extract_frame_noise(noisy)
        assert norm_map.shape == (128, 128)
        assert energy > 0.0
        assert 0.0 <= discrepancy <= 1.0

    def test_extract_frame_noise_discrepancy(self, extractor: NoiseFingerprintExtractor):
        """Manipulated central face region shows distinct residual energy from peripheral background."""
        rng = np.random.RandomState(10)
        # Background: smooth with low noise
        frame = np.ones((3, 128, 128), dtype=np.float32) * 0.2
        # Central face area: heavy noise / GAN artifact
        frame[:, 32:96, 32:96] += rng.uniform(-0.5, 0.5, (3, 64, 64)).astype(np.float32)

        norm_map, energy, discrepancy = extractor.extract_frame_noise(frame)
        assert energy > 0.0
        # Discrepancy ratio should be elevated due to central noise
        assert discrepancy > 0.30

    def test_analyze_sequence_empty_and_single(self, extractor: NoiseFingerprintExtractor):
        """Handles empty and single-frame inputs gracefully."""
        empty_frames = np.zeros((0, 3, 128, 128), dtype=np.float32)
        res_empty = extractor.analyze_sequence(empty_frames)
        assert isinstance(res_empty, NoiseFingerprintResult)
        assert res_empty.mean_noise_energy == 0.0
        assert res_empty.temporal_inconsistency_score == 0.0
        assert res_empty.frame_noise_energies == []

        single_frame = np.ones((1, 3, 128, 128), dtype=np.float32) * 0.5
        res_single = extractor.analyze_sequence(single_frame)
        assert len(res_single.frame_noise_energies) == 1
        assert res_single.temporal_inconsistency_score == 0.0

    def test_analyze_sequence_temporal_inconsistency(self, extractor: NoiseFingerprintExtractor):
        """Sequences with flickering noise patterns yield higher inconsistency scores than stable ones."""
        rng = np.random.RandomState(42)

        # Stable sequence (identical static frames with fixed noise)
        fixed_frame = rng.uniform(-1.0, 1.0, (1, 3, 128, 128)).astype(np.float32)
        stable_seq = np.repeat(fixed_frame, 5, axis=0)
        stable_result = extractor.analyze_sequence(stable_seq)

        # Flickering sequence (rapidly alternating noise levels per frame)
        flickering_seq = []
        for i in range(5):
            noise_scale = 1.0 if (i % 2 == 0) else 0.05
            f = rng.uniform(-noise_scale, noise_scale, (3, 128, 128)).astype(np.float32)
            flickering_seq.append(f)
        flickering_seq = np.stack(flickering_seq, axis=0)
        flickering_result = extractor.analyze_sequence(flickering_seq)

        assert flickering_result.temporal_inconsistency_score > stable_result.temporal_inconsistency_score

    def test_noise_fingerprint_serialization(self, extractor: NoiseFingerprintExtractor):
        frames = np.random.uniform(-1.0, 1.0, (3, 3, 128, 128)).astype(np.float32)
        result = extractor.analyze_sequence(frames)
        d = result.to_dict()

        assert "mean_noise_energy" in d
        assert "face_bg_noise_discrepancy" in d
        assert "temporal_inconsistency_score" in d
        assert "frame_noise_energies" in d
        assert len(d["frame_noise_energies"]) == 3
        assert "is_noise_anomalous" in d
        assert "metadata" in d


# ============================================================================
# Part 4: Temporal Attention & Sequence Voting Tests
# ============================================================================

class TestTemporalAttentionAggregator:
    """Validates self-attention weights and cosine feature drift across frames."""

    @pytest.fixture
    def aggregator(self) -> TemporalAttentionAggregator:
        return TemporalAttentionAggregator(embed_dim=64, seed=42)

    def test_compute_temporal_drift_identical_embeddings(self, aggregator: TemporalAttentionAggregator):
        """Static video has zero cosine drift."""
        embed = np.ones((1, 64), dtype=np.float32)
        seq_embeds = np.repeat(embed, 6, axis=0)

        norm_drift, drifts = aggregator.compute_temporal_drift(seq_embeds)
        assert len(drifts) == 6
        assert all(d < 1e-4 for d in drifts)
        # Normalized drift should be low
        assert norm_drift < 0.20

    def test_compute_temporal_drift_jittery_embeddings(self, aggregator: TemporalAttentionAggregator):
        """Alternating orthogonal vectors produce high drift."""
        v1 = np.zeros(64, dtype=np.float32)
        v1[0] = 1.0
        v2 = np.zeros(64, dtype=np.float32)
        v2[1] = 1.0

        seq_embeds = np.array([v1, v2, v1, v2, v1, v2], dtype=np.float32)
        norm_drift, drifts = aggregator.compute_temporal_drift(seq_embeds)

        assert len(drifts) == 6
        # Consecutive orthogonal vectors have cosine_similarity = 0 -> drift = 1.0
        for d in drifts[1:]:
            assert abs(d - 1.0) < 1e-4
        assert norm_drift > 0.80

    def test_aggregate_attention_weights_sum_to_one(self, aggregator: TemporalAttentionAggregator):
        """Attention weights across frame sequence must sum to 1.0."""
        rng = np.random.RandomState(42)
        seq_embeds = rng.randn(8, 64).astype(np.float32)
        raw_scores = [0.1 * i for i in range(8)]

        refined_scores, frame_weights = aggregator.aggregate_attention(seq_embeds, raw_scores)

        assert len(refined_scores) == 8
        assert len(frame_weights) == 8
        assert abs(float(np.sum(frame_weights)) - 1.0) < 1e-5
        assert all(0.0 <= s <= 1.0 for s in refined_scores)


class TestVideoEnsemblePipeline:
    """Validates end-to-end multi-stream video ensemble orchestration."""

    @pytest.fixture
    def pipeline(self) -> VideoEnsemblePipeline:
        return VideoEnsemblePipeline(
            spatial_weight=0.50,
            noise_weight=0.25,
            temporal_weight=0.25,
        )

    def test_pipeline_empty_tensor(self, pipeline: VideoEnsemblePipeline):
        empty = np.zeros((0, 3, 224, 224), dtype=np.float32)
        verdict = pipeline.predict(empty)

        assert isinstance(verdict, VideoEnsembleVerdict)
        assert verdict.label == "real"
        assert verdict.confidence == 1.0
        assert verdict.frame_scores == []
        assert verdict.anomalous_frames_count == 0

    def test_pipeline_synthetic_sequence(self, pipeline: VideoEnsemblePipeline):
        # 4 frames of shape (4, 3, 224, 224)
        frames = np.random.uniform(-1.0, 1.0, (4, 3, 224, 224)).astype(np.float32)
        verdict = pipeline.predict(frames)

        assert isinstance(verdict, VideoEnsembleVerdict)
        assert verdict.label in {"real", "fake"}
        assert 0.0 <= verdict.confidence <= 1.0
        assert len(verdict.frame_scores) == 4
        assert 0.0 <= verdict.peak_score <= 1.0
        assert 0.0 <= verdict.mean_score <= 1.0
        assert 0.0 <= verdict.temporal_drift_score <= 1.0
        assert 0.0 <= verdict.noise_inconsistency_score <= 1.0

        d = verdict.to_dict()
        assert "face_tracking" in d["details"]
        assert "noise_fingerprint" in d["details"]
        assert "temporal_drift" in d["details"]
        assert "ensemble_weights" in d["details"]

    def test_service_integration_with_temporal_ensemble(self):
        """Ensures VideoDeepfakeService seamlessly integrates the ensemble verdict and details."""
        frames = np.random.uniform(-1.0, 1.0, (3, 3, 224, 224)).astype(np.float32)
        service = VideoDeepfakeService()
        verdict = service.predict(frames)

        assert isinstance(verdict, VideoVerdict)
        assert verdict.label in {"real", "fake"}
        assert 0.0 <= verdict.confidence <= 1.0
        assert len(verdict.frame_scores) == 3
        assert verdict.model_version == "xception-efficientnet-ensemble-v1"
        assert isinstance(verdict.details, dict)
        assert "noise_fingerprint" in verdict.details
        assert "temporal_drift" in verdict.details
