"""
TruthLens — Automated Tests for Fusion & Localization API Contracts & Transformer Scaffold
Author: TruthLens Core Team
Sprint: Architecture design for the Fusion & Localization Engine

Validates multi-branch Pydantic schemas, serialization/deserialization, and
CrossModalFusionTransformer projection and fusion behavior across active modalities.
"""
import numpy as np
import pytest

from backend.fusion.contracts import (
    FusionInput,
    ImageBranchInput,
    DiffusionBranchInput,
    VideoBranchInput,
    AnomalyMap,
    ConfidenceResult,
    FusionResult,
)
from backend.fusion.fusion_transformer import (
    CrossModalFusionTransformer,
    FusionTransformerConfig,
)
from backend.fusion.anomaly_decoder import (
    AnomalyDecoder,
    AnomalyDecoderConfig,
)
from backend.fusion.confidence_decoder import (
    ConfidenceDecoder,
    ConfidenceDecoderConfig,
)


class TestFusionContractsAndTransformer:
    """Test suite for Fusion contracts and transformer scaffold."""

    def test_image_branch_input_serialization(self):
        img_input = ImageBranchInput(
            cls_embedding=[0.1] * 768,
            probe_score=0.85,
        )
        assert len(img_input.cls_embedding) == 768
        assert img_input.probe_score == 0.85
        d = img_input.model_dump()
        assert "cls_embedding" in d
        assert "probe_score" in d

    def test_diffusion_branch_input_serialization(self):
        diff_input = DiffusionBranchInput(
            dire_residual_map=[[0.1, 0.2], [0.3, 0.4]],
            diffusion_score=0.92,
            mse=0.012,
        )
        assert diff_input.diffusion_score == 0.92
        assert diff_input.mse == 0.012
        assert len(diff_input.dire_residual_map) == 2

    def test_video_branch_input_serialization(self):
        vid_input = VideoBranchInput(
            frame_embeddings=[[0.5] * 256, [0.6] * 256],
            frame_scores=[0.75, 0.88],
        )
        assert len(vid_input.frame_embeddings) == 2
        assert len(vid_input.frame_scores) == 2

    def test_fusion_result_contract(self):
        anomaly = AnomalyMap(
            granularity="pixel",
            values=[0.1, 0.5, 0.9, 0.2],
            shape=[2, 2],
        )
        conf = ConfidenceResult(label="fake", confidence=0.89)
        res = FusionResult(
            sample_id="test_001",
            branches_used=["image", "diffusion"],
            anomaly_map=anomaly,
            confidence=conf,
            fusion_model_version="fusion-v1-scaffold",
        )
        assert res.sample_id == "test_001"
        assert res.confidence.label == "fake"
        assert res.anomaly_map.granularity == "pixel"

    def test_cross_modal_fusion_transformer_image_and_diffusion(self):
        cfg = FusionTransformerConfig(shared_embed_dim=256)
        transformer = CrossModalFusionTransformer(config=cfg)

        payload = FusionInput(
            sample_id="img_sample_1",
            image=ImageBranchInput(
                cls_embedding=list(np.random.randn(768).astype(float)),
                probe_score=0.72,
            ),
            diffusion=DiffusionBranchInput(
                dire_residual_map=[[0.05] * 14] * 14,
                diffusion_score=0.68,
                mse=0.015,
            ),
        )

        fused_vector = transformer.fuse(payload)
        assert isinstance(fused_vector, np.ndarray)
        assert fused_vector.shape == (256,)
        assert np.isfinite(fused_vector).all()

    def test_cross_modal_fusion_transformer_video_only(self):
        transformer = CrossModalFusionTransformer()
        payload = FusionInput(
            sample_id="video_sample_1",
            video=VideoBranchInput(
                frame_embeddings=[[0.1] * 128, [0.2] * 128],
                frame_scores=[0.4, 0.6],
            ),
        )
        fused = transformer.fuse(payload)
        assert fused.shape == (512,)

    def test_empty_fusion_input_raises_value_error(self):
        transformer = CrossModalFusionTransformer()
        empty_payload = FusionInput(sample_id="empty_sample")
        with pytest.raises(ValueError, match="at least one active branch"):
            transformer.fuse(empty_payload)

    def test_anomaly_decoder_diffusion_spatial_map(self):
        decoder = AnomalyDecoder()
        payload = FusionInput(
            sample_id="sample_diff",
            diffusion=DiffusionBranchInput(
                dire_residual_map=[[0.2, 0.4], [0.6, 0.8]],
                diffusion_score=0.85,
                mse=0.02,
            ),
        )
        dummy_embedding = np.random.randn(512).astype(np.float32)
        anomaly_map = decoder.decode(payload, dummy_embedding)

        assert anomaly_map.granularity == "pixel"
        assert anomaly_map.shape == [2, 2]
        assert len(anomaly_map.values) == 4
        assert min(anomaly_map.values) >= 0.0
        assert max(anomaly_map.values) <= 1.0

    def test_anomaly_decoder_image_fallback_grid(self):
        decoder = AnomalyDecoder(config=AnomalyDecoderConfig(fallback_grid=8))
        payload = FusionInput(
            sample_id="sample_img",
            image=ImageBranchInput(
                cls_embedding=[0.1] * 768,
                probe_score=0.75,
            ),
        )
        dummy_embedding = np.random.randn(512).astype(np.float32)
        anomaly_map = decoder.decode(payload, dummy_embedding)

        assert anomaly_map.granularity == "pixel"
        assert anomaly_map.shape == [8, 8]
        assert len(anomaly_map.values) == 64
        assert all(pytest.approx(v) == 0.75 for v in anomaly_map.values)

    def test_anomaly_decoder_video_frame_scores(self):
        decoder = AnomalyDecoder()
        payload = FusionInput(
            sample_id="sample_vid",
            video=VideoBranchInput(
                frame_embeddings=[[0.1] * 128, [0.2] * 128, [0.3] * 128],
                frame_scores=[0.2, 0.9, 0.5],
            ),
        )
        dummy_embedding = np.random.randn(512).astype(np.float32)
        anomaly_map = decoder.decode(payload, dummy_embedding)

        assert anomaly_map.granularity == "frame"
        assert anomaly_map.shape == [3]
        assert len(anomaly_map.values) == 3
        assert anomaly_map.values[1] == pytest.approx(0.9)

    def test_anomaly_decoder_empty_input_raises_value_error(self):
        decoder = AnomalyDecoder()
        empty_payload = FusionInput(sample_id="empty")
        with pytest.raises(ValueError, match="no usable branch data"):
            decoder.decode(empty_payload, np.zeros(512))

    def test_confidence_decoder_scoring(self):
        decoder = ConfidenceDecoder(config=ConfidenceDecoderConfig(fake_threshold=0.5))
        payload = FusionInput(
            sample_id="sample_multi",
            image=ImageBranchInput(cls_embedding=[0.0] * 768, probe_score=0.8),
            diffusion=DiffusionBranchInput(dire_residual_map=[[0.1]], diffusion_score=0.6, mse=0.01),
        )
        dummy_embedding = np.random.randn(512).astype(np.float32)
        result = decoder.decode(payload, dummy_embedding)

        # Average of 0.8 and 0.6 is 0.7 (fake)
        assert result.label == "fake"
        assert pytest.approx(result.confidence, abs=1e-4) == 0.7

    def test_confidence_decoder_real_verdict(self):
        decoder = ConfidenceDecoder(config=ConfidenceDecoderConfig(fake_threshold=0.5))
        payload = FusionInput(
            sample_id="sample_real",
            image=ImageBranchInput(cls_embedding=[0.0] * 768, probe_score=0.2),
            diffusion=DiffusionBranchInput(dire_residual_map=[[0.01]], diffusion_score=0.1, mse=0.001),
        )
        dummy_embedding = np.random.randn(512).astype(np.float32)
        result = decoder.decode(payload, dummy_embedding)

        # Average of 0.2 and 0.1 is 0.15 (real), confidence = 1.0 - 0.15 = 0.85
        assert result.label == "real"
        assert pytest.approx(result.confidence, abs=1e-4) == 0.85

    def test_confidence_decoder_empty_input_raises_value_error(self):
        decoder = ConfidenceDecoder()
        empty_payload = FusionInput(sample_id="empty")
        with pytest.raises(ValueError, match="no usable branch data"):
            decoder.decode(empty_payload, np.zeros(512))

