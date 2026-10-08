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
