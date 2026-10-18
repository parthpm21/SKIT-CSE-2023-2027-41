"""
TruthLens — Tests for the dual decoders, FusionEngine and /fusion/analyze
Author: Parth Maheshwari
Sprint: Architecture design for the Fusion & Localization Engine.
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.fusion.anomaly_decoder import AnomalyDecoder
from backend.fusion.confidence_decoder import ConfidenceDecoder
from backend.fusion.contracts import (
    DiffusionBranchInput, FusionInput, ImageBranchInput, VideoBranchInput,
)
from backend.fusion.engine import FusionEngine
from backend.ingestion.gateway import app

EMB = np.zeros(512, dtype=np.float32)


def image_branch(score=0.9):
    return ImageBranchInput(cls_embedding=[0.1] * 768, probe_score=score)


def diffusion_branch(score=0.8):
    return DiffusionBranchInput(
        dire_residual_map=[[0.0, 0.5, 1.0], [0.2, 0.4, 0.6]], diffusion_score=score, mse=0.01
    )


def video_branch(scores=(0.9, 0.7, 0.8)):
    return VideoBranchInput(frame_embeddings=[[0.1] * 8 for _ in scores], frame_scores=list(scores))


@pytest.mark.unit
class TestAnomalyDecoder:
    def test_diffusion_gives_pixel_map_with_dire_shape(self):
        out = AnomalyDecoder().decode(FusionInput(sample_id="a", diffusion=diffusion_branch()), EMB)
        assert out.granularity == "pixel"
        assert out.shape == [2, 3]
        assert len(out.values) == 6
        assert min(out.values) == 0.0 and max(out.values) == 1.0

    def test_image_only_uses_fallback_grid(self):
        out = AnomalyDecoder().decode(FusionInput(sample_id="a", image=image_branch(0.7)), EMB)
        assert out.shape == [14, 14] and len(out.values) == 196

    def test_video_gives_frame_sequence(self):
        out = AnomalyDecoder().decode(FusionInput(sample_id="a", video=video_branch()), EMB)
        assert out.granularity == "frame"
        assert out.shape == [3]

    def test_flat_residual_does_not_divide_by_zero(self):
        flat = DiffusionBranchInput(dire_residual_map=[[0.3, 0.3]], diffusion_score=0.5, mse=0.0)
        out = AnomalyDecoder().decode(FusionInput(sample_id="a", diffusion=flat), EMB)
        assert all(v == pytest.approx(0.3) for v in out.values)

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            AnomalyDecoder().decode(FusionInput(sample_id="a"), EMB)


@pytest.mark.unit
class TestConfidenceDecoder:
    def test_high_scores_label_fake(self):
        r = ConfidenceDecoder().decode(FusionInput(sample_id="a", image=image_branch(0.9)), EMB)
        assert r.label == "fake" and r.confidence == pytest.approx(0.9)

    def test_low_scores_label_real(self):
        r = ConfidenceDecoder().decode(FusionInput(sample_id="a", image=image_branch(0.1)), EMB)
        assert r.label == "real" and r.confidence == pytest.approx(0.9)

    def test_multiple_branches_are_averaged(self):
        fi = FusionInput(sample_id="a", image=image_branch(0.9), diffusion=diffusion_branch(0.3))
        r = ConfidenceDecoder().decode(fi, EMB)
        assert r.label == "fake" and r.confidence == pytest.approx(0.6)

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            ConfidenceDecoder().decode(FusionInput(sample_id="a"), EMB)


@pytest.mark.unit
class TestFusionEngine:
    def test_engine_reports_branches_and_version(self):
        res = FusionEngine().run(FusionInput(sample_id="s1", image=image_branch(), diffusion=diffusion_branch()))
        assert res.sample_id == "s1"
        assert res.branches_used == ["image", "diffusion"]
        assert res.fusion_model_version == "fusion-scaffold-v0"


@pytest.mark.integration
class TestFusionEndpoint:
    client = TestClient(app)

    def test_image_payload_returns_fusion_result(self):
        body = FusionInput(sample_id="s2", image=image_branch(), diffusion=diffusion_branch()).model_dump()
        resp = self.client.post("/api/v1/fusion/analyze", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["confidence"]["label"] in ("real", "fake")
        assert data["anomaly_map"]["granularity"] == "pixel"

    def test_video_payload_returns_frame_map(self):
        body = FusionInput(sample_id="s3", video=video_branch()).model_dump()
        resp = self.client.post("/api/v1/fusion/analyze", json=body)
        assert resp.status_code == 200
        assert resp.json()["anomaly_map"]["granularity"] == "frame"

    def test_no_branches_is_rejected_with_422(self):
        resp = self.client.post("/api/v1/fusion/analyze", json={"sample_id": "empty"})
        assert resp.status_code == 422

    def test_malformed_payload_is_rejected_with_422(self):
        resp = self.client.post("/api/v1/fusion/analyze", json={"sample_id": "x", "image": {"probe_score": 0.5}})
        assert resp.status_code == 422