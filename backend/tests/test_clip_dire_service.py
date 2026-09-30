"""
TruthLens — Automated Tests for Dual-Branch Image Detection Service & Inference Router
Author: Pratham Yadav
Sprint 2: User Story 2 - Part 4 Validation

Validates dual-branch score fusion, subtype attribution, composite heatmap blending,
classification service compatibility, and end-to-end inference routing.
"""
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from backend.inference.clip_dire_service import (
    CLIPDIREImageService,
    DualBranchVerdict,
)
from backend.inference.classification_service import (
    ClassificationService,
    ClassificationVerdict,
)
from backend.inference.router import (
    route_inference,
    detect_media_type,
    InferenceResponse,
)


class TestCLIPDIREServiceAndRouter:
    """Test suite for dual-branch image service and inference routing integration."""

    @pytest.fixture
    def dual_service(self) -> CLIPDIREImageService:
        return CLIPDIREImageService(clip_weight=0.5, dire_weight=0.5)

    @pytest.fixture
    def sample_image_array(self) -> np.ndarray:
        rng = np.random.RandomState(42)
        return rng.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    def test_heatmap_fusion(self, dual_service: CLIPDIREImageService):
        # 14x14 attention map and 224x224 DIRE map
        attn_map = np.random.uniform(0.0, 1.0, (14, 14)).astype(np.float32)
        dire_map = np.random.uniform(0.0, 1.0, (224, 224)).astype(np.float32)

        fused = dual_service._fuse_heatmaps(attn_map, dire_map, target_size=(224, 224))

        assert fused.shape == (224, 224)
        assert fused.min() >= 0.0
        assert fused.max() <= 1.0
        assert np.isfinite(fused).all()

    def test_dual_branch_predict(self, dual_service: CLIPDIREImageService, sample_image_array: np.ndarray):
        verdict = dual_service.predict(sample_image_array, num_dire_steps=5)

        assert isinstance(verdict, DualBranchVerdict)
        assert verdict.label in {"real", "fake"}
        assert 0.0 <= verdict.confidence <= 1.0
        assert verdict.synthetic_subtype in {"authentic", "diffusion", "gan", "hybrid"}
        assert verdict.fused_heatmap.shape == (224, 224)
        assert "clip_probe" in verdict.details
        assert "dire" in verdict.details
        assert "fusion_weights" in verdict.details

        # Check serialization
        res_dict = verdict.to_dict()
        assert "label" in res_dict
        assert "synthetic_subtype" in res_dict
        assert "dire_mse" in res_dict

    def test_classification_service_wrapper(self, sample_image_array: np.ndarray):
        service = ClassificationService()
        result = service.predict(sample_image_array, num_dire_steps=5)

        assert isinstance(result, ClassificationVerdict)
        assert result.label in {"real", "fake"}
        assert 0.0 <= result.confidence <= 1.0
        assert "synthetic_subtype" in result.details

    def test_detect_media_type(self):
        assert detect_media_type("sample.jpg") == "image"
        assert detect_media_type("photo.PNG") == "image"
        assert detect_media_type("clip.webp") == "image"
        assert detect_media_type("video.mp4") == "video"
        assert detect_media_type("record.MOV") == "video"

        with pytest.raises(ValueError, match="Unsupported file type"):
            detect_media_type("document.pdf")

    def test_end_to_end_route_inference_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test_probe.png"
            img = Image.new("RGB", (224, 224), color=(100, 150, 200))
            img.save(img_path)

            response = route_inference(img_path, "test_probe.png")

            assert isinstance(response, InferenceResponse)
            assert response.media_type == "image"
            assert response.label in {"real", "fake"}
            assert 0.0 <= response.confidence <= 1.0
            assert "synthetic_subtype" in response.details
            assert "clip_probe" in response.details
            assert "dire" in response.details
