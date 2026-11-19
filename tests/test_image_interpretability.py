"""
TruthLens - Unit Tests for Image Interpretability Integration
Author: TruthLens Development Team
User Story 3: Interpretability Support (Part 2 - Image Interpretability Integration)

These tests verify:
1. Pipeline initialization and defaults.
2. Truthful reporting when no models are loaded (no fake heatmaps or predictions).
3. Image input support across formats (PIL, bytes, numpy, ProcessedImage, disk file).
4. Grad-CAM integration with mock CNN model engine (generating valid heatmaps, target class).
5. Attention Rollout integration with synthetic attention matrices and mock Transformer engine.
6. Simultaneous Grad-CAM and Attention Rollout heatmap generation.
7. Error isolation (model/matrix failures do not crash the pipeline).
8. Spatial alignment and letterbox unpadding back to original image resolution.
9. Safe handling of invalid inputs (None, missing file, unsupported type, NaN/corrupt tensor).
10. Serialization of ImageExplanation via to_dict().
11. Package exports in backend.interpretability.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from backend.data.image_preprocessor import (
    ImagePreprocessor,
    NormalizationMode,
    ProcessedImage,
    ResizeMode,
)
from backend.interpretability.attention_rollout import AttentionRolloutEngine
from backend.interpretability.gradcam import GradCAMEngine
from backend.interpretability.image_integration import (
    ImageExplanation,
    ImageInterpretabilityPipeline,
    explain_image,
    unpad_and_resize_heatmap,
)


@pytest.fixture
def sample_pil_image() -> Image.Image:
    """Generate a 300x200 RGB PIL test image."""
    arr = np.zeros((200, 300, 3), dtype=np.uint8)
    arr[:100, :150] = [255, 0, 0]
    arr[:100, 150:] = [0, 255, 0]
    arr[100:, :150] = [0, 0, 255]
    arr[100:, 150:] = [255, 255, 0]
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def sample_processed_image() -> ProcessedImage:
    """Generate a preprocessed image container."""
    tensor = np.full((3, 224, 224), fill_value=0.5, dtype=np.float32)
    return ProcessedImage(
        tensor=tensor,
        original_size=(640, 480),
        target_size=(224, 224),
        exif_present=True,
        exif_data={"Make": "Canon", "has_editing_signature": False},
        noise_residual=np.zeros((224, 224), dtype=np.float32),
        scale_factor=0.35,
        pad_offsets=(28, 28, 0, 0),
    )


@pytest.fixture
def synthetic_attention_layers() -> List[np.ndarray]:
    """
    Generate 3 layers of synthetic multi-head attention matrices.
    17 tokens = 1 [CLS] token + 16 patch tokens (4x4 spatial patch grid).
    """
    rng = np.random.RandomState(42)
    layers = []
    for _ in range(3):
        # 4 heads, 17x17 tokens
        raw = rng.uniform(0.1, 1.0, (4, 17, 17)).astype(np.float32)
        # Softmax normalize rows across last dim
        exp = np.exp(raw - np.max(raw, axis=-1, keepdims=True))
        softmax_attn = exp / np.sum(exp, axis=-1, keepdims=True)
        layers.append(softmax_attn)
    return layers


class TestImageInterpretabilityWithoutModels:
    """Tests representing the current real repository state where no model is loaded."""

    def test_pipeline_initialization_defaults(self):
        pipeline = ImageInterpretabilityPipeline()
        assert pipeline.gradcam_engine is None
        assert pipeline.attention_rollout_engine is None
        assert isinstance(pipeline.image_preprocessor, ImagePreprocessor)

    def test_process_image_without_models_is_truthful(self, sample_pil_image):
        pipeline = ImageInterpretabilityPipeline()
        result = pipeline.process_image(sample_pil_image)

        assert isinstance(result, ImageExplanation)
        assert result.gradcam_available is False
        assert result.gradcam_heatmap is None
        assert "unavailable" in result.gradcam_status.lower()

        assert result.attention_available is False
        assert result.attention_heatmap is None
        assert result.attention_rollout is None
        assert "unavailable" in result.attention_status.lower()

        assert result.error is None
        assert "no compatible model" in result.status_message.lower()
        assert "transformer model" in result.status_message.lower()

        # Metadata from ImagePreprocessor preserved
        assert result.metadata["original_size"] == (300, 200)
        assert result.metadata["target_size"] == (224, 224)
        assert "pad_offsets" in result.metadata
        assert "scale_factor" in result.metadata

    def test_process_image_from_various_input_types(
        self, sample_pil_image, sample_processed_image
    ):
        pipeline = ImageInterpretabilityPipeline()

        # 1. From PIL Image
        res_pil = pipeline.process_image(sample_pil_image)
        assert res_pil.error is None
        assert res_pil.metadata["original_size"] == (300, 200)

        # 2. From raw bytes
        import io
        buf = io.BytesIO()
        sample_pil_image.save(buf, format="PNG")
        res_bytes = pipeline.process_image(buf.getvalue())
        assert res_bytes.error is None
        assert res_bytes.metadata["original_size"] == (300, 200)

        # 3. From NumPy array (H, W, C)
        np_arr = np.array(sample_pil_image)
        res_np = pipeline.process_image(np_arr)
        assert res_np.error is None
        assert res_np.metadata["original_size"] == (300, 200)

        # 4. From ProcessedImage container directly
        res_proc = pipeline.process_image(sample_processed_image)
        assert res_proc.error is None
        assert res_proc.metadata["original_size"] == (640, 480)
        assert res_proc.metadata["pad_offsets"] == (28, 28, 0, 0)

        # 5. From file on disk
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            sample_pil_image.save(tmp.name, format="PNG")
            tmp_path = tmp.name

        try:
            res_file = pipeline.process_image(tmp_path)
            assert res_file.error is None
            assert res_file.metadata["original_size"] == (300, 200)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_to_dict_serialization(self, sample_processed_image):
        pipeline = ImageInterpretabilityPipeline()
        result = pipeline.process_image(sample_processed_image)
        data = result.to_dict()

        assert isinstance(data, dict)
        assert data["gradcam_available"] is False
        assert "unavailable" in data["gradcam_status"].lower()
        assert data["gradcam_shape"] is None
        assert data["attention_available"] is False
        assert "unavailable" in data["attention_status"].lower()
        assert data["attention_heatmap_shape"] is None
        assert data["attention_rollout_shape"] is None
        assert data["prediction"] is None
        assert data["error"] is None
        assert data["metadata"]["original_size"] == (640, 480)
        assert "status_message" in data


class TestMockModelInterpretability:
    """Unit tests using mock engines and synthetic attention to verify explanation generation."""

    def test_mock_gradcam_engine_generation(self, sample_processed_image):
        mock_engine = MagicMock(spec=GradCAMEngine)
        mock_engine.is_available.return_value = True

        def mock_generate(input_tensor, target_class=None, target_size=None):
            return np.ones((224, 224), dtype=np.float32) * 0.8

        mock_engine.generate_heatmap.side_effect = mock_generate

        pipeline = ImageInterpretabilityPipeline(gradcam_engine=mock_engine)
        result = pipeline.process_image(sample_processed_image, target_class=1)

        assert result.gradcam_available is True
        assert result.gradcam_status == "generated"
        assert result.gradcam_heatmap is not None
        assert result.gradcam_heatmap.shape == (224, 224)
        assert np.isclose(np.mean(result.gradcam_heatmap), 0.8)
        assert result.prediction == {"target_class": 1}
        assert "Grad-CAM heatmap generated" in result.status_message

    def test_mock_gradcam_error_handling(self, sample_processed_image):
        mock_engine = MagicMock(spec=GradCAMEngine)
        mock_engine.is_available.return_value = True
        mock_engine.generate_heatmap.side_effect = RuntimeError("Backward pass failed")

        pipeline = ImageInterpretabilityPipeline(gradcam_engine=mock_engine)
        result = pipeline.process_image(sample_processed_image)

        assert result.gradcam_available is False
        assert result.gradcam_heatmap is None
        assert result.gradcam_status.startswith("error:")
        assert "Backward pass failed" in result.gradcam_status
        assert result.error is None  # Image itself was fine, only model errored

    def test_synthetic_attention_matrices_generation(
        self, sample_processed_image, synthetic_attention_layers
    ):
        pipeline = ImageInterpretabilityPipeline()
        result = pipeline.process_image(
            sample_processed_image,
            attention_matrices=synthetic_attention_layers,
        )

        assert result.attention_available is True
        assert result.attention_status == "generated"
        assert result.attention_rollout is not None
        assert result.attention_rollout.shape == (17, 17)
        # Verify row sums equal 1.0 (stochastic matrix)
        np.testing.assert_allclose(
            np.sum(result.attention_rollout, axis=-1), 1.0, rtol=1e-4
        )

        assert result.attention_heatmap is not None
        assert result.attention_heatmap.shape == (224, 224)
        assert np.all(np.isfinite(result.attention_heatmap))
        assert np.all(result.attention_heatmap >= 0.0)
        assert np.all(result.attention_heatmap <= 1.0)
        assert "Attention Rollout heatmap generated" in result.status_message

    def test_mock_attention_rollout_engine(self, sample_processed_image):
        mock_attn_engine = MagicMock(spec=AttentionRolloutEngine)
        mock_attn_engine.is_available.return_value = True
        mock_attn_engine.generate_rollout.return_value = np.eye(17, dtype=np.float32)
        mock_attn_engine.generate_heatmap.return_value = np.full(
            (224, 224), fill_value=0.5, dtype=np.float32
        )

        pipeline = ImageInterpretabilityPipeline(
            attention_rollout_engine=mock_attn_engine
        )
        result = pipeline.process_image(sample_processed_image)

        assert result.attention_available is True
        assert result.attention_status == "generated"
        assert result.attention_rollout.shape == (17, 17)
        assert result.attention_heatmap.shape == (224, 224)
        assert np.isclose(np.mean(result.attention_heatmap), 0.5)

    def test_both_gradcam_and_attention_generated(
        self, sample_processed_image, synthetic_attention_layers
    ):
        mock_engine = MagicMock(spec=GradCAMEngine)
        mock_engine.is_available.return_value = True
        mock_engine.generate_heatmap.return_value = np.ones((224, 224), dtype=np.float32)

        pipeline = ImageInterpretabilityPipeline(gradcam_engine=mock_engine)
        result = pipeline.process_image(
            sample_processed_image,
            attention_matrices=synthetic_attention_layers,
            target_class=0,
        )

        assert result.gradcam_available is True
        assert result.gradcam_status == "generated"
        assert result.attention_available is True
        assert result.attention_status == "generated"
        assert "Grad-CAM heatmap generated" in result.status_message
        assert "Attention Rollout heatmap generated" in result.status_message

    def test_attention_matrix_error_handling(self, sample_processed_image):
        # Non-square matrix must fail rollout cleanly without crashing
        bad_matrix = [np.ones((4, 7), dtype=np.float32)]

        pipeline = ImageInterpretabilityPipeline()
        result = pipeline.process_image(
            sample_processed_image,
            attention_matrices=bad_matrix,
        )

        assert result.attention_available is False
        assert result.attention_heatmap is None
        assert result.attention_rollout is None
        assert result.attention_status.startswith("error:")
        assert "square" in result.attention_status.lower()


class TestSpatialAlignmentAndHelpers:
    """Tests for spatial heatmap letterbox unpadding and resizing."""

    def test_unpad_and_resize_heatmap(self):
        heatmap = np.zeros((224, 224), dtype=np.float32)
        heatmap[28:196, :] = 1.0  # Center active region

        pad_offsets = (28, 28, 0, 0)
        original_size = (640, 480)

        aligned = unpad_and_resize_heatmap(
            heatmap=heatmap,
            pad_offsets=pad_offsets,
            original_size=original_size,
        )

        assert aligned.shape == (480, 640)
        assert aligned.dtype == np.float32
        assert np.isclose(np.mean(aligned), 1.0, atol=1e-3)

    def test_unpad_and_resize_heatmap_no_padding(self):
        heatmap = np.ones((224, 224), dtype=np.float32)
        aligned = unpad_and_resize_heatmap(
            heatmap=heatmap,
            pad_offsets=(0, 0, 0, 0),
            original_size=(300, 200),
        )
        assert aligned.shape == (200, 300)

    def test_unpad_and_resize_heatmap_invalid_input(self):
        assert unpad_and_resize_heatmap(None, (0, 0, 0, 0), (100, 100)) is None
        bad_3d = np.ones((3, 10, 10))
        assert unpad_and_resize_heatmap(bad_3d, (0, 0, 0, 0), (100, 100)) is bad_3d
        zero_size = np.ones((10, 10))
        assert unpad_and_resize_heatmap(zero_size, (0, 0, 0, 0), (0, 0)) is zero_size

    def test_align_to_original_resolution_gradcam(self, sample_processed_image):
        mock_engine = MagicMock(spec=GradCAMEngine)
        mock_engine.is_available.return_value = True
        mock_engine.generate_heatmap.return_value = np.ones((224, 224), dtype=np.float32)

        pipeline = ImageInterpretabilityPipeline(gradcam_engine=mock_engine)
        result = pipeline.process_image(
            sample_processed_image,
            align_to_original_resolution=True,
        )

        assert result.gradcam_available is True
        # Original size is (640, 480) -> shape is (480, 640)
        assert result.gradcam_heatmap.shape == (480, 640)

    def test_align_to_original_resolution_attention(
        self, sample_processed_image, synthetic_attention_layers
    ):
        pipeline = ImageInterpretabilityPipeline()
        result = pipeline.process_image(
            sample_processed_image,
            attention_matrices=synthetic_attention_layers,
            align_to_original_resolution=True,
        )

        assert result.attention_available is True
        assert result.attention_heatmap.shape == (480, 640)


class TestEdgeCasesAndInputValidation:
    """Tests verifying safety against corrupt inputs, bad shapes, and wrong types."""

    def test_none_image_input_raises_value_error(self):
        pipeline = ImageInterpretabilityPipeline()
        with pytest.raises(ValueError, match="cannot be None"):
            pipeline.process_image(None)

    def test_nonexistent_file_path_raises_filenotfound(self):
        pipeline = ImageInterpretabilityPipeline()
        with pytest.raises(FileNotFoundError):
            pipeline.process_image("missing_image_file_xyz_987.png")

    def test_unsupported_input_type_raises_typeerror(self):
        pipeline = ImageInterpretabilityPipeline()
        with pytest.raises(TypeError, match="Unsupported image_input type"):
            pipeline.process_image(12345)

    def test_invalid_tensor_nan_isolated(self):
        nan_tensor = np.full((3, 224, 224), fill_value=np.nan, dtype=np.float32)
        bad_proc = ProcessedImage(
            tensor=nan_tensor,
            original_size=(224, 224),
            target_size=(224, 224),
        )
        pipeline = ImageInterpretabilityPipeline()
        result = pipeline.process_image(bad_proc)

        assert result.error is not None
        assert "non-finite" in result.error.lower()
        assert result.gradcam_status == "skipped"
        assert result.attention_status == "skipped"
        assert result.gradcam_heatmap is None
        assert result.attention_heatmap is None

    def test_invalid_tensor_wrong_channels(self):
        wrong_chan = np.zeros((1, 224, 224), dtype=np.float32)
        bad_proc = ProcessedImage(
            tensor=wrong_chan,
            original_size=(224, 224),
            target_size=(224, 224),
        )
        pipeline = ImageInterpretabilityPipeline()
        result = pipeline.process_image(bad_proc)

        assert result.error is not None
        assert "channels" in result.error.lower()
        assert result.gradcam_status == "skipped"

    def test_invalid_tensor_wrong_dims(self):
        wrong_dims = np.zeros((224, 224), dtype=np.float32)
        bad_proc = ProcessedImage(
            tensor=wrong_dims,
            original_size=(224, 224),
            target_size=(224, 224),
        )
        pipeline = ImageInterpretabilityPipeline()
        result = pipeline.process_image(bad_proc)

        assert result.error is not None
        assert "expected 3d" in result.error.lower()
        assert result.gradcam_status == "skipped"


class TestConvenienceFunctionAndExports:
    """Tests for explain_image top-level function and package exports."""

    def test_explain_image_convenience_function(self, sample_processed_image):
        result = explain_image(sample_processed_image)
        assert isinstance(result, ImageExplanation)
        assert result.gradcam_available is False
        assert result.attention_available is False
        assert result.error is None

    def test_package_exports(self):
        import backend.interpretability as interp

        # Original exports
        assert hasattr(interp, "compute_attention_rollout")
        assert hasattr(interp, "rollout_to_heatmap")
        assert hasattr(interp, "compute_attention_rollout_heatmap")
        assert hasattr(interp, "AttentionRolloutEngine")
        assert hasattr(interp, "compute_gradcam_heatmap")
        assert hasattr(interp, "GradCAMEngine")

        # New Image Interpretability Integration exports
        assert hasattr(interp, "ImageExplanation")
        assert hasattr(interp, "ImageInterpretabilityPipeline")
        assert hasattr(interp, "explain_image")
        assert hasattr(interp, "unpad_and_resize_heatmap")
