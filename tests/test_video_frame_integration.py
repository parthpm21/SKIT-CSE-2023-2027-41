"""
TruthLens - Unit Tests for Video/Frame Interpretability Integration
Author: TruthLens Development Team
Sprint 2: Interpretability (Part 4 - Video/Frame Integration & Testing)

These tests verify:
1. Multi-frame processing order and temporal association.
2. Preservation of frame indices and timestamps from VideoPreprocessor.
3. Accurate association of explanations with each frame.
4. Safe handling of invalid or empty frame inputs.
5. Truthful handling of missing models (no fake predictions/heatmaps).
6. Safe handling of missing or invalid transformer attention matrices.
7. Error isolation across frames (errors in one frame do not contaminate others).
8. Spatial unpadding and alignment to original frame resolution.
9. Reusability of existing pipeline components without code duplication.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest

from backend.data.image_preprocessor import ProcessedImage
from backend.data.video_preprocessor import (
    ProcessedVideo,
    VideoFrameMetadata,
    VideoPreprocessor,
)
from backend.interpretability.gradcam import GradCAMEngine
from backend.interpretability.video_integration import (
    FrameExplanation,
    VideoExplanationResult,
    VideoInterpretabilityPipeline,
    explain_video_frames,
    unpad_and_resize_heatmap,
)


@pytest.fixture
def sample_valid_frames() -> List[np.ndarray]:
    """Generate 3 valid (3, 224, 224) float32 frame tensors."""
    frames = []
    for i in range(3):
        # Distinct pattern for each frame
        frame = np.full((3, 224, 224), fill_value=0.1 * (i + 1), dtype=np.float32)
        frames.append(frame)
    return frames


@pytest.fixture
def sample_frame_metadata() -> List[VideoFrameMetadata]:
    """Generate corresponding metadata for 3 frames."""
    return [
        VideoFrameMetadata(
            frame_index=0,
            timestamp_seconds=0.0,
            original_size=(640, 480),
            scale_factor=0.35,
            pad_offsets=(28, 28, 0, 0),
        ),
        VideoFrameMetadata(
            frame_index=15,
            timestamp_seconds=0.625,
            original_size=(640, 480),
            scale_factor=0.35,
            pad_offsets=(28, 28, 0, 0),
        ),
        VideoFrameMetadata(
            frame_index=30,
            timestamp_seconds=1.25,
            original_size=(640, 480),
            scale_factor=0.35,
            pad_offsets=(28, 28, 0, 0),
        ),
    ]


@pytest.fixture
def sample_processed_video(
    sample_valid_frames: List[np.ndarray],
    sample_frame_metadata: List[VideoFrameMetadata],
) -> ProcessedVideo:
    """Create a ProcessedVideo container matching the 3 sample frames."""
    stacked_tensor = np.stack(sample_valid_frames, axis=0)  # (3, 3, 224, 224)
    return ProcessedVideo(
        tensor=stacked_tensor,
        frame_count=3,
        original_fps=24.0,
        duration_seconds=1.25,
        original_resolution=(640, 480),
        target_resolution=(224, 224),
        frame_metadata=sample_frame_metadata,
        has_audio_track=False,
    )


class TestVideoFrameIntegrationWithoutModels:
    """Tests representing the current real repository state where no model is loaded."""

    def test_pipeline_initialization_defaults(self):
        pipeline = VideoInterpretabilityPipeline()
        assert pipeline.gradcam_engine is None
        assert pipeline.attention_rollout_engine is None
        assert isinstance(pipeline.video_preprocessor, VideoPreprocessor)

    def test_process_frames_without_models_is_truthful(self, sample_valid_frames):
        pipeline = VideoInterpretabilityPipeline()
        result = pipeline.process_frames(sample_valid_frames)

        assert isinstance(result, VideoExplanationResult)
        assert result.frame_count == 3
        assert len(result.frames) == 3
        assert result.gradcam_available is False
        assert result.attention_available is False

        # Status must explicitly state that no model is loaded
        assert "no compatible model" in result.status_message.lower()
        assert "no transformer model" in result.status_message.lower()

        # Check each frame explanation
        for idx, frame_exp in enumerate(result.frames):
            assert frame_exp.frame_index == idx
            assert frame_exp.gradcam_heatmap is None
            assert "unavailable" in frame_exp.gradcam_status.lower()
            assert frame_exp.attention_rollout is None
            assert "unavailable" in frame_exp.attention_status.lower()
            assert frame_exp.error is None

        # Verify summary counts
        assert result.summary["total_frames"] == 3
        assert result.summary["valid_frames"] == 3
        assert result.summary["gradcam_generated_count"] == 0
        assert result.summary["attention_generated_count"] == 0
        assert result.summary["errors_count"] == 0

    def test_frame_indices_and_timestamps_preserved(self, sample_processed_video):
        pipeline = VideoInterpretabilityPipeline()
        result = pipeline.process_video(sample_processed_video)

        assert result.frame_count == 3
        assert len(result.frames) == 3

        # Verify exact preservation of indices and timestamps from VideoFrameMetadata
        expected_indices = [0, 15, 30]
        expected_timestamps = [0.0, 0.625, 1.25]

        for i in range(3):
            assert result.frames[i].frame_index == expected_indices[i]
            assert result.frames[i].timestamp_seconds == expected_timestamps[i]
            assert result.frames[i].metadata["original_size"] == (640, 480)
            assert result.frames[i].metadata["pad_offsets"] == (28, 28, 0, 0)

    def test_process_video_from_synthetic_video_bytes(self, sample_synthetic_video_bytes: bytes):
        """Verify end-to-end integration with VideoPreprocessor decoding real MP4 bytes."""
        pipeline = VideoInterpretabilityPipeline()
        result = pipeline.process_video(sample_synthetic_video_bytes)

        assert result.frame_count > 0
        assert len(result.frames) == result.frame_count
        assert result.gradcam_available is False
        assert result.attention_available is False
        # Timestamps must be monotonically non-decreasing
        timestamps = [f.timestamp_seconds for f in result.frames if f.timestamp_seconds is not None]
        assert timestamps == sorted(timestamps)


class TestMockModelInterpretability:
    """Unit tests using explicitly labeled synthetic/mock models to verify explanation flow."""

    def test_mock_gradcam_engine_association(self, sample_valid_frames, sample_frame_metadata):
        # Create a mock GradCAMEngine
        mock_engine = MagicMock(spec=GradCAMEngine)
        mock_engine.is_available.return_value = True

        # Generate distinct heatmaps per call
        def mock_generate(input_tensor, target_class=None, target_size=None):
            val = float(np.mean(input_tensor))
            return np.full((224, 224), fill_value=val, dtype=np.float32)

        mock_engine.generate_heatmap.side_effect = mock_generate

        pipeline = VideoInterpretabilityPipeline(gradcam_engine=mock_engine)
        result = pipeline.process_frames(
            frames=sample_valid_frames,
            frame_metadata=sample_frame_metadata,
        )

        assert result.gradcam_available is True
        assert result.summary["gradcam_generated_count"] == 3

        for idx, frame_exp in enumerate(result.frames):
            assert frame_exp.gradcam_status == "generated"
            assert frame_exp.gradcam_heatmap is not None
            assert frame_exp.gradcam_heatmap.shape == (224, 224)
            # Ensure correct association with frame input value
            expected_val = pytest.approx(0.1 * (idx + 1), abs=1e-3)
            assert np.mean(frame_exp.gradcam_heatmap) == expected_val

    def test_mock_gradcam_error_isolation_across_frames(self, sample_valid_frames):
        """Confirm that a failure on one frame does not break or misassociate other frames."""
        mock_engine = MagicMock(spec=GradCAMEngine)
        mock_engine.is_available.return_value = True

        def mock_generate_with_failure(input_tensor, target_class=None, target_size=None):
            val = float(np.mean(input_tensor))
            # Simulate failure on frame index 1 (val approx 0.2)
            if np.isclose(val, 0.2, atol=1e-2):
                raise RuntimeError("Simulated target layer backward pass failure on frame 1")
            return np.full((224, 224), fill_value=val, dtype=np.float32)

        mock_engine.generate_heatmap.side_effect = mock_generate_with_failure

        pipeline = VideoInterpretabilityPipeline(gradcam_engine=mock_engine)
        result = pipeline.process_frames(sample_valid_frames)

        # Frame 0: success
        assert result.frames[0].gradcam_status == "generated"
        assert result.frames[0].gradcam_heatmap is not None

        # Frame 1: isolated error
        assert "error:" in result.frames[1].gradcam_status.lower()
        assert result.frames[1].gradcam_heatmap is None

        # Frame 2: success
        assert result.frames[2].gradcam_status == "generated"
        assert result.frames[2].gradcam_heatmap is not None

        # Summary tracks error count
        assert result.summary["errors_count"] == 1
        assert result.summary["gradcam_generated_count"] == 2

    def test_synthetic_attention_matrices_per_frame(self, sample_valid_frames):
        """Verify that per-frame attention matrices are correctly rolled out."""
        # 3 layers of (4, 4) synthetic attention matrices for frame 0 and frame 2
        rng = np.random.RandomState(42)
        mat0 = [rng.uniform(0.1, 1.0, (4, 4)).astype(np.float32) for _ in range(3)]
        mat2 = [rng.uniform(0.1, 1.0, (4, 4)).astype(np.float32) for _ in range(3)]

        # Frame 1 has no attention matrices
        frame_attn = [mat0, None, mat2]

        pipeline = VideoInterpretabilityPipeline()
        result = pipeline.process_frames(
            frames=sample_valid_frames,
            frame_attention_matrices=frame_attn,
        )

        assert result.attention_available is True
        assert result.summary["attention_generated_count"] == 2

        # Frame 0: generated
        assert result.frames[0].attention_status == "generated"
        assert result.frames[0].attention_rollout is not None
        assert result.frames[0].attention_rollout.shape == (4, 4)
        np.testing.assert_allclose(np.sum(result.frames[0].attention_rollout, axis=-1), 1.0, rtol=1e-4)

        # Frame 1: unavailable for this frame
        assert "unavailable" in result.frames[1].attention_status.lower()
        assert result.frames[1].attention_rollout is None

        # Frame 2: generated
        assert result.frames[2].attention_status == "generated"
        assert result.frames[2].attention_rollout is not None
        assert result.frames[2].attention_rollout.shape == (4, 4)

    def test_attention_matrix_error_isolation(self, sample_valid_frames):
        """Verify that an invalid attention matrix on frame 0 does not affect frame 1."""
        # Frame 0 has invalid non-square matrix
        bad_matrix = [np.ones((4, 6), dtype=np.float32)]
        # Frame 1 has valid square matrix
        good_matrix = [np.eye(4, dtype=np.float32)]

        frame_attn = [bad_matrix, good_matrix, None]

        pipeline = VideoInterpretabilityPipeline()
        result = pipeline.process_frames(
            frames=sample_valid_frames,
            frame_attention_matrices=frame_attn,
        )

        # Frame 0: error isolated
        assert "error:" in result.frames[0].attention_status.lower()
        assert result.frames[0].attention_rollout is None

        # Frame 1: generated successfully
        assert result.frames[1].attention_status == "generated"
        assert result.frames[1].attention_rollout is not None

        # Frame 2: unavailable
        assert "unavailable" in result.frames[2].attention_status.lower()


class TestEdgeCasesAndErrorHandling:
    """Tests verifying safety against corrupt inputs, empty collections, and bad types."""

    def test_invalid_frame_tensor_isolated_safely(self):
        valid_frame = np.ones((3, 224, 224), dtype=np.float32)
        wrong_channels = np.ones((1, 224, 224), dtype=np.float32)
        nan_frame = np.full((3, 224, 224), np.nan, dtype=np.float32)
        wrong_dims = np.ones((224, 224), dtype=np.float32)

        frames = [valid_frame, wrong_channels, nan_frame, wrong_dims]

        pipeline = VideoInterpretabilityPipeline()
        result = pipeline.process_frames(frames)

        assert result.frame_count == 4
        assert result.frames[0].error is None

        # Invalid frames must be flagged with descriptive errors and skipped
        assert result.frames[1].error is not None
        assert "channels" in result.frames[1].error.lower()
        assert result.frames[1].gradcam_status == "skipped"

        assert result.frames[2].error is not None
        assert "non-finite" in result.frames[2].error.lower()
        assert result.frames[2].gradcam_status == "skipped"

        assert result.frames[3].error is not None
        assert "expected 3d" in result.frames[3].error.lower()
        assert result.frames[3].gradcam_status == "skipped"

        assert result.summary["valid_frames"] == 1
        assert result.summary["errors_count"] == 3

    def test_empty_frames_list(self):
        pipeline = VideoInterpretabilityPipeline()
        result = pipeline.process_frames([])

        assert result.frame_count == 0
        assert len(result.frames) == 0
        assert result.gradcam_available is False
        assert result.attention_available is False
        assert "empty" in result.status_message.lower()

    def test_none_input_raises_value_error(self):
        pipeline = VideoInterpretabilityPipeline()
        with pytest.raises(ValueError, match="cannot be None"):
            pipeline.process_frames(None)

        with pytest.raises(ValueError, match="cannot be None"):
            pipeline.process_video(None)

    def test_nonexistent_video_path_raises_filenotfound(self):
        pipeline = VideoInterpretabilityPipeline()
        with pytest.raises(FileNotFoundError):
            pipeline.process_video("nonexistent_video_file_12345.mp4")

    def test_unsupported_video_type_raises_typeerror(self):
        pipeline = VideoInterpretabilityPipeline()
        with pytest.raises(TypeError, match="Unsupported video_input type"):
            pipeline.process_video(12345)


class TestSpatialAlignmentAndHelpers:
    """Tests for spatial heatmap unpadding and resizing."""

    def test_unpad_and_resize_heatmap(self):
        # Create a synthetic 224x224 heatmap with a hot circle in the center
        heatmap = np.zeros((224, 224), dtype=np.float32)
        heatmap[50:174, 50:174] = 1.0

        # Letterbox offsets: 28px top, 28px bottom
        pad_offsets = (28, 28, 0, 0)
        original_size = (640, 480)  # (w, h)

        aligned = unpad_and_resize_heatmap(
            heatmap=heatmap,
            pad_offsets=pad_offsets,
            original_size=original_size,
        )

        assert aligned.shape == (480, 640)  # (H, W)
        assert aligned.dtype == np.float32

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
        bad = np.ones((5, 5, 5))
        assert unpad_and_resize_heatmap(bad, (0, 0, 0, 0), (100, 100)) is bad

    def test_processed_image_input_conversion(self):
        # Create a dummy ProcessedImage
        img_tensor = np.zeros((3, 224, 224), dtype=np.float32)
        proc_img = ProcessedImage(
            tensor=img_tensor,
            original_size=(400, 300),
            target_size=(224, 224),
            scale_factor=0.5,
            pad_offsets=(10, 10, 0, 0),
        )

        result = explain_video_frames([proc_img])
        assert result.frame_count == 1
        assert result.frames[0].frame_index == 0
        assert result.frames[0].metadata["original_size"] == (400, 300)
        assert result.frames[0].metadata["pad_offsets"] == (10, 10, 0, 0)

    def test_to_dict_serialization(self, sample_processed_video):
        pipeline = VideoInterpretabilityPipeline()
        result = pipeline.process_video(sample_processed_video)

        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["frame_count"] == 3
        assert result_dict["gradcam_available"] is False
        assert result_dict["attention_available"] is False
        assert len(result_dict["frames"]) == 3
        assert result_dict["frames"][0]["frame_index"] == 0
        assert "metadata" in result_dict["frames"][0]
