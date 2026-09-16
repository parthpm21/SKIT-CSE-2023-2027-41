"""
TruthLens - Automated Unit Tests for Image & Video Preprocessing Pipeline
Author: Pratham Yadav (Deep Learning & Computer Vision Lead)
Sprint 1: Benchmark Assembly & Ingestion API (User Story 1 - Item 5)

This module provides automated unit tests verifying:
1. Image preprocessing output validity and tensor normalization.
2. Channel consistency across RGB, RGBA, and Grayscale sources.
3. Corrupt/truncated image and video media handling.
4. Video temporal batching, keyframe sampling, and timestamp alignment.
"""

import io
import os
import tempfile
import cv2
import numpy as np
import pytest
from PIL import Image

from backend.data.dataset_manifest import MediaType
from backend.data.image_preprocessor import (
    ImagePreprocessor,
    NormalizationMode,
    ProcessedImage,
    ResizeMode,
)
from backend.data.validator import PreprocessingValidator, ValidationResult
from backend.data.video_preprocessor import (
    ProcessedVideo,
    SamplingStrategy,
    VideoPreprocessor,
)


@pytest.fixture
def synthetic_test_image() -> Image.Image:
    """Generate a synthetic RGB PIL test image."""
    arr = np.random.randint(0, 256, (180, 240, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def synthetic_test_video_path() -> str:
    """Generate a temporary synthetic MP4 video file."""
    fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tmp_path, fourcc, 24.0, (160, 120))
    for i in range(24):
        frame = np.full((120, 160, 3), (i * 10) % 255, dtype=np.uint8)
        out.write(frame)
    out.release()
    yield tmp_path
    if os.path.exists(tmp_path):
        os.remove(tmp_path)


class TestPreprocessingPipeline:
    """Automated test suite for User Story 1 Item 5."""

    def test_image_preprocessing_output_validity(self, synthetic_test_image: Image.Image):
        """Verify image tensor dimensions (C, H, W), dtype, and batch expansion."""
        preprocessor = ImagePreprocessor(
            target_size=(224, 224),
            normalization=NormalizationMode.ZERO_TO_ONE,
            resize_mode=ResizeMode.LETTERBOX,
        )
        processed: ProcessedImage = preprocessor.process(synthetic_test_image)

        assert processed.tensor.shape == (3, 224, 224)
        assert processed.tensor.dtype == np.float32
        assert processed.to_batch().shape == (1, 3, 224, 224)
        assert processed.original_size == (240, 180)

        validator = PreprocessingValidator()
        tensor_res: ValidationResult = validator.validate_tensor(
            processed.tensor, norm_mode=NormalizationMode.ZERO_TO_ONE
        )
        assert tensor_res.is_valid is True

    def test_image_channel_consistency(self):
        """Verify 3-channel RGB output consistency across RGB, RGBA, and Grayscale inputs."""
        preprocessor = ImagePreprocessor(target_size=(128, 128))

        # 1. RGBA Image (with alpha channel)
        rgba_img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        res_rgba = preprocessor.process(rgba_img)
        assert res_rgba.tensor.shape[0] == 3

        # 2. Grayscale Image
        gray_img = Image.new("L", (100, 100), 128)
        res_gray = preprocessor.process(gray_img)
        assert res_gray.tensor.shape[0] == 3

        # 3. 2D NumPy Array
        np_2d = np.zeros((100, 100), dtype=np.uint8)
        res_np = preprocessor.process(np_2d)
        assert res_np.tensor.shape[0] == 3

    def test_image_corrupt_media_handling(self):
        """Verify graceful error rejection for corrupted image payloads."""
        preprocessor = ImagePreprocessor()
        validator = PreprocessingValidator()

        # Corrupt bytes
        corrupt_bytes = b"CORRUPT_INVALID_JPEG_PAYLOAD_BYTE_STREAM"
        with pytest.raises(Exception):
            preprocessor.process(corrupt_bytes)

        # File validation rejection
        fd, tmp_corrupt = tempfile.mkstemp(suffix=".jpg")
        with os.fdopen(fd, "wb") as f:
            f.write(corrupt_bytes)
        try:
            val_res = validator.validate_image_file(tmp_corrupt)
            assert val_res.is_valid is False
            assert any(i.code == "IMAGE_DECODE_FAILED" for i in val_res.issues)
        finally:
            if os.path.exists(tmp_corrupt):
                os.remove(tmp_corrupt)

    def test_video_preprocessing_output_validity(self, synthetic_test_video_path: str):
        """Verify video temporal tensor batching (T, C, H, W) and 3D backbone conversion."""
        preprocessor = VideoPreprocessor(
            target_size=(112, 112),
            num_frames=8,
            sampling_strategy=SamplingStrategy.UNIFORM,
        )
        processed: ProcessedVideo = preprocessor.process(synthetic_test_video_path)

        assert processed.tensor.shape == (8, 3, 112, 112)
        assert processed.to_batch().shape == (1, 8, 3, 112, 112)
        assert processed.to_channel_first_temporal().shape == (3, 8, 112, 112)
        assert processed.frame_count == 8

        validator = PreprocessingValidator()
        val_res = validator.validate_tensor(processed.tensor)
        assert val_res.is_valid is True

    def test_video_corrupt_media_handling(self):
        """Verify corrupt or truncated video files are caught without crashing."""
        validator = PreprocessingValidator()
        fd, tmp_corrupt = tempfile.mkstemp(suffix=".mp4")
        with os.fdopen(fd, "wb") as f:
            f.write(b"NOT_A_VALID_MP4_VIDEO_CONTAINER_DATA")

        try:
            val_res = validator.validate_video_file(tmp_corrupt)
            assert val_res.is_valid is False
            assert any(
                i.code in ("VIDEO_DECODE_FAILED", "FIRST_FRAME_DECODE_ERROR", "NO_FRAMES_DETECTED")
                for i in val_res.issues
            )
        finally:
            if os.path.exists(tmp_corrupt):
                os.remove(tmp_corrupt)

    def test_video_keyframe_and_timestamp_alignment(self, synthetic_test_video_path: str):
        """Verify timestamp alignment schema matching AnalysisResult.videoFrames."""
        preprocessor = VideoPreprocessor(
            target_size=(64, 64),
            num_frames=6,
            sampling_strategy=SamplingStrategy.UNIFORM,
        )
        processed: ProcessedVideo = preprocessor.process(synthetic_test_video_path)
        analysis_frames = processed.to_analysis_frames(default_trust_score=94.0)

        assert len(analysis_frames) == 6
        for f in analysis_frames:
            assert "timestamp" in f
            assert "timestampSeconds" in f
            assert f["trustScore"] == 94.0
            assert "isKeyframe" in f
            assert "frameIndex" in f
