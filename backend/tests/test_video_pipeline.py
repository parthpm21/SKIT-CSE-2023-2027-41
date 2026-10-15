"""
TruthLens — Automated Tests for Face Alignment & Video Deepfake Backbone
Author: Pratham Yadav
Sprint 3: User Story 3 - Parts 1 & 2 Validation

Validates 5-point facial landmark estimation, affine rotation alignment,
canonical 224x224 crop generation, Xception depthwise separable convolutions,
EfficientNet squeeze-and-excitation blocks, and dual-stream video sequence prediction.
"""
import numpy as np
import pytest
from PIL import Image

from backend.inference.face_aligner import (
    FaceLandmarks,
    FaceDetection,
    FaceDetector,
    AffineFaceAligner,
    VideoFacePipeline,
    AlignedVideoSequence,
)
from backend.inference.video_backbone import (
    VideoBackboneConfig,
    DepthwiseSeparableConv2D,
    XceptionResidualBlock,
    SqueezeAndExcitationBlock,
    EfficientNetBlock,
    DualStreamVideoBackbone,
    FrameInferenceResult,
)
from backend.inference.video_deepfake_service import (
    VideoDeepfakeService,
    VideoVerdict,
)


class TestFaceAlignmentPipeline:
    """Test suite for Face Detection and 5-Point Affine Alignment (Part 1)."""

    @pytest.fixture
    def detector(self) -> FaceDetector:
        return FaceDetector(min_face_size=30, confidence_threshold=0.5)

    @pytest.fixture
    def aligner(self) -> AffineFaceAligner:
        return AffineFaceAligner(target_size=224)

    @pytest.fixture
    def synthetic_face_frame(self) -> np.ndarray:
        """Constructs an RGB frame with a central warm-skin tone region."""
        frame = np.ones((400, 400, 3), dtype=np.uint8) * 40
        # Warm skin-tone block in center
        frame[100:300, 120:280, 0] = 190  # R
        frame[100:300, 120:280, 1] = 130  # G
        frame[100:300, 120:280, 2] = 100  # B
        return frame

    def test_face_detector_detects_skin_tone_face(self, detector: FaceDetector, synthetic_face_frame: np.ndarray):
        det = detector.detect(synthetic_face_frame)

        assert isinstance(det, FaceDetection)
        assert det.is_fallback is False
        assert det.confidence > 0.4
        assert det.width >= 30 and det.height >= 30

        # Check landmarks
        lm = det.landmarks
        assert isinstance(lm, FaceLandmarks)
        assert lm.left_eye[0] < lm.right_eye[0]  # Left eye is to the left of right eye
        assert lm.left_eye[1] < lm.nose[1]       # Eyes are above nose
        assert lm.nose[1] < lm.mouth_left[1]     # Nose is above mouth

    def test_face_detector_fallback_on_blank_frame(self, detector: FaceDetector):
        blank = np.zeros((300, 300, 3), dtype=np.uint8)
        det = detector.detect(blank)

        assert det.is_fallback is True
        assert det.confidence == 0.40
        assert det.bbox == (45, 45, 255, 255)

    def test_affine_face_aligner_output_shape_and_range(self, aligner: AffineFaceAligner, synthetic_face_frame: np.ndarray, detector: FaceDetector):
        det = detector.detect(synthetic_face_frame)
        crop = aligner.align_face(synthetic_face_frame, det)

        assert crop.shape == (3, 224, 224)
        assert crop.dtype == np.float32
        assert crop.min() >= -1.0
        assert crop.max() <= 1.0

    def test_video_face_pipeline_batch_processing(self, synthetic_face_frame: np.ndarray):
        pipeline = VideoFacePipeline(target_size=224)
        # Sequence of 4 frames
        frames = [synthetic_face_frame.copy() for _ in range(4)]
        seq = pipeline.process_frames(frames)

        assert isinstance(seq, AlignedVideoSequence)
        assert seq.num_frames == 4
        assert seq.aligned_frames.shape == (4, 3, 224, 224)
        assert seq.faces_detected_count == 4
        assert seq.fallback_count == 0
        assert len(seq.detections) == 4

        d = seq.to_dict()
        assert d["num_frames"] == 4
        assert d["detection_rate"] == 1.0

    def test_video_face_pipeline_empty_sequence(self):
        pipeline = VideoFacePipeline(target_size=224)
        seq = pipeline.process_frames([])
        assert seq.num_frames == 0
        assert seq.aligned_frames.shape == (0, 3, 224, 224)


class TestDualStreamVideoBackbone:
    """Test suite for Xception & EfficientNet Spatial Deepfake Backbone (Part 2)."""

    @pytest.fixture
    def test_backbone(self) -> DualStreamVideoBackbone:
        cfg = VideoBackboneConfig(
            in_channels=3,
            base_dim=8,         # Fast configuration for unit tests
            embed_dim=64,
            xception_depth=1,
            efficientnet_depth=1,
        )
        return DualStreamVideoBackbone(config=cfg, seed=42)

    def test_depthwise_separable_conv2d_forward(self):
        conv = DepthwiseSeparableConv2D(in_ch=4, out_ch=8, stride=1, seed=42)
        x = np.random.randn(2, 4, 32, 32).astype(np.float32)
        out = conv.forward(x)

        assert out.shape == (2, 8, 32, 32)
        assert np.isfinite(out).all()
        assert (out >= 0.0).all()  # ReLU activation

    def test_depthwise_separable_conv2d_stride2(self):
        conv = DepthwiseSeparableConv2D(in_ch=4, out_ch=8, stride=2, seed=42)
        x = np.random.randn(2, 4, 32, 32).astype(np.float32)
        out = conv.forward(x)

        assert out.shape == (2, 8, 16, 16)

    def test_xception_residual_block(self):
        block = XceptionResidualBlock(channels=8, seed=42)
        x = np.random.randn(2, 8, 16, 16).astype(np.float32)
        out = block.forward(x)

        assert out.shape == (2, 8, 16, 16)
        assert not np.array_equal(out, x)

    def test_squeeze_and_excitation_block(self):
        se = SqueezeAndExcitationBlock(channels=8, reduction=2, seed=42)
        x = np.random.randn(2, 8, 16, 16).astype(np.float32)
        out = se.forward(x)

        assert out.shape == (2, 8, 16, 16)
        assert np.isfinite(out).all()

    def test_efficientnet_mbconv_block(self):
        block = EfficientNetBlock(in_ch=8, out_ch=8, expansion=2, seed=42)
        x = np.random.randn(2, 8, 16, 16).astype(np.float32)
        out = block.forward(x)

        assert out.shape == (2, 8, 16, 16)

    def test_dual_stream_backbone_forward_frames(self, test_backbone: DualStreamVideoBackbone):
        aligned_frames = np.random.uniform(-1.0, 1.0, (3, 3, 224, 224)).astype(np.float32)
        embeddings, logits = test_backbone.forward_frames(aligned_frames)

        assert embeddings.shape == (3, 64)
        assert logits.shape == (3,)
        assert np.isfinite(embeddings).all()
        assert np.isfinite(logits).all()

        # Embeddings must be L2 normalized
        norms = np.linalg.norm(embeddings, axis=-1)
        np.testing.assert_allclose(norms, np.ones(3), atol=1e-4)

    def test_dual_stream_backbone_predict_video_sequence(self, test_backbone: DualStreamVideoBackbone):
        aligned_frames = np.random.uniform(-1.0, 1.0, (5, 3, 224, 224)).astype(np.float32)
        res = test_backbone.predict_video_sequence(aligned_frames)

        assert isinstance(res, FrameInferenceResult)
        assert res.label in {"real", "fake"}
        assert 0.0 <= res.confidence <= 1.0
        assert len(res.frame_scores) == 5
        assert all(0.0 <= s <= 1.0 for s in res.frame_scores)
        assert res.frame_embeddings.shape == (5, 64)

        d = res.to_dict()
        assert d["num_frames"] == 5
        assert "mean_score" in d
        assert "peak_score" in d


class TestVideoDeepfakeServiceIntegration:
    """End-to-end integration test with the service layer."""

    def test_service_predict_with_dual_stream_backbone(self):
        # Frame tensor stack (3 frames, 3 channels, 224, 224)
        frames = np.random.uniform(-1.0, 1.0, (3, 3, 224, 224)).astype(np.float32)
        service = VideoDeepfakeService()
        verdict = service.predict(frames)

        assert isinstance(verdict, VideoVerdict)
        assert verdict.label in {"real", "fake"}
        assert 0.0 <= verdict.confidence <= 1.0
        assert len(verdict.frame_scores) == 3
        assert verdict.model_version == "xception-efficientnet-ensemble-v1"

    def test_service_predict_empty_tensor(self):
        empty = np.zeros((0, 3, 224, 224), dtype=np.float32)
        service = VideoDeepfakeService()
        verdict = service.predict(empty)

        assert verdict.label == "real"
        assert verdict.confidence == 1.0
        assert verdict.frame_scores == []
