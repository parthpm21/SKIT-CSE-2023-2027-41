"""
TruthLens — Tests for the fusion training-data schema
Author: Parth Maheshwari
"""
import numpy as np
import pytest

from backend.fusion.contracts import FusionInput, ImageBranchInput, VideoBranchInput
from backend.fusion.training.schema import (
    MASK_SIZE, FusionSample, read_jsonl, weak_frame_labels, weak_pixel_mask, write_jsonl,
)


def image_input():
    return FusionInput(sample_id="i", image=ImageBranchInput(cls_embedding=[0.1] * 4, probe_score=0.9))


def video_input(t=3):
    return FusionInput(sample_id="v", video=VideoBranchInput(
        frame_embeddings=[[0.1] * 4 for _ in range(t)], frame_scores=[0.5] * t))


@pytest.mark.unit
class TestFusionSample:
    def test_jsonl_roundtrip(self, tmp_path):
        samples = [
            FusionSample(image_input(), 1, pixel_mask=weak_pixel_mask(1), source="GenImage"),
            FusionSample(video_input(3), 0, frame_labels=weak_frame_labels(0, 3)),
        ]
        path = tmp_path / "nested" / "train.jsonl"
        assert write_jsonl(samples, path) == 2
        loaded = read_jsonl(path)
        assert len(loaded) == 2
        assert loaded[0].verdict == 1 and loaded[0].source == "GenImage"
        np.testing.assert_array_equal(loaded[0].pixel_mask, np.ones((MASK_SIZE, MASK_SIZE)))
        np.testing.assert_array_equal(loaded[1].frame_labels, np.zeros(3))
        assert loaded[1].pixel_mask is None

    def test_bad_verdict_rejected(self):
        with pytest.raises(ValueError):
            FusionSample(image_input(), 2)

    def test_bad_mask_shape_rejected(self):
        with pytest.raises(ValueError):
            FusionSample(image_input(), 1, pixel_mask=np.ones((8, 8)))

    def test_mask_out_of_range_rejected(self):
        with pytest.raises(ValueError):
            FusionSample(image_input(), 1, pixel_mask=np.full((MASK_SIZE, MASK_SIZE), 2.0))

    def test_non_binary_frame_labels_rejected(self):
        with pytest.raises(ValueError):
            FusionSample(video_input(3), 1, frame_labels=np.array([0.0, 0.5, 1.0]))

    def test_frame_label_length_must_match_video(self):
        with pytest.raises(ValueError):
            FusionSample(video_input(3), 1, frame_labels=np.array([1.0, 1.0]))

    def test_weak_labels_follow_verdict(self):
        assert weak_pixel_mask(0).sum() == 0
        assert weak_pixel_mask(1).sum() == MASK_SIZE * MASK_SIZE
        assert weak_frame_labels(1, 5).tolist() == [1.0] * 5