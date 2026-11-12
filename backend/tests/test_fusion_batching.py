"""
TruthLens — Tests for fusion config validation and batching
Author: Parth Maheshwari
"""
import numpy as np
import pytest

from backend.fusion.model.batching import (
    collate_inputs_np, collate_samples_np, fit_dim, resize_map, time_index,
)
from backend.fusion.model.config import FusionModelConfig
from backend.fusion.training.synthetic import make_synthetic_samples

CFG = FusionModelConfig(d_model=32, num_heads=4, clip_dim=16, video_dim=8, max_frames=12)


@pytest.mark.unit
class TestConfig:
    def test_defaults_valid_and_grid(self):
        assert FusionModelConfig().grid == 4

    @pytest.mark.parametrize("kw", [
        {"d_model": 30, "num_heads": 4}, {"patch": 6}, {"patch": 1}, {"map_size": 16}, {"max_frames": 0},
    ])
    def test_invalid_rejected(self, kw):
        with pytest.raises(ValueError):
            FusionModelConfig(**kw)

    def test_dict_roundtrip(self):
        assert FusionModelConfig.from_dict(CFG.to_dict()) == CFG


@pytest.mark.unit
class TestHelpers:
    def test_fit_dim_pads_and_truncates(self):
        assert fit_dim([1, 2], 4).tolist() == [1, 2, 0, 0]
        assert fit_dim([1, 2, 3], 2).tolist() == [1, 2]

    def test_time_index(self):
        assert time_index(5, 8).tolist() == [0, 1, 2, 3, 4]
        idx = time_index(100, 10)
        assert len(idx) == 10 and idx[0] == 0 and idx[-1] == 99

    def test_resize_map_up_down_and_constant(self):
        assert resize_map(np.ones((14, 14)), 32).shape == (32, 32)
        assert resize_map(np.ones((14, 14)), 32) == pytest.approx(1.0)
        assert resize_map(np.random.rand(64, 64), 32).shape == (32, 32)

    def test_resize_map_rejects_non_2d(self):
        with pytest.raises(ValueError):
            resize_map(np.ones((2, 2, 2)), 32)


@pytest.mark.unit
class TestCollate:
    def setup_method(self):
        self.samples = make_synthetic_samples(9, seed=0, clip_dim=CFG.clip_dim, video_dim=CFG.video_dim)

    def test_shapes_and_presence_flags(self):
        b = collate_samples_np(self.samples, CFG)
        B, T, S = 9, CFG.max_frames, CFG.map_size
        assert b["clip_vec"].shape == (B, CFG.clip_dim)
        assert b["dire_map"].shape == (B, 1, S, S)
        assert b["vid_emb"].shape == (B, T, CFG.video_dim)
        assert b["pixel_mask"].shape == (B, 1, S, S) and b["frame_labels"].shape == (B, T)
        for i, s in enumerate(self.samples):
            fi = s.fusion_input
            assert b["clip_present"][i] == (fi.image is not None)
            assert b["dire_present"][i] == (fi.diffusion is not None)
            assert b["vid_present"][i] == (fi.video is not None)
            assert b["noise_present"][i] == (fi.noise is not None)
            assert b["pixel_valid"][i] == (s.pixel_mask is not None)

    def test_frame_valid_matches_clip_length(self):
        b = collate_samples_np(self.samples, CFG)
        for i, s in enumerate(self.samples):
            if s.fusion_input.video is not None:
                n = min(len(s.fusion_input.video.frame_scores), CFG.max_frames)
                assert b["vid_valid"][i].sum() == n and b["frame_valid"][i].sum() == n

    def test_long_video_is_subsampled_to_max_frames(self):
        s = make_synthetic_samples(3, seed=1, clip_dim=16, video_dim=8)[2]
        small = FusionModelConfig(d_model=32, num_heads=4, clip_dim=16, video_dim=8, max_frames=4)
        b = collate_samples_np([s], small)
        assert b["vid_valid"][0].sum() == 4 and b["frame_valid"][0].sum() == 4

    def test_padding_is_zero(self):
        b = collate_samples_np(self.samples, CFG)
        assert not b["vid_emb"][~b["vid_valid"]].any()
        assert not b["clip_vec"][~b["clip_present"]].any()

    def test_inputs_only_has_no_targets(self):
        b = collate_inputs_np([s.fusion_input for s in self.samples], CFG)
        assert "verdict" not in b and "pixel_mask" not in b