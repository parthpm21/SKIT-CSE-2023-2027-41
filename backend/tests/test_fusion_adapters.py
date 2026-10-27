"""
TruthLens — Tests for the fusion feature adapters
Author: Parth Maheshwari
"""
import numpy as np
import pytest

from backend.fusion.adapters import (
    MAP_SIZE, build_fusion_input, downsample_map, video_to_branch,
)
from backend.inference.clip_vit_probe import CLIPProbeResult
from backend.inference.dire_engine import DIREResult
from backend.inference.noise_fingerprint import NoiseFingerprintResult


def clip_result(prob=0.8):
    return CLIPProbeResult(
        is_synthetic=prob >= 0.5, label="synthetic", confidence=prob, synthetic_prob=prob,
        authentic_prob=1 - prob, cls_embedding=np.ones(768, dtype=np.float32),
        patch_attention_map=np.zeros((14, 14), dtype=np.float32), feature_dim=768,
    )


def dire_result(h=224, w=224):
    return DIREResult(
        is_diffusion_generated=True, confidence=0.7, diffusion_score=0.7, mse=0.02, mae=0.1,
        psnr=20.0, ssim_score=0.8, high_freq_energy=0.1, residual_std=0.05,
        residual_map=np.random.default_rng(0).random((h, w)).astype(np.float32),
        channel_residuals=np.zeros((3, h, w), dtype=np.float32),
        reconstructed_image=np.zeros((3, h, w), dtype=np.float32),
    )


def noise_result():
    return NoiseFingerprintResult(
        temporal_inconsistency_score=0.6, mean_noise_energy=0.02, face_bg_noise_discrepancy=0.3,
        frame_noise_energies=[0.01, 0.02, 0.03], residual_maps=np.zeros((3, 8, 8)),
        is_noise_anomalous=True,
    )


@pytest.mark.unit
class TestAdapters:
    def test_downsample_large_map(self):
        out = downsample_map(np.random.default_rng(1).random((224, 224)))
        assert out.shape == (MAP_SIZE, MAP_SIZE)

    def test_downsample_preserves_mean(self):
        arr = np.random.default_rng(2).random((64, 64)).astype(np.float32)
        assert downsample_map(arr).mean() == pytest.approx(arr.mean(), abs=1e-4)

    def test_small_map_unchanged(self):
        arr = np.ones((4, 5), dtype=np.float32)
        assert downsample_map(arr).shape == (4, 5)

    def test_non_2d_map_rejected(self):
        with pytest.raises(ValueError):
            downsample_map(np.ones((3, 3, 3)))

    def test_build_with_all_branches(self):
        fi = build_fusion_input(
            "s1", clip_result(), dire_result(), np.zeros((3, 512)), [0.1, 0.2, 0.3], noise_result(),
        )
        assert fi.image.probe_score == pytest.approx(0.8)
        assert len(fi.image.cls_embedding) == 768
        assert np.asarray(fi.diffusion.dire_residual_map).shape == (MAP_SIZE, MAP_SIZE)
        assert len(fi.video.frame_embeddings) == 3
        assert fi.noise.frame_noise_energies == [0.01, 0.02, 0.03]

    def test_build_with_subset(self):
        fi = build_fusion_input("s2", clip_result=clip_result())
        assert fi.diffusion is None and fi.video is None and fi.noise is None

    def test_build_with_nothing_raises(self):
        with pytest.raises(ValueError):
            build_fusion_input("s3")

    def test_video_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            video_to_branch(np.zeros((3, 8)), [0.1, 0.2])