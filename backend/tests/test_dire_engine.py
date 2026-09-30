"""
TruthLens — Automated Tests for DIRE (Diffusion Reconstruction Error) Engine
Author: Pratham Yadav
Sprint 2: User Story 2 - Part 3 Validation

Validates SSIM calculation, Laplacian high-frequency residual energy,
gradient residual maps, spatial error heatmaps, and diffusion authenticity scoring.
"""
import math
import numpy as np
import pytest
from PIL import Image

from backend.inference.ddim_inversion import DDIMConfig, DDIMInversionEngine
from backend.inference.dire_engine import (
    DIREEngine,
    DIREResult,
    _compute_ssim,
    _compute_laplacian_energy,
    _compute_gradient_residual,
)


class TestDIREEngine:
    """Test suite for DIRE residual map generation and error metrics."""

    @pytest.fixture
    def test_inversion_engine(self) -> DDIMInversionEngine:
        # Fast config for testing
        cfg = DDIMConfig(
            num_train_timesteps=500,
            num_inversion_steps=5,
            image_size=64,
            base_dim=16,
            time_embed_dim=32,
        )
        return DDIMInversionEngine(config=cfg, seed=42)

    @pytest.fixture
    def dire_engine_instance(self, test_inversion_engine: DDIMInversionEngine) -> DIREEngine:
        return DIREEngine(
            inversion_engine=test_inversion_engine,
            num_inversion_steps=5,
            mse_threshold=0.03,
            ssim_threshold=0.90,
        )

    def test_ssim_identical_and_perturbed_images(self):
        rng = np.random.RandomState(42)
        img = rng.uniform(-1.0, 1.0, (3, 64, 64)).astype(np.float32)

        # Identical image must yield SSIM = 1.0
        ssim_identical = _compute_ssim(img, img)
        assert pytest.approx(ssim_identical, abs=1e-4) == 1.0

        # Perturbed image must yield lower SSIM
        noisy = img + rng.normal(0.0, 0.3, img.shape).astype(np.float32)
        ssim_noisy = _compute_ssim(img, noisy)
        assert ssim_noisy < 1.0
        assert ssim_noisy >= 0.0

    def test_laplacian_energy_and_gradient_residuals(self):
        # Smooth image vs high-frequency pattern
        smooth = np.ones((64, 64), dtype=np.float32) * 0.5
        lap_smooth = _compute_laplacian_energy(smooth)
        assert pytest.approx(lap_smooth, abs=1e-5) == 0.0

        rng = np.random.RandomState(42)
        checker = (rng.rand(64, 64) > 0.5).astype(np.float32)
        lap_checker = _compute_laplacian_energy(checker)
        assert lap_checker > lap_smooth

        # Gradient residual calculation
        x0 = rng.uniform(-1.0, 1.0, (3, 64, 64)).astype(np.float32)
        x_recon = x0 + rng.normal(0.0, 0.1, (3, 64, 64)).astype(np.float32)
        grad_res = _compute_gradient_residual(x0, x_recon)
        assert grad_res.shape == (64, 64)
        assert (grad_res >= 0.0).all()

    def test_compute_residual_map_normalization(self, dire_engine_instance: DIREEngine):
        rng = np.random.RandomState(42)
        x0 = rng.uniform(-1.0, 1.0, (3, 64, 64)).astype(np.float32)
        x_recon = x0 + rng.normal(0.0, 0.05, (3, 64, 64)).astype(np.float32)

        heatmap, channel_res = dire_engine_instance.compute_residual_map(x0, x_recon)

        assert heatmap.shape == (64, 64)
        assert channel_res.shape == (3, 64, 64)
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0
        assert (channel_res >= 0.0).all()

    def test_analyze_residual_metrics(self, dire_engine_instance: DIREEngine):
        rng = np.random.RandomState(42)
        x0 = rng.uniform(-1.0, 1.0, (3, 64, 64)).astype(np.float32)
        x_recon = x0 + rng.normal(0.0, 0.05, (3, 64, 64)).astype(np.float32)

        result = dire_engine_instance.analyze_residual(x0, x_recon, num_steps_used=5)

        assert isinstance(result, DIREResult)
        assert 0.0 <= result.diffusion_score <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert result.mse >= 0.0
        assert result.mae >= 0.0
        assert result.psnr > 0.0
        assert 0.0 <= result.ssim_score <= 1.0
        assert result.residual_map.shape == (64, 64)
        assert result.channel_residuals.shape == (3, 64, 64)

        # Dictionary serialization
        res_dict = result.to_dict()
        assert "is_diffusion_generated" in res_dict
        assert "diffusion_score" in res_dict
        assert "ssim_score" in res_dict
        assert "high_freq_energy" in res_dict

    def test_end_to_end_dire_analyze(self, dire_engine_instance: DIREEngine):
        pil_img = Image.new("RGB", (64, 64), color=(180, 120, 70))
        result = dire_engine_instance.analyze(pil_img, num_steps=5)

        assert isinstance(result, DIREResult)
        assert result.residual_map.shape == (64, 64)
        assert np.isfinite(result.mse)
        assert np.isfinite(result.ssim_score)
