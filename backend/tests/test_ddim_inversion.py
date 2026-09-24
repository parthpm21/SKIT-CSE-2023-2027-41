"""
TruthLens — Automated Tests for Deterministic DDIM Inversion & Reconstruction Engine
Author: Pratham Yadav
Sprint 2: User Story 2 - Part 2 Validation

Validates DDIM variance schedules, deterministic ODE forward trajectory inversion,
reverse reconstruction steps, multi-scale noise estimator, and reconstruction metrics (L2, PSNR).
"""
import numpy as np
import pytest
from PIL import Image

from backend.inference.ddim_inversion import (
    DDIMConfig,
    DDIMScheduler,
    SinusoidalTimeEmbedding,
    DiffusionUNetNoiseEstimator,
    DDIMInversionEngine,
    DDIMInversionResult,
)


class TestDDIMInversionEngine:
    """Test suite for DDIM Inversion and Reverse Reconstruction algorithms."""

    @pytest.fixture
    def default_config(self) -> DDIMConfig:
        return DDIMConfig(
            num_train_timesteps=1000,
            num_inversion_steps=10,  # Fast step count for unit tests
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            image_size=64,  # Compact resolution for high-speed deterministic test execution
            base_dim=16,
            time_embed_dim=32,
        )

    @pytest.fixture
    def sample_image(self, default_config: DDIMConfig) -> np.ndarray:
        rng = np.random.RandomState(42)
        return rng.uniform(-1.0, 1.0, (3, default_config.image_size, default_config.image_size)).astype(np.float32)

    def test_ddim_scheduler_variance_invariants(self, default_config: DDIMConfig):
        scheduler = DDIMScheduler(default_config)

        # 1. Betas must be positive and bounded
        assert scheduler.betas.shape == (1000,)
        assert (scheduler.betas > 0.0).all()
        assert (scheduler.betas < 1.0).all()

        # 2. Cumulative alpha products must decrease monotonically
        assert scheduler.alphas_cumprod.shape == (1000,)
        assert scheduler.alphas_cumprod[0] > scheduler.alphas_cumprod[-1]
        assert np.all(np.diff(scheduler.alphas_cumprod) <= 0)

        # 3. Timesteps subsampling must match configured inversion steps
        assert len(scheduler.timesteps) == 10
        assert scheduler.timesteps[0] == 0
        assert scheduler.timesteps[-1] < 1000

    def test_cosine_and_scaled_linear_schedules(self):
        # Cosine schedule
        cfg_cos = DDIMConfig(num_train_timesteps=500, num_inversion_steps=5, beta_schedule="cosine")
        sched_cos = DDIMScheduler(cfg_cos)
        assert len(sched_cos.betas) == 500
        assert (sched_cos.alphas_cumprod > 0.0).all()

        # Scaled linear schedule
        cfg_scaled = DDIMConfig(num_train_timesteps=500, num_inversion_steps=5, beta_schedule="scaled_linear")
        sched_scaled = DDIMScheduler(cfg_scaled)
        assert len(sched_scaled.betas) == 500

    def test_sinusoidal_time_embedding(self):
        time_embed = SinusoidalTimeEmbedding(embed_dim=64)
        emb_0 = time_embed.forward(0)
        emb_500 = time_embed.forward(500)

        assert emb_0.shape == (1, 64)
        assert emb_500.shape == (1, 64)
        assert not np.allclose(emb_0, emb_500), "Different timesteps must have distinct embeddings"
        assert np.all(emb_0 >= -1.0) and np.all(emb_0 <= 1.0)

    def test_noise_estimator_forward(self, default_config: DDIMConfig, sample_image: np.ndarray):
        estimator = DiffusionUNetNoiseEstimator(default_config, seed=42)
        noise = estimator.forward(sample_image, timestep=200)

        assert noise.shape == sample_image.shape
        assert np.isfinite(noise).all(), "Predicted noise contains non-finite values"

    def test_forward_invert_and_reverse_reconstruct(self, default_config: DDIMConfig, sample_image: np.ndarray):
        engine = DDIMInversionEngine(config=default_config, seed=42)

        # 1. Forward Inversion
        latent_xT, fwd_traj = engine.forward_invert(sample_image, num_steps=5, save_trajectory=True)
        assert latent_xT.shape == sample_image.shape
        assert len(fwd_traj) == 5
        assert np.isfinite(latent_xT).all()

        # 2. Reverse Reconstruction
        recon_x0, rev_traj = engine.reverse_reconstruct(latent_xT, num_steps=5, save_trajectory=True)
        assert recon_x0.shape == sample_image.shape
        assert len(rev_traj) == 5
        assert np.isfinite(recon_x0).all()

    def test_full_inversion_and_reconstruction_pipeline(self, default_config: DDIMConfig, sample_image: np.ndarray):
        engine = DDIMInversionEngine(config=default_config, seed=42)
        result = engine.invert_and_reconstruct(sample_image, num_steps=8, save_trajectory=True)

        assert isinstance(result, DDIMInversionResult)
        assert result.original_image.shape == (3, 64, 64)
        assert result.latent_noise.shape == (3, 64, 64)
        assert result.reconstructed_image.shape == (3, 64, 64)
        assert result.inversion_error_l2 >= 0.0
        assert result.reconstruction_psnr > 0.0
        assert result.num_steps == 8
        assert len(result.intermediate_latents) == 16  # 8 forward + 8 reverse

        # Serialization to dict
        res_dict = result.to_dict()
        assert "inversion_error_l2" in res_dict
        assert "reconstruction_psnr" in res_dict
        assert res_dict["num_steps"] == 8

    def test_engine_handles_pil_image_and_rescaling(self, default_config: DDIMConfig):
        engine = DDIMInversionEngine(config=default_config, seed=42)

        # PIL image with non-matching resolution (128x128)
        pil_img = Image.new("RGB", (128, 128), color=(200, 100, 50))
        result = engine.invert_and_reconstruct(pil_img, num_steps=5)

        assert result.original_image.shape == (3, 64, 64)
        assert result.reconstructed_image.shape == (3, 64, 64)
        assert result.inversion_error_l2 >= 0.0
