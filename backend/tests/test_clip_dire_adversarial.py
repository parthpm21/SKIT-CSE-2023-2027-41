"""
TruthLens — Automated Adversarial, Edge-Case & Stress Test Suite
Author: Pratham Yadav
Sprint 2: User Story 2 - Part 5 (Robustness & Adversarial Testing)

Validates CLIP:ViT probing and DDIM/DIRE reconstruction against adversarial perturbations,
JPEG compression artifacts, Gaussian noise, extreme aspect ratios, resolution scaling,
and multi-generator synthetic media profiles (Diffusion vs GAN vs Authentic).
"""
import io
import math
import numpy as np
import pytest
from PIL import Image, ImageFilter, ImageEnhance

from backend.inference.clip_vit_probe import (
    ViTConfig,
    FrozenCLIPViTProbe,
    CLIPProbeResult,
)
from backend.inference.ddim_inversion import (
    DDIMConfig,
    DDIMInversionEngine,
    DDIMInversionResult,
)
from backend.inference.dire_engine import (
    DIREEngine,
    DIREResult,
)
from backend.inference.clip_dire_service import (
    CLIPDIREImageService,
    DualBranchVerdict,
)


class TestCLIPDIREAdversarialAndEdgeCases:
    """Stress and adversarial test suite for dual-branch image detection pipeline."""

    @pytest.fixture
    def service(self) -> CLIPDIREImageService:
        vit_cfg = ViTConfig(num_layers=2, mlp_dim=512)
        ddim_cfg = DDIMConfig(num_train_timesteps=200, num_inversion_steps=5, image_size=64)
        probe = FrozenCLIPViTProbe(config=vit_cfg, seed=42)
        inversion = DDIMInversionEngine(config=ddim_cfg, seed=42)
        dire = DIREEngine(inversion_engine=inversion, num_inversion_steps=5)
        return CLIPDIREImageService(clip_probe=probe, dire_analyzer=dire)

    def test_heavy_jpeg_compression_robustness(self, service: CLIPDIREImageService):
        """Tests that JPEG compression artifacts do not crash the pipeline or produce NaNs."""
        base_img = Image.new("RGB", (224, 224), color=(140, 90, 210))
        
        # Test extreme JPEG quality levels (10, 30, 75, 95)
        for quality in [10, 30, 75, 95]:
            buf = io.BytesIO()
            base_img.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            compressed_img = Image.open(buf)

            verdict = service.predict(compressed_img, num_dire_steps=5)
            assert isinstance(verdict, DualBranchVerdict)
            assert np.isfinite(verdict.confidence)
            assert np.isfinite(verdict.dire_mse)
            assert np.isfinite(verdict.dire_ssim)
            assert not np.isnan(verdict.fused_heatmap).any()

    def test_gaussian_noise_injection(self, service: CLIPDIREImageService):
        """Tests pipeline resilience under heavy additive Gaussian noise."""
        rng = np.random.RandomState(42)
        clean_arr = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        noise = rng.normal(0, 35, (224, 224, 3))
        noisy_arr = np.clip(clean_arr + noise, 0, 255).astype(np.uint8)

        verdict = service.predict(noisy_arr, num_dire_steps=5)
        assert verdict.label in {"real", "fake"}
        assert 0.0 <= verdict.confidence <= 1.0
        assert verdict.fused_heatmap.shape == (224, 224)

    def test_gaussian_blur_and_contrast_extremes(self, service: CLIPDIREImageService):
        """Tests pipeline with blurred images and extreme contrast/brightness."""
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        
        # 1. Heavily blurred image
        blurred = img.filter(ImageFilter.GaussianBlur(radius=8))
        res_blur = service.predict(blurred, num_dire_steps=5)
        assert np.isfinite(res_blur.confidence)

        # 2. Overexposed / maximum brightness
        enhancer = ImageEnhance.Brightness(img)
        overexposed = enhancer.enhance(3.0)
        res_bright = service.predict(overexposed, num_dire_steps=5)
        assert np.isfinite(res_bright.confidence)

        # 3. Pure black and pure white extremes
        black_img = Image.new("RGB", (224, 224), color=(0, 0, 0))
        white_img = Image.new("RGB", (224, 224), color=(255, 255, 255))
        assert np.isfinite(service.predict(black_img, num_dire_steps=5).confidence)
        assert np.isfinite(service.predict(white_img, num_dire_steps=5).confidence)

    def test_extreme_aspect_ratios_and_resolutions(self, service: CLIPDIREImageService):
        """Tests non-square aspect ratios (panoramic and vertical strips) and tiny inputs."""
        # 1. Panoramic banner (800 x 80)
        panoramic = Image.new("RGB", (800, 80), color=(100, 180, 120))
        res_pano = service.predict(panoramic, num_dire_steps=5)
        assert res_pano.fused_heatmap.shape == (224, 224)

        # 2. Vertical strip (60 x 600)
        vertical = Image.new("RGB", (60, 600), color=(200, 80, 100))
        res_vert = service.predict(vertical, num_dire_steps=5)
        assert res_vert.fused_heatmap.shape == (224, 224)

        # 3. Micro image (8 x 8)
        tiny = Image.new("RGB", (8, 8), color=(50, 100, 150))
        res_tiny = service.predict(tiny, num_dire_steps=5)
        assert res_tiny.fused_heatmap.shape == (224, 224)

    def test_rgba_and_palette_color_modes(self, service: CLIPDIREImageService):
        """Tests RGBA (transparency) and P (palette-indexed) image formats."""
        # RGBA
        rgba_img = Image.new("RGBA", (128, 128), color=(100, 150, 200, 128))
        res_rgba = service.predict(rgba_img, num_dire_steps=5)
        assert res_rgba.label in {"real", "fake"}

        # Palette P-mode
        p_img = Image.new("P", (128, 128))
        res_p = service.predict(p_img, num_dire_steps=5)
        assert res_p.label in {"real", "fake"}

    def test_ddim_inversion_numerical_stability(self):
        """Tests DDIM scheduler numerical invariants and tensor clipping bounds."""
        cfg = DDIMConfig(num_train_timesteps=500, num_inversion_steps=10, image_size=32)
        engine = DDIMInversionEngine(config=cfg, seed=42)

        test_tensor = np.random.uniform(-1.0, 1.0, (3, 32, 32)).astype(np.float32)
        res = engine.invert_and_reconstruct(test_tensor, num_steps=10, save_trajectory=True)

        # Latent noise and reconstruction must strictly remain within [-1, 1] bounds
        assert np.all(res.latent_noise >= -1.0) and np.all(res.latent_noise <= 1.0)
        assert np.all(res.reconstructed_image >= -1.0) and np.all(res.reconstructed_image <= 1.0)
        for latent in res.intermediate_latents:
            assert np.all(latent >= -1.0) and np.all(latent <= 1.0)
            assert np.isfinite(latent).all()

    def test_batch_synthetic_attribution_profiles(self, service: CLIPDIREImageService):
        """Tests that the subtype attribution correctly yields valid diagnostic labels."""
        rng = np.random.RandomState(42)
        
        # Run multiple simulated synthetic and natural patterns
        subtypes_seen = set()
        for i in range(5):
            arr = rng.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            verdict = service.predict(arr, num_dire_steps=5)
            assert verdict.synthetic_subtype in {"authentic", "diffusion", "gan", "hybrid"}
            subtypes_seen.add(verdict.synthetic_subtype)

        assert len(subtypes_seen) >= 1
