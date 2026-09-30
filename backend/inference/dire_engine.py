"""
TruthLens — DIRE (Diffusion Reconstruction Error) Residual Engine
Author: Pratham Yadav
Sprint 2: CLIP:ViT probing & DDIM/DIRE reconstruction branch (User Story 2 - Part 3)

This module computes and analyzes Diffusion Reconstruction Error (DIRE) residual maps.
By comparing an original image x_0 with its deterministic DDIM reconstruction x'_0,
it extracts pixel-level, gradient, frequency-domain, and structural anomaly signatures
to detect diffusion-generated synthetic media.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from backend.inference.ddim_inversion import (
    DDIMConfig,
    DDIMInversionEngine,
    DDIMInversionResult,
    ddim_inversion_engine,
)


@dataclass
class DIREResult:
    """Structured container holding DIRE error metrics, residual maps, and scoring."""
    is_diffusion_generated: bool
    confidence: float
    diffusion_score: float
    mse: float
    mae: float
    psnr: float
    ssim_score: float
    high_freq_energy: float
    residual_std: float
    residual_map: np.ndarray        # (H, W) normalized [0, 1] spatial error heatmap
    channel_residuals: np.ndarray   # (3, H, W) absolute per-channel error
    reconstructed_image: np.ndarray # (3, H, W) in [-1, 1]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_diffusion_generated": self.is_diffusion_generated,
            "confidence": round(float(self.confidence), 4),
            "diffusion_score": round(float(self.diffusion_score), 4),
            "mse": round(float(self.mse), 6),
            "mae": round(float(self.mae), 6),
            "psnr": round(float(self.psnr), 2),
            "ssim_score": round(float(self.ssim_score), 4),
            "high_freq_energy": round(float(self.high_freq_energy), 6),
            "residual_std": round(float(self.residual_std), 6),
            "metadata": self.metadata,
        }


def _compute_ssim(
    img1: np.ndarray,
    img2: np.ndarray,
    window_size: int = 7,
    k1: float = 0.01,
    k2: float = 0.03,
    data_range: float = 2.0,
) -> float:
    """
    Computes Structural Similarity Index (SSIM) between two (3, H, W) image arrays.
    """
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    # Mean across spatial dimensions
    mu1 = np.mean(img1, axis=(1, 2), keepdims=True)
    mu2 = np.mean(img2, axis=(1, 2), keepdims=True)

    sigma1_sq = np.var(img1, axis=(1, 2), keepdims=True)
    sigma2_sq = np.var(img2, axis=(1, 2), keepdims=True)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2), axis=(1, 2), keepdims=True)

    num = (2.0 * mu1 * mu2 + c1) * (2.0 * sigma12 + c2)
    den = (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = num / (den + 1e-8)
    return float(np.clip(np.mean(ssim_map), 0.0, 1.0))


def _compute_laplacian_energy(residual: np.ndarray) -> float:
    """
    Computes high-frequency spectral residual energy using a 2D Laplacian operator.
    """
    # 3x3 Laplacian kernel
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    padded = np.pad(residual, ((1, 1), (1, 1)), mode="reflect")
    H, W = residual.shape
    
    lap = np.zeros((H, W), dtype=np.float32)
    for di in range(3):
        for dj in range(3):
            lap += padded[di : di + H, dj : dj + W] * kernel[di, dj]
            
    return float(np.mean(np.abs(lap)))


def _compute_gradient_residual(x0: np.ndarray, x_recon: np.ndarray) -> np.ndarray:
    """
    Computes edge/gradient magnitude difference between original and reconstruction.
    """
    diff = np.mean(np.abs(x0 - x_recon), axis=0)  # (H, W)
    
    # Finite differences along Y and X
    grad_y = np.abs(np.diff(diff, axis=0, append=diff[-1:, :]))
    grad_x = np.abs(np.diff(diff, axis=1, append=diff[:, -1:]))
    grad_mag = np.sqrt(grad_y ** 2 + grad_x ** 2)
    return grad_mag.astype(np.float32)


class DIREEngine:
    """
    Diffusion Reconstruction Error (DIRE) Analysis Engine.
    Executes DDIM inversion + reconstruction, extracts multi-scale error residuals,
    and classifies diffusion generation patterns.
    """

    def __init__(
        self,
        inversion_engine: Optional[DDIMInversionEngine] = None,
        num_inversion_steps: int = 20,
        mse_threshold: float = 0.025,
        ssim_threshold: float = 0.92,
    ) -> None:
        self.inversion_engine = inversion_engine or ddim_inversion_engine
        self.num_inversion_steps = num_inversion_steps
        self.mse_threshold = mse_threshold
        self.ssim_threshold = ssim_threshold

    def compute_residual_map(self, x0: np.ndarray, x_recon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes normalized spatial residual map and per-channel residuals.
        
        Args:
            x0: Original tensor (3, H, W) in [-1, 1].
            x_recon: Reconstructed tensor (3, H, W) in [-1, 1].
        Returns:
            heatmap: (H, W) normalized [0, 1].
            channel_res: (3, H, W) absolute differences.
        """
        channel_res = np.abs(x0 - x_recon)  # (3, H, W)
        mean_res = np.mean(channel_res, axis=0)  # (H, W)
        
        # Include gradient error signature
        grad_res = _compute_gradient_residual(x0, x_recon)
        combined_res = 0.7 * mean_res + 0.3 * grad_res

        # Spatial normalization to [0, 1]
        r_min = float(np.min(combined_res))
        r_max = float(np.max(combined_res))
        if r_max > r_min:
            heatmap = (combined_res - r_min) / (r_max - r_min)
        else:
            heatmap = np.zeros_like(combined_res)

        return heatmap.astype(np.float32), channel_res.astype(np.float32)

    def analyze_residual(
        self,
        x0: np.ndarray,
        x_recon: np.ndarray,
        num_steps_used: int,
    ) -> DIREResult:
        """
        Calculates all error metrics and performs diffusion manifold scoring.
        """
        # 1. Error residual maps
        heatmap, channel_res = self.compute_residual_map(x0, x_recon)

        # 2. Mathematical error metrics
        l2_diff = (x0 - x_recon) ** 2
        mse = float(np.mean(l2_diff))
        mae = float(np.mean(channel_res))
        mse_safe = max(mse, 1e-10)
        psnr = float(10.0 * math.log10(4.0 / mse_safe))
        ssim_val = _compute_ssim(x0, x_recon)
        hf_energy = _compute_laplacian_energy(heatmap)
        res_std = float(np.std(heatmap))

        # 3. Diffusion Manifold Scoring Model
        # Diffusion images reconstruct with specific low/structured error distribution
        # Calibrated logistic score based on MSE, SSIM, and High-Frequency Energy
        z = (
            -15.0 * (mse - self.mse_threshold)
            + 8.0 * (ssim_val - self.ssim_threshold)
            - 12.0 * (hf_energy - 0.08)
            + 4.0 * (res_std - 0.15)
        )
        diffusion_prob = float(1.0 / (1.0 + math.exp(-np.clip(z, -10.0, 10.0))))

        is_diffusion = diffusion_prob >= 0.5
        confidence = diffusion_prob if is_diffusion else (1.0 - diffusion_prob)

        return DIREResult(
            is_diffusion_generated=is_diffusion,
            confidence=confidence,
            diffusion_score=diffusion_prob,
            mse=mse,
            mae=mae,
            psnr=psnr,
            ssim_score=ssim_val,
            high_freq_energy=hf_energy,
            residual_std=res_std,
            residual_map=heatmap,
            channel_residuals=channel_res,
            reconstructed_image=x_recon,
            metadata={
                "inversion_steps": num_steps_used,
                "mse_threshold": self.mse_threshold,
                "ssim_threshold": self.ssim_threshold,
            },
        )

    def analyze(
        self,
        input_data: Union[np.ndarray, Image.Image, str, Path],
        num_steps: Optional[int] = None,
    ) -> DIREResult:
        """
        End-to-end execution: DDIM inversion + reconstruction + DIRE residual analysis.
        """
        steps = num_steps or self.num_inversion_steps
        inv_result: DDIMInversionResult = self.inversion_engine.invert_and_reconstruct(
            input_data, num_steps=steps, save_trajectory=False
        )
        return self.analyze_residual(
            inv_result.original_image,
            inv_result.reconstructed_image,
            num_steps_used=steps,
        )


# Global singleton instance
dire_engine = DIREEngine()
