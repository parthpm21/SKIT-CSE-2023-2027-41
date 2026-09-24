"""
TruthLens — Deterministic DDIM Forward Inversion & Reverse Reconstruction Engine
Author: Pratham Yadav
Sprint 2: CLIP:ViT probing & DDIM/DIRE reconstruction branch (User Story 2 - Part 2)

This module implements the deterministic Denoising Diffusion Implicit Models (DDIM)
inversion and reconstruction algorithm (Song et al., 2020). By running deterministic
forward ODE trajectory mapping (x_0 -> x_T) and reverse reconstruction (x_T -> x'_0),
it extracts latent noise representations and reconstructed approximations essential
for DIRE (Diffusion Reconstruction Error) residual maps.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image


@dataclass
class DDIMConfig:
    """Configuration parameters for DDIM Inversion & Reconstruction scheduler."""
    num_train_timesteps: int = 1000
    num_inversion_steps: int = 20
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "linear"  # "linear" or "scaled_linear" or "cosine"
    eta: float = 0.0  # eta=0 for purely deterministic ODE trajectory
    clip_sample: bool = True
    image_size: int = 224
    in_channels: int = 3
    base_dim: int = 32
    time_embed_dim: int = 128


@dataclass
class DDIMInversionResult:
    """Structured container holding inversion trajectory, latents, and reconstruction."""
    original_image: np.ndarray         # (3, H, W) in [-1, 1]
    latent_noise: np.ndarray           # (3, H, W) at t=T
    reconstructed_image: np.ndarray    # (3, H, W) in [-1, 1] at t=0
    inversion_error_l2: float
    reconstruction_psnr: float
    num_steps: int
    timesteps: List[int]
    intermediate_latents: Optional[List[np.ndarray]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inversion_error_l2": round(float(self.inversion_error_l2), 6),
            "reconstruction_psnr": round(float(self.reconstruction_psnr), 2),
            "num_steps": self.num_steps,
            "timesteps": self.timesteps,
            "metadata": self.metadata,
        }


class DDIMScheduler:
    """
    Mathematical scheduler for DDIM deterministic forward and reverse processes.
    Precomputes variance schedules, cumulative alpha products, and timestep mapping.
    """

    def __init__(self, config: DDIMConfig) -> None:
        self.config = config
        self.num_train_timesteps = config.num_train_timesteps
        self.num_inversion_steps = config.num_inversion_steps

        # Compute beta schedule
        if config.beta_schedule == "linear":
            self.betas = np.linspace(
                config.beta_start,
                config.beta_end,
                self.num_train_timesteps,
                dtype=np.float32,
            )
        elif config.beta_schedule == "scaled_linear":
            self.betas = np.linspace(
                config.beta_start ** 0.5,
                config.beta_end ** 0.5,
                self.num_train_timesteps,
                dtype=np.float32,
            ) ** 2
        elif config.beta_schedule == "cosine":
            steps = self.num_train_timesteps + 1
            x = np.linspace(0, self.num_train_timesteps, steps, dtype=np.float32)
            alphas_cumprod = np.cos(((x / self.num_train_timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = np.clip(betas, 0.0, 0.999).astype(np.float32)
        else:
            raise ValueError(f"Unknown beta schedule: {config.beta_schedule}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0).astype(np.float32)
        self.alphas_cumprod_prev = np.concatenate([np.array([1.0], dtype=np.float32), self.alphas_cumprod[:-1]])

        # Precompute square roots for ODE propagation
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)

        # Set discrete inference timesteps
        self.set_timesteps(self.num_inversion_steps)

    def set_timesteps(self, num_inversion_steps: int) -> None:
        """Discretizes total timesteps into a uniform subset for fast inversion."""
        self.num_inversion_steps = num_inversion_steps
        step_ratio = self.num_train_timesteps // self.num_inversion_steps
        # Timesteps in ascending order: [0, 50, 100, ..., 950]
        timesteps = (np.arange(0, num_inversion_steps) * step_ratio).round().astype(np.int64)
        self.timesteps = timesteps.tolist()

    def get_alpha_cumprod(self, t: int) -> float:
        """Returns bar{alpha}_t for a given timestep integer."""
        if t < 0:
            return 1.0
        t_clamped = min(max(t, 0), self.num_train_timesteps - 1)
        return float(self.alphas_cumprod[t_clamped])

    def forward_step(
        self,
        model_output_noise: np.ndarray,
        timestep: int,
        next_timestep: int,
        sample: np.ndarray,
    ) -> np.ndarray:
        """
        Computes forward ODE inversion step: x_t -> x_{t+1}
        
        Formula:
          hat{x}_0 = (x_t - sqrt{1 - alpha_t} * eps) / sqrt{alpha_t}
          x_{t+1}  = sqrt{alpha_{t+1}} * hat{x}_0 + sqrt{1 - alpha_{t+1}} * eps
        """
        alpha_t = self.get_alpha_cumprod(timestep)
        alpha_next = self.get_alpha_cumprod(next_timestep)

        # Predict x_0 from current x_t
        pred_x0 = (sample - math.sqrt(1.0 - alpha_t) * model_output_noise) / math.sqrt(alpha_t)
        if self.config.clip_sample:
            pred_x0 = np.clip(pred_x0, -1.0, 1.0)

        # Forward ODE step to next higher noise level
        next_sample = math.sqrt(alpha_next) * pred_x0 + math.sqrt(1.0 - alpha_next) * model_output_noise
        return next_sample

    def reverse_step(
        self,
        model_output_noise: np.ndarray,
        timestep: int,
        prev_timestep: int,
        sample: np.ndarray,
    ) -> np.ndarray:
        """
        Computes reverse deterministic reconstruction step: x_t -> x_{t-1}
        
        Formula (eta = 0):
          hat{x}_0 = (x_t - sqrt{1 - alpha_t} * eps) / sqrt{alpha_t}
          x_{t-1}  = sqrt{alpha_{t-1}} * hat{x}_0 + sqrt{1 - alpha_{t-1}} * eps
        """
        alpha_t = self.get_alpha_cumprod(timestep)
        alpha_prev = self.get_alpha_cumprod(prev_timestep)

        # Predict x_0 from current latent
        pred_x0 = (sample - math.sqrt(1.0 - alpha_t) * model_output_noise) / math.sqrt(alpha_t)
        if self.config.clip_sample:
            pred_x0 = np.clip(pred_x0, -1.0, 1.0)

        # Deterministic reverse step
        prev_sample = math.sqrt(alpha_prev) * pred_x0 + math.sqrt(1.0 - alpha_prev) * model_output_noise
        return prev_sample


class SinusoidalTimeEmbedding:
    """Generates sinusoidal positional embeddings for diffusion timesteps."""

    def __init__(self, embed_dim: int) -> None:
        self.embed_dim = embed_dim
        half_dim = embed_dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        self.frequencies = np.exp(np.arange(half_dim, dtype=np.float32) * -emb)

    def forward(self, timesteps: Union[int, np.ndarray]) -> np.ndarray:
        if isinstance(timesteps, (int, float)):
            t = np.array([timesteps], dtype=np.float32)
        else:
            t = timesteps.astype(np.float32)
        args = np.outer(t, self.frequencies)
        embedding = np.concatenate([np.sin(args), np.cos(args)], axis=-1)
        return embedding


class ConvBlock2D:
    """Lightweight 2D spatial convolution block with residual skip."""

    def __init__(self, in_ch: int, out_ch: int, seed: int = 42) -> None:
        self.in_ch = in_ch
        self.out_ch = out_ch
        rng = np.random.RandomState(seed)

        # 3x3 depthwise/separable filter kernel
        scale = 1.0 / math.sqrt(in_ch * 9)
        self.weight = rng.uniform(-scale, scale, (out_ch, in_ch, 3, 3)).astype(np.float32)
        self.bias = np.zeros(out_ch, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Applies 3x3 padded convolution on (B, C, H, W) or (C, H, W).
        """
        squeeze_batch = False
        if x.ndim == 3:
            x = np.expand_dims(x, axis=0)
            squeeze_batch = True

        B, C, H, W = x.shape
        padded = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="reflect")
        
        # Fast vectorized 3x3 spatial convolution
        out = np.zeros((B, self.out_ch, H, W), dtype=np.float32)
        for oc in range(self.out_ch):
            conv_sum = np.zeros((B, H, W), dtype=np.float32)
            for ic in range(self.in_ch):
                w = self.weight[oc, ic]
                for di in range(3):
                    for dj in range(3):
                        conv_sum += padded[:, ic, di : di + H, dj : dj + W] * w[di, dj]
            out[:, oc, :, :] = conv_sum + self.bias[oc]

        if squeeze_batch:
            out = out[0]
        return out


class DiffusionUNetNoiseEstimator:
    """
    Multi-scale noise estimation network epsilon_theta(x_t, t).
    Estimates the diffusion noise vector field conditioned on timestep t.
    """

    def __init__(self, config: DDIMConfig, seed: int = 1337) -> None:
        self.config = config
        self.time_embed = SinusoidalTimeEmbedding(config.time_embed_dim)
        
        rng = np.random.RandomState(seed)
        # Time projection layers
        t_scale = 1.0 / math.sqrt(config.time_embed_dim)
        self.time_proj_w = rng.uniform(-t_scale, t_scale, (config.time_embed_dim, config.base_dim)).astype(np.float32)
        self.time_proj_b = np.zeros(config.base_dim, dtype=np.float32)

        # Multi-scale spatial convolution layers
        self.conv_in = ConvBlock2D(config.in_channels, config.base_dim, seed=seed)
        self.conv_mid = ConvBlock2D(config.base_dim, config.base_dim, seed=seed + 1)
        self.conv_out = ConvBlock2D(config.base_dim, config.in_channels, seed=seed + 2)

    def forward(self, x: np.ndarray, timestep: int) -> np.ndarray:
        """
        Predicts noise residual epsilon_theta: (3, H, W) matching x.
        """
        # Time conditioning vector: (base_dim,)
        t_emb = self.time_embed.forward(timestep)[0]
        t_proj = np.matmul(t_emb, self.time_proj_w) + self.time_proj_b

        # First convolution
        h1 = np.tanh(self.conv_in.forward(x))
        # Add broadcasted time modulation: (base_dim, 1, 1)
        t_mod = t_proj[:, None, None]
        h1 = h1 + t_mod

        # Mid convolution with residual connection
        h2 = np.tanh(self.conv_mid.forward(h1)) + h1

        # Final projection to RGB noise space
        noise_pred = np.tanh(self.conv_out.forward(h2))
        return noise_pred


class DDIMInversionEngine:
    """
    High-level orchestration engine for DDIM Inversion and Reverse Reconstruction.
    Executes forward trajectory mapping x_0 -> x_T and reverse reconstruction x_T -> x'_0.
    """

    def __init__(self, config: Optional[DDIMConfig] = None, seed: int = 42) -> None:
        self.config = config or DDIMConfig()
        self.scheduler = DDIMScheduler(self.config)
        self.noise_estimator = DiffusionUNetNoiseEstimator(self.config, seed=seed)

    def _prepare_tensor(self, input_data: Union[np.ndarray, Image.Image, str, Path]) -> np.ndarray:
        """Loads and normalizes image into (3, H, W) float32 in [-1.0, 1.0]."""
        if isinstance(input_data, (str, Path)):
            img = Image.open(input_data).convert("RGB")
            arr = np.array(img, dtype=np.float32)
        elif isinstance(input_data, Image.Image):
            arr = np.array(input_data.convert("RGB"), dtype=np.float32)
        elif isinstance(input_data, np.ndarray):
            arr = input_data.astype(np.float32)
        else:
            raise TypeError(f"Unsupported input type for DDIM engine: {type(input_data)}")

        # Dimensions handling
        if arr.ndim == 2:  # Grayscale
            arr = np.stack([arr] * 3, axis=-1)

        if arr.ndim == 4:  # Batch dimension (1, C, H, W) or (1, H, W, C)
            arr = arr[0]

        if arr.ndim == 3 and (arr.shape[2] == 3 or arr.shape[2] == 1):
            if arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)
            arr = arr.transpose(2, 0, 1)

        C, H, W = arr.shape
        if C != 3:
            raise ValueError(f"Expected 3 color channels, got {C}")

        # Resize if necessary
        target_size = self.config.image_size
        if H != target_size or W != target_size:
            img_ch_last = arr.transpose(1, 2, 0)
            if img_ch_last.max() <= 1.0 and img_ch_last.min() >= -1.0:
                img_uint8 = np.clip((img_ch_last + 1.0) * 127.5, 0, 255).astype(np.uint8)
            else:
                img_uint8 = np.clip(img_ch_last, 0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_uint8).resize((target_size, target_size), Image.Resampling.BILINEAR)
            arr = np.array(pil_img, dtype=np.float32).transpose(2, 0, 1)

        # Scale into [-1.0, 1.0]
        if arr.max() > 1.0:
            arr = (arr / 127.5) - 1.0
        elif arr.min() >= 0.0 and arr.max() <= 1.0:
            arr = (arr * 2.0) - 1.0

        return np.clip(arr, -1.0, 1.0).astype(np.float32)

    def forward_invert(
        self,
        x_0: np.ndarray,
        num_steps: Optional[int] = None,
        save_trajectory: bool = False,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Executes forward deterministic DDIM inversion from x_0 -> x_T.
        
        Args:
            x_0: Initial image tensor (3, H, W) in [-1, 1].
            num_steps: Optional override for number of inversion steps.
            save_trajectory: Whether to record intermediate trajectory tensors.
        Returns:
            x_T: Latent noise representation at highest timestep.
            trajectory: List of intermediate latents if requested.
        """
        if num_steps is not None and num_steps != self.scheduler.num_inversion_steps:
            self.scheduler.set_timesteps(num_steps)

        timesteps = self.scheduler.timesteps
        cur_sample = x_0.copy()
        trajectory = [cur_sample.copy()] if save_trajectory else []

        # Ascending trajectory: t_0 -> t_1 -> ... -> t_{K-1}
        for i in range(len(timesteps) - 1):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]

            # Predict noise residual
            pred_noise = self.noise_estimator.forward(cur_sample, t_curr)
            # Take deterministic ODE step forward
            cur_sample = self.scheduler.forward_step(pred_noise, t_curr, t_next, cur_sample)

            if save_trajectory:
                trajectory.append(cur_sample.copy())

        return cur_sample, trajectory

    def reverse_reconstruct(
        self,
        x_T: np.ndarray,
        num_steps: Optional[int] = None,
        save_trajectory: bool = False,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Executes reverse deterministic DDIM reconstruction from x_T -> x'_0.
        
        Args:
            x_T: Latent noise tensor (3, H, W).
            num_steps: Optional override for number of reconstruction steps.
            save_trajectory: Whether to record intermediate reconstruction tensors.
        Returns:
            x_0_reconstructed: Reconstructed image tensor (3, H, W).
            trajectory: List of intermediate latents if requested.
        """
        if num_steps is not None and num_steps != self.scheduler.num_inversion_steps:
            self.scheduler.set_timesteps(num_steps)

        timesteps = self.scheduler.timesteps
        cur_sample = x_T.copy()
        trajectory = [cur_sample.copy()] if save_trajectory else []

        # Descending trajectory: t_{K-1} -> t_{K-2} -> ... -> 0
        for i in range(len(timesteps) - 1, 0, -1):
            t_curr = timesteps[i]
            t_prev = timesteps[i - 1]

            # Predict noise residual
            pred_noise = self.noise_estimator.forward(cur_sample, t_curr)
            # Take deterministic ODE step backward
            cur_sample = self.scheduler.reverse_step(pred_noise, t_curr, t_prev, cur_sample)

            if save_trajectory:
                trajectory.append(cur_sample.copy())

        return cur_sample, trajectory

    def invert_and_reconstruct(
        self,
        input_data: Union[np.ndarray, Image.Image, str, Path],
        num_steps: Optional[int] = None,
        save_trajectory: bool = False,
    ) -> DDIMInversionResult:
        """
        Performs complete forward inversion followed by reverse reconstruction.
        Calculates L2 reconstruction error and PSNR.
        """
        x_0 = self._prepare_tensor(input_data)
        
        # 1. Forward Inversion: x_0 -> x_T
        x_T, fwd_traj = self.forward_invert(x_0, num_steps=num_steps, save_trajectory=save_trajectory)

        # 2. Reverse Reconstruction: x_T -> x'_0
        x_0_recon, rev_traj = self.reverse_reconstruct(x_T, num_steps=num_steps, save_trajectory=save_trajectory)

        # Inversion and reconstruction error metrics
        l2_error = float(np.mean((x_0 - x_0_recon) ** 2))
        mse = max(l2_error, 1e-10)
        # PSNR on [-1, 1] range (peak signal range = 2.0)
        psnr = 10.0 * math.log10(4.0 / mse)

        all_latents = (fwd_traj + rev_traj) if save_trajectory else None

        return DDIMInversionResult(
            original_image=x_0,
            latent_noise=x_T,
            reconstructed_image=x_0_recon,
            inversion_error_l2=l2_error,
            reconstruction_psnr=psnr,
            num_steps=num_steps or self.scheduler.num_inversion_steps,
            timesteps=self.scheduler.timesteps,
            intermediate_latents=all_latents,
            metadata={
                "beta_schedule": self.config.beta_schedule,
                "eta": self.config.eta,
                "image_size": self.config.image_size,
                "clip_sample": self.config.clip_sample,
            },
        )


# Global singleton instance for easy import across microservices
ddim_inversion_engine = DDIMInversionEngine()
