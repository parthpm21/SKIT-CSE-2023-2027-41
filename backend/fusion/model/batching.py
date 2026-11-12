"""
TruthLens — Batching: FusionInput / FusionSample -> fixed-shape arrays and tensors
Author: Parth Maheshwari
Sprint: Development of Cross-Modal Fusion Transformer with dual Decoders.

Every sample is padded to the same layout so a batch is a dict of arrays:

  features                                   shape
    clip_vec / clip_score / clip_present     (B, clip_dim) (B, 1) (B,)
    dire_map / dire_scalars / dire_present   (B, 1, S, S) (B, 2) (B,)
    vid_emb / vid_scores / vid_valid         (B, T, video_dim) (B, T) (B, T)
    vid_present                              (B,)
    noise_scalars / noise_energy             (B, 3) (B, T)
    noise_valid / noise_present              (B, T) (B,)
  targets (collate_samples only)
    verdict (B,)  pixel_mask (B, 1, S, S)  pixel_valid (B,)
    frame_labels (B, T)  frame_valid (B, T)

S = config.map_size, T = config.max_frames. The array builders are pure NumPy
(testable without PyTorch); to_tensors() converts them for the model.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from backend.fusion.contracts import FusionInput
from backend.fusion.model.config import FusionModelConfig
from backend.fusion.training.schema import FusionSample

FEATURE_KEYS = (
    "clip_vec", "clip_score", "clip_present", "dire_map", "dire_scalars", "dire_present",
    "vid_emb", "vid_scores", "vid_valid", "vid_present",
    "noise_scalars", "noise_energy", "noise_valid", "noise_present",
)
PRESENCE_KEYS = ("clip_present", "dire_present", "vid_present", "noise_present")


def _energy(x):
    """Noise energies / MSE are tiny (~0.01-0.05); log1p(100x) brings them to O(1)."""
    return np.log1p(np.asarray(x, dtype=np.float32) * 100.0)


def fit_dim(vec, dim: int) -> np.ndarray:
    """Zero-pad or truncate a vector to exactly `dim` entries."""
    v = np.asarray(vec, dtype=np.float32).ravel()
    out = np.zeros(dim, dtype=np.float32)
    n = min(dim, v.size)
    out[:n] = v[:n]
    return out


def time_index(n: int, t: int) -> np.ndarray:
    """Indices selecting at most t of n frames (uniformly sub-sampled when n > t)."""
    if n <= t:
        return np.arange(n)
    return np.linspace(0, n - 1, t).round().astype(int)


def resize_map(arr, size: int) -> np.ndarray:
    """Bilinear resize of a 2D map to (size, size), pure NumPy (separable interpolation)."""
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim != 2 or a.size == 0:
        raise ValueError("expected a non-empty 2D map")
    if a.shape == (size, size):
        return a

    def interp_axis(x, new_n, axis):
        old_n = x.shape[axis]
        if old_n == 1:
            return np.repeat(x, new_n, axis=axis)
        # align_corners=False sampling positions
        pos = (np.arange(new_n) + 0.5) * old_n / new_n - 0.5
        pos = np.clip(pos, 0, old_n - 1)
        lo = np.floor(pos).astype(int)
        hi = np.minimum(lo + 1, old_n - 1)
        w = (pos - lo).astype(np.float32)
        x_lo, x_hi = np.take(x, lo, axis=axis), np.take(x, hi, axis=axis)
        shape = [1, 1]
        shape[axis] = new_n
        w = w.reshape(shape)
        return x_lo * (1 - w) + x_hi * w

    return interp_axis(interp_axis(a, size, 0), size, 1).astype(np.float32)


def collate_inputs_np(inputs: Sequence[FusionInput], cfg: FusionModelConfig) -> Dict[str, np.ndarray]:
    B, T, S = len(inputs), cfg.max_frames, cfg.map_size
    b = {
        "clip_vec": np.zeros((B, cfg.clip_dim), np.float32),
        "clip_score": np.zeros((B, 1), np.float32),
        "clip_present": np.zeros(B, bool),
        "dire_map": np.zeros((B, 1, S, S), np.float32),
        "dire_scalars": np.zeros((B, 2), np.float32),
        "dire_present": np.zeros(B, bool),
        "vid_emb": np.zeros((B, T, cfg.video_dim), np.float32),
        "vid_scores": np.zeros((B, T), np.float32),
        "vid_valid": np.zeros((B, T), bool),
        "vid_present": np.zeros(B, bool),
        "noise_scalars": np.zeros((B, 3), np.float32),
        "noise_energy": np.zeros((B, T), np.float32),
        "noise_valid": np.zeros((B, T), bool),
        "noise_present": np.zeros(B, bool),
    }
    for i, fi in enumerate(inputs):
        if fi.image is not None:
            b["clip_vec"][i] = fit_dim(fi.image.cls_embedding, cfg.clip_dim)
            b["clip_score"][i, 0] = fi.image.probe_score
            b["clip_present"][i] = True
        if fi.diffusion is not None:
            b["dire_map"][i, 0] = resize_map(fi.diffusion.dire_residual_map, S)
            b["dire_scalars"][i] = [fi.diffusion.diffusion_score, float(_energy(fi.diffusion.mse))]
            b["dire_present"][i] = True
        if fi.video is not None and len(fi.video.frame_scores) > 0:
            idx = time_index(len(fi.video.frame_scores), T)
            emb = np.asarray(fi.video.frame_embeddings, dtype=np.float32)[idx]
            for j in range(len(idx)):
                b["vid_emb"][i, j] = fit_dim(emb[j], cfg.video_dim)
            b["vid_scores"][i, :len(idx)] = np.asarray(fi.video.frame_scores, dtype=np.float32)[idx]
            b["vid_valid"][i, :len(idx)] = True
            b["vid_present"][i] = True
        if fi.noise is not None:
            n = fi.noise
            b["noise_scalars"][i] = [n.temporal_inconsistency_score, float(_energy(n.mean_noise_energy)),
                                     n.face_bg_noise_discrepancy]
            b["noise_present"][i] = True
            if n.frame_noise_energies:
                idx = time_index(len(n.frame_noise_energies), T)
                b["noise_energy"][i, :len(idx)] = _energy(np.asarray(n.frame_noise_energies)[idx])
                b["noise_valid"][i, :len(idx)] = True
    return b


def collate_samples_np(samples: Sequence[FusionSample], cfg: FusionModelConfig) -> Dict[str, np.ndarray]:
    b = collate_inputs_np([s.fusion_input for s in samples], cfg)
    B, T, S = len(samples), cfg.max_frames, cfg.map_size
    b["verdict"] = np.array([s.verdict for s in samples], dtype=np.float32)
    b["pixel_mask"] = np.zeros((B, 1, S, S), np.float32)
    b["pixel_valid"] = np.zeros(B, bool)
    b["frame_labels"] = np.zeros((B, T), np.float32)
    b["frame_valid"] = np.zeros((B, T), bool)
    for i, s in enumerate(samples):
        if s.pixel_mask is not None:
            b["pixel_mask"][i, 0] = s.pixel_mask
            b["pixel_valid"][i] = True
        if s.frame_labels is not None and len(s.frame_labels) > 0:
            idx = time_index(len(s.frame_labels), T)
            b["frame_labels"][i, :len(idx)] = s.frame_labels[idx]
            b["frame_valid"][i, :len(idx)] = True
    return b


def to_tensors(batch: Dict[str, np.ndarray], device: str = "cpu"):
    """NumPy batch -> dict of torch tensors (imports torch lazily)."""
    import torch
    return {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in batch.items()}


def collate_inputs(inputs, cfg, device: str = "cpu"):
    return to_tensors(collate_inputs_np(inputs, cfg), device)


def collate_samples(samples, cfg, device: str = "cpu"):
    return to_tensors(collate_samples_np(samples, cfg), device)