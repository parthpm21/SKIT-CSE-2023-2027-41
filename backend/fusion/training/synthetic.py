"""
TruthLens — Synthetic fusion samples (SMOKE TESTS ONLY)
Author: Parth Maheshwari

Generates small FusionSample sets with planted structure (fake = higher scores,
a localized residual blob, a manipulated frame range). They let the training
pipeline, CI and the demo run end-to-end without the benchmark data.
Results measured on this data say nothing about real deepfake detection - never
report them as model accuracy.
"""
from __future__ import annotations

from typing import List

import numpy as np

from backend.fusion.contracts import (
    DiffusionBranchInput, FusionInput, ImageBranchInput, NoiseBranchInput, VideoBranchInput,
)
from backend.fusion.training.schema import MASK_SIZE, FusionSample


def _blob(rng: np.random.Generator, size: int = MASK_SIZE):
    cy, cx = rng.integers(6, size - 6, size=2)
    r = int(rng.integers(4, 8))
    yy, xx = np.mgrid[:size, :size]
    return (((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r).astype(np.float32)


def _image_sample(rng, idx, clip_dim, with_dire):
    fake = int(rng.random() < 0.5)
    signature = np.linspace(-1.0, 1.0, clip_dim, dtype=np.float32)
    emb = rng.normal(0, 0.3, clip_dim).astype(np.float32) + (0.8 * signature if fake else 0.0)
    probe = float(np.clip(rng.normal(0.75 if fake else 0.25, 0.15), 0.0, 1.0))
    image = ImageBranchInput(cls_embedding=emb.tolist(), probe_score=probe)
    mask = _blob(rng) if fake else np.zeros((MASK_SIZE, MASK_SIZE), dtype=np.float32)
    diffusion = None
    if with_dire:
        resid = np.clip(rng.normal(0.1, 0.05, (MASK_SIZE, MASK_SIZE)) + 0.6 * mask, 0.0, 1.0)
        score = float(np.clip(rng.normal(0.7 if fake else 0.2, 0.15), 0.0, 1.0))
        diffusion = DiffusionBranchInput(
            dire_residual_map=resid.astype(np.float32).tolist(), diffusion_score=score,
            mse=float(0.02 + 0.03 * fake),
        )
    return FusionSample(
        FusionInput(sample_id=f"syn_img_{idx}", image=image, diffusion=diffusion),
        verdict=fake, pixel_mask=mask, source="synthetic",
    )


def _video_sample(rng, idx, video_dim):
    fake = int(rng.random() < 0.5)
    t = int(rng.integers(8, 17))
    labels = np.zeros(t, dtype=np.float32)
    if fake:
        start = int(rng.integers(0, t // 2))
        labels[start:start + int(rng.integers(3, t - start + 1))] = 1.0
    signature = np.linspace(1.0, -1.0, video_dim, dtype=np.float32)
    emb = rng.normal(0, 0.3, (t, video_dim)).astype(np.float32) + 0.8 * labels[:, None] * signature
    scores = np.clip(rng.normal(0.2, 0.1, t) + 0.55 * labels, 0.0, 1.0)
    energies = np.clip(rng.normal(0.02, 0.005, t) + 0.03 * labels, 0.0, None)
    video = VideoBranchInput(frame_embeddings=emb.tolist(), frame_scores=scores.tolist())
    noise = NoiseBranchInput(
        temporal_inconsistency_score=float(np.clip(0.2 + 0.5 * labels.mean() + rng.normal(0, 0.05), 0, 1)),
        mean_noise_energy=float(energies.mean()), face_bg_noise_discrepancy=float(0.1 + 0.3 * fake),
        frame_noise_energies=energies.tolist(),
    )
    return FusionSample(
        FusionInput(sample_id=f"syn_vid_{idx}", video=video, noise=noise),
        verdict=int(labels.max() > 0), frame_labels=labels, source="synthetic",
    )


def make_synthetic_samples(n: int, seed: int = 0, clip_dim: int = 768, video_dim: int = 512) -> List[FusionSample]:
    """Mix of image (CLIP only / CLIP+DIRE) and video (video+noise) samples."""
    rng = np.random.default_rng(seed)
    out: List[FusionSample] = []
    for i in range(n):
        if i % 3 == 2:
            out.append(_video_sample(rng, i, video_dim))
        else:
            out.append(_image_sample(rng, i, clip_dim, with_dire=(i % 3 == 0)))
    return out