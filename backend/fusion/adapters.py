"""
TruthLens — Fusion Feature Adapters
Author: Parth Maheshwari
Sprint: Development of Cross-Modal Fusion Transformer with dual Decoders.

Converts the result objects of the upstream branches (CLIP:ViT probe, DIRE
engine, video backbone, noise-fingerprint extractor) into the FusionInput
contract. Duck-typed on purpose: this module imports none of the upstream
modules, so the fusion package stays light and free of circular imports.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from backend.fusion.contracts import (
    DiffusionBranchInput,
    FusionInput,
    ImageBranchInput,
    NoiseBranchInput,
    VideoBranchInput,
)

# Residual maps are block-averaged down to at most this size so the JSON
# payload and the model's patchifier see a fixed, small grid.
MAP_SIZE = 32


def downsample_map(arr: np.ndarray, size: int = MAP_SIZE) -> np.ndarray:
    """Block-average a 2D map to (size, size); smaller maps are returned unchanged."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("expected a non-empty 2D map")
    h, w = arr.shape
    if h <= size and w <= size:
        return arr
    ys = np.array_split(np.arange(h), min(size, h))
    xs = np.array_split(np.arange(w), min(size, w))
    out = np.empty((len(ys), len(xs)), dtype=np.float32)
    for i, yi in enumerate(ys):
        for j, xj in enumerate(xs):
            out[i, j] = arr[np.ix_(yi, xj)].mean()
    return out


def clip_to_branch(clip_result) -> ImageBranchInput:
    """CLIPProbeResult -> ImageBranchInput (synthetic probability becomes probe_score)."""
    return ImageBranchInput(
        cls_embedding=np.asarray(clip_result.cls_embedding, dtype=np.float32).ravel().tolist(),
        probe_score=float(clip_result.synthetic_prob),
    )


def dire_to_branch(dire_result) -> DiffusionBranchInput:
    """DIREResult -> DiffusionBranchInput with a down-sampled residual map."""
    return DiffusionBranchInput(
        dire_residual_map=downsample_map(dire_result.residual_map).tolist(),
        diffusion_score=float(dire_result.diffusion_score),
        mse=float(dire_result.mse),
    )


def video_to_branch(frame_embeddings: np.ndarray, frame_scores: Sequence[float]) -> VideoBranchInput:
    """Per-frame embeddings (T, D) + per-frame scores (T,) -> VideoBranchInput."""
    emb = np.asarray(frame_embeddings, dtype=np.float32)
    if emb.ndim != 2:
        raise ValueError("frame_embeddings must be 2D (T, D)")
    if emb.shape[0] != len(frame_scores):
        raise ValueError(f"{emb.shape[0]} embeddings but {len(frame_scores)} frame scores")
    return VideoBranchInput(frame_embeddings=emb.tolist(), frame_scores=[float(s) for s in frame_scores])


def noise_to_branch(noise_result) -> NoiseBranchInput:
    """NoiseFingerprintResult -> NoiseBranchInput."""
    return NoiseBranchInput(
        temporal_inconsistency_score=float(noise_result.temporal_inconsistency_score),
        mean_noise_energy=float(noise_result.mean_noise_energy),
        face_bg_noise_discrepancy=float(noise_result.face_bg_noise_discrepancy),
        frame_noise_energies=[float(e) for e in noise_result.frame_noise_energies],
    )


def build_fusion_input(
    sample_id: str,
    clip_result=None,
    dire_result=None,
    frame_embeddings: Optional[np.ndarray] = None,
    frame_scores: Optional[Sequence[float]] = None,
    noise_result=None,
) -> FusionInput:
    """Assemble a FusionInput from whichever upstream results are available."""
    video = None
    if frame_embeddings is not None and frame_scores is not None:
        video = video_to_branch(frame_embeddings, frame_scores)
    fusion_input = FusionInput(
        sample_id=sample_id,
        image=clip_to_branch(clip_result) if clip_result is not None else None,
        diffusion=dire_to_branch(dire_result) if dire_result is not None else None,
        video=video,
        noise=noise_to_branch(noise_result) if noise_result is not None else None,
    )
    if not any((fusion_input.image, fusion_input.diffusion, fusion_input.video, fusion_input.noise)):
        raise ValueError("build_fusion_input needs at least one upstream result")
    return fusion_input