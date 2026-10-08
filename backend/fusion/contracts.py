"""
TruthLens — Fusion & Localization Engine API Contracts
Author: Parth Maheshwari / TruthLens Core Team
Sprint: Architecture design for the Fusion & Localization Engine.

Defines Pydantic data schemas for multi-branch inputs (Image CLIP:ViT, Diffusion DIRE,
Video Xception/EfficientNet) and decoder outputs (AnomalyMap, ConfidenceResult, FusionResult).
"""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class ImageBranchInput(BaseModel):
    """Features from the frozen CLIP:ViT probing branch."""
    cls_embedding: List[float] = Field(..., description="CLS token embedding vector (e.g. 768-dim)")
    probe_score: float = Field(..., description="Synthetic probability from linear/MLP probe [0, 1]")


class DiffusionBranchInput(BaseModel):
    """Features from the DDIM / DIRE reconstruction branch."""
    dire_residual_map: List[List[float]] = Field(..., description="2D spatial residual error heatmap")
    diffusion_score: float = Field(..., description="Diffusion manifold probability score [0, 1]")
    mse: float = Field(..., description="Reconstruction mean squared error")


class VideoBranchInput(BaseModel):
    """Features from the Xception / EfficientNet video deepfake branch."""
    frame_embeddings: List[List[float]] = Field(..., description="Per-frame feature embedding vectors")
    frame_scores: List[float] = Field(..., description="Per-frame anomaly scores")


class FusionInput(BaseModel):
    """Multi-branch input payload passed to the Cross-Modal Fusion Transformer."""
    sample_id: str = Field(..., description="Unique media/sample identifier")
    image: Optional[ImageBranchInput] = None
    diffusion: Optional[DiffusionBranchInput] = None
    video: Optional[VideoBranchInput] = None


class AnomalyMap(BaseModel):
    """Output of the Anomaly Decoder — spatial or temporal localization."""
    granularity: str = Field(..., description="'pixel' for spatial image heatmap or 'frame' for temporal sequence")
    values: List[float] = Field(..., description="Flattened heatmap or per-frame score sequence")
    shape: List[int] = Field(..., description="Original (H, W) for pixel maps, or [T] for frame sequences")


class ConfidenceResult(BaseModel):
    """Output of the Confidence Decoder — scalar authenticity verdict."""
    label: str = Field(..., description="'real' | 'fake'")
    confidence: float = Field(..., description="Verdict confidence in range [0, 1]")


class FusionResult(BaseModel):
    """Unified output from the Cross-Modal Fusion & Localization Engine."""
    sample_id: str
    branches_used: List[str]
    anomaly_map: AnomalyMap
    confidence: ConfidenceResult
    fusion_model_version: str