"""
TruthLens — Dual-Branch Image Detection Service (CLIP:ViT + DIRE)
Author: Pratham Yadav
Sprint 2: CLIP:ViT probing & DDIM/DIRE reconstruction branch (User Story 2 - Part 4)

This microservice combines frozen semantic representations from the CLIP:ViT Probing
backbone with reconstruction anomaly statistics from the DDIM/DIRE Residual engine.
It produces unified authenticity verdicts, synthetic subtype attribution (Diffusion vs GAN),
calibrated confidence scores, and composite explainable anomaly heatmaps.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from backend.inference.clip_vit_probe import (
    CLIPProbeResult,
    FrozenCLIPViTProbe,
    clip_vit_probe_engine,
)
from backend.inference.dire_engine import (
    DIREResult,
    DIREEngine,
    dire_engine,
)


@dataclass
class DualBranchVerdict:
    """Unified verdict from the dual-branch image detection microservice."""
    label: str                   # "real" | "fake"
    confidence: float            # [0.0, 1.0]
    synthetic_subtype: str       # "authentic" | "diffusion" | "gan" | "hybrid"
    model_version: str
    clip_probe_score: float      # Synthetic probability from CLIP:ViT
    dire_score: float            # Diffusion probability from DIRE
    dire_mse: float
    dire_ssim: float
    fused_heatmap: np.ndarray    # (224, 224) composite visual anomaly map
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "synthetic_subtype": self.synthetic_subtype,
            "model_version": self.model_version,
            "clip_probe_score": round(float(self.clip_probe_score), 4),
            "dire_score": round(float(self.dire_score), 4),
            "dire_mse": round(float(self.dire_mse), 6),
            "dire_ssim": round(float(self.dire_ssim), 4),
            "details": self.details,
        }


class CLIPDIREImageService:
    """
    Dual-branch image deepfake & synthetic media detection microservice.
    Coordinates CLIP:ViT semantic probing and DDIM/DIRE residual reconstruction.
    """

    def __init__(
        self,
        clip_probe: Optional[FrozenCLIPViTProbe] = None,
        dire_analyzer: Optional[DIREEngine] = None,
        clip_weight: float = 0.55,
        dire_weight: float = 0.45,
    ) -> None:
        self.clip_probe = clip_probe or clip_vit_probe_engine
        self.dire_analyzer = dire_analyzer or dire_engine
        self.clip_weight = clip_weight
        self.dire_weight = dire_weight
        self.model_version = "clip-vit-dire-dual-v1"

    def _fuse_heatmaps(
        self,
        clip_attn_map: np.ndarray,   # (14, 14)
        dire_residual_map: np.ndarray, # (H, W) e.g. (224, 224)
        target_size: Tuple[int, int] = (224, 224),
    ) -> np.ndarray:
        """
        Combines low-resolution CLIP attention heatmap with high-resolution DIRE residual map.
        """
        # Upsample 14x14 attention map to target size via bilinear interpolation
        attn_pil = Image.fromarray((clip_attn_map * 255.0).astype(np.uint8))
        attn_upsampled = np.array(
            attn_pil.resize(target_size, Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0

        # Resize DIRE map if needed
        if dire_residual_map.shape != target_size:
            dire_pil = Image.fromarray((dire_residual_map * 255.0).astype(np.uint8))
            dire_upsampled = np.array(
                dire_pil.resize(target_size, Image.Resampling.BILINEAR),
                dtype=np.float32,
            ) / 255.0
        else:
            dire_upsampled = dire_residual_map.astype(np.float32)

        # Weighted blend
        fused = self.clip_weight * attn_upsampled + self.dire_weight * dire_upsampled
        
        # Min-max normalization
        f_min = float(np.min(fused))
        f_max = float(np.max(fused))
        if f_max > f_min:
            fused_norm = (fused - f_min) / (f_max - f_min)
        else:
            fused_norm = np.zeros_like(fused)

        return fused_norm.astype(np.float32)

    def predict(
        self,
        input_data: Union[np.ndarray, Image.Image, str, Path],
        num_dire_steps: int = 15,
    ) -> DualBranchVerdict:
        """
        Executes dual-branch analysis:
        1. CLIP:ViT generator-agnostic semantic probing.
        2. DDIM deterministic inversion + DIRE error reconstruction.
        3. Multi-branch score fusion and subtype attribution.
        """
        # Branch 1: Frozen CLIP:ViT Probing
        clip_res: CLIPProbeResult = self.clip_probe.probe(input_data)

        # Branch 2: DDIM Inversion + DIRE Residual Engine
        dire_res: DIREResult = self.dire_analyzer.analyze(input_data, num_steps=num_dire_steps)

        # Multi-Branch Fusion Decision Logic
        # Calculate ensemble synthetic score
        p_clip = clip_res.synthetic_prob
        p_dire = dire_res.diffusion_score

        # Dynamic confidence-weighted gating
        w_c = self.clip_weight * (1.0 + abs(p_clip - 0.5))
        w_d = self.dire_weight * (1.0 + abs(p_dire - 0.5))
        fused_synth_score = (w_c * p_clip + w_d * p_dire) / (w_c + w_d)

        is_fake = fused_synth_score >= 0.50
        label = "fake" if is_fake else "real"
        confidence = float(fused_synth_score if is_fake else (1.0 - fused_synth_score))

        # Synthetic Subtype Attribution
        if not is_fake:
            subtype = "authentic"
        elif dire_res.is_diffusion_generated and p_dire > 0.60:
            subtype = "diffusion"
        elif clip_res.is_synthetic and p_dire <= 0.45:
            subtype = "gan"
        else:
            subtype = "hybrid"

        # Composite spatial anomaly heatmap
        fused_map = self._fuse_heatmaps(
            clip_res.patch_attention_map,
            dire_res.residual_map,
            target_size=(224, 224),
        )

        details = {
            "synthetic_subtype": subtype,
            "clip_probe": clip_res.to_dict(),
            "dire": dire_res.to_dict(),
            "fusion_weights": {
                "clip_weight": round(float(w_c / (w_c + w_d)), 4),
                "dire_weight": round(float(w_d / (w_c + w_d)), 4),
            },
            "heatmap_preview": {
                "mean_anomaly": round(float(np.mean(fused_map)), 4),
                "peak_anomaly": round(float(np.max(fused_map)), 4),
            },
        }

        return DualBranchVerdict(
            label=label,
            confidence=confidence,
            synthetic_subtype=subtype,
            model_version=self.model_version,
            clip_probe_score=p_clip,
            dire_score=p_dire,
            dire_mse=dire_res.mse,
            dire_ssim=dire_res.ssim_score,
            fused_heatmap=fused_map,
            details=details,
        )


# Global singleton instance
clip_dire_service = CLIPDIREImageService()
