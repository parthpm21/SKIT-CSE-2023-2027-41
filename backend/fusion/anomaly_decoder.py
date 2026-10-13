"""
TruthLens — Anomaly Decoder (localization scaffold)
Author: Parth Maheshwari
Sprint: Architecture design for the Fusion & Localization Engine.

Answers "WHERE does the media look manipulated?". Output follows the
AnomalyMap contract: a pixel-level heatmap for images, or a frame-level
score sequence for video.

Scaffold only: values are derived from the upstream branch outputs, not
from a trained decoder. Learned upsampling lands in the next sprint.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.fusion.contracts import AnomalyMap, FusionInput


@dataclass
class AnomalyDecoderConfig:
    # Grid used when an image has no spatial DIRE map to localize from.
    fallback_grid: int = 14


class AnomalyDecoder:
    """Turns a fused embedding + branch inputs into a localization map."""

    def __init__(self, config: AnomalyDecoderConfig = AnomalyDecoderConfig()):
        self.config = config

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        """Scale to [0, 1]; a flat map stays flat instead of dividing by zero."""
        lo, hi = float(arr.min()), float(arr.max())
        if hi - lo < 1e-8:
            return np.clip(arr, 0.0, 1.0)
        return (arr - lo) / (hi - lo)

    def decode(self, fusion_input: FusionInput, fused_embedding: np.ndarray) -> AnomalyMap:
        # 1) Best case: DIRE residual map gives a real spatial heatmap.
        if fusion_input.diffusion is not None:
            residual = np.asarray(fusion_input.diffusion.dire_residual_map, dtype=np.float32)
            if residual.ndim == 2 and residual.size > 0:
                heat = self._normalize(residual)
                return AnomalyMap(granularity="pixel", values=heat.flatten().tolist(),
                                  shape=list(heat.shape))

        # 2) Image with no spatial map: uniform grid at the probe score
        #    (placeholder until a trained decoder can upsample the embedding).
        if fusion_input.image is not None:
            g = self.config.fallback_grid
            score = float(np.clip(fusion_input.image.probe_score, 0.0, 1.0))
            return AnomalyMap(granularity="pixel", values=[score] * (g * g), shape=[g, g])

        # 3) Video: one anomaly score per frame.
        if fusion_input.video is not None and fusion_input.video.frame_scores:
            scores = np.clip(np.asarray(fusion_input.video.frame_scores, dtype=np.float32), 0.0, 1.0)
            return AnomalyMap(granularity="frame", values=scores.tolist(), shape=[int(scores.shape[0])])

        raise ValueError("Cannot localize: FusionInput has no usable branch data")