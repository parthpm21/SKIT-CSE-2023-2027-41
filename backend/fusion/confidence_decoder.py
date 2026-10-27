"""
TruthLens — Confidence Decoder (verdict scaffold)
Author: Parth Maheshwari
Sprint: Architecture design for the Fusion & Localization Engine.

Answers "IS the media fake, and how sure are we?". Output follows the
ConfidenceResult contract (label + confidence in [0, 1]).

Scaffold only: averages the branch scores that are present. A learned
decoder over the fused embedding replaces this in the next sprint.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from backend.fusion.contracts import ConfidenceResult, FusionInput


@dataclass
class ConfidenceDecoderConfig:
    fake_threshold: float = 0.5


class ConfidenceDecoder:
    """Pools branch evidence into one real/fake verdict."""

    def __init__(self, config: ConfidenceDecoderConfig = ConfidenceDecoderConfig()):
        self.config = config

    @staticmethod
    def _branch_scores(fusion_input: FusionInput) -> List[float]:
        scores: List[float] = []
        if fusion_input.image is not None:
            scores.append(fusion_input.image.probe_score)
        if fusion_input.diffusion is not None:
            scores.append(fusion_input.diffusion.diffusion_score)
        if fusion_input.video is not None and fusion_input.video.frame_scores:
            scores.append(float(np.mean(fusion_input.video.frame_scores)))
        return [float(np.clip(s, 0.0, 1.0)) for s in scores]

    def decode(self, fusion_input: FusionInput, fused_embedding: np.ndarray) -> ConfidenceResult:
        scores = self._branch_scores(fusion_input)
        if not scores:
            raise ValueError("Cannot score: FusionInput has no usable branch data")

        fake_prob = float(np.mean(scores))
        is_fake = fake_prob >= self.config.fake_threshold
        # Confidence = how far the probability sits on the winning side.
        confidence = fake_prob if is_fake else 1.0 - fake_prob
        if fusion_input.noise is not None:
            scores.append(fusion_input.noise.temporal_inconsistency_score)
        return ConfidenceResult(label="fake" if is_fake else "real", confidence=confidence)