"""
TruthLens — Fusion & Localization Engine (orchestrator)
Author: Parth Maheshwari
Sprint: Architecture design for the Fusion & Localization Engine.

Single entry point: FusionInput -> Fusion Transformer -> (Anomaly Decoder,
Confidence Decoder) -> FusionResult.
"""
from __future__ import annotations

from backend.fusion.anomaly_decoder import AnomalyDecoder
from backend.fusion.confidence_decoder import ConfidenceDecoder
from backend.fusion.contracts import FusionInput, FusionResult
from backend.fusion.fusion_transformer import CrossModalFusionTransformer

FUSION_MODEL_VERSION = "fusion-scaffold-v0"


class FusionEngine:
    def __init__(self):
        self.transformer = CrossModalFusionTransformer()
        self.anomaly_decoder = AnomalyDecoder()
        self.confidence_decoder = ConfidenceDecoder()

    def run(self, fusion_input: FusionInput) -> FusionResult:
        branches = [name for name in ("image", "diffusion", "video", "noise")
                    if getattr(fusion_input, name) is not None]
        fused = self.transformer.fuse(fusion_input)  # raises if no branch present
        return FusionResult(
            sample_id=fusion_input.sample_id,
            branches_used=branches,
            anomaly_map=self.anomaly_decoder.decode(fusion_input, fused),
            confidence=self.confidence_decoder.decode(fusion_input, fused),
            fusion_model_version=FUSION_MODEL_VERSION,
        )


fusion_engine = FusionEngine()