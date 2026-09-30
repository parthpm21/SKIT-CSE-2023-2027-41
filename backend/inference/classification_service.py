"""
TruthLens — Image Classification Microservice Wrapper
Author: Parth Maheshwari / Pratham Yadav
Sprint 2: Backend integration of classification and video deepfake services.

Exposes the CLIP:ViT Probing & DDIM/DIRE Dual-Branch image detection engine
under a stable classification service contract compatible with FastAPI and the
inference routing layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from backend.inference.clip_dire_service import (
    CLIPDIREImageService,
    DualBranchVerdict,
    clip_dire_service,
)


@dataclass
class ClassificationVerdict:
    """Standard image classification verdict schema."""
    label: str          # "real" | "fake"
    confidence: float
    model_version: str
    details: Dict[str, Any] = field(default_factory=dict)


class ClassificationService:
    """Wraps CLIP:ViT + DIRE dual-branch engine behind classification service interface."""

    def __init__(self, backend_service: Optional[CLIPDIREImageService] = None) -> None:
        self.service = backend_service or clip_dire_service
        self.model_version = self.service.model_version

    def predict(
        self,
        image_tensor: Union[np.ndarray, str, Path],
        num_dire_steps: int = 15,
    ) -> ClassificationVerdict:
        """
        Runs dual-branch prediction on preprocessed (C, H, W) or (B, C, H, W) image tensor.
        """
        verdict: DualBranchVerdict = self.service.predict(
            image_tensor, num_dire_steps=num_dire_steps
        )
        return ClassificationVerdict(
            label=verdict.label,
            confidence=verdict.confidence,
            model_version=verdict.model_version,
            details=verdict.details,
        )


# Global singleton instance for inference router
classification_service = ClassificationService()
