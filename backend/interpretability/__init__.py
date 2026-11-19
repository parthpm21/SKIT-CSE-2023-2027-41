"""
TruthLens Interpretability Module.

This package provides visual explainability utilities including Grad-CAM,
Attention Rollout, and multimodal interpretability integrations for deep
learning forensic analysis.
"""

from backend.interpretability.gradcam import compute_gradcam_heatmap, GradCAMEngine
from backend.interpretability.attention_rollout import (
    compute_attention_rollout,
    rollout_to_heatmap,
    compute_attention_rollout_heatmap,
    AttentionRolloutEngine,
)
from backend.interpretability.image_integration import (
    ImageExplanation,
    ImageInterpretabilityPipeline,
    explain_image,
    unpad_and_resize_heatmap,
)

__all__ = [
    "compute_gradcam_heatmap",
    "GradCAMEngine",
    "compute_attention_rollout",
    "rollout_to_heatmap",
    "compute_attention_rollout_heatmap",
    "AttentionRolloutEngine",
    "ImageExplanation",
    "ImageInterpretabilityPipeline",
    "explain_image",
    "unpad_and_resize_heatmap",
]
