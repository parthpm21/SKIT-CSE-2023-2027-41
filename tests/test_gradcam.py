"""
TruthLens - Grad-CAM Utility Tests
Author: TruthLens Development Team
Sprint 2: Interpretability (Part 2 - Grad-CAM Implementation / Readiness)

Tests for GradCAMEngine interface behavior, error handling, and model availability checks.
"""

import pytest
from backend.interpretability.gradcam import GradCAMEngine, compute_gradcam_heatmap


def test_gradcam_engine_initialization():
    """Verify GradCAMEngine initializes with default None model and target layer."""
    engine = GradCAMEngine()
    assert engine.model is None
    assert engine.target_layer is None
    assert engine.activations is None
    assert engine.gradients is None


def test_gradcam_engine_is_available_without_model():
    """Verify is_available() returns False when no PyTorch model is loaded."""
    engine = GradCAMEngine(model=None, target_layer=None)
    assert engine.is_available() is False


def test_gradcam_engine_raises_runtime_error_without_model():
    """Verify generate_heatmap() raises RuntimeError when no suitable model is present."""
    engine = GradCAMEngine(model=None, target_layer=None)
    with pytest.raises(RuntimeError) as exc_info:
        engine.generate_heatmap(input_tensor=None)
    assert "Grad-CAM cannot be executed" in str(exc_info.value)
    assert "No supported model/inference pipeline" in str(exc_info.value)


def test_compute_gradcam_heatmap_wrapper_without_model():
    """Verify convenience wrapper compute_gradcam_heatmap raises RuntimeError without model."""
    with pytest.raises(RuntimeError) as exc_info:
        compute_gradcam_heatmap(model=None, target_layer=None, input_tensor=None)
    assert "No supported model/inference pipeline" in str(exc_info.value)
