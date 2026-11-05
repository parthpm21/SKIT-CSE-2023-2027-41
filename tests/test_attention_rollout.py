"""
TruthLens - Attention Rollout Utility Tests
Author: TruthLens Development Team
Sprint 2: Interpretability (Part 3 - Attention Rollout Implementation)

Tests for Attention Rollout mathematical utility and AttentionRolloutEngine interface.
Uses small, explicitly synthetic attention matrices labeled as test data.
"""

import numpy as np
import pytest

from backend.interpretability.attention_rollout import (
    AttentionRolloutEngine,
    compute_attention_rollout,
    compute_attention_rollout_heatmap,
    rollout_to_heatmap,
)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


# ============================================================================
# Synthetic Test Data Helpers (Explicitly Synthetic Unit Test Matrices)
# ============================================================================

def make_synthetic_softmax_attention(n_tokens: int, heads: int = 1) -> np.ndarray:
    """
    Generate synthetic test attention matrix where rows sum to 1.0.
    Clearly marked as synthetic unit test data, NOT real model outputs.
    """
    if heads > 1:
        # Shape: (heads, n_tokens, n_tokens)
        raw = np.random.RandomState(42).uniform(0.1, 1.0, size=(heads, n_tokens, n_tokens))
        sums = np.sum(raw, axis=-1, keepdims=True)
        return (raw / sums).astype(np.float32)
    else:
        # Shape: (n_tokens, n_tokens)
        raw = np.random.RandomState(42).uniform(0.1, 1.0, size=(n_tokens, n_tokens))
        sums = np.sum(raw, axis=-1, keepdims=True)
        return (raw / sums).astype(np.float32)


# ============================================================================
# Mathematical Utility Tests
# ============================================================================

def test_single_layer_valid_matrix():
    """Verify compute_attention_rollout produces valid normalized output for a single layer."""
    n_tokens = 4
    # Synthetic test data: 1 layer, shape (4, 4)
    synth_matrix = make_synthetic_softmax_attention(n_tokens=n_tokens)

    rollout = compute_attention_rollout([synth_matrix])

    # Check output shape
    assert rollout.shape == (n_tokens, n_tokens)
    # Check all values are finite and non-negative
    assert np.all(np.isfinite(rollout))
    assert np.all(rollout >= 0.0)
    # Check rows sum to 1.0 (stochastic matrix property)
    row_sums = np.sum(rollout, axis=-1)
    np.testing.assert_allclose(row_sums, np.ones(n_tokens), atol=1e-5)


def test_multiple_attention_layers():
    """Verify Attention Rollout combines multiple layers correctly."""
    n_tokens = 5
    num_layers = 3
    # Synthetic test data: 3 layers of shape (5, 5)
    synth_layers = [
        make_synthetic_softmax_attention(n_tokens=n_tokens)
        for _ in range(num_layers)
    ]

    rollout = compute_attention_rollout(synth_layers)

    assert rollout.shape == (n_tokens, n_tokens)
    assert np.all(np.isfinite(rollout))
    assert np.all(rollout >= 0.0)
    # Each row must sum to 1.0
    row_sums = np.sum(rollout, axis=-1)
    np.testing.assert_allclose(row_sums, np.ones(n_tokens), atol=1e-5)


def test_mathematical_step_by_step_exactness():
    """
    Verify step-by-step Attention Rollout formula on known 2x2 synthetic matrices.
    
    Formula check:
    A_0 = [[0.6, 0.4], [0.2, 0.8]]
    Add Identity:
    A_0 + I = [[1.6, 0.4], [0.2, 1.8]] (row sums = 2.0)
    Normalize rows:
    A_0_norm = [[0.8, 0.2], [0.1, 0.9]]

    A_1 = [[0.5, 0.5], [0.3, 0.7]]
    A_1 + I = [[1.5, 0.5], [0.3, 1.7]] (row sums = 2.0)
    A_1_norm = [[0.75, 0.25], [0.15, 0.85]]

    Rollout = A_1_norm @ A_0_norm:
    Row 0: [0.75*0.8 + 0.25*0.1, 0.75*0.2 + 0.25*0.9] = [0.60 + 0.025, 0.15 + 0.225] = [0.625, 0.375]
    Row 1: [0.15*0.8 + 0.85*0.1, 0.15*0.2 + 0.85*0.9] = [0.12 + 0.085, 0.03 + 0.765] = [0.205, 0.795]
    """
    layer_0 = np.array([[0.6, 0.4], [0.2, 0.8]], dtype=np.float32)
    layer_1 = np.array([[0.5, 0.5], [0.3, 0.7]], dtype=np.float32)

    expected = np.array([[0.625, 0.375], [0.205, 0.795]], dtype=np.float32)

    rollout = compute_attention_rollout([layer_0, layer_1])

    np.testing.assert_allclose(rollout, expected, atol=1e-5)


def test_multiple_attention_heads():
    """Verify compute_attention_rollout correctly averages across multiple attention heads."""
    n_tokens = 6
    num_heads = 4
    num_layers = 2

    # Synthetic test data: 2 layers, each with 4 heads and 6 tokens
    synth_layers = [
        make_synthetic_softmax_attention(n_tokens=n_tokens, heads=num_heads)
        for _ in range(num_layers)
    ]

    rollout = compute_attention_rollout(synth_layers)

    assert rollout.shape == (n_tokens, n_tokens)
    assert np.all(np.isfinite(rollout))
    # Check row normalization
    row_sums = np.sum(rollout, axis=-1)
    np.testing.assert_allclose(row_sums, np.ones(n_tokens), atol=1e-5)


def test_4d_batch_attention_matrix():
    """Verify 4D attention matrix of shape (1, heads, N, N) is supported."""
    n_tokens = 4
    num_heads = 2
    # Synthetic test data: (1, 2, 4, 4)
    synth_4d = make_synthetic_softmax_attention(n_tokens=n_tokens, heads=num_heads)[np.newaxis, ...]
    assert synth_4d.shape == (1, num_heads, n_tokens, n_tokens)

    rollout = compute_attention_rollout([synth_4d])

    assert rollout.shape == (n_tokens, n_tokens)
    row_sums = np.sum(rollout, axis=-1)
    np.testing.assert_allclose(row_sums, np.ones(n_tokens), atol=1e-5)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is not available in environment")
def test_pytorch_tensor_inputs():
    """Verify PyTorch tensor inputs are processed and return a PyTorch tensor."""
    n_tokens = 5
    num_heads = 3
    synth_np = make_synthetic_softmax_attention(n_tokens=n_tokens, heads=num_heads)
    synth_tensor_0 = torch.tensor(synth_np)
    synth_tensor_1 = torch.tensor(synth_np)

    rollout = compute_attention_rollout([synth_tensor_0, synth_tensor_1])

    assert isinstance(rollout, torch.Tensor)
    assert rollout.shape == (n_tokens, n_tokens)
    assert bool(torch.all(torch.isfinite(rollout)))
    row_sums = torch.sum(rollout, dim=-1)
    assert torch.allclose(row_sums, torch.ones(n_tokens), atol=1e-5)


# ============================================================================
# Invalid and Empty Input Handling Tests
# ============================================================================

def test_empty_input_handling():
    """Verify empty sequence raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        compute_attention_rollout([])
    assert "cannot be empty" in str(exc_info.value)


def test_none_input_handling():
    """Verify None input raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        compute_attention_rollout(None)  # type: ignore
    assert "cannot be None" in str(exc_info.value)


def test_non_square_attention_matrix():
    """Verify non-square token dimensions raise ValueError."""
    non_square = np.ones((4, 5), dtype=np.float32)
    with pytest.raises(ValueError) as exc_info:
        compute_attention_rollout([non_square])
    assert "must be square" in str(exc_info.value)


def test_inconsistent_sequence_length_across_layers():
    """Verify mismatched token sequence lengths across layers raise ValueError."""
    layer_0 = np.ones((4, 4), dtype=np.float32)
    layer_1 = np.ones((6, 6), dtype=np.float32)
    with pytest.raises(ValueError) as exc_info:
        compute_attention_rollout([layer_0, layer_1])
    assert "Inconsistent token sequence length" in str(exc_info.value)


def test_non_finite_values_handling():
    """Verify attention matrix with NaN or Inf raises ValueError."""
    nan_matrix = np.array([[1.0, np.nan], [0.5, 0.5]], dtype=np.float32)
    with pytest.raises(ValueError) as exc_info:
        compute_attention_rollout([nan_matrix])
    assert "contains non-finite values" in str(exc_info.value)


def test_invalid_dimension_count():
    """Verify 1D or 5D attention matrices raise ValueError."""
    mat_1d = np.array([0.5, 0.5], dtype=np.float32)
    with pytest.raises(ValueError) as exc_info:
        compute_attention_rollout([mat_1d])
    assert "must have 2, 3, or 4 dimensions" in str(exc_info.value)


# ============================================================================
# AttentionRolloutEngine Interface & Model Availability Tests
# ============================================================================

def test_attention_rollout_engine_initialization():
    """Verify AttentionRolloutEngine initializes with default None state."""
    engine = AttentionRolloutEngine()
    assert engine.model is None
    assert engine.target_layers == []
    assert engine.captured_matrices == []
    assert engine.is_available() is False


def test_attention_rollout_engine_raises_without_model():
    """Verify generate_rollout() raises descriptive RuntimeError when no model is integrated."""
    engine = AttentionRolloutEngine(model=None, target_layers=None)
    with pytest.raises(RuntimeError) as exc_info:
        engine.generate_rollout(input_tensor=None)
    assert "Attention Rollout cannot be executed" in str(exc_info.value)
    assert "No supported Transformer model" in str(exc_info.value)


# ============================================================================
# ViT Spatial Heatmap & Image Region Highlighting Tests
# ============================================================================

def test_rollout_to_heatmap_valid_square_grid():
    """
    Verify rollout_to_heatmap converts a ViT rollout matrix (1 CLS + 16 patches)
    into a valid normalized 2D spatial heatmap with inferred square grid (4x4).
    """
    # 1 [CLS] token + 16 patch tokens = 17 tokens
    n_tokens = 17
    synth_matrix = make_synthetic_softmax_attention(n_tokens=n_tokens)
    rollout = compute_attention_rollout([synth_matrix])

    target_size = (64, 64)
    heatmap = rollout_to_heatmap(rollout, target_size=target_size)

    # Check output properties
    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == target_size
    assert heatmap.dtype == np.float32
    assert np.all(np.isfinite(heatmap))
    assert np.all(heatmap >= 0.0)
    assert np.all(heatmap <= 1.0)
    # Check normalization spread
    assert np.isclose(np.max(heatmap), 1.0, atol=1e-5)
    assert np.isclose(np.min(heatmap), 0.0, atol=1e-5)


def test_rollout_to_heatmap_unscaled_grid():
    """Verify rollout_to_heatmap returns raw patch grid shape when target_size is None."""
    # 1 [CLS] token + 9 patch tokens (3x3 grid) = 10 tokens
    n_tokens = 10
    synth_matrix = make_synthetic_softmax_attention(n_tokens=n_tokens)
    rollout = compute_attention_rollout([synth_matrix])

    heatmap = rollout_to_heatmap(rollout, target_size=None)

    assert heatmap.shape == (3, 3)
    assert np.all(heatmap >= 0.0)
    assert np.all(heatmap <= 1.0)


def test_rollout_to_heatmap_explicit_non_square_grid():
    """Verify rollout_to_heatmap supports non-square patch grids (e.g., 2x8 = 16 patches)."""
    n_tokens = 17  # 1 CLS + 16 patches
    synth_matrix = make_synthetic_softmax_attention(n_tokens=n_tokens)
    rollout = compute_attention_rollout([synth_matrix])

    heatmap = rollout_to_heatmap(
        rollout,
        grid_size=(2, 8),
        target_size=(30, 120),
    )

    assert heatmap.shape == (30, 120)
    assert np.isclose(np.max(heatmap), 1.0, atol=1e-5)
    assert np.isclose(np.min(heatmap), 0.0, atol=1e-5)


def test_compute_attention_rollout_heatmap_end_to_end():
    """Verify end-to-end compute_attention_rollout_heatmap from multi-layer attention matrices."""
    # 3 layers of 1 CLS + 16 patches (17 tokens) with 4 heads
    layers = [
        make_synthetic_softmax_attention(n_tokens=17, heads=4)
        for _ in range(3)
    ]

    heatmap = compute_attention_rollout_heatmap(
        layers,
        target_size=(224, 224),
    )

    assert heatmap.shape == (224, 224)
    assert np.all(np.isfinite(heatmap))
    assert np.all(heatmap >= 0.0)
    assert np.all(heatmap <= 1.0)


def test_rollout_to_heatmap_invalid_inputs():
    """Verify descriptive ValueErrors for invalid inputs to rollout_to_heatmap."""
    # None input
    with pytest.raises(ValueError) as exc:
        rollout_to_heatmap(None)  # type: ignore
    assert "cannot be None" in str(exc.value)

    # 1D input
    with pytest.raises(ValueError) as exc:
        rollout_to_heatmap(np.array([1.0, 2.0]))
    assert "must be a 2D matrix" in str(exc.value)

    # Fewer than 2 tokens
    with pytest.raises(ValueError) as exc:
        rollout_to_heatmap(np.ones((1, 1)))
    assert "at least 2 tokens" in str(exc.value)

    # Out-of-bounds CLS index
    with pytest.raises(ValueError) as exc:
        rollout_to_heatmap(np.ones((5, 5)), cls_token_index=10)
    assert "out of bounds" in str(exc.value)

    # Incompatible explicit grid size (3x3 = 9 != 16 patches)
    with pytest.raises(ValueError) as exc:
        rollout_to_heatmap(np.ones((17, 17)), grid_size=(3, 3))
    assert "does not match the number of patch tokens" in str(exc.value)

    # Non-square patch count with no explicit grid size (10 patches is not a square)
    with pytest.raises(ValueError) as exc:
        rollout_to_heatmap(np.ones((11, 11)))
    assert "Cannot infer square patch grid" in str(exc.value)


def test_attention_rollout_engine_generate_heatmap_raises_without_model():
    """
    Verify generate_heatmap() raises descriptive RuntimeError when no model is integrated,
    ensuring TruthLens never manufactures fake heatmaps.
    """
    engine = AttentionRolloutEngine(model=None, target_layers=None)
    with pytest.raises(RuntimeError) as exc_info:
        engine.generate_heatmap(input_tensor=None)
    assert "Attention Rollout cannot be executed" in str(exc_info.value)
    assert "No supported Transformer model" in str(exc_info.value)


def test_package_exports():
    """Verify backend.interpretability exports all required public utilities."""
    import backend.interpretability as interp

    assert hasattr(interp, "compute_attention_rollout")
    assert hasattr(interp, "rollout_to_heatmap")
    assert hasattr(interp, "compute_attention_rollout_heatmap")
    assert hasattr(interp, "AttentionRolloutEngine")
    assert hasattr(interp, "compute_gradcam_heatmap")
    assert hasattr(interp, "GradCAMEngine")

