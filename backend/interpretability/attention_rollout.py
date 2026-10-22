"""
TruthLens - Attention Rollout Interpretability Utility
Author: TruthLens Development Team
Sprint 2: Interpretability (Part 3 - Attention Rollout Implementation)

This module provides the AttentionRolloutEngine interface and the standalone
`compute_attention_rollout` mathematical utility for propagating self-attention
matrices across Transformer layers.

Attention Rollout Flow:
Layer Attention Matrices [A_0, A_1, ..., A_{L-1}]
  -> Average across attention heads (if multi-head)
  -> Add Identity matrix (residual connections: A + I)
  -> Normalize across last dimension (row sum = 1)
  -> Recursive layer multiplication: R_l = A_l @ R_{l-1}
  -> Cumulative Rollout Matrix R of shape (N, N)
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    HAS_TORCH = False

logger = logging.getLogger(__name__)


def compute_attention_rollout(
    attention_matrices: Sequence[Union[np.ndarray, "torch.Tensor"]],
) -> Union[np.ndarray, "torch.Tensor"]:
    """
    Compute cumulative Attention Rollout across Transformer layers.

    Standard Simplified Procedure:
    1. Average attention across heads when the input includes multiple heads.
    2. Add an identity matrix to account for residual connections (A + I).
    3. Normalize each attention matrix across its last dimension (row sum = 1).
    4. Combine attention across layers by multiplying normalized matrices in layer order:
       R_0 = A_0_norm
       R_l = A_l_norm @ R_{l-1}
    5. Return the resulting cumulative rollout attention matrix.

    Args:
        attention_matrices: Non-empty sequence of attention matrices from Transformer layers.
            Each matrix can be either a 2D matrix (N, N) or a 3D matrix (heads, N, N),
            or a 4D batch matrix (1, heads, N, N), where N is the token sequence length.
            Inputs can be PyTorch tensors or NumPy arrays.

    Returns:
        Union[np.ndarray, torch.Tensor]: Cumulative rollout matrix of shape (N, N).
            Matches the tensor type (PyTorch Tensor or NumPy ndarray) of the input.

    Raises:
        ValueError: If attention_matrices is empty, has invalid dimensions, non-square
            token dimensions, non-finite values, or inconsistent sequence lengths across layers.
        TypeError: If attention_matrices is not a sequence or contains unsupported types.
    """
    # 0. Input validation
    if attention_matrices is None:
        raise ValueError("attention_matrices cannot be None. Provide a non-empty sequence of attention matrices.")

    if not isinstance(attention_matrices, (list, tuple)):
        raise TypeError(f"attention_matrices must be a list or tuple of matrices, got {type(attention_matrices).__name__}.")

    if len(attention_matrices) == 0:
        raise ValueError("attention_matrices sequence cannot be empty. Provide at least one layer attention matrix.")

    # Determine whether input is PyTorch tensor or NumPy array
    is_torch = HAS_TORCH and any(
        (torch is not None and isinstance(m, torch.Tensor)) for m in attention_matrices
    )

    expected_seq_len: Optional[int] = None
    normalized_layers: List[Any] = []

    for layer_idx, raw_matrix in enumerate(attention_matrices):
        if raw_matrix is None:
            raise ValueError(f"Attention matrix at layer index {layer_idx} is None.")

        # Convert to working tensor type
        if is_torch:
            if not isinstance(raw_matrix, torch.Tensor):
                try:
                    mat = torch.as_tensor(raw_matrix, dtype=torch.float32)
                except Exception as e:
                    raise TypeError(f"Could not convert matrix at layer {layer_idx} to torch.Tensor: {e}") from e
            else:
                mat = raw_matrix.float()

            # Check for non-finite values
            if not bool(torch.all(torch.isfinite(mat))):
                raise ValueError(f"Attention matrix at layer {layer_idx} contains non-finite values (NaN or Inf).")

            ndim = mat.ndim
            shape = mat.shape

            # Validate dimensions: (N, N), (H, N, N), or (1, H, N, N)
            if ndim < 2 or ndim > 4:
                raise ValueError(
                    f"Attention matrix at layer {layer_idx} must have 2, 3, or 4 dimensions, "
                    f"got shape {tuple(shape)} with {ndim} dimensions."
                )

            # Step 1: Average attention across heads if multiple heads are present
            if ndim == 4:
                if shape[0] != 1:
                    raise ValueError(
                        f"4D attention matrix at layer {layer_idx} must have batch size 1, got shape {tuple(shape)}."
                    )
                # Shape: (1, H, N, N) -> average across heads -> (N, N)
                mat_squeezed = mat.squeeze(0)
                mat_avg = torch.mean(mat_squeezed, dim=0)
            elif ndim == 3:
                # Shape: (H, N, N) -> average across heads -> (N, N)
                mat_avg = torch.mean(mat, dim=0)
            else:
                # ndim == 2: Shape (N, N)
                mat_avg = mat

            # Validate square token dimensions
            n_rows, n_cols = mat_avg.shape[-2], mat_avg.shape[-1]
            if n_rows != n_cols:
                raise ValueError(
                    f"Attention matrix at layer {layer_idx} must be square in token dimensions, "
                    f"got ({n_rows}, {n_cols})."
                )

            if n_rows <= 0:
                raise ValueError(
                    f"Attention matrix at layer {layer_idx} has invalid token sequence length: {n_rows}."
                )

            if expected_seq_len is None:
                expected_seq_len = n_rows
            elif n_rows != expected_seq_len:
                raise ValueError(
                    f"Inconsistent token sequence length across layers: Layer 0 has sequence length {expected_seq_len}, "
                    f"but layer {layer_idx} has sequence length {n_rows}."
                )

            # Step 2: Add identity matrix to account for residual connections
            # Residual connection gives x_{l} = x_{l-1} + Attention(x_{l-1}), so A_residual = A + I
            eye = torch.eye(n_rows, dtype=mat_avg.dtype, device=mat_avg.device)
            mat_residual = mat_avg + eye

            # Step 3: Normalize each attention matrix across its last dimension (rows sum to 1)
            row_sums = torch.sum(mat_residual, dim=-1, keepdim=True)
            # Guard against division by zero with small epsilon
            mat_normalized = mat_residual / torch.clamp(row_sums, min=1e-12)

            normalized_layers.append(mat_normalized)

        else:
            # NumPy processing path
            if not isinstance(raw_matrix, np.ndarray):
                try:
                    mat_np = np.asarray(raw_matrix, dtype=np.float32)
                except Exception as e:
                    raise TypeError(f"Could not convert matrix at layer {layer_idx} to np.ndarray: {e}") from e
            else:
                mat_np = raw_matrix.astype(np.float32)

            if not np.all(np.isfinite(mat_np)):
                raise ValueError(f"Attention matrix at layer {layer_idx} contains non-finite values (NaN or Inf).")

            ndim = mat_np.ndim
            shape = mat_np.shape

            if ndim < 2 or ndim > 4:
                raise ValueError(
                    f"Attention matrix at layer {layer_idx} must have 2, 3, or 4 dimensions, "
                    f"got shape {tuple(shape)} with {ndim} dimensions."
                )

            # Step 1: Average attention across heads if multi-head
            if ndim == 4:
                if shape[0] != 1:
                    raise ValueError(
                        f"4D attention matrix at layer {layer_idx} must have batch size 1, got shape {tuple(shape)}."
                    )
                mat_squeezed = np.squeeze(mat_np, axis=0)
                mat_avg = np.mean(mat_squeezed, axis=0)
            elif ndim == 3:
                mat_avg = np.mean(mat_np, axis=0)
            else:
                mat_avg = mat_np

            n_rows, n_cols = mat_avg.shape[-2], mat_avg.shape[-1]
            if n_rows != n_cols:
                raise ValueError(
                    f"Attention matrix at layer {layer_idx} must be square in token dimensions, "
                    f"got ({n_rows}, {n_cols})."
                )

            if n_rows <= 0:
                raise ValueError(
                    f"Attention matrix at layer {layer_idx} has invalid token sequence length: {n_rows}."
                )

            if expected_seq_len is None:
                expected_seq_len = n_rows
            elif n_rows != expected_seq_len:
                raise ValueError(
                    f"Inconsistent token sequence length across layers: Layer 0 has sequence length {expected_seq_len}, "
                    f"but layer {layer_idx} has sequence length {n_rows}."
                )

            # Step 2: Add identity matrix for residual connections
            eye = np.eye(n_rows, dtype=mat_avg.dtype)
            mat_residual = mat_avg + eye

            # Step 3: Normalize each attention matrix across last dimension (row sum = 1)
            row_sums = np.sum(mat_residual, axis=-1, keepdims=True)
            mat_normalized = mat_residual / np.clip(row_sums, a_min=1e-12, a_max=None)

            normalized_layers.append(mat_normalized)

    # Step 4: Combine attention across layers by multiplying normalized matrices in layer order:
    # R_0 = A_0_norm
    # R_l = A_l_norm @ R_{l-1}
    rollout = normalized_layers[0]

    for next_layer in normalized_layers[1:]:
        if is_torch:
            rollout = torch.matmul(next_layer, rollout)
        else:
            rollout = np.matmul(next_layer, rollout)

    # Step 5: Return resulting rollout attention matrix of shape (N, N)
    return rollout


class AttentionRolloutEngine:
    """
    Attention Rollout explanation engine for Transformer vision backbones.

    Provides hook management to capture attention weight matrices across Transformer
    layers and compute recursive Attention Rollout.

    Note:
    Like GradCAMEngine, this engine is designed to interface with PyTorch Transformer
    models (e.g., CLIP, ViT, DINOv2). When no model is loaded in TruthLens, is_available()
    returns False and generate_rollout() raises a descriptive RuntimeError.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        target_layers: Optional[List[Any]] = None,
    ):
        """
        Initialize the AttentionRolloutEngine.

        Args:
            model: PyTorch model instance (torch.nn.Module), or None if no model is loaded.
            target_layers: List of Transformer self-attention modules to extract attention from.
        """
        self.model = model
        self.target_layers = target_layers or []
        self.captured_matrices: List[Any] = []
        self._hooks: List[Any] = []

    def is_available(self) -> bool:
        """
        Check if a valid PyTorch Transformer model and target layers are present.
        """
        return (
            HAS_TORCH
            and self.model is not None
            and isinstance(self.model, nn.Module)
            and len(self.target_layers) > 0
            and all(isinstance(layer, nn.Module) for layer in self.target_layers)
        )

    def _register_hooks(self) -> None:
        """Register forward hooks on each target attention layer."""
        if not self.is_available():
            raise RuntimeError(
                "Cannot register hooks: No valid PyTorch model and target layers are loaded."
            )

        self.captured_matrices.clear()
        self._remove_hooks()

        for layer in self.target_layers:
            def hook_fn(module, input_tensor, output_tensor):
                # Standard Transformer attention layers output (attn_output, attn_weights)
                # or store attention weights. Handle tuple output or raw tensor.
                if isinstance(output_tensor, tuple) and len(output_tensor) > 1:
                    attn_weights = output_tensor[1]
                else:
                    attn_weights = output_tensor
                if attn_weights is not None:
                    self.captured_matrices.append(attn_weights.detach())

            handle = layer.register_forward_hook(hook_fn)
            self._hooks.append(handle)

    def _remove_hooks(self) -> None:
        """Remove registered hooks to prevent memory leaks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def generate_rollout(
        self,
        input_tensor: Optional[Any] = None,
    ) -> Any:
        """
        Generate Attention Rollout matrix for the given input tensor.

        Args:
            input_tensor: PyTorch tensor input batch of shape (1, C, H, W).

        Returns:
            Rollout attention matrix of shape (N, N).

        Raises:
            RuntimeError: If no suitable Transformer model is loaded in the codebase.
        """
        if not self.is_available():
            raise RuntimeError(
                "Attention Rollout cannot be executed: No supported Transformer model "
                "or inference pipeline is currently loaded in TruthLens codebase. "
                "Please integrate a Transformer vision backbone (e.g., CLIP, ViT, or DINOv2) "
                "before generating live Attention Rollout explanations."
            )

        self.model.eval()
        self._register_hooks()

        try:
            # Forward pass to trigger attention hooks
            if torch is not None:
                with torch.no_grad():
                    self.model(input_tensor)

            if len(self.captured_matrices) == 0:
                raise RuntimeError("Failed to capture attention matrices from target layers during forward pass.")

            return compute_attention_rollout(self.captured_matrices)

        finally:
            self._remove_hooks()
