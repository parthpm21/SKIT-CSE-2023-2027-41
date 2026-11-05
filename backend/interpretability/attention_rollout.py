"""
TruthLens - Attention Rollout Interpretability Utility
Author: TruthLens Development Team
Sprint 2: Interpretability (Part 3 - Attention Rollout Implementation)

This module provides the AttentionRolloutEngine interface, the standalone
`compute_attention_rollout` mathematical utility for propagating self-attention
matrices across Transformer layers, and the `rollout_to_heatmap` utility
to highlight important image regions that a Vision Transformer (ViT) focuses on.

Attention Rollout Flow:
Layer Attention Matrices [A_0, A_1, ..., A_{L-1}]
  -> Average across attention heads (if multi-head)
  -> Add Identity matrix (residual connections: A + I)
  -> Normalize across last dimension (row sum = 1)
  -> Recursive layer multiplication: R_l = A_l @ R_{l-1}
  -> Cumulative Rollout Matrix R of shape (N, N)
  -> Extract [CLS] row over patch tokens (N - 1)
  -> Reshape to 2D patch grid (grid_h, grid_w)
  -> Upsample to target image resolution (target_h, target_w)
  -> Normalized 2D Attention Heatmap [0.0, 1.0]
"""

from __future__ import annotations

import logging
import math
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

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


def rollout_to_heatmap(
    rollout_matrix: Union[np.ndarray, "torch.Tensor"],
    grid_size: Optional[Tuple[int, int]] = None,
    target_size: Optional[Tuple[int, int]] = None,
    cls_token_index: int = 0,
) -> np.ndarray:
    """
    Convert a cumulative Attention Rollout matrix into a 2D spatial heatmap
    highlighting the image regions that a Vision Transformer (ViT) focuses on.

    ViT Spatial Mapping Flow:
    1. Extract the attention row corresponding to the [CLS] token (default index 0).
    2. Exclude the [CLS] token itself to isolate attention weights across patch tokens.
    3. Reshape the 1D patch attention vector into a 2D spatial grid (grid_h, grid_w).
    4. Upsample the spatial grid to the target image dimensions (target_h, target_w)
       via bilinear interpolation.
    5. Min-max normalize the heatmap values into the [0.0, 1.0] range.

    Args:
        rollout_matrix: 2D square rollout matrix of shape (N, N).
        grid_size: Optional (grid_h, grid_w) tuple representing the ViT patch grid dimensions.
            If None, square dimensions are automatically inferred from the number of patches.
        target_size: Optional (target_h, target_w) tuple to resize the heatmap to (e.g., original
            or preprocessed image dimensions). If None, the un-interpolated patch grid is returned.
        cls_token_index: Index of the class token ([CLS]) in the sequence (default 0).

    Returns:
        np.ndarray: 2D float32 numpy array with values normalized in [0.0, 1.0].

    Raises:
        ValueError: If rollout_matrix is None, non-2D, non-square, has fewer than 2 tokens,
            cls_token_index is out of range, or grid_size does not match the patch token count.
        TypeError: If rollout_matrix cannot be converted to a numerical array.
    """
    if rollout_matrix is None:
        raise ValueError("rollout_matrix cannot be None.")

    # Convert PyTorch Tensor or array-like to NumPy array
    if HAS_TORCH and torch is not None and isinstance(rollout_matrix, torch.Tensor):
        rollout_np = rollout_matrix.detach().cpu().numpy()
    elif isinstance(rollout_matrix, np.ndarray):
        rollout_np = rollout_matrix
    else:
        try:
            rollout_np = np.asarray(rollout_matrix, dtype=np.float32)
        except Exception as e:
            raise TypeError(f"Could not convert rollout_matrix to np.ndarray: {e}") from e

    if rollout_np.ndim != 2:
        raise ValueError(
            f"rollout_matrix must be a 2D matrix (N, N), got shape {rollout_np.shape} with {rollout_np.ndim} dimensions."
        )

    n_rows, n_cols = rollout_np.shape
    if n_rows != n_cols:
        raise ValueError(
            f"rollout_matrix must be square, got shape ({n_rows}, {n_cols})."
        )

    if n_rows < 2:
        raise ValueError(
            f"rollout_matrix must have at least 2 tokens (1 [CLS] token + patch tokens), got {n_rows}."
        )

    if cls_token_index < 0 or cls_token_index >= n_rows:
        raise ValueError(
            f"cls_token_index {cls_token_index} is out of bounds for sequence length {n_rows}."
        )

    # Extract attention row from [CLS] token to all tokens
    cls_attention = rollout_np[cls_token_index, :]

    # Remove the self-attention of the [CLS] token to isolate spatial patch attention
    patch_attention = np.delete(cls_attention, cls_token_index)
    num_patches = len(patch_attention)

    # Determine spatial grid dimensions (grid_h, grid_w)
    if grid_size is None:
        side = int(math.isqrt(num_patches))
        if side * side != num_patches:
            raise ValueError(
                f"Cannot infer square patch grid for {num_patches} patches. "
                f"Please provide an explicit grid_size=(grid_h, grid_w)."
            )
        grid_h, grid_w = side, side
    else:
        if len(grid_size) != 2 or grid_size[0] <= 0 or grid_size[1] <= 0:
            raise ValueError(
                f"grid_size must be a tuple of two positive integers, got {grid_size}."
            )
        grid_h, grid_w = int(grid_size[0]), int(grid_size[1])
        if grid_h * grid_w != num_patches:
            raise ValueError(
                f"grid_size ({grid_h}, {grid_w}) product {grid_h * grid_w} does not match "
                f"the number of patch tokens {num_patches}."
            )

    # Reshape 1D patch attention into 2D grid
    spatial_map = patch_attention.reshape((grid_h, grid_w)).astype(np.float32)

    # Upsample/interpolate to target resolution if specified
    if target_size is not None:
        if len(target_size) != 2 or target_size[0] <= 0 or target_size[1] <= 0:
            raise ValueError(
                f"target_size must be a tuple of two positive integers (height, width), got {target_size}."
            )
        target_h, target_w = int(target_size[0]), int(target_size[1])

        if HAS_CV2 and cv2 is not None:
            # cv2.resize expects dsize=(width, height)
            spatial_map = cv2.resize(
                spatial_map,
                (target_w, target_h),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            # Fallback interpolation using numpy index grids
            row_indices = np.linspace(0, grid_h - 1, target_h).round().astype(int)
            col_indices = np.linspace(0, grid_w - 1, target_w).round().astype(int)
            spatial_map = spatial_map[np.ix_(row_indices, col_indices)]

    # Normalize heatmap to [0.0, 1.0]
    min_val = float(np.min(spatial_map))
    max_val = float(np.max(spatial_map))
    if max_val > min_val:
        heatmap = (spatial_map - min_val) / (max_val - min_val)
    else:
        heatmap = np.zeros_like(spatial_map, dtype=np.float32)

    return heatmap.astype(np.float32)


def compute_attention_rollout_heatmap(
    attention_matrices: Sequence[Union[np.ndarray, "torch.Tensor"]],
    grid_size: Optional[Tuple[int, int]] = None,
    target_size: Optional[Tuple[int, int]] = None,
    cls_token_index: int = 0,
) -> np.ndarray:
    """
    Convenience wrapper to compute cumulative Attention Rollout and convert
    it directly to a 2D spatial heatmap highlighting image regions.

    Args:
        attention_matrices: Non-empty sequence of attention matrices across Transformer layers.
        grid_size: Optional (grid_h, grid_w) tuple for patch dimensions.
        target_size: Optional (target_h, target_w) tuple to resize heatmap to (e.g. image size).
        cls_token_index: Index of the [CLS] token (default 0).

    Returns:
        np.ndarray: 2D normalized heatmap [0.0, 1.0] of shape target_size or grid_size.
    """
    rollout = compute_attention_rollout(attention_matrices)
    return rollout_to_heatmap(
        rollout_matrix=rollout,
        grid_size=grid_size,
        target_size=target_size,
        cls_token_index=cls_token_index,
    )


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

    def generate_heatmap(
        self,
        input_tensor: Optional[Any] = None,
        target_size: Optional[Tuple[int, int]] = None,
        grid_size: Optional[Tuple[int, int]] = None,
        cls_token_index: int = 0,
    ) -> np.ndarray:
        """
        Generate a 2D Attention Rollout heatmap for the given input tensor.

        Args:
            input_tensor: PyTorch tensor input batch of shape (1, C, H, W).
            target_size: Optional (height, width) tuple to resize output heatmap to.
            grid_size: Optional (grid_h, grid_w) tuple representing Transformer patch grid.
            cls_token_index: Index of the [CLS] token (default 0).

        Returns:
            np.ndarray: 2D normalized heatmap [0.0, 1.0].

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

        rollout = self.generate_rollout(input_tensor=input_tensor)
        if target_size is None and input_tensor is not None and hasattr(input_tensor, "shape") and len(input_tensor.shape) >= 4:
            target_size = (int(input_tensor.shape[2]), int(input_tensor.shape[3]))

        return rollout_to_heatmap(
            rollout_matrix=rollout,
            grid_size=grid_size,
            target_size=target_size,
            cls_token_index=cls_token_index,
        )

