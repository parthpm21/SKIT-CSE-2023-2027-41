"""
TruthLens - Grad-CAM Visual Interpretability Utility
Author: TruthLens Development Team
Sprint 2: Interpretability (Part 2 - Grad-CAM Readiness & Implementation)

This module provides the GradCAMEngine interface and helper functions
for registering PyTorch layer hooks, capturing activation maps and gradients,
and generating normalized 2D Grad-CAM heatmaps for image forensic models.

Grad-CAM (Gradient-weighted Class Activation Mapping) Flow:
Image -> Model -> Target Layer -> Activations + Gradients -> Grad-CAM Weights -> Heatmap -> Highlighted Image
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    F = None
    HAS_TORCH = False

logger = logging.getLogger(__name__)


class GradCAMEngine:
    """
    Grad-CAM explanation engine for PyTorch vision models.
    
    Captures forward activations and backward gradients from a specified target layer
    to construct spatial heatmap explanations of model predictions.
    """

    def __init__(self, model: Optional[Any] = None, target_layer: Optional[Any] = None):
        """
        Initialize the GradCAMEngine with a model and target layer.
        
        Args:
            model: PyTorch model instance (torch.nn.Module), or None if no model is loaded.
            target_layer: The target module/layer within the model to hook into.
        """
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[Any] = None
        self.gradients: Optional[Any] = None
        self._forward_handle: Optional[Any] = None
        self._backward_handle: Optional[Any] = None

    def is_available(self) -> bool:
        """
        Check if a valid PyTorch model and target layer are present and ready for Grad-CAM.
        """
        return (
            HAS_TORCH
            and self.model is not None
            and self.target_layer is not None
            and isinstance(self.model, nn.Module)
            and isinstance(self.target_layer, nn.Module)
        )

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on the target layer."""
        if not self.is_available():
            raise RuntimeError(
                "Cannot register hooks: No valid PyTorch model and target layer are loaded."
            )

        def forward_hook(module, input_tensor, output_tensor):
            self.activations = output_tensor.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self._backward_handle = self.target_layer.register_full_backward_hook(backward_hook)

    def _remove_hooks(self) -> None:
        """Remove registered hooks to prevent memory leaks."""
        if self._forward_handle is not None:
            self._forward_handle.remove()
            self._forward_handle = None
        if self._backward_handle is not None:
            self._backward_handle.remove()
            self._backward_handle = None

    def generate_heatmap(
        self,
        input_tensor: Any,
        target_class: Optional[int] = None,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Generate a 2D Grad-CAM heatmap for the given input tensor.
        
        Args:
            input_tensor: PyTorch tensor input batch of shape (1, C, H, W).
            target_class: Target class index for backpropagation. If None, uses argmax.
            target_size: Optional (height, width) tuple to resize output heatmap to.
            
        Returns:
            np.ndarray: 2D heatmap normalized to [0.0, 1.0] with shape (H, W).
            
        Raises:
            RuntimeError: If no suitable model is loaded in the codebase.
        """
        if not self.is_available():
            raise RuntimeError(
                "Grad-CAM cannot be executed: No supported model/inference pipeline is currently "
                "loaded in TruthLens codebase. Please integrate a PyTorch vision backbone "
                "(e.g., CLIP, ViT, DINOv2, or ResNet) before generating Grad-CAM heatmaps."
            )

        self.model.eval()
        self._register_hooks()

        try:
            # Forward pass
            output = self.model(input_tensor)
            if target_class is None:
                target_class = int(torch.argmax(output, dim=1).item())

            score = output[0, target_class]
            
            # Zero gradients and backpropagate
            self.model.zero_grad()
            score.backward(retain_graph=True)

            if self.activations is None or self.gradients is None:
                raise RuntimeError("Failed to capture activations or gradients from target layer.")

            # Calculate Grad-CAM channel importance weights: mean gradient per feature map
            # gradients shape: (1, C, H_feat, W_feat)
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

            # Weighted combination of activation maps
            cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

            # Apply ReLU to focus on features with positive influence on prediction
            cam = F.relu(cam)

            # Resize heatmap to input/target spatial resolution
            if target_size is not None:
                h, w = target_size
            else:
                h, w = input_tensor.shape[2], input_tensor.shape[3]

            cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
            
            # Squeeze to 2D numpy array
            heatmap = cam.squeeze().cpu().numpy()

            # Normalize to [0, 1]
            max_val = np.max(heatmap)
            min_val = np.min(heatmap)
            if max_val > min_val:
                heatmap = (heatmap - min_val) / (max_val - min_val)
            else:
                heatmap = np.zeros_like(heatmap, dtype=np.float32)

            return heatmap.astype(np.float32)

        finally:
            self._remove_hooks()


def compute_gradcam_heatmap(
    model: Optional[Any] = None,
    target_layer: Optional[Any] = None,
    input_tensor: Optional[Any] = None,
    target_class: Optional[int] = None,
    target_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Convenience wrapper to compute Grad-CAM heatmap.
    
    Args:
        model: PyTorch model (torch.nn.Module) or None.
        target_layer: Target layer inside model or None.
        input_tensor: Input tensor batch (1, C, H, W).
        target_class: Optional target class index.
        target_size: Optional target size (H, W).
        
    Returns:
        np.ndarray: 2D normalized heatmap [0, 1].
    """
    engine = GradCAMEngine(model=model, target_layer=target_layer)
    return engine.generate_heatmap(
        input_tensor=input_tensor,
        target_class=target_class,
        target_size=target_size,
    )
