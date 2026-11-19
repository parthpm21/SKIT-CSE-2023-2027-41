"""
TruthLens - Image Interpretability Integration
Author: TruthLens Development Team
User Story 3: Interpretability Support (Part 2 - Image Interpretability Integration)

This module provides a unified, modular integration connecting the TruthLens
image preprocessing pipeline with visual interpretability utilities (Grad-CAM
for CNN backbones and Attention Rollout for Vision Transformer architectures).

Workflow:
1. Receive image input (file path, raw bytes, PIL Image, NumPy array, or ProcessedImage).
2. Standardize and preprocess via ImagePreprocessor (preserving original resolution,
   aspect ratio, letterbox padding offsets, EXIF metadata, and noise residuals).
3. Generate Grad-CAM heatmaps when a compatible model and target layer are present.
4. Generate Attention Rollout heatmaps and rollout matrices when Transformer attention
   matrices or a compatible Transformer model are available.
5. Spatially unpad and align heatmaps to original image resolution if requested.
6. Handle missing models, unsupported inputs, non-finite values, and errors safely
   and truthfully without fabricating predictions or fake heatmaps.
7. Return a structured ImageExplanation container with JSON serialization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from backend.data.image_preprocessor import (
    ImagePreprocessor,
    NormalizationMode,
    ProcessedImage,
    ResizeMode,
)
from backend.interpretability.attention_rollout import (
    AttentionRolloutEngine,
    compute_attention_rollout,
    rollout_to_heatmap,
)
from backend.interpretability.gradcam import GradCAMEngine

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    HAS_TORCH = False

logger = logging.getLogger(__name__)


def unpad_and_resize_heatmap(
    heatmap: np.ndarray,
    pad_offsets: Tuple[int, int, int, int],
    original_size: Tuple[int, int],
) -> np.ndarray:
    """
    Remove letterbox padding offsets and resize heatmap back to original image resolution.

    Args:
        heatmap: 2D numpy array (H, W).
        pad_offsets: (top, bottom, left, right) padding pixel offsets.
        original_size: (width, height) of original image before letterboxing.

    Returns:
        np.ndarray: Rescaled 2D heatmap aligned with the original unpadded image.
    """
    if heatmap is None or not isinstance(heatmap, np.ndarray) or heatmap.ndim != 2:
        return heatmap

    orig_w, orig_h = original_size
    if orig_w <= 0 or orig_h <= 0:
        return heatmap

    h, w = heatmap.shape
    top, bottom, left, right = pad_offsets

    bottom_idx = h - bottom
    right_idx = w - right

    if top < bottom_idx and left < right_idx and (top > 0 or bottom > 0 or left > 0 or right > 0):
        active_crop = heatmap[top:bottom_idx, left:right_idx]
    else:
        active_crop = heatmap

    resized = cv2.resize(active_crop, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float32)


@dataclass
class ImageExplanation:
    """
    Interpretability results for an analyzed still image.

    Attributes:
        gradcam_heatmap: Normalized 2D Grad-CAM heatmap array (H, W) or None.
        gradcam_status: Status describing Grad-CAM execution
            ("generated", "unavailable: ...", "error: ...", "skipped").
        attention_heatmap: Normalized 2D Attention Rollout spatial heatmap array (H, W) or None.
        attention_rollout: Cumulative attention rollout matrix (N, N) or None.
        attention_status: Status describing Attention Rollout execution
            ("generated", "unavailable: ...", "error: ...", "skipped").
        prediction: Optional model prediction data (e.g. target_class, predicted_class, score).
        error: Descriptive error message if image loading or tensor validation failed.
        metadata: Forensic image metadata (e.g., original_size, target_size, pad_offsets,
            scale_factor, exif_present, has_editing_signature, noise_residual_present).
        status_message: Human-readable summary of explainability execution.
    """
    gradcam_heatmap: Optional[np.ndarray] = None
    gradcam_status: str = "unavailable"
    attention_heatmap: Optional[np.ndarray] = None
    attention_rollout: Optional[Any] = None
    attention_status: str = "unavailable"
    prediction: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    status_message: str = ""

    @property
    def gradcam_available(self) -> bool:
        """Return True if a valid Grad-CAM heatmap was generated."""
        return self.gradcam_heatmap is not None

    @property
    def attention_available(self) -> bool:
        """Return True if an Attention Rollout heatmap or rollout matrix was generated."""
        return self.attention_heatmap is not None or self.attention_rollout is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert image explanation result to a serializable dictionary."""
        gradcam_shape = (
            list(self.gradcam_heatmap.shape)
            if self.gradcam_heatmap is not None
            else None
        )
        attn_heat_shape = (
            list(self.attention_heatmap.shape)
            if self.attention_heatmap is not None
            else None
        )
        attn_roll_shape = (
            list(self.attention_rollout.shape)
            if hasattr(self.attention_rollout, "shape")
            else None
        )
        return {
            "gradcam_available": self.gradcam_available,
            "gradcam_status": self.gradcam_status,
            "gradcam_shape": gradcam_shape,
            "attention_available": self.attention_available,
            "attention_status": self.attention_status,
            "attention_heatmap_shape": attn_heat_shape,
            "attention_rollout_shape": attn_roll_shape,
            "prediction": self.prediction,
            "error": self.error,
            "metadata": self.metadata,
            "status_message": self.status_message,
        }


class ImageInterpretabilityPipeline:
    """
    Forensic integration pipeline connecting image preprocessing with
    interpretability utilities (Grad-CAM and Attention Rollout).
    """

    def __init__(
        self,
        gradcam_engine: Optional[GradCAMEngine] = None,
        attention_rollout_engine: Optional[AttentionRolloutEngine] = None,
        image_preprocessor: Optional[ImagePreprocessor] = None,
    ):
        """
        Initialize the ImageInterpretabilityPipeline.

        Args:
            gradcam_engine: Optional GradCAMEngine instance.
            attention_rollout_engine: Optional AttentionRolloutEngine instance.
            image_preprocessor: Optional ImagePreprocessor instance.
        """
        self.gradcam_engine = gradcam_engine
        self.attention_rollout_engine = attention_rollout_engine
        self.image_preprocessor = image_preprocessor or ImagePreprocessor()

    def process_image(
        self,
        image_input: Union[str, Path, bytes, Image.Image, np.ndarray, ProcessedImage],
        attention_matrices: Optional[Sequence[Any]] = None,
        target_class: Optional[int] = None,
        grid_size: Optional[Tuple[int, int]] = None,
        align_to_original_resolution: bool = False,
    ) -> ImageExplanation:
        """
        Execute visual interpretability on an input image.

        Args:
            image_input: Image file path, raw bytes, PIL Image, NumPy array, or ProcessedImage.
            attention_matrices: Optional sequence of Transformer layer self-attention matrices.
            target_class: Optional target class index for Grad-CAM explanation.
            grid_size: Optional (grid_h, grid_w) patch grid for Vision Transformer Attention Rollout.
            align_to_original_resolution: If True, crops padding and scales heatmaps back
                to the original unpadded image resolution.

        Returns:
            ImageExplanation containing generated heatmaps, execution statuses, and metadata.

        Raises:
            ValueError: If image_input is None.
            FileNotFoundError: If a file path is provided but does not exist.
            TypeError: If image_input has an unsupported type.
        """
        # 1. Validate input source
        if image_input is None:
            raise ValueError("image_input cannot be None.")

        if isinstance(image_input, (str, Path)):
            path_obj = Path(image_input)
            if not path_obj.exists():
                raise FileNotFoundError(f"Image file does not exist: {image_input}")

        # 2. Preprocess or extract ProcessedImage
        if isinstance(image_input, ProcessedImage):
            processed = image_input
        elif isinstance(image_input, (str, Path, bytes, Image.Image, np.ndarray)):
            processed = self.image_preprocessor.process(image_input)
        else:
            raise TypeError(
                f"Unsupported image_input type: {type(image_input).__name__}. "
                "Expected str, Path, bytes, PIL.Image, np.ndarray, or ProcessedImage."
            )

        tensor = processed.tensor
        meta_dict: Dict[str, Any] = {
            "original_size": processed.original_size,
            "target_size": processed.target_size,
            "pad_offsets": processed.pad_offsets,
            "scale_factor": processed.scale_factor,
            "exif_present": processed.exif_present,
            "has_editing_signature": (
                processed.exif_data.get("has_editing_signature", False)
                if processed.exif_data
                else False
            ),
            "noise_residual_present": processed.noise_residual is not None,
        }

        # 3. Validate image tensor shape, channels, and finiteness
        is_valid, validation_error = self._validate_image_tensor(tensor)
        if not is_valid:
            err_msg = f"Invalid image tensor: {validation_error}"
            return ImageExplanation(
                gradcam_heatmap=None,
                gradcam_status="skipped",
                attention_heatmap=None,
                attention_rollout=None,
                attention_status="skipped",
                prediction=None,
                error=err_msg,
                metadata=meta_dict,
                status_message=f"Image interpretability skipped: {err_msg}",
            )

        # Ensure 3D tensor (C, H, W)
        tensor_3d = tensor.squeeze(0) if tensor.ndim == 4 and tensor.shape[0] == 1 else tensor
        h, w = int(tensor_3d.shape[1]), int(tensor_3d.shape[2])

        # 4. Grad-CAM Generation
        gradcam_is_ready = self.gradcam_engine is not None and self.gradcam_engine.is_available()
        gradcam_heatmap: Optional[np.ndarray] = None
        gradcam_status: str

        if not gradcam_is_ready:
            gradcam_status = "unavailable: no compatible model/target layer loaded in TruthLens"
        else:
            try:
                # Prepare batch tensor (1, C, H, W)
                if hasattr(processed, "to_batch") and callable(processed.to_batch):
                    batch_input = processed.to_batch()
                elif tensor_3d.ndim == 3:
                    batch_input = np.expand_dims(tensor_3d, axis=0)
                else:
                    batch_input = tensor_3d

                if HAS_TORCH and torch is not None and not isinstance(batch_input, torch.Tensor):
                    batch_input = torch.from_numpy(batch_input).float()

                heatmap = self.gradcam_engine.generate_heatmap(
                    input_tensor=batch_input,
                    target_class=target_class,
                    target_size=(h, w),
                )

                if align_to_original_resolution and processed.pad_offsets and processed.original_size:
                    heatmap = unpad_and_resize_heatmap(
                        heatmap=heatmap,
                        pad_offsets=processed.pad_offsets,
                        original_size=processed.original_size,
                    )

                gradcam_heatmap = heatmap
                gradcam_status = "generated"
            except Exception as e:
                logger.warning("Grad-CAM generation failed: %s", e)
                gradcam_status = f"error: {str(e)}"
                gradcam_heatmap = None

        # 5. Attention Rollout Generation
        attention_engine_is_ready = (
            self.attention_rollout_engine is not None
            and self.attention_rollout_engine.is_available()
        )
        attention_heatmap: Optional[np.ndarray] = None
        attention_rollout: Optional[Any] = None
        attention_status: str

        if attention_matrices is not None:
            # Explicit Transformer layer attention matrices supplied
            try:
                rollout = compute_attention_rollout(attention_matrices)
                attention_rollout = rollout

                # Convert rollout matrix into 2D spatial heatmap
                attn_map = rollout_to_heatmap(
                    rollout_matrix=rollout,
                    grid_size=grid_size,
                    target_size=(h, w),
                )

                if align_to_original_resolution and processed.pad_offsets and processed.original_size:
                    attn_map = unpad_and_resize_heatmap(
                        heatmap=attn_map,
                        pad_offsets=processed.pad_offsets,
                        original_size=processed.original_size,
                    )

                attention_heatmap = attn_map
                attention_status = "generated"
            except Exception as e:
                logger.warning("Attention Rollout generation from matrices failed: %s", e)
                attention_status = f"error: {str(e)}"
                attention_heatmap = None
                attention_rollout = None

        elif attention_engine_is_ready:
            # Transformer model hooked in engine
            try:
                if hasattr(processed, "to_batch") and callable(processed.to_batch):
                    batch_input = processed.to_batch()
                elif tensor_3d.ndim == 3:
                    batch_input = np.expand_dims(tensor_3d, axis=0)
                else:
                    batch_input = tensor_3d

                if HAS_TORCH and torch is not None and not isinstance(batch_input, torch.Tensor):
                    batch_input = torch.from_numpy(batch_input).float()

                rollout = self.attention_rollout_engine.generate_rollout(batch_input)
                attention_rollout = rollout

                attn_map = self.attention_rollout_engine.generate_heatmap(
                    input_tensor=batch_input,
                    target_size=(h, w),
                    grid_size=grid_size,
                )

                if align_to_original_resolution and processed.pad_offsets and processed.original_size:
                    attn_map = unpad_and_resize_heatmap(
                        heatmap=attn_map,
                        pad_offsets=processed.pad_offsets,
                        original_size=processed.original_size,
                    )

                attention_heatmap = attn_map
                attention_status = "generated"
            except Exception as e:
                logger.warning("Attention Rollout generation from engine failed: %s", e)
                attention_status = f"error: {str(e)}"
                attention_heatmap = None
                attention_rollout = None
        else:
            attention_status = (
                "unavailable: no compatible Transformer model or attention matrices available"
            )
            attention_heatmap = None
            attention_rollout = None

        # 6. Formulate status summary message
        status_parts: List[str] = []
        if gradcam_heatmap is not None:
            status_parts.append("Grad-CAM heatmap generated.")
        else:
            status_parts.append(f"Grad-CAM: {gradcam_status}.")

        if attention_heatmap is not None:
            status_parts.append("Attention Rollout heatmap generated.")
        elif attention_rollout is not None:
            status_parts.append("Attention Rollout matrix generated.")
        else:
            status_parts.append(f"Attention Rollout: {attention_status}.")

        status_message = " ".join(status_parts)

        # 7. Package prediction metadata if target_class was specified
        prediction_payload: Optional[Dict[str, Any]] = None
        if target_class is not None:
            prediction_payload = {"target_class": target_class}

        return ImageExplanation(
            gradcam_heatmap=gradcam_heatmap,
            gradcam_status=gradcam_status,
            attention_heatmap=attention_heatmap,
            attention_rollout=attention_rollout,
            attention_status=attention_status,
            prediction=prediction_payload,
            error=None,
            metadata=meta_dict,
            status_message=status_message,
        )

    @staticmethod
    def _validate_image_tensor(tensor: Any) -> Tuple[bool, Optional[str]]:
        """Validate shape, channels, and finiteness of an image tensor."""
        if tensor is None:
            return False, "Tensor is None"

        if isinstance(tensor, np.ndarray):
            if not np.all(np.isfinite(tensor)):
                return False, "Tensor contains non-finite values (NaN or Inf)"
            ndim = tensor.ndim
            if ndim == 3:
                c, h, w = tensor.shape
                if c != 3:
                    return False, f"Expected 3 color channels, got {c}"
                if h <= 0 or w <= 0:
                    return False, f"Invalid spatial dimensions: ({h}, {w})"
            elif ndim == 4 and tensor.shape[0] == 1:
                _, c, h, w = tensor.shape
                if c != 3:
                    return False, f"Expected 3 color channels, got {c}"
                if h <= 0 or w <= 0:
                    return False, f"Invalid spatial dimensions: ({h}, {w})"
            else:
                return False, f"Expected 3D tensor (C, H, W), got {ndim}D tensor with shape {tensor.shape}"
            return True, None

        if HAS_TORCH and torch is not None and isinstance(tensor, torch.Tensor):
            if not bool(torch.all(torch.isfinite(tensor))):
                return False, "Tensor contains non-finite values (NaN or Inf)"
            ndim = tensor.ndim
            if ndim == 3:
                c, h, w = tensor.shape
                if c != 3:
                    return False, f"Expected 3 color channels, got {c}"
                if h <= 0 or w <= 0:
                    return False, f"Invalid spatial dimensions: ({h}, {w})"
            elif ndim == 4 and tensor.shape[0] == 1:
                _, c, h, w = tensor.shape
                if c != 3:
                    return False, f"Expected 3 color channels, got {c}"
                if h <= 0 or w <= 0:
                    return False, f"Invalid spatial dimensions: ({h}, {w})"
            else:
                return False, f"Expected 3D tensor (C, H, W), got {ndim}D tensor with shape {tuple(tensor.shape)}"
            return True, None

        return False, f"Unsupported tensor type: {type(tensor).__name__}"


def explain_image(
    image_input: Union[str, Path, bytes, Image.Image, np.ndarray, ProcessedImage],
    gradcam_engine: Optional[GradCAMEngine] = None,
    attention_rollout_engine: Optional[AttentionRolloutEngine] = None,
    attention_matrices: Optional[Sequence[Any]] = None,
    image_preprocessor: Optional[ImagePreprocessor] = None,
    target_class: Optional[int] = None,
    grid_size: Optional[Tuple[int, int]] = None,
    align_to_original_resolution: bool = False,
) -> ImageExplanation:
    """
    Convenience function to run image interpretability integration.

    Args:
        image_input: Image file path, raw bytes, PIL Image, NumPy array, or ProcessedImage.
        gradcam_engine: Optional GradCAMEngine instance.
        attention_rollout_engine: Optional AttentionRolloutEngine instance.
        attention_matrices: Optional Transformer layer self-attention matrices.
        image_preprocessor: Optional ImagePreprocessor instance.
        target_class: Optional target class for Grad-CAM.
        grid_size: Optional (grid_h, grid_w) tuple for patch grid.
        align_to_original_resolution: If True, crops padding and scales heatmaps to original size.

    Returns:
        ImageExplanation container.
    """
    pipeline = ImageInterpretabilityPipeline(
        gradcam_engine=gradcam_engine,
        attention_rollout_engine=attention_rollout_engine,
        image_preprocessor=image_preprocessor,
    )
    return pipeline.process_image(
        image_input=image_input,
        attention_matrices=attention_matrices,
        target_class=target_class,
        grid_size=grid_size,
        align_to_original_resolution=align_to_original_resolution,
    )
