"""
TruthLens - Video & Frame Interpretability Integration
Author: TruthLens Development Team
Sprint 2: Interpretability (Part 4 - Video/Frame Integration & Testing)

This module provides a modular, reusable integration between the TruthLens
video/frame preprocessing pipeline and the interpretability utilities (Grad-CAM
and Attention Rollout).

Workflow:
1. Receive video input (file path, bytes, or preprocessed frames/ProcessedVideo).
2. Obtain valid frames via VideoPreprocessor / ImagePreprocessor, preserving
   frame indices and timestamps.
3. Pass compatible frames to Grad-CAM if a compatible model and target layer exist.
4. Pass attention matrices to Attention Rollout when available.
5. Associate explanations with each frame in strict temporal order.
6. Handle missing models, invalid frames, missing attention matrices, and per-frame
   errors safely and truthfully without fabricating predictions or heatmaps.
7. Return a structured VideoExplanationResult.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from backend.data.image_preprocessor import ProcessedImage
from backend.data.video_preprocessor import (
    ProcessedVideo,
    VideoFrameMetadata,
    VideoPreprocessor,
)
from backend.interpretability.attention_rollout import (
    AttentionRolloutEngine,
    compute_attention_rollout,
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


@dataclass
class FrameExplanation:
    """
    Interpretability results for a single video frame.

    Attributes:
        frame_index: The sequential index or original video frame index.
        timestamp_seconds: Timestamp of the frame in seconds (if available).
        gradcam_heatmap: Normalized 2D Grad-CAM heatmap array or None.
        gradcam_status: Status describing Grad-CAM execution ("generated", "unavailable: ...", "error: ...", "skipped").
        attention_rollout: Cumulative attention rollout matrix or None.
        attention_status: Status describing Attention Rollout execution ("generated", "unavailable: ...", "error: ...", "skipped").
        error: Descriptive error message if the frame itself was invalid.
        metadata: Associated forensic metadata (e.g. pad_offsets, original_size, scale_factor).
    """
    frame_index: int
    timestamp_seconds: Optional[float] = None
    gradcam_heatmap: Optional[np.ndarray] = None
    gradcam_status: str = "unavailable"
    attention_rollout: Optional[Any] = None
    attention_status: str = "unavailable"
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert frame explanation to serializable dictionary."""
        gradcam_shape = list(self.gradcam_heatmap.shape) if self.gradcam_heatmap is not None else None
        attention_shape = (
            list(self.attention_rollout.shape)
            if hasattr(self.attention_rollout, "shape")
            else None
        )
        return {
            "frame_index": self.frame_index,
            "timestamp_seconds": self.timestamp_seconds,
            "gradcam_available": self.gradcam_heatmap is not None,
            "gradcam_status": self.gradcam_status,
            "gradcam_shape": gradcam_shape,
            "attention_available": self.attention_rollout is not None,
            "attention_status": self.attention_status,
            "attention_shape": attention_shape,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class VideoExplanationResult:
    """
    Aggregated interpretability explanations for an entire video or sequence of frames.

    Attributes:
        frame_count: Total number of frames processed.
        frames: Chronological list of FrameExplanation objects.
        gradcam_available: True if at least one Grad-CAM heatmap was generated.
        attention_available: True if at least one Attention Rollout matrix was generated.
        status_message: Human-readable summary of explainability readiness and execution.
        summary: Structured summary statistics dictionary.
    """
    frame_count: int
    frames: List[FrameExplanation] = field(default_factory=list)
    gradcam_available: bool = False
    attention_available: bool = False
    status_message: str = ""
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert video explanation result to serializable dictionary."""
        return {
            "frame_count": self.frame_count,
            "gradcam_available": self.gradcam_available,
            "attention_available": self.attention_available,
            "status_message": self.status_message,
            "summary": self.summary,
            "frames": [f.to_dict() for f in self.frames],
        }


def unpad_and_resize_heatmap(
    heatmap: np.ndarray,
    pad_offsets: Tuple[int, int, int, int],
    original_size: Tuple[int, int],
) -> np.ndarray:
    """
    Remove letterbox padding offsets and resize heatmap back to original frame resolution.

    Args:
        heatmap: 2D numpy array (H, W).
        pad_offsets: (top, bottom, left, right) padding pixel offsets.
        original_size: (width, height) of original frame before letterboxing.

    Returns:
        np.ndarray: Rescaled 2D heatmap aligned with the original unpadded frame.
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


class VideoInterpretabilityPipeline:
    """
    Forensic integration pipeline connecting video frame extraction with
    interpretability utilities (Grad-CAM and Attention Rollout).
    """

    def __init__(
        self,
        gradcam_engine: Optional[GradCAMEngine] = None,
        attention_rollout_engine: Optional[AttentionRolloutEngine] = None,
        video_preprocessor: Optional[VideoPreprocessor] = None,
    ):
        """
        Initialize the VideoInterpretabilityPipeline.

        Args:
            gradcam_engine: Optional GradCAMEngine instance.
            attention_rollout_engine: Optional AttentionRolloutEngine instance.
            video_preprocessor: Optional VideoPreprocessor instance.
        """
        self.gradcam_engine = gradcam_engine
        self.attention_rollout_engine = attention_rollout_engine
        self.video_preprocessor = video_preprocessor or VideoPreprocessor()

    def process_video(
        self,
        video_input: Union[str, Path, bytes, ProcessedVideo],
        frame_attention_matrices: Optional[Sequence[Optional[Sequence[Any]]]] = None,
        target_class: Optional[int] = None,
        align_to_original_resolution: bool = False,
    ) -> VideoExplanationResult:
        """
        Process a video input source and produce frame-associated interpretability explanations.

        Args:
            video_input: Video file path, raw bytes, or already preprocessed ProcessedVideo.
            frame_attention_matrices: Optional list of attention matrices per frame.
            target_class: Target class for Grad-CAM.
            align_to_original_resolution: If True, crops padding and resizes heatmaps to original size.

        Returns:
            VideoExplanationResult with frame explanations and execution summary.
        """
        if video_input is None:
            raise ValueError("video_input cannot be None.")

        if isinstance(video_input, ProcessedVideo):
            processed_video = video_input
        elif isinstance(video_input, (str, Path, bytes)):
            processed_video = self.video_preprocessor.process(video_input)
        else:
            raise TypeError(f"Unsupported video_input type: {type(video_input).__name__}")

        return self._explain_processed_video(
            processed_video=processed_video,
            frame_attention_matrices=frame_attention_matrices,
            target_class=target_class,
            align_to_original_resolution=align_to_original_resolution,
        )

    def process_frames(
        self,
        frames: Sequence[Union[np.ndarray, ProcessedImage]],
        frame_metadata: Optional[Sequence[Optional[VideoFrameMetadata]]] = None,
        frame_attention_matrices: Optional[Sequence[Optional[Sequence[Any]]]] = None,
        target_class: Optional[int] = None,
        align_to_original_resolution: bool = False,
    ) -> VideoExplanationResult:
        """
        Process a pre-extracted sequence of frame tensors or ProcessedImages.

        Args:
            frames: Sequence of (C, H, W) numpy tensors or ProcessedImage instances.
            frame_metadata: Optional sequence of VideoFrameMetadata for each frame.
            frame_attention_matrices: Optional sequence of attention matrices per frame.
            target_class: Target class for Grad-CAM.
            align_to_original_resolution: If True, unpads and resizes heatmaps to original resolution.

        Returns:
            VideoExplanationResult.
        """
        if frames is None:
            raise ValueError("frames cannot be None.")

        if len(frames) == 0:
            return VideoExplanationResult(
                frame_count=0,
                frames=[],
                gradcam_available=False,
                attention_available=False,
                status_message="Empty frame sequence provided.",
                summary={
                    "total_frames": 0,
                    "valid_frames": 0,
                    "gradcam_generated_count": 0,
                    "attention_generated_count": 0,
                    "errors_count": 0,
                },
            )

        extracted_tensors: List[np.ndarray] = []
        resolved_metadata: List[Optional[VideoFrameMetadata]] = []

        for idx, item in enumerate(frames):
            if isinstance(item, ProcessedImage):
                extracted_tensors.append(item.tensor)
                meta = VideoFrameMetadata(
                    frame_index=idx,
                    timestamp_seconds=round(float(idx), 3),
                    original_size=item.original_size,
                    scale_factor=item.scale_factor,
                    pad_offsets=item.pad_offsets,
                )
                resolved_metadata.append(meta)
            elif isinstance(item, np.ndarray):
                extracted_tensors.append(item)
                meta = frame_metadata[idx] if (frame_metadata and idx < len(frame_metadata)) else None
                resolved_metadata.append(meta)
            else:
                # Store placeholder for invalid item
                extracted_tensors.append(item)  # Will be flagged as invalid during per-frame validation
                meta = frame_metadata[idx] if (frame_metadata and idx < len(frame_metadata)) else None
                resolved_metadata.append(meta)

        return self._generate_explanations(
            frame_tensors=extracted_tensors,
            metadata_list=resolved_metadata,
            frame_attention_matrices=frame_attention_matrices,
            target_class=target_class,
            align_to_original_resolution=align_to_original_resolution,
        )

    def _explain_processed_video(
        self,
        processed_video: ProcessedVideo,
        frame_attention_matrices: Optional[Sequence[Optional[Sequence[Any]]]] = None,
        target_class: Optional[int] = None,
        align_to_original_resolution: bool = False,
    ) -> VideoExplanationResult:
        """Internal helper to explain a ProcessedVideo container."""
        if processed_video.frame_count == 0 or len(processed_video.tensor) == 0:
            return VideoExplanationResult(
                frame_count=0,
                frames=[],
                gradcam_available=False,
                attention_available=False,
                status_message="Empty video: no frames extracted.",
                summary={
                    "total_frames": 0,
                    "valid_frames": 0,
                    "gradcam_generated_count": 0,
                    "attention_generated_count": 0,
                    "errors_count": 0,
                },
            )

        frame_tensors = [processed_video.tensor[i] for i in range(processed_video.frame_count)]
        return self._generate_explanations(
            frame_tensors=frame_tensors,
            metadata_list=processed_video.frame_metadata,
            frame_attention_matrices=frame_attention_matrices,
            target_class=target_class,
            align_to_original_resolution=align_to_original_resolution,
        )

    def _generate_explanations(
        self,
        frame_tensors: Sequence[Any],
        metadata_list: Sequence[Optional[VideoFrameMetadata]],
        frame_attention_matrices: Optional[Sequence[Optional[Sequence[Any]]]] = None,
        target_class: Optional[int] = None,
        align_to_original_resolution: bool = False,
    ) -> VideoExplanationResult:
        """
        Generate explanations frame by frame, isolating errors and maintaining temporal association.
        """
        frame_explanations: List[FrameExplanation] = []
        total_frames = len(frame_tensors)

        gradcam_is_ready = self.gradcam_engine is not None and self.gradcam_engine.is_available()
        attention_engine_is_ready = (
            self.attention_rollout_engine is not None and self.attention_rollout_engine.is_available()
        )

        for idx in range(total_frames):
            tensor = frame_tensors[idx]
            meta = metadata_list[idx] if idx < len(metadata_list) else None

            frame_idx = meta.frame_index if meta is not None else idx
            timestamp = meta.timestamp_seconds if meta is not None else None
            meta_dict: Dict[str, Any] = {}
            if meta is not None:
                meta_dict = {
                    "original_size": meta.original_size,
                    "scale_factor": meta.scale_factor,
                    "pad_offsets": meta.pad_offsets,
                }

            # 1. Validate individual frame tensor
            is_valid, validation_error = self._validate_frame_tensor(tensor)
            if not is_valid:
                frame_explanations.append(
                    FrameExplanation(
                        frame_index=frame_idx,
                        timestamp_seconds=timestamp,
                        gradcam_heatmap=None,
                        gradcam_status="skipped",
                        attention_rollout=None,
                        attention_status="skipped",
                        error=f"Invalid frame tensor at position {idx}: {validation_error}",
                        metadata=meta_dict,
                    )
                )
                continue

            # Ensure (C, H, W)
            frame_tensor_3d = tensor.squeeze() if tensor.ndim == 4 and tensor.shape[0] == 1 else tensor

            # 2. Grad-CAM Generation
            gradcam_heatmap: Optional[np.ndarray] = None
            if not gradcam_is_ready:
                gradcam_status = (
                    "unavailable: no compatible model/target layer loaded in TruthLens"
                )
            else:
                try:
                    # Prepare input batch (1, C, H, W)
                    if HAS_TORCH and torch is not None and not isinstance(frame_tensor_3d, torch.Tensor):
                        batch_input = torch.from_numpy(frame_tensor_3d).unsqueeze(0).float()
                    elif isinstance(frame_tensor_3d, np.ndarray):
                        batch_input = np.expand_dims(frame_tensor_3d, axis=0)
                    else:
                        batch_input = frame_tensor_3d

                    heatmap = self.gradcam_engine.generate_heatmap(
                        input_tensor=batch_input,
                        target_class=target_class,
                    )

                    if align_to_original_resolution and meta is not None:
                        heatmap = unpad_and_resize_heatmap(
                            heatmap=heatmap,
                            pad_offsets=meta.pad_offsets,
                            original_size=meta.original_size,
                        )

                    gradcam_heatmap = heatmap
                    gradcam_status = "generated"
                except Exception as e:
                    gradcam_status = f"error: {str(e)}"
                    gradcam_heatmap = None

            # 3. Attention Rollout Generation
            attention_rollout: Optional[Any] = None
            if frame_attention_matrices is not None and idx < len(frame_attention_matrices):
                # Frame-specific attention matrices were provided
                attn_mats = frame_attention_matrices[idx]
                if attn_mats is not None:
                    try:
                        attention_rollout = compute_attention_rollout(attn_mats)
                        attention_status = "generated"
                    except Exception as e:
                        attention_status = f"error: {str(e)}"
                        attention_rollout = None
                else:
                    attention_status = "unavailable: no attention matrices provided for frame"
            elif attention_engine_is_ready:
                try:
                    if HAS_TORCH and torch is not None and not isinstance(frame_tensor_3d, torch.Tensor):
                        batch_input = torch.from_numpy(frame_tensor_3d).unsqueeze(0).float()
                    else:
                        batch_input = frame_tensor_3d
                    attention_rollout = self.attention_rollout_engine.generate_rollout(batch_input)
                    attention_status = "generated"
                except Exception as e:
                    attention_status = f"error: {str(e)}"
                    attention_rollout = None
            else:
                attention_status = (
                    "unavailable: no compatible Transformer model or attention matrices available"
                )

            frame_explanations.append(
                FrameExplanation(
                    frame_index=frame_idx,
                    timestamp_seconds=timestamp,
                    gradcam_heatmap=gradcam_heatmap,
                    gradcam_status=gradcam_status,
                    attention_rollout=attention_rollout,
                    attention_status=attention_status,
                    error=None,
                    metadata=meta_dict,
                )
            )

        gradcam_count = sum(1 for f in frame_explanations if f.gradcam_heatmap is not None)
        attention_count = sum(1 for f in frame_explanations if f.attention_rollout is not None)
        valid_count = sum(1 for f in frame_explanations if f.error is None)
        error_count = sum(
            1 for f in frame_explanations
            if f.error is not None or f.gradcam_status.startswith("error") or f.attention_status.startswith("error")
        )

        # Formulate status message
        status_parts: List[str] = [f"Processed {total_frames} frames ({valid_count} valid)."]
        if gradcam_count > 0:
            status_parts.append(f"Grad-CAM generated for {gradcam_count}/{total_frames} frames.")
        else:
            status_parts.append("Grad-CAM: unavailable (no compatible model/target layer loaded).")

        if attention_count > 0:
            status_parts.append(f"Attention Rollout generated for {attention_count}/{total_frames} frames.")
        else:
            status_parts.append("Attention Rollout: unavailable (no Transformer model or attention matrices).")

        status_message = " ".join(status_parts)

        summary = {
            "total_frames": total_frames,
            "valid_frames": valid_count,
            "gradcam_generated_count": gradcam_count,
            "attention_generated_count": attention_count,
            "errors_count": error_count,
            "gradcam_engine_available": gradcam_is_ready,
            "attention_engine_available": attention_engine_is_ready,
        }

        return VideoExplanationResult(
            frame_count=total_frames,
            frames=frame_explanations,
            gradcam_available=gradcam_count > 0,
            attention_available=attention_count > 0,
            status_message=status_message,
            summary=summary,
        )

    @staticmethod
    def _validate_frame_tensor(tensor: Any) -> Tuple[bool, Optional[str]]:
        """Validate shape, type, and finiteness of a frame tensor."""
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
            else:
                return False, f"Expected 3D tensor (C, H, W), got {ndim}D tensor with shape {tuple(tensor.shape)}"
            return True, None

        return False, f"Unsupported tensor type: {type(tensor).__name__}"


def explain_video_frames(
    video_or_frames: Union[
        str,
        Path,
        bytes,
        ProcessedVideo,
        Sequence[Union[np.ndarray, ProcessedImage]],
    ],
    gradcam_engine: Optional[GradCAMEngine] = None,
    attention_rollout_engine: Optional[AttentionRolloutEngine] = None,
    frame_attention_matrices: Optional[Sequence[Optional[Sequence[Any]]]] = None,
    frame_metadata: Optional[Sequence[Optional[VideoFrameMetadata]]] = None,
    video_preprocessor: Optional[VideoPreprocessor] = None,
    target_class: Optional[int] = None,
    align_to_original_resolution: bool = False,
) -> VideoExplanationResult:
    """
    Convenience function to run video/frame interpretability integration.

    Args:
        video_or_frames: Video file path, raw video bytes, ProcessedVideo container,
                         or sequence of frame tensors / ProcessedImages.
        gradcam_engine: Optional GradCAMEngine instance.
        attention_rollout_engine: Optional AttentionRolloutEngine instance.
        frame_attention_matrices: Optional list of attention matrices per frame.
        frame_metadata: Optional list of VideoFrameMetadata per frame.
        video_preprocessor: Optional VideoPreprocessor instance.
        target_class: Target class for Grad-CAM backpropagation.
        align_to_original_resolution: If True, crops padding and resizes heatmaps to original size.

    Returns:
        VideoExplanationResult.
    """
    pipeline = VideoInterpretabilityPipeline(
        gradcam_engine=gradcam_engine,
        attention_rollout_engine=attention_rollout_engine,
        video_preprocessor=video_preprocessor,
    )

    if isinstance(video_or_frames, (str, Path, bytes, ProcessedVideo)):
        return pipeline.process_video(
            video_input=video_or_frames,
            frame_attention_matrices=frame_attention_matrices,
            target_class=target_class,
            align_to_original_resolution=align_to_original_resolution,
        )
    elif isinstance(video_or_frames, (list, tuple)):
        return pipeline.process_frames(
            frames=video_or_frames,
            frame_metadata=frame_metadata,
            frame_attention_matrices=frame_attention_matrices,
            target_class=target_class,
            align_to_original_resolution=align_to_original_resolution,
        )
    else:
        raise TypeError(f"Unsupported input type for video_or_frames: {type(video_or_frames).__name__}")
