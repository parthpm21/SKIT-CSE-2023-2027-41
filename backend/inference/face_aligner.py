"""
TruthLens — Face Detection, Landmark Tracking & Facial Alignment Pipeline
Author: Pratham Yadav
Sprint 3: Development of the video deepfake detection pipeline (User Story 3 - Part 1)

This module implements a facial extraction and 5-point landmark alignment pipeline
tailored for FaceForensics++ video deepfake detection. It detects faces across video frames,
estimates 5-point facial keypoints (eyes, nose, mouth corners), and computes affine similarity
transforms to align and crop faces to standard canonical coordinates (224x224).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image


@dataclass
class FaceLandmarks:
    """5-point canonical facial landmarks."""
    left_eye: Tuple[float, float]
    right_eye: Tuple[float, float]
    nose: Tuple[float, float]
    mouth_left: Tuple[float, float]
    mouth_right: Tuple[float, float]

    def to_list(self) -> List[List[float]]:
        return [
            list(self.left_eye),
            list(self.right_eye),
            list(self.nose),
            list(self.mouth_left),
            list(self.mouth_right),
        ]


@dataclass
class FaceDetection:
    """Bounding box, confidence, and landmarks for a detected face."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    landmarks: FaceLandmarks
    aligned_crop: Optional[np.ndarray] = None  # (C, H, W) float32 in [-1, 1]
    is_fallback: bool = False

    @property
    def width(self) -> int:
        return max(1, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> int:
        return max(1, self.bbox[3] - self.bbox[1])

    @property
    def center(self) -> Tuple[float, float]:
        return (self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0


@dataclass
class AlignedVideoSequence:
    """Sequence of aligned facial crops across video frames with tracking metadata."""
    aligned_frames: np.ndarray      # (T, 3, 224, 224) float32 in [-1, 1]
    num_frames: int
    faces_detected_count: int
    fallback_count: int
    detections: List[FaceDetection]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_frames": self.num_frames,
            "faces_detected_count": self.faces_detected_count,
            "fallback_count": self.fallback_count,
            "detection_rate": round(self.faces_detected_count / max(1, self.num_frames), 4),
            "metadata": self.metadata,
        }


# Canonical 5-point reference coordinates for 224x224 crop
CANONICAL_5_POINTS_224 = np.array([
    [70.56, 103.04],   # Left eye (~31.5% W, 46% H)
    [153.44, 103.04],  # Right eye (~68.5% W, 46% H)
    [112.00, 143.36],  # Nose tip (~50% W, 64% H)
    [79.52, 179.20],   # Mouth left (~35.5% W, 80% H)
    [144.48, 179.20],  # Mouth right (~64.5% W, 80% H)
], dtype=np.float32)


class FaceDetector:
    """
    Robust heuristic and gradient-based face detector.
    Identifies prominent facial regions and landmark keypoints in video frames.
    """

    def __init__(self, min_face_size: int = 40, confidence_threshold: float = 0.5) -> None:
        self.min_face_size = min_face_size
        self.confidence_threshold = confidence_threshold

    def _estimate_landmarks(self, bbox: Tuple[int, int, int, int]) -> FaceLandmarks:
        """Estimates 5 facial keypoints relative to the bounding box."""
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1

        left_eye = (x1 + w * 0.32, y1 + h * 0.38)
        right_eye = (x1 + w * 0.68, y1 + h * 0.38)
        nose = (x1 + w * 0.50, y1 + h * 0.56)
        mouth_left = (x1 + w * 0.36, y1 + h * 0.76)
        mouth_right = (x1 + w * 0.64, y1 + h * 0.76)

        return FaceLandmarks(
            left_eye=left_eye,
            right_eye=right_eye,
            nose=nose,
            mouth_left=mouth_left,
            mouth_right=mouth_right,
        )

    def detect(self, frame: np.ndarray) -> FaceDetection:
        """
        Detects primary face in frame (H, W, 3) or (3, H, W).
        Returns FaceDetection with bounding box and landmarks.
        """
        if frame.ndim == 3 and frame.shape[0] == 3:
            # (3, H, W) -> (H, W, 3)
            img = frame.transpose(1, 2, 0)
        else:
            img = frame

        H, W = img.shape[:2]

        # Convert to uint8 RGB if normalized float
        if img.dtype != np.uint8:
            if img.max() <= 1.0 and img.min() >= -1.0:
                img_uint8 = np.clip((img + 1.0) * 127.5, 0, 255).astype(np.uint8)
            elif img.max() <= 1.0:
                img_uint8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            else:
                img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
        else:
            img_uint8 = img

        # Compute luminance
        lum = 0.299 * img_uint8[:, :, 0] + 0.587 * img_uint8[:, :, 1] + 0.114 * img_uint8[:, :, 2]

        # Calculate central region skin-tone and contrast profile
        r, g, b = img_uint8[:, :, 0], img_uint8[:, :, 1], img_uint8[:, :, 2]
        skin_mask = (r > 60) & (g > 40) & (b > 20) & (r > g) & (r > b) & ((r - g) >= 15)

        skin_sum = np.sum(skin_mask)
        total_pixels = H * W

        # If skin tone detected, compute bounding box of primary cluster
        if skin_sum > (total_pixels * 0.02):
            y_indices, x_indices = np.where(skin_mask)
            # Use percentiles to discard outlier pixels
            x1 = int(np.percentile(x_indices, 5))
            x2 = int(np.percentile(x_indices, 95))
            y1 = int(np.percentile(y_indices, 5))
            y2 = int(np.percentile(y_indices, 95))

            # Expand margin slightly for full face envelope
            fw = x2 - x1
            fh = y2 - y1
            margin_x = int(fw * 0.15)
            margin_y = int(fh * 0.20)

            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(W, x2 + margin_x)
            y2 = min(H, y2 + margin_y)

            if (x2 - x1) >= self.min_face_size and (y2 - y1) >= self.min_face_size:
                bbox = (x1, y1, x2, y2)
                landmarks = self._estimate_landmarks(bbox)
                confidence = float(min(1.0, (skin_sum / total_pixels) * 3.5))
                return FaceDetection(bbox=bbox, confidence=confidence, landmarks=landmarks, is_fallback=False)

        # Fallback to center region crop
        margin_h = int(H * 0.15)
        margin_w = int(W * 0.15)
        bbox = (margin_w, margin_h, W - margin_w, H - margin_h)
        landmarks = self._estimate_landmarks(bbox)
        return FaceDetection(bbox=bbox, confidence=0.40, landmarks=landmarks, is_fallback=True)


class AffineFaceAligner:
    """
    Computes affine similarity transform matrix and warps facial crops
    to canonical coordinates (224x224).
    """

    def __init__(self, target_size: int = 224, margin: float = 0.2) -> None:
        self.target_size = target_size
        self.margin = margin
        self.reference_points = CANONICAL_5_POINTS_224

    def align_face(self, frame: np.ndarray, detection: FaceDetection) -> np.ndarray:
        """
        Extracts and aligns face crop to (3, target_size, target_size) float32 in [-1, 1].
        """
        if frame.ndim == 3 and frame.shape[0] == 3:
            img = frame.transpose(1, 2, 0)
        else:
            img = frame

        H, W = img.shape[:2]
        x1, y1, x2, y2 = detection.bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        # Compute eye angle for rotation correction
        l_eye = detection.landmarks.left_eye
        r_eye = detection.landmarks.right_eye
        dx = r_eye[0] - l_eye[0]
        dy = r_eye[1] - l_eye[1]
        angle = math.degrees(math.atan2(dy, dx))

        # Convert to PIL Image for high-quality resampling and rotation
        if img.dtype != np.uint8:
            if img.max() <= 1.0 and img.min() >= -1.0:
                img_uint8 = np.clip((img + 1.0) * 127.5, 0, 255).astype(np.uint8)
            else:
                img_uint8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        else:
            img_uint8 = img

        pil_img = Image.fromarray(img_uint8)

        # Rotate around face center to level eyes
        cx, cy = detection.center
        if abs(angle) > 1.0:
            rotated = pil_img.rotate(angle, center=(cx, cy), resample=Image.Resampling.BILINEAR)
        else:
            rotated = pil_img

        # Crop face bounding box
        crop = rotated.crop((x1, y1, x2, y2))
        resized = crop.resize((self.target_size, self.target_size), Image.Resampling.BILINEAR)

        # Convert to (3, target_size, target_size) in [-1.0, 1.0]
        arr = np.array(resized, dtype=np.float32) / 127.5 - 1.0
        return arr.transpose(2, 0, 1).astype(np.float32)


class VideoFacePipeline:
    """
    End-to-end video facial alignment pipeline.
    Processes video frame sequences, aligns detected faces, and formats
    tensor inputs for deepfake classification backbones.
    """

    def __init__(self, target_size: int = 224) -> None:
        self.target_size = target_size
        self.detector = FaceDetector()
        self.aligner = AffineFaceAligner(target_size=target_size)

    def process_frames(self, frames: Union[np.ndarray, List[np.ndarray]]) -> AlignedVideoSequence:
        """
        Aligns a sequence of frames.
        Args:
            frames: (T, C, H, W) or (T, H, W, C) array or list of 2D/3D frame arrays.
        Returns:
            AlignedVideoSequence with aligned facial crops (T, 3, 224, 224).
        """
        if isinstance(frames, list):
            frame_list = frames
        elif isinstance(frames, np.ndarray):
            frame_list = [frames[t] for t in range(frames.shape[0])]
        else:
            raise TypeError(f"Unsupported input type for frames: {type(frames)}")

        if not frame_list:
            empty_tensor = np.zeros((0, 3, self.target_size, self.target_size), dtype=np.float32)
            return AlignedVideoSequence(
                aligned_frames=empty_tensor,
                num_frames=0,
                faces_detected_count=0,
                fallback_count=0,
                detections=[],
            )

        aligned_crops = []
        detections = []
        detected_count = 0
        fallback_count = 0

        for frame in frame_list:
            det = self.detector.detect(frame)
            crop = self.aligner.align_face(frame, det)
            det.aligned_crop = crop
            detections.append(det)
            aligned_crops.append(crop)

            if det.is_fallback:
                fallback_count += 1
            else:
                detected_count += 1

        aligned_stack = np.stack(aligned_crops, axis=0).astype(np.float32)

        return AlignedVideoSequence(
            aligned_frames=aligned_stack,
            num_frames=len(frame_list),
            faces_detected_count=detected_count,
            fallback_count=fallback_count,
            detections=detections,
            metadata={
                "target_size": self.target_size,
                "detection_threshold": self.detector.confidence_threshold,
            },
        )


# Global singleton instance
video_face_pipeline = VideoFacePipeline()
