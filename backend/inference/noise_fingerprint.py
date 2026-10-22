"""
TruthLens — Noise-Fingerprint Residual Extractor for Video Deepfakes
Author: Pratham Yadav
Sprint 3: Development of the video deepfake detection pipeline (User Story 3 - Part 3)

This module extracts sensor pattern noise residuals and Spatial Rich Model (SRM)
high-frequency fingerprints across video frames. Deepfake synthesis (FaceForensics++,
FaceSwap, Deepfakes, NeuralTextures) disrupts physical camera sensor noise (PRNU)
and introduces localized noise suppression and boundary blending artifacts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class NoiseFingerprintResult:
    """Structured container holding noise residuals, spectral energy, and temporal variance."""
    temporal_inconsistency_score: float  # [0, 1] - higher indicates erratic inter-frame noise
    mean_noise_energy: float             # Average residual variance across frames
    face_bg_noise_discrepancy: float     # Ratio/difference between central facial noise and peripheral noise
    frame_noise_energies: List[float]    # Per-frame noise energy sequence
    residual_maps: np.ndarray            # (T, H, W) normalized high-pass noise heatmaps
    is_noise_anomalous: bool             # Whether noise statistics deviate from natural sensor PRNU
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "temporal_inconsistency_score": round(float(self.temporal_inconsistency_score), 4),
            "mean_noise_energy": round(float(self.mean_noise_energy), 6),
            "face_bg_noise_discrepancy": round(float(self.face_bg_noise_discrepancy), 4),
            "frame_noise_energies": [round(float(e), 6) for e in self.frame_noise_energies],
            "is_noise_anomalous": self.is_noise_anomalous,
            "metadata": self.metadata,
        }


# Standard 3x3 and 5x5 Spatial Rich Model (SRM) high-pass filtering kernels
SRM_KERNEL_1ST_ORDER = np.array([
    [0, 0, 0],
    [-1, 2, -1],
    [0, 0, 0]
], dtype=np.float32) * 0.5

SRM_KERNEL_2ND_ORDER = np.array([
    [-1, 2, -1],
    [2, -4, 2],
    [-1, 2, -1]
], dtype=np.float32) * 0.25

SRM_KERNEL_SQUARE_3X3 = np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
], dtype=np.float32) * 0.125


class NoiseFingerprintExtractor:
    """
    Extracts high-frequency noise residuals and analyzes inter-frame sensor consistency.
    Detects PRNU disruptions, blending boundary artifacts, and temporal noise flicker.
    """

    def __init__(
        self,
        inconsistency_threshold: float = 0.45,
        discrepancy_threshold: float = 0.35,
    ) -> None:
        self.inconsistency_threshold = inconsistency_threshold
        self.discrepancy_threshold = discrepancy_threshold
        self.kernels = [
            SRM_KERNEL_1ST_ORDER,
            SRM_KERNEL_2ND_ORDER,
            SRM_KERNEL_SQUARE_3X3,
        ]

    def _apply_kernel_2d(self, channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Applies 3x3 convolution with reflection padding on a 2D single-channel matrix."""
        H, W = channel.shape
        padded = np.pad(channel, ((1, 1), (1, 1)), mode="reflect")
        out = np.zeros((H, W), dtype=np.float32)
        for di in range(3):
            for dj in range(3):
                out += padded[di : di + H, dj : dj + W] * kernel[di, dj]
        return out

    def extract_frame_noise(self, frame: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Extracts high-frequency noise residual map and energy metrics for a single frame.
        
        Args:
            frame: (3, H, W) float32 in [-1, 1] or uint8.
        Returns:
            residual_map: (H, W) normalized [0, 1] spatial noise intensity.
            noise_energy: scalar variance of high-frequency components.
            face_bg_discrepancy: ratio of central facial region noise to boundary region noise.
        """
        if frame.ndim == 3 and frame.shape[0] == 3:
            # (3, H, W) -> grayscale luminance
            gray = 0.299 * frame[0] + 0.587 * frame[1] + 0.114 * frame[2]
        elif frame.ndim == 3 and frame.shape[2] == 3:
            gray = 0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]
        elif frame.ndim == 2:
            gray = frame.astype(np.float32)
        else:
            raise ValueError(f"Unsupported frame shape: {frame.shape}")

        # Multi-kernel SRM filtering
        residuals = []
        for kernel in self.kernels:
            res = self._apply_kernel_2d(gray, kernel)
            residuals.append(res)

        # Combined magnitude across filter responses
        combined = np.mean(np.stack([np.abs(r) for r in residuals], axis=0), axis=0)

        # Global high-frequency noise energy (variance)
        noise_energy = float(np.var(combined))

        # Central facial ROI vs peripheral background discrepancy
        H, W = combined.shape
        ch_start, ch_end = int(H * 0.25), int(H * 0.75)
        cw_start, cw_end = int(W * 0.25), int(W * 0.75)

        center_region = combined[ch_start:ch_end, cw_start:cw_end]
        border_mask = np.ones((H, W), dtype=bool)
        border_mask[ch_start:ch_end, cw_start:cw_end] = False
        border_region = combined[border_mask]

        center_var = float(np.var(center_region)) if center_region.size > 0 else 1e-6
        border_var = float(np.var(border_region)) if border_region.size > 0 else 1e-6

        # In deepfakes, synthesized face regions frequently have suppressed or mismatched variance
        discrepancy = abs(center_var - border_var) / (center_var + border_var + 1e-8)

        # Spatial map normalized to [0, 1]
        c_min, c_max = float(np.min(combined)), float(np.max(combined))
        if c_max > c_min:
            norm_map = (combined - c_min) / (c_max - c_min)
        else:
            norm_map = np.zeros_like(combined)

        return norm_map.astype(np.float32), noise_energy, float(discrepancy)

    def analyze_sequence(self, frames: np.ndarray) -> NoiseFingerprintResult:
        """
        Analyzes noise fingerprint consistency across an aligned video frame sequence.
        
        Args:
            frames: (T, 3, H, W) or (T, H, W, 3) frame tensor stack.
        Returns:
            NoiseFingerprintResult with temporal inconsistency and discrepancy metrics.
        """
        T = frames.shape[0]
        if T == 0:
            return NoiseFingerprintResult(
                temporal_inconsistency_score=0.0,
                mean_noise_energy=0.0,
                face_bg_noise_discrepancy=0.0,
                frame_noise_energies=[],
                residual_maps=np.zeros((0, 224, 224), dtype=np.float32),
                is_noise_anomalous=False,
            )

        residual_maps = []
        frame_energies = []
        discrepancies = []

        for t in range(T):
            res_map, energy, disc = self.extract_frame_noise(frames[t])
            residual_maps.append(res_map)
            frame_energies.append(energy)
            discrepancies.append(disc)

        stack_maps = np.stack(residual_maps, axis=0)  # (T, H, W)
        mean_energy = float(np.mean(frame_energies))
        mean_discrepancy = float(np.mean(discrepancies))

        # Temporal inter-frame noise inconsistency
        if T > 1:
            # Measure frame-to-frame residual variance jumps: Delta N(t) = ||N(t) - N(t-1)||_2
            frame_diffs = []
            for t in range(1, T):
                diff = np.mean((stack_maps[t] - stack_maps[t - 1]) ** 2)
                frame_diffs.append(diff)
            
            # Normalize temporal jump score via logistic scaling
            avg_diff = float(np.mean(frame_diffs))
            temp_inconsistency = float(1.0 / (1.0 + math.exp(-12.0 * (avg_diff - 0.15))))
        else:
            temp_inconsistency = 0.0

        is_anomalous = (
            temp_inconsistency >= self.inconsistency_threshold or
            mean_discrepancy >= self.discrepancy_threshold
        )

        return NoiseFingerprintResult(
            temporal_inconsistency_score=temp_inconsistency,
            mean_noise_energy=mean_energy,
            face_bg_noise_discrepancy=mean_discrepancy,
            frame_noise_energies=frame_energies,
            residual_maps=stack_maps,
            is_noise_anomalous=is_anomalous,
            metadata={
                "num_frames": T,
                "inconsistency_threshold": self.inconsistency_threshold,
                "discrepancy_threshold": self.discrepancy_threshold,
            },
        )


# Global singleton instance
noise_fingerprint_extractor = NoiseFingerprintExtractor()
