"""
TruthLens — Training Sample Schema for Cross-Modal Fusion
Author: Pratham Yadav
Sprint: Feature-branch support for the Fusion Engine (User Story 4 - Part 1).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from backend.fusion.contracts import FusionInput

MASK_SIZE: int = 32


def weak_pixel_mask(verdict: int) -> np.ndarray:
    """Generate a weak supervision binary pixel mask of shape (MASK_SIZE, MASK_SIZE)."""
    val = 1.0 if verdict else 0.0
    return np.full((MASK_SIZE, MASK_SIZE), val, dtype=np.float32)


def weak_frame_labels(verdict: int, num_frames: int) -> np.ndarray:
    """Generate weak supervision binary frame labels of length num_frames."""
    val = 1.0 if verdict else 0.0
    return np.full((num_frames,), val, dtype=np.float32)


@dataclass
class FusionSample:
    fusion_input: FusionInput
    verdict: int
    pixel_mask: Optional[np.ndarray] = None
    frame_labels: Optional[np.ndarray] = None
    source: Optional[str] = None

    def __post_init__(self):
        if self.verdict not in (0, 1):
            raise ValueError(f"verdict must be 0 or 1, got {self.verdict}")

        if self.pixel_mask is not None:
            self.pixel_mask = np.asarray(self.pixel_mask, dtype=np.float32)
            if self.pixel_mask.shape != (MASK_SIZE, MASK_SIZE):
                raise ValueError(
                    f"pixel_mask shape must be ({MASK_SIZE}, {MASK_SIZE}), got {self.pixel_mask.shape}"
                )
            if (self.pixel_mask < 0.0).any() or (self.pixel_mask > 1.0).any():
                raise ValueError("pixel_mask values must be in [0.0, 1.0]")

        if self.frame_labels is not None:
            self.frame_labels = np.asarray(self.frame_labels, dtype=np.float32)
            if not np.all(np.isin(self.frame_labels, [0.0, 1.0])):
                raise ValueError("frame_labels must be binary (0.0 or 1.0)")
            if self.fusion_input.video is None or len(self.frame_labels) != len(
                self.fusion_input.video.frame_embeddings
            ):
                raise ValueError("frame_labels length must match video frame_embeddings length")

    def to_dict(self) -> Dict[str, Any]:
        input_dict = (
            self.fusion_input.model_dump()
            if hasattr(self.fusion_input, "model_dump")
            else self.fusion_input.dict()
        )
        return {
            "fusion_input": input_dict,
            "verdict": int(self.verdict),
            "pixel_mask": self.pixel_mask.tolist() if self.pixel_mask is not None else None,
            "frame_labels": self.frame_labels.tolist() if self.frame_labels is not None else None,
            "source": self.source,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FusionSample":
        fusion_input = FusionInput(**d["fusion_input"])
        pixel_mask = (
            np.asarray(d["pixel_mask"], dtype=np.float32)
            if d.get("pixel_mask") is not None
            else None
        )
        frame_labels = (
            np.asarray(d["frame_labels"], dtype=np.float32)
            if d.get("frame_labels") is not None
            else None
        )
        return cls(
            fusion_input=fusion_input,
            verdict=d["verdict"],
            pixel_mask=pixel_mask,
            frame_labels=frame_labels,
            source=d.get("source"),
        )

    @classmethod
    def from_json(cls, s: str) -> "FusionSample":
        return cls.from_dict(json.loads(s))


def write_jsonl(samples: Sequence[FusionSample], path: Union[str, Path]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(sample.to_json() + "\n")
    return len(samples)


def read_jsonl(path: Union[str, Path]) -> List[FusionSample]:
    path = Path(path)
    samples: List[FusionSample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(FusionSample.from_json(line))
    return samples
