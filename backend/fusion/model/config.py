"""
TruthLens — Fusion Model Configuration
Author: Parth Maheshwari
Sprint: Development of Cross-Modal Fusion Transformer with dual Decoders.

Torch-free on purpose: the config (and the batching code that depends on it)
can be imported, validated and tested without PyTorch installed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

from backend.fusion.training.schema import MASK_SIZE

# Modality ids used for the type embedding and the cross-modal attention mask.
MOD_FUSION, MOD_IMAGE, MOD_DIFFUSION, MOD_VIDEO, MOD_NOISE = range(5)
NUM_MODALITIES = 5


@dataclass
class FusionModelConfig:
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 2
    ffn_mult: int = 4
    dropout: float = 0.1
    clip_dim: int = 768        # CLIP:ViT [CLS] embedding size
    video_dim: int = 512       # per-frame embedding size from the dual-stream backbone
    map_size: int = MASK_SIZE  # residual map / pixel-mask resolution
    patch: int = 8             # DIRE patchify stride; must be a power of two
    max_frames: int = 32       # longer clips are uniformly sub-sampled to this length
    version: str = "fusion-transformer-v1"

    def __post_init__(self):
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.map_size != MASK_SIZE:
            raise ValueError(f"map_size must equal the training mask size ({MASK_SIZE})")
        if self.patch < 2 or self.patch & (self.patch - 1) or self.map_size % self.patch:
            raise ValueError("patch must be a power of two (>= 2) that divides map_size")
        if self.max_frames < 1 or self.num_layers < 1:
            raise ValueError("max_frames and num_layers must be >= 1")

    @property
    def grid(self) -> int:
        """Side of the DIRE token grid (and of the pixel decoder's seed grid)."""
        return self.map_size // self.patch

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FusionModelConfig":
        return cls(**d)