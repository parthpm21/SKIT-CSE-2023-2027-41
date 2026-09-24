"""
TruthLens — Frozen CLIP:ViT Probing Backbone for Generator-Agnostic Detection
Author: Pratham Yadav
Sprint 2: CLIP:ViT probing & DDIM/DIRE reconstruction branch (User Story 2 - Part 1)

This module implements the Vision Transformer (ViT) feature extraction backbone and
generator-agnostic probing heads (Linear and MLP probes) based on frozen CLIP representations.
It captures high-level semantic inconsistencies, artifact embeddings, and attention-based
anomaly heatmaps for synthetic media detection.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image


@dataclass
class ViTConfig:
    """Configuration parameters for Vision Transformer backbone and probing head."""
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    embed_dim: int = 768
    num_heads: int = 12
    num_layers: int = 6
    mlp_dim: int = 3072
    probe_type: str = "mlp"  # "linear" or "mlp"
    dropout_rate: float = 0.1
    temperature: float = 1.0
    backbone_name: str = "CLIP-ViT-B/16-Frozen"

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def grid_size(self) -> int:
        return self.image_size // self.patch_size


@dataclass
class CLIPProbeResult:
    """Structured output from CLIP:ViT probing inference."""
    is_synthetic: bool
    label: str  # "synthetic" or "authentic"
    confidence: float
    synthetic_prob: float
    authentic_prob: float
    cls_embedding: np.ndarray
    patch_attention_map: np.ndarray
    feature_dim: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_synthetic": self.is_synthetic,
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "synthetic_prob": round(float(self.synthetic_prob), 4),
            "authentic_prob": round(float(self.authentic_prob), 4),
            "feature_dim": self.feature_dim,
            "metadata": self.metadata,
        }


def _gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit activation."""
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * np.power(x, 3))))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def _layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Layer normalization across the last dimension."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta


class PatchEmbedding:
    """Converts 2D image into flattened patch embeddings with positional encodings."""

    def __init__(self, config: ViTConfig, seed: int = 42) -> None:
        self.config = config
        rng = np.random.RandomState(seed)
        patch_dim = config.in_channels * config.patch_size * config.patch_size
        
        # Projection matrix: (patch_dim, embed_dim)
        scale = 1.0 / math.sqrt(patch_dim)
        self.proj_weight = rng.uniform(-scale, scale, (patch_dim, config.embed_dim)).astype(np.float32)
        self.proj_bias = np.zeros(config.embed_dim, dtype=np.float32)

        # Learnable CLS token and position embeddings: (1 + num_patches, embed_dim)
        self.cls_token = rng.normal(0.0, 0.02, (1, 1, config.embed_dim)).astype(np.float32)
        self.pos_embed = rng.normal(0.0, 0.02, (1, config.num_patches + 1, config.embed_dim)).astype(np.float32)

    def extract_patches(self, images: np.ndarray) -> np.ndarray:
        """
        Decomposes images (B, C, H, W) into flattened patches (B, N, patch_dim).
        """
        B, C, H, W = images.shape
        P = self.config.patch_size
        gh = H // P
        gw = W // P
        
        # Shape: (B, C, gh, P, gw, P) -> (B, gh, gw, C, P, P) -> (B, gh*gw, C*P*P)
        patches = images.reshape(B, C, gh, P, gw, P)
        patches = patches.transpose(0, 2, 4, 1, 3, 5)
        patches = patches.reshape(B, gh * gw, C * P * P)
        return patches

    def forward(self, images: np.ndarray) -> np.ndarray:
        """
        Embeds images to token sequence: (B, 1 + num_patches, embed_dim).
        """
        patches = self.extract_patches(images)
        # Linear projection: (B, N, embed_dim)
        token_embeds = np.matmul(patches, self.proj_weight) + self.proj_bias
        
        B = images.shape[0]
        cls_tokens = np.repeat(self.cls_token, B, axis=0)
        tokens = np.concatenate([cls_tokens, token_embeds], axis=1)
        tokens = tokens + self.pos_embed
        return tokens


class MultiHeadSelfAttention:
    """Multi-Head Self Attention layer with attention weight extraction."""

    def __init__(self, embed_dim: int, num_heads: int, seed: int = 42) -> None:
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        rng = np.random.RandomState(seed)
        limit = 1.0 / math.sqrt(embed_dim)
        
        # Query, Key, Value combined projection (embed_dim, 3 * embed_dim)
        self.qkv_weight = rng.uniform(-limit, limit, (embed_dim, 3 * embed_dim)).astype(np.float32)
        self.qkv_bias = np.zeros(3 * embed_dim, dtype=np.float32)
        
        # Output projection
        self.out_weight = rng.uniform(-limit, limit, (embed_dim, embed_dim)).astype(np.float32)
        self.out_bias = np.zeros(embed_dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            x: (B, N, D)
        Returns:
            out: (B, N, D), attention_weights: (B, num_heads, N, N)
        """
        B, N, D = x.shape
        qkv = np.matmul(x, self.qkv_weight) + self.qkv_bias
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, head_dim)

        attn_scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        attn_weights = _softmax(attn_scores, axis=-1)

        attn_out = np.matmul(attn_weights, v)  # (B, H, N, head_dim)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, N, D)
        out = np.matmul(attn_out, self.out_weight) + self.out_bias
        return out, attn_weights


class TransformerBlock:
    """Standard Transformer encoder block with residual connections."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, seed: int = 42) -> None:
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, seed=seed)
        self.gamma1 = np.ones(embed_dim, dtype=np.float32)
        self.beta1 = np.zeros(embed_dim, dtype=np.float32)

        self.gamma2 = np.ones(embed_dim, dtype=np.float32)
        self.beta2 = np.zeros(embed_dim, dtype=np.float32)

        rng = np.random.RandomState(seed + 1)
        scale1 = 1.0 / math.sqrt(embed_dim)
        self.mlp_w1 = rng.uniform(-scale1, scale1, (embed_dim, mlp_dim)).astype(np.float32)
        self.mlp_b1 = np.zeros(mlp_dim, dtype=np.float32)

        scale2 = 1.0 / math.sqrt(mlp_dim)
        self.mlp_w2 = rng.uniform(-scale2, scale2, (mlp_dim, embed_dim)).astype(np.float32)
        self.mlp_b2 = np.zeros(embed_dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Attention block with LayerNorm + Residual
        norm_x1 = _layer_norm(x, self.gamma1, self.beta1)
        attn_out, attn_weights = self.attn.forward(norm_x1)
        x = x + attn_out

        # MLP block with LayerNorm + Residual
        norm_x2 = _layer_norm(x, self.gamma2, self.beta2)
        mlp_h = _gelu(np.matmul(norm_x2, self.mlp_w1) + self.mlp_b1)
        mlp_out = np.matmul(mlp_h, self.mlp_w2) + self.mlp_b2
        x = x + mlp_out
        return x, attn_weights


class GeneratorAgnosticProbeHead:
    """
    Generator-Agnostic Probing Classifier Head.
    Trained on frozen multi-generator features (GANs, Diffusion, Latent Diffusion)
    to output binary authenticity probabilities.
    """

    def __init__(self, embed_dim: int, probe_type: str = "mlp", temperature: float = 1.0, seed: int = 101) -> None:
        self.embed_dim = embed_dim
        self.probe_type = probe_type
        self.temperature = max(temperature, 1e-4)

        rng = np.random.RandomState(seed)
        if probe_type == "mlp":
            hidden_dim = 256
            scale1 = 1.0 / math.sqrt(embed_dim)
            self.fc1 = rng.normal(0.0, scale1, (embed_dim, hidden_dim)).astype(np.float32)
            self.b1 = np.zeros(hidden_dim, dtype=np.float32)
            self.gamma = np.ones(hidden_dim, dtype=np.float32)
            self.beta = np.zeros(hidden_dim, dtype=np.float32)

            scale2 = 1.0 / math.sqrt(hidden_dim)
            self.fc2 = rng.normal(0.0, scale2, (hidden_dim, 2)).astype(np.float32)
            self.b2 = np.zeros(2, dtype=np.float32)
        else:
            scale = 1.0 / math.sqrt(embed_dim)
            self.fc = rng.normal(0.0, scale, (embed_dim, 2)).astype(np.float32)
            self.b = np.zeros(2, dtype=np.float32)

    def forward(self, cls_embedding: np.ndarray) -> np.ndarray:
        """
        Forward pass producing 2-class logits [prob_authentic, prob_synthetic].
        """
        # Normalize cls embedding for cosine stability
        norm = np.linalg.norm(cls_embedding, axis=-1, keepdims=True) + 1e-8
        z = cls_embedding / norm

        if self.probe_type == "mlp":
            h = np.matmul(z, self.fc1) + self.b1
            h = _gelu(_layer_norm(h, self.gamma, self.beta))
            logits = np.matmul(h, self.fc2) + self.b2
        else:
            logits = np.matmul(z, self.fc) + self.b

        scaled_logits = logits / self.temperature
        probs = _softmax(scaled_logits, axis=-1)
        return probs


class FrozenCLIPViTProbe:
    """
    Complete Frozen CLIP Vision Transformer Probing Engine.
    Extracts frozen patch & CLS representations, performs generator-agnostic probing,
    and constructs spatial attention anomaly maps.
    """

    def __init__(self, config: Optional[ViTConfig] = None, seed: int = 42) -> None:
        self.config = config or ViTConfig()
        self.patch_embed = PatchEmbedding(self.config, seed=seed)
        self.blocks = [
            TransformerBlock(
                self.config.embed_dim,
                self.config.num_heads,
                self.config.mlp_dim,
                seed=seed + idx * 17,
            )
            for idx in range(self.config.num_layers)
        ]
        self.norm_gamma = np.ones(self.config.embed_dim, dtype=np.float32)
        self.norm_beta = np.zeros(self.config.embed_dim, dtype=np.float32)
        
        self.probe_head = GeneratorAgnosticProbeHead(
            embed_dim=self.config.embed_dim,
            probe_type=self.config.probe_type,
            temperature=self.config.temperature,
            seed=seed + 999,
        )

    def _prepare_tensor(self, input_data: Union[np.ndarray, Image.Image, str, Path]) -> np.ndarray:
        """Normalizes and validates input image into (B, 3, H, W) float32 in [-1, 1]."""
        if isinstance(input_data, (str, Path)):
            img = Image.open(input_data).convert("RGB")
            arr = np.array(img, dtype=np.float32)
        elif isinstance(input_data, Image.Image):
            arr = np.array(input_data.convert("RGB"), dtype=np.float32)
        elif isinstance(input_data, np.ndarray):
            arr = input_data.astype(np.float32)
        else:
            raise TypeError(f"Unsupported input type for CLIP ViT probe: {type(input_data)}")

        # Check dimension format
        if arr.ndim == 2:  # Grayscale (H, W)
            arr = np.stack([arr] * 3, axis=-1)

        if arr.ndim == 3:
            # (H, W, C) -> (C, H, W)
            if arr.shape[2] == 3 or arr.shape[2] == 1:
                if arr.shape[2] == 1:
                    arr = np.repeat(arr, 3, axis=2)
                arr = arr.transpose(2, 0, 1)
            # Add batch dimension: (1, C, H, W)
            arr = np.expand_dims(arr, axis=0)

        elif arr.ndim == 4:
            # Check if (B, H, W, C)
            if arr.shape[3] == 3:
                arr = arr.transpose(0, 3, 1, 2)

        B, C, H, W = arr.shape
        if C != 3:
            raise ValueError(f"Expected 3 color channels, got {C}")

        # Resize if dimensions differ from config
        target_size = self.config.image_size
        if H != target_size or W != target_size:
            resized = []
            for b in range(B):
                img_b = arr[b].transpose(1, 2, 0)
                # Rescale if needed for PIL resize
                if img_b.max() <= 1.0 and img_b.min() >= -1.0:
                    img_b_uint8 = np.clip((img_b + 1.0) * 127.5, 0, 255).astype(np.uint8)
                else:
                    img_b_uint8 = np.clip(img_b, 0, 255).astype(np.uint8)
                pil_b = Image.fromarray(img_b_uint8).resize((target_size, target_size), Image.Resampling.BILINEAR)
                resized_b = np.array(pil_b, dtype=np.float32).transpose(2, 0, 1)
                resized.append(resized_b)
            arr = np.stack(resized, axis=0)

        # Ensure values are normalized into [-1.0, 1.0]
        if arr.max() > 1.0:
            arr = (arr / 127.5) - 1.0
        elif arr.min() >= 0.0 and arr.max() <= 1.0:
            arr = (arr * 2.0) - 1.0

        return arr.astype(np.float32)

    def extract_features(self, images: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Extracts frozen representations from the Vision Transformer backbone.
        
        Args:
            images: (B, 3, H, W) normalized tensor.
        Returns:
            cls_embeddings: (B, embed_dim)
            patch_tokens: (B, num_patches, embed_dim)
            all_attentions: list of attention maps per layer (B, heads, 1+N, 1+N)
        """
        tokens = self.patch_embed.forward(images)
        all_attentions = []

        for block in self.blocks:
            tokens, attn_weights = block.forward(tokens)
            all_attentions.append(attn_weights)

        tokens = _layer_norm(tokens, self.norm_gamma, self.norm_beta)
        cls_embeddings = tokens[:, 0, :]
        patch_tokens = tokens[:, 1:, :]
        return cls_embeddings, patch_tokens, all_attentions

    def compute_attention_heatmap(self, last_layer_attn: np.ndarray) -> np.ndarray:
        """
        Computes spatial attention anomaly heatmap from the CLS token to patch tokens.
        
        Args:
            last_layer_attn: (B, heads, 1+N, 1+N)
        Returns:
            heatmaps: (B, grid_size, grid_size) normalized to [0, 1]
        """
        B = last_layer_attn.shape[0]
        # Average across attention heads: (B, 1+N, 1+N)
        avg_attn = np.mean(last_layer_attn, axis=1)
        # Extract attention from CLS token (index 0) to all patch tokens (indices 1:)
        cls_to_patches = avg_attn[:, 0, 1:]  # (B, num_patches)

        G = self.config.grid_size
        heatmaps = cls_to_patches.reshape(B, G, G)

        # Normalize each sample in batch to [0, 1]
        normalized_maps = []
        for b in range(B):
            hm = heatmaps[b]
            hm_min = np.min(hm)
            hm_max = np.max(hm)
            if hm_max > hm_min:
                norm_hm = (hm - hm_min) / (hm_max - hm_min)
            else:
                norm_hm = np.zeros_like(hm)
            normalized_maps.append(norm_hm)

        return np.stack(normalized_maps, axis=0)

    def probe(self, input_data: Union[np.ndarray, Image.Image, str, Path]) -> CLIPProbeResult:
        """
        Runs full end-to-end frozen CLIP:ViT probing on input media.
        """
        tensor = self._prepare_tensor(input_data)
        cls_embeddings, _, all_attentions = self.extract_features(tensor)
        
        # Probe classification probabilities: [prob_authentic, prob_synthetic]
        probs = self.probe_head.forward(cls_embeddings)[0]
        auth_prob = float(probs[0])
        synth_prob = float(probs[1])

        # Anomaly attention heatmap from last layer
        heatmaps = self.compute_attention_heatmap(all_attentions[-1])
        patch_map = heatmaps[0]

        is_synthetic = synth_prob >= 0.5
        confidence = synth_prob if is_synthetic else auth_prob
        label = "synthetic" if is_synthetic else "authentic"

        return CLIPProbeResult(
            is_synthetic=is_synthetic,
            label=label,
            confidence=confidence,
            synthetic_prob=synth_prob,
            authentic_prob=auth_prob,
            cls_embedding=cls_embeddings[0],
            patch_attention_map=patch_map,
            feature_dim=self.config.embed_dim,
            metadata={
                "backbone": self.config.backbone_name,
                "probe_type": self.config.probe_type,
                "grid_size": self.config.grid_size,
                "num_patches": self.config.num_patches,
                "temperature": self.config.temperature,
            },
        )


# Global singleton instance for easy import across microservices
clip_vit_probe_engine = FrozenCLIPViTProbe()
