"""
TruthLens — Automated Tests for Frozen CLIP:ViT Probing Backbone
Author: Pratham Yadav
Sprint 2: User Story 2 - Part 1 Validation

Validates Vision Transformer patch tokenization, multi-head attention weights,
generator-agnostic probing heads, probability calibration, and attention anomaly heatmaps.
"""
import numpy as np
import pytest
from PIL import Image

from backend.inference.clip_vit_probe import (
    ViTConfig,
    PatchEmbedding,
    MultiHeadSelfAttention,
    TransformerBlock,
    GeneratorAgnosticProbeHead,
    FrozenCLIPViTProbe,
    CLIPProbeResult,
    _softmax,
    _layer_norm,
)


class TestCLIPViTProbingBackbone:
    """Test suite for Vision Transformer backbone and generator-agnostic probe."""

    @pytest.fixture
    def default_config(self) -> ViTConfig:
        return ViTConfig(
            image_size=224,
            patch_size=16,
            embed_dim=768,
            num_heads=12,
            num_layers=2,  # Compact depth for fast unit test execution
            mlp_dim=1536,
            probe_type="mlp",
            temperature=1.0,
        )

    @pytest.fixture
    def sample_image_tensor(self) -> np.ndarray:
        # Batch of 2 RGB images (2, 3, 224, 224) in [-1, 1]
        rng = np.random.RandomState(42)
        return rng.uniform(-1.0, 1.0, (2, 3, 224, 224)).astype(np.float32)

    def test_patch_embedding_shape_and_tokens(self, default_config: ViTConfig, sample_image_tensor: np.ndarray):
        patch_embed = PatchEmbedding(default_config, seed=42)
        tokens = patch_embed.forward(sample_image_tensor)

        # 224 / 16 = 14 -> 14*14 = 196 patches + 1 CLS token = 197 tokens
        assert tokens.shape == (2, 197, 768)
        assert np.isfinite(tokens).all(), "Patch tokens contain NaN or Inf values"

    def test_multi_head_self_attention_invariants(self, default_config: ViTConfig):
        attn = MultiHeadSelfAttention(embed_dim=768, num_heads=12, seed=42)
        dummy_tokens = np.random.randn(2, 197, 768).astype(np.float32)

        out, attn_weights = attn.forward(dummy_tokens)

        # Output shape must match input token shape
        assert out.shape == (2, 197, 768)
        # Attention weights shape: (B, num_heads, N, N)
        assert attn_weights.shape == (2, 12, 197, 197)

        # Attention weights must sum to 1.0 along the last axis (softmax property)
        sums = np.sum(attn_weights, axis=-1)
        np.testing.assert_allclose(sums, np.ones_like(sums), atol=1e-5)

    def test_transformer_block_residual_connections(self, default_config: ViTConfig):
        block = TransformerBlock(embed_dim=768, num_heads=12, mlp_dim=1536, seed=42)
        dummy_tokens = np.random.randn(2, 197, 768).astype(np.float32)

        out, attn_weights = block.forward(dummy_tokens)
        assert out.shape == (2, 197, 768)
        assert attn_weights.shape == (2, 12, 197, 197)
        assert not np.array_equal(out, dummy_tokens), "Transformer block must transform tokens"

    def test_generator_agnostic_probe_heads(self):
        cls_embeddings = np.random.randn(4, 768).astype(np.float32)

        # 1. Test MLP Probe Head
        mlp_head = GeneratorAgnosticProbeHead(embed_dim=768, probe_type="mlp", temperature=1.0)
        mlp_probs = mlp_head.forward(cls_embeddings)
        assert mlp_probs.shape == (4, 2)
        np.testing.assert_allclose(np.sum(mlp_probs, axis=-1), np.ones(4), atol=1e-5)
        assert (mlp_probs >= 0.0).all() and (mlp_probs <= 1.0).all()

        # 2. Test Linear Probe Head
        lin_head = GeneratorAgnosticProbeHead(embed_dim=768, probe_type="linear", temperature=1.2)
        lin_probs = lin_head.forward(cls_embeddings)
        assert lin_probs.shape == (4, 2)
        np.testing.assert_allclose(np.sum(lin_probs, axis=-1), np.ones(4), atol=1e-5)

    def test_end_to_end_probe_inference(self, default_config: ViTConfig):
        engine = FrozenCLIPViTProbe(config=default_config, seed=42)
        dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        result = engine.probe(dummy_img)

        assert isinstance(result, CLIPProbeResult)
        assert result.label in {"synthetic", "authentic"}
        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.synthetic_prob <= 1.0
        assert 0.0 <= result.authentic_prob <= 1.0
        np.testing.assert_allclose(result.synthetic_prob + result.authentic_prob, 1.0, atol=1e-4)

        # Validate CLS embedding shape and norm
        assert result.cls_embedding.shape == (768,)
        assert result.feature_dim == 768

        # Validate Attention Anomaly Heatmap
        assert result.patch_attention_map.shape == (14, 14)
        assert result.patch_attention_map.min() >= 0.0
        assert result.patch_attention_map.max() <= 1.0

        # Validate serialization
        res_dict = result.to_dict()
        assert "confidence" in res_dict
        assert "is_synthetic" in res_dict
        assert res_dict["metadata"]["backbone"] == default_config.backbone_name

    def test_probe_handles_pil_image_and_grayscale(self, default_config: ViTConfig):
        engine = FrozenCLIPViTProbe(config=default_config, seed=42)

        # 1. PIL Image with arbitrary resolution (300 x 400)
        pil_img = Image.new("RGB", (300, 400), color=(128, 64, 32))
        res_pil = engine.probe(pil_img)
        assert res_pil.label in {"synthetic", "authentic"}

        # 2. 2D Grayscale array
        gray_arr = np.random.uniform(0.0, 1.0, (120, 120)).astype(np.float32)
        res_gray = engine.probe(gray_arr)
        assert res_gray.label in {"synthetic", "authentic"}
        assert res_gray.patch_attention_map.shape == (14, 14)
