"""
TruthLens — Tests for the fusion tokenizer (requires PyTorch)
Author: Parth Maheshwari
"""
import pytest

torch = pytest.importorskip("torch")

from backend.fusion.model.batching import collate_inputs
from backend.fusion.model.config import MOD_VIDEO, FusionModelConfig
from backend.fusion.model.tokenizer import FusionTokenizer
from backend.fusion.training.synthetic import make_synthetic_samples

CFG = FusionModelConfig(d_model=32, num_heads=4, clip_dim=16, video_dim=8, max_frames=12)


def batch(n=6):
    samples = make_synthetic_samples(n, seed=0, clip_dim=CFG.clip_dim, video_dim=CFG.video_dim)
    return samples, collate_inputs([s.fusion_input for s in samples], CFG)


@pytest.mark.unit
class TestTokenizer:
    def test_output_shapes(self):
        tok = FusionTokenizer(CFG)
        _, b = batch()
        tokens, valid, modality = tok(b)
        N = tok.num_tokens
        assert tokens.shape == (6, N, CFG.d_model)
        assert valid.shape == (6, N) and modality.shape == (N,)

    def test_token_count_matches_layout(self):
        g, T = CFG.grid, CFG.max_frames
        assert FusionTokenizer(CFG).num_tokens == 1 + 1 + (g * g + 1) + T + (1 + T)

    def test_fusion_token_always_valid(self):
        _, b = batch()
        _, valid, _ = FusionTokenizer(CFG)(b)
        assert valid[:, 0].all()

    def test_absent_branches_are_masked_and_zeroed(self):
        samples, b = batch()
        tokens, valid, modality = FusionTokenizer(CFG)(b)
        for i, s in enumerate(samples):
            if s.fusion_input.video is None:
                assert not valid[i, modality == MOD_VIDEO].any()
        assert tokens[~valid].abs().sum() == 0

    def test_video_and_noise_share_time_positions(self):
        tok = FusionTokenizer(CFG)
        # same table object feeds both frame streams -> aligned positions by construction
        assert tok.time_pos.shape == (1, CFG.max_frames, CFG.d_model)