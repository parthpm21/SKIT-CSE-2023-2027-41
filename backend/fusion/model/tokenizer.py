"""
TruthLens — Modality Tokenizer
Author: Parth Maheshwari
Sprint: Development of Cross-Modal Fusion Transformer with dual Decoders.

Turns the four branches into one token sequence in a shared d_model space:

  [FUS] | CLIP token | DIRE patch tokens (grid^2) + DIRE scalar token
        | video frame tokens (T) | noise scalar token + noise frame tokens (T)

* Video and noise frame tokens share one temporal position table, so frame t of
  the video and frame t of the noise fingerprint land at the same position -
  this is the "alignment" the cross-modal attention then exploits.
* Every token also gets a modality-type embedding.
* A mask marks which tokens are real; absent branches are simply masked out.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from backend.fusion.model.config import (
    MOD_DIFFUSION, MOD_FUSION, MOD_IMAGE, MOD_NOISE, MOD_VIDEO, NUM_MODALITIES, FusionModelConfig,
)


class FusionTokenizer(nn.Module):
    def __init__(self, cfg: FusionModelConfig):
        super().__init__()
        d, g, T = cfg.d_model, cfg.grid, cfg.max_frames
        self.cfg = cfg

        self.fus_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        self.clip_norm = nn.LayerNorm(cfg.clip_dim)
        self.clip_proj = nn.Linear(cfg.clip_dim, d)
        self.clip_score = nn.Linear(1, d)

        self.dire_patch = nn.Conv2d(1, d, kernel_size=cfg.patch, stride=cfg.patch)
        self.dire_pos = nn.Parameter(torch.randn(1, g * g, d) * 0.02)
        self.dire_scalar = nn.Linear(2, d)

        self.vid_norm = nn.LayerNorm(cfg.video_dim)
        self.vid_proj = nn.Linear(cfg.video_dim + 1, d)
        self.time_pos = nn.Parameter(torch.randn(1, T, d) * 0.02)  # shared by video + noise

        self.noise_scalar = nn.Linear(3, d)
        self.noise_frame = nn.Linear(1, d)

        self.type_emb = nn.Embedding(NUM_MODALITIES, d)
        self.out_norm = nn.LayerNorm(d)

        layout = ([MOD_FUSION] + [MOD_IMAGE] + [MOD_DIFFUSION] * (g * g + 1)
                  + [MOD_VIDEO] * T + [MOD_NOISE] * (1 + T))
        self.register_buffer("modality", torch.tensor(layout, dtype=torch.long), persistent=False)

    @property
    def num_tokens(self) -> int:
        return int(self.modality.numel())

    def _type(self, mod: int, like: torch.Tensor) -> torch.Tensor:
        return self.type_emb(torch.tensor(mod, device=like.device))

    def forward(self, b):
        B = b["clip_vec"].size(0)
        dev = b["clip_vec"].device

        fus = self.fus_token.expand(B, -1, -1) + self._type(MOD_FUSION, b["clip_vec"])
        fus_valid = torch.ones(B, 1, dtype=torch.bool, device=dev)

        clip = (self.clip_proj(self.clip_norm(b["clip_vec"])) + self.clip_score(b["clip_score"])).unsqueeze(1)
        clip = clip + self._type(MOD_IMAGE, clip)
        clip_valid = b["clip_present"].unsqueeze(1)

        patches = self.dire_patch(b["dire_map"]).flatten(2).transpose(1, 2) + self.dire_pos
        dscalar = self.dire_scalar(b["dire_scalars"]).unsqueeze(1)
        dire = torch.cat([patches, dscalar], dim=1) + self._type(MOD_DIFFUSION, patches)
        dire_valid = b["dire_present"].unsqueeze(1).expand(-1, dire.size(1))

        vid_in = torch.cat([self.vid_norm(b["vid_emb"]), b["vid_scores"].unsqueeze(-1)], dim=-1)
        vid = self.vid_proj(vid_in) + self.time_pos + self._type(MOD_VIDEO, vid_in)
        vid_valid = b["vid_valid"]

        nscalar = self.noise_scalar(b["noise_scalars"]).unsqueeze(1)
        nframes = self.noise_frame(b["noise_energy"].unsqueeze(-1)) + self.time_pos
        noise = torch.cat([nscalar, nframes], dim=1) + self._type(MOD_NOISE, nframes)
        noise_valid = torch.cat([b["noise_present"].unsqueeze(1), b["noise_valid"]], dim=1)

        tokens = torch.cat([fus, clip, dire, vid, noise], dim=1)
        valid = torch.cat([fus_valid, clip_valid, dire_valid, vid_valid, noise_valid], dim=1)
        tokens = self.out_norm(tokens) * valid.unsqueeze(-1)  # zero out masked tokens
        return tokens, valid, self.modality