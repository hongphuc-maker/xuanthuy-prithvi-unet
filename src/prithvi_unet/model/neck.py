"""Neck: turn four same-size ViT readouts into a four-level feature pyramid.

1. ``TokensToFeatureMap`` (FIXED, no weights). Drop CLS, lay the tokens back on
   their 14 x 14 grid and keep the target date. Replaces two necks of the completed runs:
   - TerraTorch ``ReshapeTokensToImage`` (rearrange "batch (t h w) e -> batch (t e) h w"):
     https://github.com/IBM/terratorch/blob/v1.2.12/terratorch/models/necks.py#L224-L319
   - project ``TargetTimeSlice`` (reshape [B, T, D, H, W], take the last date):
     https://github.com/hongphuc-maker/xuan-thuy-prithvi-segmentation/blob/42290823beeb42e44ec87bb8845e439e10c18735/reference/wce_completed/prithvi_xtseg/model.py#L15-L32

2. ``SimpleFeaturePyramid`` (LEARNED). Resample the four 14 x 14 maps to strides
   4, 8, 16, 32 so that a U-Net decoder has skip connections. Same layers as
   TerraTorch ``LearnedInterpolateToPyramidal``:
   https://github.com/IBM/terratorch/blob/v1.2.12/terratorch/models/necks.py#L343-L384
   which follows the "simple feature pyramid" of ViTDet (Li et al. 2022, Fig. 2 and Sec. 3),
   https://arxiv.org/abs/2203.16527
   GroupNorm replaces BatchNorm because training uses a micro-batch of one image
   (project ``replace_batch_norm``, same 32-group rule):
   https://github.com/hongphuc-maker/xuan-thuy-prithvi-segmentation/blob/42290823beeb42e44ec87bb8845e439e10c18735/reference/wce_completed/prithvi_xtseg/model.py#L44-L56
   GroupNorm paper (Wu and He 2018): https://arxiv.org/abs/1803.08494
   GELU paper (Hendrycks and Gimpel 2016): https://arxiv.org/abs/1606.08415

Shapes for one readout, D = 1024:
    tokens [B, 1 + T*196, D] -> feature map [B, D, 14, 14]
Pyramid:
    block 5  -> ConvT x2, GN, GELU, ConvT x2 -> [B, 256, 56, 56]   stride 4
    block 11 -> ConvT x2                     -> [B, 512, 28, 28]   stride 8
    block 17 -> identity                     -> [B, 1024, 14, 14]  stride 16
    block 23 -> MaxPool 2x2                  -> [B, 1024, 7, 7]    stride 32
"""

from __future__ import annotations

import math

import torch
from einops import rearrange
from torch import nn


def group_norm(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """GroupNorm with gcd(32, C) groups and at least two channels per group.

    Same rule as ``replace_batch_norm`` of the completed runs (link above); gives 32
    groups for every layer of this model (64 ... 1024 channels).
    """
    groups = math.gcd(max_groups, channels)
    while channels // groups < 2 and groups > 1:
        groups //= 2
    return nn.GroupNorm(groups, channels)


class TokensToFeatureMap(nn.Module):
    """[B, 1 + T*H*W, D] tokens -> [B, D, H, W] feature map of one date. No weights.

    Tokens are ordered (date, row, column) because Conv3d output [B, D, T, H, W] is
    flattened in that order (``PatchEmbed.forward``: x.flatten(2).transpose(1, 2)).
    Each token keeps its own grid cell; the 24 blocks mix content between tokens but
    never move them.
    """

    def __init__(self, num_frames: int, grid_size: tuple[int, int] = (14, 14), target_index: int = -1) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.grid_h, self.grid_w = grid_size
        self.target_index = target_index % num_frames

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        patch_tokens = tokens[:, 1:, :]  # drop CLS -> [B, T*H*W, D]
        maps = rearrange(
            patch_tokens, "b (t h w) d -> b t d h w", t=self.num_frames, h=self.grid_h, w=self.grid_w
        )  # [B, T, D, H, W]
        # Keep the target date. Its tokens have already attended to every date (P2).
        return maps[:, self.target_index]  # [B, D, H, W]


class SimpleFeaturePyramid(nn.Module):
    """Four 14 x 14 maps -> pyramid at strides 4, 8, 16, 32 (LEARNED except pool/identity).

    Attribute names follow the stride they produce; TerraTorch calls them fpn1..fpn4
    (``LearnedInterpolateToPyramidal.__init__``, necks.py L349-L363).
    """

    def __init__(self, embed_dim: int = 1024) -> None:
        super().__init__()
        d = embed_dim
        # fpn1. Shallowest readout -> finest level: two learned x2 upsamplings, 14 -> 28 -> 56.
        self.to_stride4 = nn.Sequential(
            nn.ConvTranspose2d(d, d // 2, kernel_size=2, stride=2),
            group_norm(d // 2),
            nn.GELU(),
            nn.ConvTranspose2d(d // 2, d // 4, kernel_size=2, stride=2),
        )
        self.to_stride8 = nn.ConvTranspose2d(d, d // 2, kernel_size=2, stride=2)  # fpn2: 14 -> 28
        self.to_stride16 = nn.Identity()  # fpn3: stays 14
        self.to_stride32 = nn.MaxPool2d(kernel_size=2, stride=2)  # fpn4: 14 -> 7, no weights
        self.out_channels = (d // 4, d // 2, d, d)

    def forward(self, maps: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(maps) != 4:
            raise ValueError("SimpleFeaturePyramid needs exactly four readouts (one per pyramid level)")
        return [
            self.to_stride4(maps[0]),  # [B, 256, 56, 56]
            self.to_stride8(maps[1]),  # [B, 512, 28, 28]
            self.to_stride16(maps[2]),  # [B, 1024, 14, 14]
            self.to_stride32(maps[3]),  # [B, 1024, 7, 7]
        ]
