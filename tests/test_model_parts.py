"""Neck, decoder and head without the 300M encoder (fast, no TerraTorch needed)."""

import torch
from einops import rearrange

from prithvi_unet.model.decoder import SegmentationHead, UNetDecoder
from prithvi_unet.model.neck import SimpleFeaturePyramid, TokensToFeatureMap, group_norm


def count(module):
    return sum(p.numel() for p in module.parameters())


def test_token_k_lands_on_its_grid_cell():
    T, H, W, D = 6, 14, 14, 8
    tokens = torch.randn(2, 1 + T * H * W, D)
    fmap = TokensToFeatureMap(T, (H, W), target_index=-1)(tokens)
    t, row, col = 5, 2, 6
    k = 1 + t * H * W + row * W + col  # = 1015
    assert k == 1015
    assert torch.equal(fmap[:, :, row, col], tokens[:, k, :])


def test_same_as_terratorch_reshape_then_target_slice():
    T, D = 6, 16
    tokens = torch.randn(2, 1 + T * 196, D)
    terratorch_map = rearrange(tokens[:, 1:], "b (t h w) e -> b (t e) h w", t=T, h=14)  # ReshapeTokensToImage
    target_slice = terratorch_map.reshape(2, T, D, 14, 14)[:, T - 1]  # TargetTimeSlice
    assert torch.equal(TokensToFeatureMap(T)(tokens), target_slice)


def test_pyramid_and_decoder_shapes_and_parameters():
    maps = [torch.randn(1, 1024, 14, 14) for _ in range(4)]
    neck = SimpleFeaturePyramid(1024)
    pyramid = neck(maps)
    assert [tuple(p.shape) for p in pyramid] == [(1, 256, 56, 56), (1, 512, 28, 28), (1, 1024, 14, 14), (1, 1024, 7, 7)]
    assert count(neck) == 4_720_896

    decoder = UNetDecoder(neck.out_channels, (512, 256, 128, 64))
    # conv weights 9*Cin*Cout (no bias) + GroupNorm gamma/beta 2*Cout, twice per block
    assert [count(b) for b in decoder.blocks] == [11_798_528, 2_950_144, 737_792, 110_848]
    assert count(decoder) == 15_597_312
    out = decoder(pyramid)
    assert tuple(out.shape) == (1, 64, 56, 56)

    head = SegmentationHead(64, 11, 0.1)
    assert count(head) == 715
    assert tuple(head(out).shape) == (1, 11, 56, 56)


def test_group_norm_rule():
    assert [group_norm(c).num_groups for c in (64, 128, 256, 512, 1024)] == [32] * 5
    assert group_norm(48).num_groups == 16
