"""Frozen copy of the model builder used by the completed P2/P3 runs.

Source: xuan-thuy-prithvi-segmentation @ 4229082,
``reference/wce_completed/prithvi_xtseg/model.py`` (``TargetTimeSlice``,
``CheckpointBlock``, ``replace_batch_norm``, ``build_model``). Only the pretrained
download was removed. Used by tests to prove the explicit model computes the same
function. Do not edit.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from terratorch.models.encoder_decoder_factory import EncoderDecoderFactory
from terratorch.models.necks import Neck
from terratorch.registry import BACKBONE_REGISTRY, TERRATORCH_NECK_REGISTRY

if "TargetTimeSlice" not in TERRATORCH_NECK_REGISTRY:

    @TERRATORCH_NECK_REGISTRY.register
    class TargetTimeSlice(Neck):
        def __init__(self, channel_list, frames, target_index=-1):
            super().__init__(channel_list)
            self.frames = frames
            self.target_index = target_index % frames
            if any(c % frames for c in channel_list):
                raise ValueError("Temporal channels do not divide into frames")

        def process_channel_list(self, channel_list):
            return [c // self.frames for c in channel_list]

        def forward(self, features, **kwargs):
            return [x.reshape(x.shape[0], self.frames, x.shape[1] // self.frames, *x.shape[-2:])[:, self.target_index]
                    for x in features]


class CheckpointBlock(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        if self.training and torch.is_grad_enabled():
            return checkpoint(self.inner, x, use_reentrant=False)
        return self.inner(x)


def replace_batch_norm(module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            groups = math.gcd(32, child.num_features)
            while child.num_features // groups < 2 and groups > 1:
                groups //= 2
            new = nn.GroupNorm(groups, child.num_features, eps=child.eps, affine=child.affine)
            if child.affine:
                with torch.no_grad():
                    new.weight.copy_(child.weight)
                    new.bias.copy_(child.bias)
            setattr(module, name, new)
        else:
            replace_batch_norm(child)


def build_legacy_model(config):
    m = config["model"]
    frames = len(config["active_dates"])
    kwargs = dict(pretrained=False, num_frames=frames,
                  bands=["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"],
                  out_indices=m["out_indices"], coords_encoding=["time", "location"])
    backbone = BACKBONE_REGISTRY.build(m["backbone"], **kwargs)
    backbone.out_channels = [backbone.out_channels[i] for i in kwargs["out_indices"]]
    if m["gradient_checkpointing"]:
        backbone.blocks = nn.ModuleList([CheckpointBlock(b) for b in backbone.blocks])
    model = EncoderDecoderFactory().build_model(
        task="segmentation", backbone=backbone, decoder=m["decoder"], num_classes=11,
        decoder_channels=m["decoder_channels"], head_dropout=m["head_dropout"], rescale=True,
        necks=[{"name": "ReshapeTokensToImage", "effective_time_dim": frames},
               {"name": "TargetTimeSlice", "frames": frames, "target_index": frames - 1},
               {"name": "LearnedInterpolateToPyramidal"}])
    replace_batch_norm(model)
    return model


def legacy_logits(model, image, temporal_coords, location_coords):
    return model(image, temporal_coords=temporal_coords, location_coords=location_coords).output
