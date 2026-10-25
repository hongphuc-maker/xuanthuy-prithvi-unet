"""Prithvi-UNet: Prithvi-EO-2.0 encoder + simple feature pyramid + U-Net decoder.

Full forward pass for P3 (T = 1, C = 6 bands); P2 differs only in T = 6 before the neck.

    image             [B, C, T, 224, 224]
    encoder readouts  4 x [B, 1 + 196*T, 1024]          blocks 5, 11, 17, 23       encoder.py
    feature maps      4 x [B, 1024, 14, 14]             drop CLS, grid, target date neck.py
    pyramid           256@56, 512@28, 1024@14, 1024@7    neck.py
    decoder           [B, 64, 56, 56]                    3 skips + 1 refinement     decoder.py
    head              [B, K, 56, 56]                     Dropout + Conv 1x1         decoder.py
    logits            [B, K, 224, 224]                   bilinear x4 (no weights)

The completed runs assembled the same parts with TerraTorch ``EncoderDecoderFactory``
(``PixelWiseModel.forward``: encoder -> neck -> decoder -> head -> F.interpolate(..., "bilinear")):
https://github.com/IBM/terratorch/blob/v1.2.12/terratorch/models/pixel_wise_model.py#L120-L150
The old builder: https://github.com/hongphuc-maker/xuan-thuy-prithvi-segmentation/blob/42290823beeb42e44ec87bb8845e439e10c18735/reference/wce_completed/prithvi_xtseg/model.py#L59-L91

``forward_trace`` returns every intermediate tensor for inspection.
To change the architecture, edit ``PrithviUNet.__init__`` / ``forward_trace``: each stage is
one attribute (encoder, to_feature_map, pyramid, decoder, head). See docs/EXTENDING.md.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .decoder import SegmentationHead, UNetDecoder
from .encoder import PrithviEncoder
from .neck import SimpleFeaturePyramid, TokensToFeatureMap


class PrithviUNet(nn.Module):
    def __init__(
        self,
        num_frames: int,
        num_classes: int = 11,
        out_indices: tuple[int, ...] = (5, 11, 17, 23),
        decoder_channels: tuple[int, ...] = (512, 256, 128, 64),
        head_dropout: float = 0.1,
        target_index: int = -1,
        gradient_checkpointing: bool = False,
        bands: list[str] | None = None,
        final_upsample: str = "bilinear",
    ) -> None:
        super().__init__()
        if len(out_indices) != 4:
            raise ValueError("The pyramid has four levels, so the encoder must give four readouts")
        if final_upsample not in ("bilinear", "nearest"):
            raise ValueError("final_upsample must be 'bilinear' or 'nearest'")
        self.num_classes = num_classes
        self.final_upsample = final_upsample
        self.encoder = PrithviEncoder(num_frames, out_indices, gradient_checkpointing, bands)
        self.to_feature_map = TokensToFeatureMap(num_frames, self.encoder.grid_size, target_index)
        self.pyramid = SimpleFeaturePyramid(self.encoder.embed_dim)
        self.decoder = UNetDecoder(self.pyramid.out_channels, tuple(decoder_channels))
        self.head = SegmentationHead(self.decoder.out_channels, num_classes, head_dropout)

    def forward(self, image: torch.Tensor, temporal_coords: torch.Tensor, location_coords: torch.Tensor
                ) -> torch.Tensor:
        return self.forward_trace(image, temporal_coords, location_coords)["logits"]

    def forward_trace(self, image: torch.Tensor, temporal_coords: torch.Tensor, location_coords: torch.Tensor
                      ) -> dict[str, Any]:
        readouts = self.encoder(image, temporal_coords, location_coords)
        feature_maps = [self.to_feature_map(tokens) for tokens in readouts]
        pyramid = self.pyramid(feature_maps)
        decoded = self.decoder(pyramid)
        logits_low = self.head(decoded)
        # Resize class scores back to the input size (no weights). "bilinear" is what the
        # completed runs used (PixelWiseModel, pixel_wise_model.py L136); "nearest" is only
        # for comparisons and needs no retraining.
        if self.final_upsample == "bilinear":
            logits = F.interpolate(logits_low, size=image.shape[-2:], mode="bilinear", align_corners=False)
        else:
            logits = F.interpolate(logits_low, size=image.shape[-2:], mode="nearest")
        return {"readouts": readouts, "feature_maps": feature_maps, "pyramid": pyramid, "decoded": decoded,
                "logits_low": logits_low, "logits": logits}


def build_model(config: dict, pretrained: bool = True, prithvi_weights: str | None = None) -> PrithviUNet:
    """Build the model from an experiment config (``configs/*.json``).

    ``prithvi_weights``: local copy of Prithvi_EO_V2_300M_TL.pt (else $PRITHVI_WEIGHTS, else Hugging Face).
    """
    m = config["model"]
    if m.get("backbone", "prithvi_eo_v2_300_tl") != "prithvi_eo_v2_300_tl":
        raise ValueError("Only prithvi_eo_v2_300_tl is implemented")
    if m.get("decoder", "UNetDecoder") != "UNetDecoder":
        raise ValueError("Only the U-Net decoder is implemented")
    model = PrithviUNet(
        num_frames=len(config["active_dates"]),
        num_classes=len(config["class_map"]),
        out_indices=tuple(m["out_indices"]),
        decoder_channels=tuple(m["decoder_channels"]),
        head_dropout=float(m["head_dropout"]),
        target_index=-1,  # the target date is the last active date (checked in config.py)
        gradient_checkpointing=bool(m.get("gradient_checkpointing", False)),
        bands=list(config["bands"]),
        final_upsample=m.get("final_upsample", "bilinear"),
    )
    provenance: dict[str, Any] = {"pretrained": bool(pretrained)}
    if pretrained:
        provenance.update(model.encoder.load_pretrained(config["model_hf_revision"], prithvi_weights))
    model.pretrained_provenance = provenance
    return model


def parameter_table(model: PrithviUNet) -> list[dict[str, Any]]:
    """Learned parameters per stage (for documentation and tests)."""
    def count(module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters())

    vit = model.encoder.vit
    return [
        {"stage": "patch embedding (Conv3d)", "params": count(vit.patch_embed)},
        {"stage": "CLS token + date/location scales", "params": vit.cls_token.numel()
         + count(vit.temporal_embed_enc) + count(vit.location_embed_enc)},
        {"stage": "24 Transformer blocks + final LayerNorm", "params": count(vit.blocks) + count(vit.norm)},
        {"stage": "tokens -> feature map", "params": count(model.to_feature_map)},
        {"stage": "feature pyramid (neck)", "params": count(model.pyramid)},
        {"stage": "U-Net decoder", "params": count(model.decoder)},
        {"stage": "segmentation head", "params": count(model.head)},
    ]
