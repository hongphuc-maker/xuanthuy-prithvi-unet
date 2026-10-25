"""U-Net decoder (the right arm of the U) and the segmentation head.

Same computation as the completed runs, which used segmentation_models_pytorch 0.5.0
``UnetDecoder`` through TerraTorch's ``UNetDecoder`` wrapper. Written out here so the
upsampling, concatenation and convolutions are visible.

Sources
    U-Net (Ronneberger et al. 2015): https://arxiv.org/abs/1505.04597
    SMP UnetDecoderBlock (interpolate -> cat -> conv1 -> conv2):
      https://github.com/qubvel-org/segmentation_models.pytorch/blob/v0.5.0/segmentation_models_pytorch/decoders/unet/decoder.py#L10-L61
    SMP UnetDecoder (drops the first feature, reverses, head = deepest):
      https://github.com/qubvel-org/segmentation_models.pytorch/blob/v0.5.0/segmentation_models_pytorch/decoders/unet/decoder.py#L90-L170
    SMP Conv2dReLU (Conv without bias -> norm -> ReLU):
      https://github.com/qubvel-org/segmentation_models.pytorch/blob/v0.5.0/segmentation_models_pytorch/base/modules.py#L89-L114
    TerraTorch UNetDecoder wrapper (duplicates the first feature so SMP keeps it):
      https://github.com/IBM/terratorch/blob/v1.2.12/terratorch/models/decoders/unet_decoder.py#L12-L48
    TerraTorch SegmentationHead (Dropout -> Conv 1x1):
      https://github.com/IBM/terratorch/blob/v1.2.12/terratorch/models/heads/segmentation_head.py#L6-L49

Each decoder block:
    nearest upsampling x2  ->  concat skip (channel axis)  ->  [Conv 3x3 -> GroupNorm -> ReLU] x 2

With decoder_channels = (512, 256, 128, 64) and the pyramid (256@56, 512@28, 1024@14, 1024@7):

    block 1: 1024@7  -> up 14 -> concat 1024@14 -> 2048 -> 512@14
    block 2: 512@14  -> up 28 -> concat 512@28  -> 1024 -> 256@28
    block 3: 256@28  -> up 56 -> concat 256@56  -> 512  -> 128@56
    block 4: 128@56  -> (no upsampling, no skip)        -> 64@56
    head:    Dropout(0.1) -> Conv 1x1 64 -> 11           -> 11@56
    then (in PrithviUNet) bilinear x4 -> 11@224

Only three concatenations: the finest pyramid level is stride 4 (56 x 56). There
is no stride-1 or stride-2 feature to concatenate, because a ViT has no detail
finer than its 16 x 16 patch. Block 4 exists because SMP builds one block per
decoder channel and its last target size equals the duplicated first feature (56).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .neck import group_norm


class ConvGNReLU(nn.Sequential):
    """Conv 3x3 (padding 1, no bias) -> GroupNorm -> ReLU. Padding keeps H x W.

    SMP ``Conv2dReLU`` with use_norm set: the conv has no bias because the norm's beta replaces it.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            group_norm(out_channels),
            nn.ReLU(inplace=True),
        )


class UNetDecoderBlock(nn.Module):
    """SMP ``UnetDecoderBlock`` with attention_type=None (its attention layers are identities)."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = ConvGNReLU(in_channels + skip_channels, out_channels)  # fuses skip + upsampled
        self.conv2 = ConvGNReLU(out_channels, out_channels)  # refines

    def forward(self, x: torch.Tensor, size: tuple[int, int], skip: torch.Tensor | None) -> torch.Tensor:
        x = F.interpolate(x, size=size, mode="nearest")  # x2 (identity when size is unchanged)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)  # skip connection: channel concatenation
        return self.conv2(self.conv1(x))


class UNetDecoder(nn.Module):
    """Bottom-up decoder over the pyramid (stride 32 -> 16 -> 8 -> 4 -> 4)."""

    def __init__(self, pyramid_channels: tuple[int, int, int, int], decoder_channels: tuple[int, ...]) -> None:
        super().__init__()
        if len(decoder_channels) != 4:
            raise ValueError("This decoder has four blocks; give four decoder_channels")
        c4, c8, c16, c32 = pyramid_channels
        d1, d2, d3, d4 = decoder_channels
        self.blocks = nn.ModuleList(
            [
                UNetDecoderBlock(c32, c16, d1),  # 7 -> 14, skip stride 16
                UNetDecoderBlock(d1, c8, d2),  # 14 -> 28, skip stride 8
                UNetDecoderBlock(d2, c4, d3),  # 28 -> 56, skip stride 4
                UNetDecoderBlock(d3, 0, d4),  # 56 -> 56, no skip
            ]
        )
        self.out_channels = d4
        self._init_weights()

    def _init_weights(self) -> None:
        """Same initialisation as SMP ``initialize_decoder``:
        https://github.com/qubvel-org/segmentation_models.pytorch/blob/v0.5.0/segmentation_models_pytorch/base/initialization.py#L4-L20
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, pyramid: list[torch.Tensor]) -> torch.Tensor:
        p4, p8, p16, p32 = pyramid
        x = self.blocks[0](p32, size=p16.shape[-2:], skip=p16)  # [B, 512, 14, 14]
        x = self.blocks[1](x, size=p8.shape[-2:], skip=p8)  # [B, 256, 28, 28]
        x = self.blocks[2](x, size=p4.shape[-2:], skip=p4)  # [B, 128, 56, 56]
        x = self.blocks[3](x, size=p4.shape[-2:], skip=None)  # [B, 64, 56, 56]
        return x


class SegmentationHead(nn.Module):
    """Dropout -> Conv 1x1: per-pixel linear classifier from 64 features to class scores."""

    def __init__(self, in_channels: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(x))
