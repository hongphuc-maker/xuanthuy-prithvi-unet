# Architecture

Shapes for P3 (T = 1, batch B = 1). P2 (T = 6) differs only before the neck:
1,177 tokens instead of 197. Output of `python scripts/trace_shapes.py`.

| # | Stage | Operation | Output shape | Learned parameters | Origin |
| --- | --- | --- | --- | ---: | --- |
| 0 | Radiometry | ρ = (DN − 1000) / 10000; z = (ρ − μ/10⁴) / (σ/10⁴) | 6 × T × 224 × 224 | fixed | ESA L2A offset; Prithvi HLS statistics |
| 1 | Patch embedding | Conv3d 6 → 1024, kernel = stride = (1, 16, 16), flatten | 196T × 1024 | 1,573,888 | Prithvi (TerraTorch `PatchEmbed`) |
| 2 | Position | + 3D sin-cos table over (date index, row, column) | 196T × 1024 | fixed | Prithvi |
| 3 | Date, location | + scale_t · sincos(year, day of year) + scale_l · sincos(lat, lon) | 196T × 1024 | 2 scalars | Prithvi-EO-2.0 "TL" |
| 4 | CLS | prepend one learned token | (1 + 196T) × 1024 | 1,024 | Prithvi |
| 5 | Encoder | 24 × [LayerNorm → 16-head self-attention → residual; LayerNorm → MLP 1024-4096-1024 (GELU) → residual] | (1 + 196T) × 1024 | 302,311,424 (incl. final LayerNorm) | timm ViT `Block` |
| 6 | Readouts | outputs of blocks 5, 11, 17, 23 (block 23 after the final LayerNorm) | 4 × (1 + 196T) × 1024 | — | config `out_indices` |
| 7 | Tokens → map | drop CLS, `b (t h w) d -> b t d h w`, keep last date | 4 × 1024 × 14 × 14 | fixed | TerraTorch `ReshapeTokensToImage` + project `TargetTimeSlice` |
| 8 | Pyramid, stride 4 | ConvT 2×2/2 1024→512, GroupNorm, GELU, ConvT 2×2/2 512→256 | 256 × 56 × 56 | 2,623,232 | ViTDet simple feature pyramid (TerraTorch `LearnedInterpolateToPyramidal`) |
| 9 | Pyramid, stride 8 | ConvT 2×2/2 1024→512 | 512 × 28 × 28 | 2,097,664 | same |
| 10 | Pyramid, stride 16 | identity | 1024 × 14 × 14 | 0 | same |
| 11 | Pyramid, stride 32 | MaxPool 2×2/2 | 1024 × 7 × 7 | 0 | same |
| 12 | Decoder block 1 | nearest ×2 (7→14), concat stride-16 skip (2048 ch), 2 × [Conv3×3 → GN → ReLU] → 512 | 512 × 14 × 14 | 11,798,528 | U-Net (SMP `UnetDecoder`) |
| 13 | Decoder block 2 | ×2, concat stride-8 skip (1024 ch) → 256 | 256 × 28 × 28 | 2,950,144 | same |
| 14 | Decoder block 3 | ×2, concat stride-4 skip (512 ch) → 128 | 128 × 56 × 56 | 737,792 | same |
| 15 | Decoder block 4 | no upsampling, no skip → 64 | 64 × 56 × 56 | 110,848 | same |
| 16 | Head | Dropout 0.1, Conv 1×1 64 → 11 | 11 × 56 × 56 | 715 | TerraTorch `SegmentationHead` |
| 17 | Resize | bilinear ×4 on class scores | 11 × 224 × 224 | fixed | TerraTorch `PixelWiseModel(rescale=True)` |

Total learned parameters: 324,205,261 (encoder 303,886,338; new layers 20,318,923).
GroupNorm uses 32 groups everywhere (micro-batch of one image makes BatchNorm statistics unusable).

## Why these choices

- **Four readouts.** The U-Net decoder needs four resolution levels. A ViT keeps one 14 × 14
  grid in every block, so four blocks are read and each is resampled to one level. Blocks 5, 11,
  17, 23 split the encoder into equal quarters (TerraTorch uses the same split for this model in
  its ViT-Adapter config). Not ablated in this project.
- **Three concatenations, not four.** The finest level is stride 4 (56 × 56). A ViT has no
  information finer than its 16-pixel patch, so there is no stride-1 or stride-2 skip.
- **Stop at 56 × 56, then bilinear ×4.** More decoder blocks at 112 and 224 would have no new
  skip, only upsampled 56 × 56 information; DeepLabv3+, SegFormer and UperNet also predict at
  stride 4 and upsample. Bilinear is applied to class scores before argmax, so class borders can
  fall between the 4 × 4-pixel cells. Objects narrower than about 40 m can be lost.
- **Decoder channels 512, 256, 128, 64.** Classic U-Net halving, same as the U-Net baseline.
  Block 1 holds 76 % of the decoder parameters because it receives 2,048 channels.
- **Padding = 1 in every 3 × 3 convolution.** Output size equals input size (unlike the original
  U-Net, which used unpadded convolutions and outputs 388 × 388 for a 572 × 572 tile). Border
  pixels of a window see less context, which the 50 % overlap of sliding windows compensates.

## Fixed versus learned

Fixed: radiometry, position table, tokens → map, MaxPool, identity, nearest upsampling,
concatenation, bilinear resize, softmax/argmax.
Learned and pretrained (fine-tuned with learning rate 1e-5): Conv3d patch embedding, CLS,
date/location scales, 24 blocks.
Learned from scratch (learning rate 1e-4): ConvTranspose layers, GroupNorm, decoder convolutions, head.
