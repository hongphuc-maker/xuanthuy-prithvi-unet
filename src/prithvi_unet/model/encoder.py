"""Prithvi-EO-2.0-300M-TL encoder with its forward pass written out step by step.

What comes from where
---------------------
* Sub-modules and pretrained weights: TerraTorch 1.2.12,
  ``terratorch.models.backbones.prithvi_mae.PrithviViT`` (registry name
  ``prithvi_eo_v2_300_tl``). Weights: Hugging Face
  ``ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL``, file ``Prithvi_EO_V2_300M_TL.pt``.
* The forward pass below re-states ``PrithviViT.forward_features`` (same file)
  line by line, so that every operation and every tensor shape is visible in
  this repository. ``tests/test_legacy_equivalence.py`` checks that the result
  is identical to the TerraTorch implementation used in the completed runs.

Shapes (B = batch, T = number of dates, D = 1024, N = 14 * 14 = 196 patches per date)
-------------------------------------------------------------------------------
image            [B, 6, T, 224, 224]   normalised reflectance (see data/radiometry.py)
patch embedding  [B, T*N, D]           Conv3d, kernel = stride = (1, 16, 16)   LEARNED
+ position       [B, T*N, D]           fixed 3D sin-cos table                  FIXED
+ date           [B, T*N, D]           sin-cos of (year, day of year) * scale  scale LEARNED
+ location       [B, T*N, D]           sin-cos of (lat, lon) * scale           scale LEARNED
+ CLS token      [B, 1 + T*N, D]                                                LEARNED
24 Transformer blocks                  LayerNorm, self-attention, MLP (GELU)    LEARNED
readouts         4 x [B, 1 + T*N, D]   outputs of blocks 5, 11, 17, 23
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

# Hyper-parameters of Prithvi-EO-2.0-300M-TL, copied from TerraTorch
# ``prithvi_cfgs["prithvi_eo_v2_300_tl"]`` so they are visible here.
PRITHVI_300M_TL = {
    "img_size": 224,
    "patch_size": (1, 16, 16),
    "in_chans": 6,
    "embed_dim": 1024,
    "depth": 24,
    "num_heads": 16,
    "mlp_ratio": 4.0,
    "coords_encoding": ["time", "location"],
    "coords_scale_learn": True,
}
HF_REPO = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL"
HF_FILE = "Prithvi_EO_V2_300M_TL.pt"
# Prithvi was pretrained on HLS bands; Sentinel-2 B02, B03, B04, B8A, B11, B12 map to these names.
HLS_BAND_NAMES = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]


class PrithviEncoder(nn.Module):
    """Prithvi ViT encoder that returns the token sequences of selected blocks.

    Args:
        num_frames: number of dates T fed to the model (P2: 6, P3: 1).
        out_indices: blocks whose output is read out (0-based). Default: the last
            block of each quarter of the 24-block encoder.
        gradient_checkpointing: recompute block activations in the backward pass
            to save GPU memory (same numbers, slower).
    """

    def __init__(
        self,
        num_frames: int,
        out_indices: tuple[int, ...] = (5, 11, 17, 23),
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        from terratorch.registry import BACKBONE_REGISTRY

        # Built through the TerraTorch registry exactly as in the completed runs,
        # so that pretrained weights load with TerraTorch's own key filter.
        self.vit = BACKBONE_REGISTRY.build(
            "prithvi_eo_v2_300_tl",
            pretrained=False,
            num_frames=num_frames,
            bands=HLS_BAND_NAMES,
            coords_encoding=PRITHVI_300M_TL["coords_encoding"],
        )
        depth = len(self.vit.blocks)
        if sorted(out_indices) != list(out_indices) or not all(0 <= i < depth for i in out_indices):
            raise ValueError(f"out_indices must be sorted block indices in [0, {depth})")
        self.num_frames = num_frames
        self.out_indices = tuple(out_indices)
        self.embed_dim = self.vit.embed_dim
        self.grid_size = tuple(self.vit.patch_embed.grid_size[1:])  # (14, 14)
        self.gradient_checkpointing = gradient_checkpointing

    # ------------------------------------------------------------------ weights
    def load_pretrained(self, revision: str) -> dict:
        """Download the Prithvi checkpoint at a pinned revision and load it strictly."""
        from huggingface_hub import hf_hub_download
        from terratorch.models.backbones.prithvi_vit import PRETRAINED_BANDS, checkpoint_filter_fn_vit

        path = Path(hf_hub_download(HF_REPO, HF_FILE, revision=revision))
        state = torch.load(path, map_location="cpu", weights_only=True)
        # TerraTorch's filter drops the MAE decoder, adapts the position table to
        # T dates and keeps Conv3d input channels by band name.
        state = checkpoint_filter_fn_vit(state, self.vit, PRETRAINED_BANDS, self.vit.model_bands)
        self.vit.load_state_dict(state, strict=True)  # never train a partly random encoder
        return {"hf_repo": HF_REPO, "file": HF_FILE, "revision": revision, "local_path": str(path)}

    # ------------------------------------------------------------------ forward
    def embed(
        self, image: torch.Tensor, temporal_coords: torch.Tensor, location_coords: torch.Tensor
    ) -> torch.Tensor:
        """Image -> token sequence with CLS, before the Transformer blocks."""
        batch = image.shape[0]

        # 1. Patch embedding (LEARNED): Conv3d 6 -> 1024 channels, kernel = stride = (1, 16, 16).
        #    Each 16 x 16 x 6-band patch of one date becomes one 1024-d token.
        #    [B, 6, T, 224, 224] -> [B, 1024, T, 14, 14] -> flatten -> [B, T*196, 1024]
        tokens = self.vit.patch_embed(image)

        # 2. Position (FIXED sin-cos table over (t, row, col)); index 0 belongs to CLS.
        pos_embed = self.vit.interpolate_pos_encoding(image.shape[-3:])  # [1, 1 + T*196, 1024]
        tokens = tokens + pos_embed[:, 1:, :]

        # 3. Acquisition date: sin-cos(year, day of year) times a LEARNED scalar (init 0.1).
        tokens_per_date = tokens.shape[1] // self.num_frames
        tokens = tokens + self.vit.temporal_embed_enc(temporal_coords, tokens_per_date)

        # 4. Location: sin-cos(latitude, longitude) of the patch centre times a LEARNED scalar.
        tokens = tokens + self.vit.location_embed_enc(location_coords)  # broadcast over all tokens

        # 5. Prepend the CLS token (LEARNED). It is dropped again before the decoder.
        cls = (self.vit.cls_token + pos_embed[:, :1, :]).expand(batch, -1, -1)
        return torch.cat([cls, tokens], dim=1)  # [B, 1 + T*196, 1024]

    def forward(
        self, image: torch.Tensor, temporal_coords: torch.Tensor, location_coords: torch.Tensor
    ) -> list[torch.Tensor]:
        """Return the outputs of ``out_indices`` blocks, each [B, 1 + T*196, 1024]."""
        if image.ndim != 5 or image.shape[2] != self.num_frames:
            raise ValueError(f"Expected image [B, 6, {self.num_frames}, H, W], got {tuple(image.shape)}")
        hidden = self.embed(image, temporal_coords, location_coords)

        readouts: list[torch.Tensor] = []
        last = len(self.vit.blocks) - 1
        for index, block in enumerate(self.vit.blocks):
            # Each block: x = x + Attention(LayerNorm(x)); x = x + MLP(LayerNorm(x)).
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                hidden = checkpoint(block, hidden, use_reentrant=False)
            else:
                hidden = block(hidden)
            if index == last:
                # TerraTorch applies the encoder's final LayerNorm to the last block only.
                readout = self.vit.norm(hidden)
            else:
                readout = hidden
            if index in self.out_indices:
                readouts.append(readout)
            if index >= self.out_indices[-1]:
                break
        return readouts
