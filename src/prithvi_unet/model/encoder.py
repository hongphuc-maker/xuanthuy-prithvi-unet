"""Prithvi-EO-2.0-300M-TL encoder with its forward pass written out step by step.

Where each part comes from (all links pinned to the versions in pyproject.toml)
-------------------------------------------------------------------------------
* Model and paper: Prithvi-EO-2.0 (Szwarcman et al., 2024), https://arxiv.org/abs/2412.02732
  Weights: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL
* Sub-modules are TerraTorch 1.2.12 ``PrithviViT``:
  https://github.com/IBM/terratorch/blob/v1.2.12/terratorch/models/backbones/prithvi_mae.py#L280-L562
* The forward pass below re-states ``PrithviViT.forward_features`` line by line:
  https://github.com/IBM/terratorch/blob/v1.2.12/terratorch/models/backbones/prithvi_mae.py#L485-L539
  ``tests/test_legacy_equivalence.py`` checks that both give identical outputs.
* Hyper-parameters (``prithvi_cfgs["prithvi_eo_v2_300_tl"]``):
  https://github.com/IBM/terratorch/blob/v1.2.12/terratorch/models/backbones/prithvi_vit.py#L58-L92

Shapes (B = batch, C = bands, T = dates, D = 1024, N = 14 * 14 = 196 patches per date)
-------------------------------------------------------------------------------------
image            [B, C, T, 224, 224]   normalised reflectance (data/radiometry.py)
patch embedding  [B, T*N, D]           Conv3d, kernel = stride = (1, 16, 16)   LEARNED
+ position       [B, T*N, D]           fixed 3D sin-cos table                  FIXED
+ date           [B, T*N, D]           sin-cos of (year, day of year) * scale  scale LEARNED
+ location       [B, T*N, D]           sin-cos of (lat, lon) * scale           scale LEARNED
+ CLS token      [B, 1 + T*N, D]                                                LEARNED
24 Transformer blocks                  LayerNorm, self-attention, MLP (GELU)    LEARNED
readouts         4 x [B, 1 + T*N, D]   outputs of blocks 5, 11, 17, 23
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..config import S2_TO_HLS, SENTINEL2_BANDS

# Copied from TerraTorch prithvi_cfgs["prithvi_eo_v2_300_tl"] (link above) so they are visible here.
PRITHVI_300M_TL = {
    "img_size": 224,
    "patch_size": (1, 16, 16),
    "embed_dim": 1024,
    "depth": 24,
    "num_heads": 16,
    "mlp_ratio": 4.0,
    "coords_encoding": ["time", "location"],
    "coords_scale_learn": True,
}
HF_REPO = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL"
HF_FILE = "Prithvi_EO_V2_300M_TL.pt"


class PrithviEncoder(nn.Module):
    """Prithvi ViT encoder that returns the token sequences of selected blocks.

    Args:
        num_frames: number of dates T fed to the model (P2: 6, P3: 1).
        out_indices: blocks whose output is read out (0-based). Default: the last
            block of each quarter of the 24-block encoder (the same split TerraTorch
            uses for this model in its ViT-Adapter config, ``interaction_indexes``:
            https://github.com/IBM/terratorch/blob/v1.2.12/terratorch/models/backbones/prithvi_vit.py#L152-L164).
        gradient_checkpointing: recompute block activations in the backward pass
            to save GPU memory (same numbers, slower).
        bands: Sentinel-2 bands fed to the model, any subset of the six pretrained
            bands. Conv3d gets one input channel per band.
    """

    def __init__(self, num_frames: int, out_indices: tuple[int, ...] = (5, 11, 17, 23),
                 gradient_checkpointing: bool = False, bands: list[str] | None = None) -> None:
        super().__init__()
        from terratorch.registry import BACKBONE_REGISTRY

        self.bands = list(bands or SENTINEL2_BANDS)
        # Built through the TerraTorch registry exactly as in the completed runs, so that
        # pretrained weights load with TerraTorch's own key filter. ``bands`` sets
        # in_chans = len(bands) (``_create_prithvi``):
        # https://github.com/IBM/terratorch/blob/v1.2.12/terratorch/models/backbones/prithvi_vit.py#L262-L381
        self.vit = BACKBONE_REGISTRY.build(
            "prithvi_eo_v2_300_tl",
            pretrained=False,
            num_frames=num_frames,
            bands=[S2_TO_HLS[b] for b in self.bands],
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
    def load_pretrained(self, revision: str, local_path: str | Path | None = None) -> dict:
        """Load Prithvi weights strictly, from ``local_path`` / $PRITHVI_WEIGHTS or Hugging Face.

        TerraTorch's ``checkpoint_filter_fn_vit`` drops the MAE decoder, rebuilds the position
        table for T dates and calls ``select_patch_embed_weights``, which copies the pretrained
        Conv3d input channel of each band BY NAME and drops the others (band subsets):
        https://github.com/IBM/terratorch/blob/v1.2.12/terratorch/models/backbones/prithvi_vit.py#L168-L201
        https://github.com/IBM/terratorch/blob/v1.2.12/terratorch/models/backbones/select_patch_embed_weights.py#L93-L157
        """
        from terratorch.models.backbones.prithvi_vit import PRETRAINED_BANDS, checkpoint_filter_fn_vit

        local_path = local_path or os.environ.get("PRITHVI_WEIGHTS")
        if local_path:
            path = Path(local_path)
        else:
            from huggingface_hub import hf_hub_download

            path = Path(hf_hub_download(HF_REPO, HF_FILE, revision=revision))
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint_filter_fn_vit(state, self.vit, PRETRAINED_BANDS, self.vit.model_bands)
        self.vit.load_state_dict(state, strict=True)  # never train a partly random encoder
        return {"hf_repo": HF_REPO, "file": HF_FILE, "revision": revision, "local_path": str(path), "bands": self.bands}

    # ------------------------------------------------------------------ forward
    def embed(self, image: torch.Tensor, temporal_coords: torch.Tensor, location_coords: torch.Tensor) -> torch.Tensor:
        """Image -> token sequence with CLS, before the Transformer blocks.

        Re-states the first half of ``PrithviViT.forward_features`` (L491-L514 of the file linked above).
        """
        batch = image.shape[0]

        # 1. Patch embedding (LEARNED): Conv3d C -> 1024 channels, kernel = stride = (1, 16, 16),
        #    then flatten(2).transpose(1, 2). Each 16 x 16 x C patch of one date becomes one token.
        #    [B, C, T, 224, 224] -> [B, 1024, T, 14, 14] -> [B, T*196, 1024]
        #    TerraTorch PatchEmbed: https://github.com/IBM/terratorch/blob/v1.2.12/terratorch/models/backbones/prithvi_mae.py#L176-L213
        tokens = self.vit.patch_embed(image)

        # 2. Position (FIXED 3D sin-cos table over (t, row, col)); index 0 belongs to CLS.
        #    get_3d_sincos_pos_embed: https://github.com/IBM/terratorch/blob/v1.2.12/terratorch/models/backbones/prithvi_mae.py#L35-L72
        pos_embed = self.vit.interpolate_pos_encoding(image.shape[-3:])  # [1, 1 + T*196, 1024]
        tokens = tokens + pos_embed[:, 1:, :]

        # 3. Acquisition date: sin-cos(year, day of year) times a LEARNED scalar (init 0.1).
        #    TemporalEncoder: https://github.com/IBM/terratorch/blob/v1.2.12/terratorch/models/backbones/prithvi_mae.py#L216-L250
        tokens_per_date = tokens.shape[1] // self.num_frames
        tokens = tokens + self.vit.temporal_embed_enc(temporal_coords, tokens_per_date)

        # 4. Location: sin-cos(latitude, longitude) of the crop centre times a LEARNED scalar.
        #    LocationEncoder: https://github.com/IBM/terratorch/blob/v1.2.12/terratorch/models/backbones/prithvi_mae.py#L253-L277
        tokens = tokens + self.vit.location_embed_enc(location_coords)  # broadcast over all tokens

        # 5. Prepend the CLS token (LEARNED). It is dropped again before the decoder.
        cls = (self.vit.cls_token + pos_embed[:, :1, :]).expand(batch, -1, -1)
        return torch.cat([cls, tokens], dim=1)  # [B, 1 + T*196, 1024]

    def forward(self, image: torch.Tensor, temporal_coords: torch.Tensor, location_coords: torch.Tensor
                ) -> list[torch.Tensor]:
        """Return the outputs of ``out_indices`` blocks, each [B, 1 + T*196, 1024].

        Re-states the second half of ``PrithviViT.forward_features`` (L516-L539). Each block is a
        timm ViT ``Block``: x = x + Attention(LayerNorm(x)); x = x + Mlp(LayerNorm(x)), with
        16 heads and an MLP 1024 -> 4096 -> 1024 using GELU.
        https://github.com/huggingface/pytorch-image-models/blob/v1.0.30/timm/models/vision_transformer.py#L131-L217
        https://github.com/huggingface/pytorch-image-models/blob/v1.0.30/timm/layers/attention.py#L43-L145
        """
        if image.ndim != 5 or image.shape[1] != len(self.bands) or image.shape[2] != self.num_frames:
            raise ValueError(f"Expected image [B, {len(self.bands)}, {self.num_frames}, H, W], got {tuple(image.shape)}")
        hidden = self.embed(image, temporal_coords, location_coords)

        readouts: list[torch.Tensor] = []
        last = len(self.vit.blocks) - 1
        for index, block in enumerate(self.vit.blocks):
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                hidden = checkpoint(block, hidden, use_reentrant=False)
            else:
                hidden = block(hidden)
            # TerraTorch applies the encoder's final LayerNorm to the last block only (L537-L538).
            readout = self.vit.norm(hidden) if index == last else hidden
            if index in self.out_indices:
                readouts.append(readout)
            if index >= self.out_indices[-1]:
                break  # later blocks do not change earlier readouts
        return readouts
