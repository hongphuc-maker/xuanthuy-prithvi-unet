"""Read the Sentinel-2 stack (and labels) and cut normalised 224 x 224 crops.

``RasterStack``     images of the active dates, no labels (used for new maps)
``TrainingScene``   + label raster, spatial split, class counts (used for training)

Input GeoTIFFs: one file per date, bands named by their description (B02, B03, ...),
an ``acquisition_date`` tag, nodata = 0. The file may hold more bands than the
config uses; ``bands`` in the config selects them by name, so the 6-band exports
also serve 4-band or 3-band experiments.

A crop is a dict ready for the model:
    image            [C, T, 224, 224] float32 (C = len(bands)), output of radiometry.dn_to_model_input
    temporal_coords  [T, 2]  (year, day of year) of each date
    location_coords  [2]     (latitude, longitude) of the crop centre
    target           [224, 224] int64, class 0..K-1, 255 = ignore   (TrainingScene only)
Pixels that are nodata (or outside the allowed domain) are set to 0 after
normalisation, i.e. to the Prithvi mean. Crops at the raster edge are zero-padded
to 224 so that the patch grid stays 14 x 14.

Ported from the completed runs, ``Scene`` / ``Scene.crop``:
https://github.com/hongphuc-maker/xuan-thuy-prithvi-segmentation/blob/42290823beeb42e44ec87bb8845e439e10c18735/reference/wce_completed/prithvi_xtseg/data.py#L50-L151
"""

from __future__ import annotations

import warnings
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from rasterio.warp import transform as transform_coords

from ..utils import sha256_file
from .radiometry import dn_to_model_input, resolve_dn_offset
from .splits import build_split_domains


def _same_grid(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (a["height"], a["width"]) == (b["height"], b["width"]) and a["crs"] == b["crs"] and a["transform"] == b["transform"]


def read_sentinel2(path: Path, bands: list[str], expected_date: str | None = None
                   ) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, str]]:
    """Return (DN [len(bands), H, W], valid mask [H, W], profile, tags).

    A pixel is valid when no band of the file is nodata, so experiments with fewer
    bands are trained and scored on the same pixels as the 6-band ones.
    """
    with rasterio.open(path) as src:
        names = list(src.descriptions)
        missing = [b for b in bands if b not in names]
        if missing:
            raise ValueError(f"{path.name}: bands {missing} not found; band descriptions are {names} "
                             "(set them with scripts/set_band_metadata.py)")
        tags = src.tags()
        if expected_date is not None and tags.get("acquisition_date") != expected_date:
            raise ValueError(f"{path.name}: acquisition_date tag {tags.get('acquisition_date')!r} != {expected_date!r}")
        dn = src.read([names.index(b) + 1 for b in bands])
        valid = (src.read_masks() > 0).all(0) & np.isfinite(dn).all(0)
        return dn, valid, src.profile.copy(), tags


class RasterStack:
    """Images of the active dates (oldest first; the last one is the target date)."""

    def __init__(self, image_specs: list[dict[str, Any]], data_root: str | Path, bands: list[str], radiometry: dict,
                 normalization: dict, patch_size: int = 224, verify_checksums: bool = True) -> None:
        self.root = Path(data_root)
        self.bands, self.radiometry, self.normalization, self.patch_size = bands, radiometry, normalization, patch_size
        arrays, masks, offsets, self.images = [], [], [], []
        for spec in image_specs:
            path = self.root / spec["path"]
            if not path.is_file():
                raise FileNotFoundError(path)
            checksum = sha256_file(path) if verify_checksums else spec.get("sha256", "")
            if verify_checksums and spec.get("sha256") and checksum != spec["sha256"]:
                raise ValueError(f"Checksum changed: {path.name}")
            dn, valid, profile, tags = read_sentinel2(path, bands, spec.get("date"))
            if not arrays:
                self.profile = profile
            elif not _same_grid(profile, self.profile):
                raise ValueError(f"Raster grid differs: {path.name}")
            offset = resolve_dn_offset(radiometry["dn_offset"], tags, spec["date"])
            arrays.append(dn)
            masks.append(valid)
            offsets.append(offset)
            self.images.append({"date": spec["date"], "file": path.name, "sha256": checksum,
                                "valid_pixels": int(valid.sum()), "dn_offset": offset})
        self.dates = [spec["date"] for spec in image_specs]
        self.raw = np.stack(arrays)  # [T, C, H, W] uint16; normalised per crop to save memory
        self.dn_offsets = np.asarray(offsets, np.float32)  # one offset per date
        self.valid_by_time = np.stack(masks)
        self.image_valid = np.logical_and.reduce(masks)  # valid on every date
        self.temporal = torch.tensor([[d.year, d.timetuple().tm_yday] for d in map(date.fromisoformat, self.dates)],
                                     dtype=torch.float32)

    @property
    def shape(self) -> tuple[int, int]:
        return self.raw.shape[-2:]

    def crop(self, row: int, col: int, height: int | None = None, width: int | None = None,
             domain: np.ndarray | None = None) -> dict[str, Any]:
        n = self.patch_size
        height = min(n, self.shape[0] - row) if height is None else int(height)
        width = min(n, self.shape[1] - col) if width is None else int(width)
        region = np.s_[row : row + height, col : col + width]

        x = dn_to_model_input(self.raw[:, :, row : row + height, col : col + width], self.radiometry,
                              self.normalization, dn_offset=self.dn_offsets[:, None, None, None])
        valid = self.image_valid[region].copy()
        if domain is not None:
            valid &= domain[region]
        x[:, :, ~valid] = 0  # nodata -> Prithvi mean
        x = np.pad(x, ((0, 0), (0, 0), (0, n - height), (0, n - width)))  # edge crops -> 224

        # Crop centre in lat/lon for the location embedding.
        xx, yy = rasterio.transform.xy(self.profile["transform"], row + height / 2, col + width / 2, offset="ul")
        lon, lat = transform_coords(self.profile["crs"], "EPSG:4326", [xx], [yy])
        return {
            "image": torch.from_numpy(x.transpose(1, 0, 2, 3).copy()),  # [T, C, ...] -> [C, T, ...]
            "temporal_coords": self.temporal.clone(),
            "location_coords": torch.tensor([lat[0], lon[0]], dtype=torch.float32),
            "row": row, "col": col, "height": height, "width": width, "valid": valid,
        }


class TrainingScene(RasterStack):
    """Active-date images + labels + split. Validity uses ALL listed images (as in the completed runs)."""

    def __init__(self, config: dict[str, Any], data_root: str | Path, verify_checksums: bool = True) -> None:
        self.config = config
        self.num_classes = len(config["class_map"])
        by_date = {spec["date"]: spec for spec in config["images"]}
        super().__init__([by_date[d] for d in config["active_dates"]], data_root, config["bands"], config["radiometry"],
                         config["normalization"], config["split"]["patch_policy"]["size"], verify_checksums)
        if list(self.shape) != config["expected_shape"] or str(self.profile["crs"]) != config["expected_crs"]:
            raise ValueError("Image shape or CRS differs from the config")

        # Common validity over every listed image (also the inactive ones), so P2 and P3
        # are trained and scored on exactly the same pixels.
        all_valid = [self.image_valid]
        for spec in config["images"]:
            if spec["date"] not in config["active_dates"]:
                path = Path(data_root) / spec["path"]
                if verify_checksums and sha256_file(path) != spec["sha256"]:
                    raise ValueError(f"Checksum changed: {path.name}")
                _, valid, profile, _ = read_sentinel2(path, config["bands"], spec["date"])
                if not _same_grid(profile, self.profile):
                    raise ValueError(f"Raster grid differs: {path.name}")
                all_valid.append(valid)
        self.image_valid = np.logical_and.reduce(all_valid)

        label_path = Path(data_root) / config["label"]["path"]
        if verify_checksums and sha256_file(label_path) != config["label"]["sha256"]:
            raise ValueError("Label checksum changed")
        with rasterio.open(label_path) as src:
            if not _same_grid(src.profile, self.profile) or src.count != 1:
                raise ValueError("Label raster must be one band on the image grid")
            self.label = src.read(1)
        k = self.num_classes
        if not set(np.unique(self.label)) <= set(range(k + 1)):
            raise ValueError(f"Label codes must be 0 (no label) or 1..{k}")

        self.common_valid = self.image_valid & np.isin(self.label, range(1, k + 1))
        self.domains = build_split_domains(self.label.shape, self.common_valid, config["split"])
        self.train_mask = self.common_valid & (self.domains["split_mask"] == 1)
        self.val_mask = self.common_valid & self.domains["validation_metric"]
        self.train_class_counts = np.bincount(self.label[self.train_mask], minlength=k + 1)[1 : k + 1]
        if np.any(self.train_class_counts == 0) or not self.val_mask.any():
            raise ValueError("A class has no training pixel, or the validation band is empty")
        if not config.get("cloud_masks"):
            warnings.warn("No per-pixel cloud mask: nodata validity is not cloud validity.", stacklevel=2)

    def crop(self, row: int, col: int, height: int | None = None, width: int | None = None,
             domain: np.ndarray | None = None) -> dict[str, Any]:
        out = super().crop(row, col, height, width, domain)
        n, h, w = self.patch_size, out["height"], out["width"]
        target = np.full((n, n), 255, np.int64)  # 255 = ignored by the loss
        labelled = self.common_valid[row : row + h, col : col + w] & out["valid"]
        target[:h, :w][labelled] = self.label[row : row + h, col : col + w][labelled].astype(np.int64) - 1
        out["target"] = torch.from_numpy(target)
        return out


def stack_crops(crops: list[dict[str, Any]], device: torch.device) -> dict[str, torch.Tensor]:
    keys = ["image", "temporal_coords", "location_coords"] + (["target"] if "target" in crops[0] else [])
    return {k: torch.stack([c[k] for c in crops]).to(device) for k in keys}
