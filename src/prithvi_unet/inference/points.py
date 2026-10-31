"""Agreement of a class map with verified reference points.

The points are one ESRI shapefile per class, named after the class (``Anm.shp``, ``Bgt.shp`` ...), given
either as a folder or as a ZIP of that folder (``data/xuanthuy/verified_points_shapefiles.zip``).
Each point is looked up in the pixel that contains it. A point is *usable* when it falls inside the
raster and the map has a class there (value 1..K; 0 = no data). Scores over the usable points:

    OA                     correct points / usable points
    macro_F1_represented   mean F1 over the classes that have points (a class without points has no F1)
    per class              support, predicted, precision, recall, F1  (same rule as inference/metrics.py)

This is agreement, not accuracy of a dated map: the date of the points is not confirmed, and most
Xuan Thuy points lie in the training area (see README). Never use it to choose a checkpoint.

Same rule as ``evaluate_reference_points`` of the completed runs (it reproduces their 955 / 1,036 = 0.922):
https://github.com/hongphuc-maker/xuan-thuy-prithvi-segmentation/blob/42290823beeb42e44ec87bb8845e439e10c18735/reference/wce_completed/prithvi_xtseg/inference.py#L126-L174

Shapefiles are read with the standard library (point geometries only), so no vector library is needed.
"""

from __future__ import annotations

import struct
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform as transform_coords

from .metrics import confusion_matrix, per_class_metrics

_POINT_TYPES = {1: "Point", 11: "PointZ", 21: "PointM"}


def _read_point_shp(raw: bytes, name: str) -> tuple[np.ndarray, np.ndarray]:
    """x, y of every record of a point shapefile (.shp bytes)."""
    if len(raw) < 100 or struct.unpack(">i", raw[0:4])[0] != 9994:
        raise ValueError(f"{name}: not a shapefile")
    shape_type = struct.unpack("<i", raw[32:36])[0]
    if shape_type not in _POINT_TYPES:
        raise ValueError(f"{name}: shape type {shape_type} is not a point type {sorted(_POINT_TYPES)}")
    xs, ys, offset = [], [], 100
    while offset + 8 <= len(raw):
        length = struct.unpack(">i", raw[offset + 4:offset + 8])[0] * 2  # content length, 16-bit words
        content = raw[offset + 8:offset + 8 + length]
        record_type = struct.unpack("<i", content[0:4])[0]
        if record_type == 0:  # null shape
            raise ValueError(f"{name}: record without geometry at byte {offset}")
        x, y = struct.unpack("<2d", content[4:20])
        xs.append(x)
        ys.append(y)
        offset += 8 + length
    return np.array(xs), np.array(ys)


def read_point_layers(source: str | Path) -> list[dict[str, Any]]:
    """[{"layer": "Anm", "x": array, "y": array, "crs": CRS}, ...] from a folder or a ZIP of shapefiles."""
    source = Path(source)
    files: dict[str, bytes] = {}
    if source.is_dir():
        for path in source.iterdir():
            if path.suffix.lower() in (".shp", ".prj"):
                files[path.name] = path.read_bytes()
    elif zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            for entry in archive.infolist():
                if Path(entry.filename).suffix.lower() in (".shp", ".prj"):
                    files[Path(entry.filename).name] = archive.read(entry)
    else:
        raise FileNotFoundError(f"{source}: expected a folder or a ZIP of point shapefiles")
    layers = []
    for name in sorted(n for n in files if n.lower().endswith(".shp")):
        stem = name[:-4]
        prj = next((files[n] for n in files if n.lower() == f"{stem.lower()}.prj"), None)
        if prj is None:
            raise ValueError(f"{name}: missing .prj (coordinate system unknown)")
        x, y = _read_point_shp(files[name], name)
        layers.append({"layer": stem, "x": x, "y": y, "crs": CRS.from_wkt(prj.decode("utf-8", "replace").strip())})
    if not layers:
        raise ValueError(f"{source}: no .shp files")
    return layers


def score_map_on_points(map_path: str | Path, points: str | Path, class_map: dict[str, str]
                        ) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """(summary, per-class table, one row per point) for one class map.

    ``class_map`` is the config's {"1": "Anm", ...}; each shapefile name must be one of its class names
    (case-insensitive), which gives the true class of its points.
    """
    by_name = {name.casefold(): int(code) for code, name in class_map.items()}
    k = len(class_map)
    rows = []
    with rasterio.open(map_path) as src:
        for layer in read_point_layers(points):
            code = by_name.get(layer["layer"].casefold())
            if code is None:
                raise ValueError(f"point layer {layer['layer']!r} is not a class of the config {sorted(class_map.values())}")
            xs, ys = layer["x"], layer["y"]
            if layer["crs"] != src.crs:
                xs, ys = map(np.asarray, transform_coords(layer["crs"], src.crs, xs.tolist(), ys.tolist()))
            for i, (x, y) in enumerate(zip(xs, ys)):
                row, col = src.index(x, y)
                inside = 0 <= row < src.height and 0 <= col < src.width
                pred = int(src.read(1, window=((row, row + 1), (col, col + 1)))[0, 0]) if inside else 0
                rows.append({"layer": layer["layer"], "id": i, "x": float(x), "y": float(y), "row": int(row),
                             "col": int(col), "truth": code, "pred": pred, "usable": inside and 1 <= pred <= k})
    frame = pd.DataFrame(rows)
    good = frame[frame.usable]
    if good.empty:
        raise ValueError("no point falls on a classified pixel of the map")
    confusion = confusion_matrix(good.truth.to_numpy() - 1, good.pred.to_numpy() - 1, k)
    summary, table = per_class_metrics(confusion, class_map)
    represented = table.support > 0
    summary = {"map": str(map_path), "points": str(points), "input_points": int(len(frame)),
               "usable_points": int(len(good)), "correct_points": int(np.trace(confusion)),
               "OA": summary["OA"], "macro_F1_represented": float(table.loc[represented, "F1"].mean()),
               "represented_classes": [str(c) for c in table.loc[represented, "class"]],
               "interpretation": "agreement with reference points of unconfirmed date; not a dated accuracy"}
    return summary, table[["code", "class", "support", "predicted", "precision", "recall", "F1"]], frame
