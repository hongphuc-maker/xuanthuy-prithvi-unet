"""Verified-point scoring: shapefile reading, CRS handling, metrics, and the repository's point sets."""

import struct
import zipfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin, xy
from rasterio.warp import transform as transform_coords

from prithvi_unet.inference.points import read_point_layers, score_map_on_points

ROOT = Path(__file__).resolve().parents[1]
CLASSES = {"1": "Anm", "2": "Bgt", "3": "Rnm"}


def write_point_shapefile(folder: Path, name: str, xy, crs: CRS) -> None:
    records = b""
    for i, (x, y) in enumerate(xy, start=1):
        content = struct.pack("<i2d", 1, x, y)
        records += struct.pack(">2i", i, len(content) // 2) + content
    xs, ys = [p[0] for p in xy], [p[1] for p in xy]
    header = struct.pack(">7i", 9994, 0, 0, 0, 0, 0, (100 + len(records)) // 2)
    header += struct.pack("<2i4d4d", 1000, 1, min(xs), min(ys), max(xs), max(ys), 0, 0, 0, 0)
    (folder / f"{name}.shp").write_bytes(header + records)
    (folder / f"{name}.prj").write_text(crs.to_wkt())


@pytest.fixture()
def scene(tmp_path):
    """10 x 10 map in UTM 48N: left half Anm (1), right half Bgt (2), top row no data (0)."""
    utm = CRS.from_epsg(32648)
    data = np.ones((10, 10), np.uint8)
    data[:, 5:] = 2
    data[0, :] = 0
    transform = from_origin(600000, 2250000, 10, 10)
    path = tmp_path / "classes.tif"
    with rasterio.open(path, "w", driver="GTiff", height=10, width=10, count=1, dtype="uint8", crs=utm,
                       transform=transform, nodata=0) as dst:
        dst.write(data, 1)
    return path, utm, transform


def centre(transform, row, col):
    return xy(transform, row, col)  # pixel centre


def test_scores_points_in_folder_and_zip(scene, tmp_path):
    path, utm, t = scene
    pts = tmp_path / "pts"
    pts.mkdir()
    write_point_shapefile(pts, "Anm", [centre(t, 3, 1), centre(t, 4, 2), centre(t, 5, 7)], utm)  # 2 right, 1 -> Bgt
    write_point_shapefile(pts, "Bgt", [centre(t, 6, 8), centre(t, 0, 8)], utm)  # 1 right, 1 on no data
    write_point_shapefile(pts, "Rnm", [(0.0, 0.0)], utm)  # outside the raster
    archive = tmp_path / "pts.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for f in pts.iterdir():
            z.write(f, f"pts/{f.name}")
    for source in (pts, archive):
        summary, table, frame = score_map_on_points(path, source, CLASSES)
        assert (summary["input_points"], summary["usable_points"], summary["correct_points"]) == (6, 4, 3)
        assert summary["OA"] == pytest.approx(0.75)
        # F1 Anm = 2*2/(3+2) = 0.8, F1 Bgt = 2*1/(1+2) = 2/3; Rnm has no usable point -> not represented
        assert summary["represented_classes"] == ["Anm", "Bgt"]
        assert summary["macro_F1_represented"] == pytest.approx((0.8 + 2 / 3) / 2)
        assert int(frame.usable.sum()) == 4


def test_points_in_another_crs_are_reprojected(scene, tmp_path):
    path, utm, t = scene
    x, y = centre(t, 5, 2)
    lon, lat = transform_coords(utm, CRS.from_epsg(4326), [x], [y])
    pts = tmp_path / "wgs84"
    pts.mkdir()
    write_point_shapefile(pts, "anm", [(lon[0], lat[0])], CRS.from_epsg(4326))  # layer names are case-insensitive
    summary, _, frame = score_map_on_points(path, pts, CLASSES)
    assert (int(frame.row[0]), int(frame.col[0])) == (5, 2)
    assert summary["OA"] == 1.0


def test_unknown_layer_name_is_refused(scene, tmp_path):
    path, utm, t = scene
    pts = tmp_path / "bad"
    pts.mkdir()
    write_point_shapefile(pts, "Forest", [centre(t, 5, 2)], utm)
    with pytest.raises(ValueError, match="not a class"):
        score_map_on_points(path, pts, CLASSES)


@pytest.mark.parametrize("area,count,layers", [
    ("xuanthuy", 1037, ["Anm", "Bgt", "Dnn", "Rnm", "Stx", "Vbn", "Vcs"]),
    ("thaithuy", 616, ["Anm", "Bgt", "Rnm", "Stx", "Vbn", "Vcs"]),
])
def test_repository_point_sets(area, count, layers):
    source = ROOT / "data" / area / "verified_points_shapefiles.zip"
    if not source.exists():
        pytest.skip(f"{source} not present")
    found = read_point_layers(source)
    assert [layer["layer"] for layer in found] == layers
    assert sum(len(layer["x"]) for layer in found) == count
    assert all(layer["crs"].to_epsg() == 32648 for layer in found)
