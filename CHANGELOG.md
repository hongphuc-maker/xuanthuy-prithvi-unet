# Changelog

## 1.1.1

- Fix: the `.gitignore` rule `data/` also matched the source package `src/prithvi_unet/data/`,
  so v1.0.0 and v1.1.0 on GitHub could not be imported (`No module named 'prithvi_unet.data'`).
  Folder rules are now anchored to the repository root (`/data/`, `/runs/`, ...).
- Test `tests/test_repo_files.py` fails if a `.gitignore` rule hides a project file again.
- Docs: broken ESA link replaced by SentiWiki; HF `config.json` cited for the Prithvi mean/std.

## 1.1.0

- Band subsets: `bands` selects bands by name from the GeoTIFFs; pretrained Conv3d channels are
  kept by band name. New configs `p3_single_date_4bands.json`, `p3_single_date_3bands.json`.
- `radiometry.dn_offset: "auto"`: per-image L2A offset from export tags, processing baseline or date
  (images before 25 Jan 2022 have no offset).
- `scripts/export_sentinel2.py`: export any area and period from Planetary Computer with the tags
  the pipeline reads. `scripts/set_band_metadata.py` for images from other sources.
- `scripts/predict.py --upsample nearest|bilinear --dn-offset ...`; memory estimate for large areas.
- `training.freeze_encoder`; `--prithvi-weights` / `$PRITHVI_WEIGHTS` for a local copy of the weights.
- Any number of classes (`class_map` 1..K).
- Pinned source links (TerraTorch 1.2.12, SMP 0.5.0, timm 1.0.30, completed-run commits) in every
  module and in `docs/REFERENCES.md`; `docs/EXTENDING.md`.
- Notebook runs in browser Colab, VS Code + Google Colab extension, or a local kernel.
- Lower peak memory when mapping right after training (checkpoint loaded with `mmap`).

## 1.0.0

- Explicit Prithvi-UNet replacing `xuan-thuy-prithvi-segmentation`; same function as the completed runs.
