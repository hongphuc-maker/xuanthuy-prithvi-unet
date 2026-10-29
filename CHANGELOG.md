# Changelog

## 1.1.3

- Executed example `notebooks/examples/xuanthuy_example.ipynb` (run on v1.1.2, Colab T4): installation
  checks, data, a 20-step run, the completed P3 runs, maps of Xuan Thuy (1 Oct 2026) and Thai Thuy
  (21 Sep 2026) with agreement between checkpoints and code versions.
- Notebooks: a clone left on the Colab machine is reused only if it is at `GIT_REF`; the quickstart
  warns when pip replaced a numpy already loaded in the kernel (restart needed).
- Package version set to 1.1.3 (`__init__.py`, `pyproject.toml`); tag v1.1.2 still reported 1.1.1.
  The version is part of a run's `method_hash`: run folders created with 1.1.1 cannot be resumed with 1.1.3.

## 1.1.2

- Notebook `notebooks/colab_train_and_predict.ipynb` rewritten as a guided quickstart: session setup,
  installation checks, image checks, mapping with any checkpoint (config read from the checkpoint),
  map comparison, training with config changes or new images/labels (`config_for_new_data`),
  20-step speed and memory test, results plots, scoring checkpoints on the validation band,
  Sentinel-2 export, checkpoint conversion. Edit only cells marked `# EDIT`.
- `scripts/clean_notebook.py`: tidies an executed notebook (progress bars, colour codes, widget state)
  and refuses errors, token-like strings and oversized files before a commit.
- Fix: `tests/test_repo_files.py` ignores the `*.egg-info` folder written by `pip install -e`.

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
