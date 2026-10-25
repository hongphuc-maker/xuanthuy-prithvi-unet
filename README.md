# Xuan Thuy Prithvi-UNet

11-class land-cover segmentation of Xuan Thuy (Nam Dinh, Vietnam) from Sentinel-2 L2A,
using the Prithvi-EO-2.0-300M-TL foundation model as encoder and a U-Net decoder.

This repository replaces `xuan-thuy-prithvi-segmentation`. The model is the same
function as in the completed P2/P3 runs (checked by `tests/test_legacy_equivalence.py`),
but every layer is now written out in plain PyTorch in `src/prithvi_unet/model/`.

```
image [B, 6 bands, T dates, 224, 224]
  │  radiometry (fixed)          DN -> reflectance -> z-score with Prithvi mean/std        data/radiometry.py
  ▼
  Conv3d 1x16x16 (learned)       patch embedding: 6 x 16 x 16 -> 1024 per patch           model/encoder.py
  + position, date, location     fixed sin-cos tables, learned scales
  24 Transformer blocks          read out after blocks 5, 11, 17, 23
  ▼
  tokens -> 14 x 14 map (fixed)  drop CLS, grid, keep target date                         model/neck.py
  feature pyramid (learned)      ConvT x4 -> 56², ConvT x2 -> 28², identity 14², MaxPool 7²
  ▼
  U-Net decoder (learned)        up x2 + concat skip + 2 x (Conv3x3-GN-ReLU), 3 skips      model/decoder.py
  head (learned)                 Dropout + Conv1x1 -> 11 classes at 56 x 56
  bilinear x4 (fixed)            -> [B, 11, 224, 224]                                     model/prithvi_unet.py
```

| Experiment | Input dates | Config | Validation macro-F1 (completed run) |
| --- | --- | --- | ---: |
| P2 multi-temporal | 6 (9 Jan to 21 Sep 2026) | `configs/p2_multitemporal.json` | 0.443 |
| P3 single-date | 1 (21 Sep 2026) | `configs/p3_single_date.json` | 0.448 |

Macro-F1 is measured on a vertical validation band against historical labels of
unknown date; no run passed the acceptance gate (all 11 classes predicted and
recall >= 0.05 on the three rare classes), so maps are exploratory.

## Install

```bash
git clone <this repository> && cd xuanthuy-prithvi-unet
pip install -e ".[test]"
pytest                     # fast tests; RUN_SLOW=1 pytest for the end-to-end CPU run
python scripts/trace_shapes.py --config configs/p3_single_date.json   # every shape and parameter count
```

## Data (not in the repository)

Six-band GeoTIFFs exported from Microsoft Planetary Computer (`sentinel-2-l2a`),
bands B02, B03, B04, B8A, B11, B12 on a 10 m EPSG:32648 grid (1,374 x 1,821),
DN with the L2A offset, an `acquisition_date` tag, and the label raster
`rasterized_vector_categorized.tif`. File names and SHA-256 checksums are in the
configs. See [`docs/DATA_AND_NORMALIZATION.md`](docs/DATA_AND_NORMALIZATION.md).

## Train and map (GPU, e.g. Colab)

```bash
python scripts/train.py --config configs/p3_single_date.json --data-root /content/data --run-dir /content/runs/p3
```

Downloads the pinned Prithvi weights from Hugging Face, prepares patches, trains
with weighted cross-entropy, keeps the best checkpoint by validation macro-F1 and
writes the 21 September map. Interrupted runs resume from `latest_resume.pt`.

## Map new dates

```bash
python scripts/predict.py --config configs/p3_single_date.json --checkpoint runs/p3/best_any.pt \
    --images data/Sentinel2_2026-10-01_S2C_MSIL2A_20261001T_HSTDNNXUANTHUY_6bands_10m.tif --out maps/
```

Checkpoints from the old repository load directly (converted on the fly, see
`src/prithvi_unet/model/legacy.py`).

## Repository layout

```
configs/                 one JSON per experiment (same schema as the completed runs)
src/prithvi_unet/
  config.py              load and validate a config
  data/radiometry.py     DN -> model input (offset, reflectance, z-score)
  data/scene.py          read GeoTIFFs, masks, 224 x 224 crops
  data/splits.py         validation band + guard
  data/sampling.py       training patches, class-complete batches, validation windows
  model/encoder.py       Prithvi encoder, forward written out
  model/neck.py          tokens -> map, simple feature pyramid
  model/decoder.py       U-Net decoder blocks and head
  model/prithvi_unet.py  assembly, forward_trace, build_model
  model/legacy.py        old checkpoint -> new layout
  training/loss.py       class weights and weighted cross-entropy
  training/trainer.py    optimisation, validation, selection, resume, final map
  inference/             sliding windows, metrics, GeoTIFF export, new-date prediction
scripts/                 command-line entry points
tests/                   unit tests, equivalence with the old model and old sampling
docs/                    architecture, data and normalisation, code walkthrough, migration
```

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): every layer, its shape, its parameter count, fixed or learned, and its origin.
- [`docs/DATA_AND_NORMALIZATION.md`](docs/DATA_AND_NORMALIZATION.md): source, offset, formula, band statistics.
- [`docs/CODE_WALKTHROUGH.md`](docs/CODE_WALKTHROUGH.md): one training step and one prediction, file by file.
- [`docs/MIGRATION.md`](docs/MIGRATION.md): what changed from the old repository and how equivalence was checked.

## License

BSD 3-Clause. Data and checkpoints are not distributed.
