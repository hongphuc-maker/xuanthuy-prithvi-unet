# Xuan Thuy Prithvi-UNet

11-class land-cover segmentation of Xuan Thuy (Nam Dinh, Vietnam) from Sentinel-2 L2A,
using the Prithvi-EO-2.0-300M-TL foundation model as encoder and a U-Net decoder.

This repository replaces `xuan-thuy-prithvi-segmentation`. The model is the same
function as in the completed P2/P3 runs (checked by `tests/test_legacy_equivalence.py`),
but every layer is now written out in plain PyTorch in `src/prithvi_unet/model/`.

```
image [B, C bands (6, or a subset), T dates, 224, 224]
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

## Results

Agreement with the **1,037 verified points** of Xuan Thuy (7 classes), on maps of **1 October 2026**,
an image that no model saw in training. OA = share of points where the map gives the point's class.

| Model | Input | Config | OA, verified points | Macro-F1, 7 point classes | Validation macro-F1 (selection) |
| --- | --- | --- | ---: | ---: | ---: |
| U-Net from scratch (`xuanthuy-segmentation`, run E0w) | 1 date, 6 bands | other repository | 0.661 | 0.554 | 0.415 |
| Prithvi P2 multi-temporal | 6 dates, 6 bands | `configs/p2_multitemporal.json` | **0.875** | **0.838** | 0.443 |
| Prithvi P3 single-date (completed run) | 1 date, 6 bands | `configs/p3_single_date.json` | 0.859 | 0.814 | 0.448 |
| Prithvi P3 single-date (re-run with this package) | 1 date, 6 bands | `configs/p3_single_date.json` | 0.866 | 0.833 | 0.429 |

Thai Thuy, 21 September 2026 (area not used in training, 616 points, 6 classes): P2 0.601,
P3 completed run 0.646, P3 re-run 0.623.

How to read it:

* On the training image (21 Sep) the U-Net (0.924) and P2 (0.922) agree equally well with the points; on
  1 Oct the U-Net drops to 0.661 while Prithvi stays near 0.87 (paired test over points: P3 re-run right
  where the U-Net is wrong 238 times, the reverse 25 times). The U-Net's errors are mostly water classes
  (Vbn, Vcs, Bgt). The two pipelines also differ outside the encoder (normalisation, decoder, training
  recipe), so the gap is not yet attributed to the pretrained encoder alone.
* The points are of unconfirmed date (agreement, not dated accuracy), cover 7 of the 11 classes (none
  for Dc, DK, trống, Gt), and 886 of them lie in the training area: the 21 Sep scores are optimistic.
  See [`data/README.md`](data/README.md).
* Checkpoints are still chosen by macro-F1 over all 11 classes on the validation band (last column),
  never by the points. No run passed the acceptance gate (all 11 classes predicted and recall >= 0.05 on
  the three rare classes), so the maps are exploratory.

Reproduce a score: `python scripts/score_points.py --points data/xuanthuy/verified_points_shapefiles.zip --maps <map>_classes.tif`
(the maps come from `scripts/predict.py`; checkpoints are not in the repository).

## Install

```bash
git clone https://github.com/hongphuc-maker/xuanthuy-prithvi-unet.git && cd xuanthuy-prithvi-unet
pip install -e ".[test]"          # add ,export for scripts/export_sentinel2.py
pytest                     # fast tests; RUN_SLOW=1 pytest for the end-to-end CPU run
python scripts/trace_shapes.py --config configs/p3_single_date.json   # every shape and parameter count
```

## Data (in `data/`, 192 MB)

Everything needed to train and map is in the repository: Sentinel-2 L2A six-band GeoTIFFs of
Xuan Thuy (10 dates) and Thai Thuy (12 dates) from Microsoft Planetary Computer (`sentinel-2-l2a`),
bands B02, B03, B04, B8A, B11, B12 on a 10 m EPSG:32648 grid, DN with the L2A offset and an
`acquisition_date` tag; the Xuan Thuy label raster `rasterized_vector_categorized.tif`; and the verified
points of both areas. Files, dates, checksums and limits: [`data/README.md`](data/README.md).
Export and normalisation: [`docs/DATA_AND_NORMALIZATION.md`](docs/DATA_AND_NORMALIZATION.md).

## Train and map (GPU, e.g. Colab)

```bash
python scripts/train.py --config configs/p3_single_date.json --data-root data/xuanthuy --run-dir runs/p3
```

Downloads the pinned Prithvi weights from Hugging Face, prepares patches, trains
with weighted cross-entropy, keeps the best checkpoint by validation macro-F1 and
writes the 21 September map. Interrupted runs resume from `latest_resume.pt`.

## Map new dates

```bash
python scripts/predict.py --config configs/p3_single_date.json --checkpoint runs/p3/best_any.pt \
    --images data/xuanthuy/Sentinel2_2026-10-01_S2C_MSIL2A_20261001T_HSTDNNXUANTHUY_6bands_10m.tif --out maps/p3
python scripts/score_points.py --points data/xuanthuy/verified_points_shapefiles.zip --maps maps/p3/2026-10-01_classes.tif
```

Checkpoints from the old repository load directly (converted on the fly, see
`src/prithvi_unet/model/legacy.py`). Other areas, other years, fewer bands, other
hyper-parameters and new modules: [`docs/EXTENDING.md`](docs/EXTENDING.md).

## Notebooks (browser Colab or VS Code)

`notebooks/colab_train_and_predict.ipynb` runs in Colab in the browser, in VS Code with
the Google Colab extension (code runs on the Colab VM; data comes with the clone, results go to Google Drive), or
on a local Python kernel (CPU is enough for prediction, a CUDA GPU is needed for training).
It walks through every feature (installation checks, image checks, mapping with any checkpoint,
scoring maps on the verified points, training with changed settings or new images and labels,
scoring checkpoints, Sentinel-2 export);
only the cells marked `# EDIT` need changes.
`notebooks/examples/xuanthuy_example.ipynb` is an executed run on the project data (Xuan Thuy and
Thai Thuy maps, P2 and P3 scores) to see the expected outputs without running anything.

## Repository layout

```
configs/                 one JSON per experiment (same schema as the completed runs)
data/                    Sentinel-2 images, label raster and verified points (data/README.md)
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
  inference/             sliding windows, metrics, GeoTIFF export, new-date prediction,
                         agreement with verified points (points.py)
scripts/                 train, predict, score_points, export_sentinel2, band_statistics,
                         set_band_metadata, trace_shapes, convert_legacy_checkpoint
tests/                   unit tests, equivalence with the old model and old sampling
docs/                    architecture, data and normalisation, code walkthrough, migration
```

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): every layer, its shape, its parameter count, fixed or learned, and its origin.
- [`docs/DATA_AND_NORMALIZATION.md`](docs/DATA_AND_NORMALIZATION.md): source, offset, formula, band statistics.
- [`docs/CODE_WALKTHROUGH.md`](docs/CODE_WALKTHROUGH.md): one training step and one prediction, file by file.
- [`docs/MIGRATION.md`](docs/MIGRATION.md): what changed from the old repository and how equivalence was checked.
- [`docs/EXTENDING.md`](docs/EXTENDING.md): new images, new areas, fewer bands, tuning, new modules.
- [`docs/REFERENCES.md`](docs/REFERENCES.md): for each module, the upstream code (pinned permalinks) and paper it follows.

## License

Code: BSD 3-Clause. Sentinel-2 images: contain modified Copernicus Sentinel data 2026.
The label raster and the verified points are project reference data, not covered by the code licence
(see `data/README.md`). Checkpoints are not distributed.
