# Extending the pipeline

Three extension points, from cheapest to most work:

| You want to | Change | Code change? |
| --- | --- | --- |
| Map another date or another area | input images only | no |
| Tune training or architecture sizes | a copy of a config | no |
| Train with fewer bands | `bands` + `normalization` in a config | no |
| Train on another area with its own labels | a new config (images, label, split, classes) | no |
| Add or replace a module (stem, neck, decoder) | `src/prithvi_unet/model/` | yes |

Any config change starts a new run folder (`RUN_LOCK.json` refuses to mix settings).

## 1. Map new images

```bash
# a) export images (any AOI, any period) from Planetary Computer
pip install -e ".[export]"
python scripts/export_sentinel2.py --aoi area.geojson --start 2026-10-01 --end 2026-10-31 --out data/area --dry-run
python scripts/export_sentinel2.py --aoi area.geojson --start 2026-10-01 --end 2026-10-31 --out data/area

# b) check radiometry (water in B11/B12 near DN 1000 with offset, near 0 without)
python scripts/band_statistics.py data/area/*.tif --offset auto

# c) predict (P3: one image; P2: six images, any order, sorted by acquisition_date)
python scripts/predict.py --config configs/p3_single_date.json --checkpoint runs/p3/best_any.pt \
    --images data/area/Sentinel2_2026-10-15_*.tif --out maps/area_20261015 --dn-offset auto
```

* Images from other sources: add band names and date with
  `scripts/set_band_metadata.py image.tif --bands B02 B03 B04 B8A B11 B12 --date 2026-10-15 --boa-add-offset 0`.
* Before 25 Jan 2022 L2A has no offset: use `--dn-offset auto` (reads the export tags) or `--dn-offset 0`.
* P2 expects six dates with a seasonal spread similar to training (Jan, May, Jul, Aug, Sep, target).
* Memory grows with area: about (pixels × (2 × bands × T + 4 × classes)) bytes plus 1.3 GB for the model;
  clip large areas into pieces. CPU inference works (P3: a few minutes for 1,374 × 1,821 px).
* A model trained on Xuan Thuy carries its 11 classes and its training distribution. On another
  area, inspect the map and the entropy layer before using it; accuracy is unknown without labels.
* Compare the final resize without retraining: `--upsample nearest` (default: bilinear, as trained).

## 2. Tune with a config

Copy a config, change values, give it a new `experiment_id`, train into a new run folder:

| Key | Default | Effect |
| --- | --- | --- |
| `training.encoder_lr`, `decoder_lr` | 1e-5, 1e-4 | learning rates of the pretrained and new layers |
| `training.freeze_encoder` | false | train only neck, decoder, head (feature-quality test) |
| `training.maximum_steps`, `validation_interval_steps`, `patience_evaluations` | 3120, 156, 6 | schedule and early stopping |
| `training.seed`, `split.sampling.*_seed` | fixed | repeat with other seeds to measure run-to-run spread |
| `model.out_indices` | [5, 11, 17, 23] | which blocks feed the four pyramid levels (sorted, 4 values) |
| `model.decoder_channels` | [512, 256, 128, 64] | width of the four decoder blocks |
| `model.head_dropout` | 0.1 | dropout before the 1×1 classifier |
| `model.final_upsample` | bilinear | final resize (nearest only for comparisons) |
| `model.gradient_checkpointing` | true | less GPU memory, slower; same numbers |
| `split.parameters` | columns 1122–1388, guard 48 | validation band |
| `split.sampling.n_patches`, `rare_codes`, `rare_quota_each` | 4992, [5, 6, 7], 416 | patch sampling |

```bash
python scripts/trace_shapes.py --config configs/my_variant.json      # shapes and parameters
python scripts/train.py --config configs/my_variant.json --data-root DATA --run-dir runs/my_variant --steps 2
python scripts/train.py --config configs/my_variant.json --data-root DATA --run-dir runs/my_variant
```

## 3. Fewer bands

`configs/p3_single_date_4bands.json` (B02 B03 B04 B8A) and `p3_single_date_3bands.json`
(B02 B03 B04) use the same 6-band files; the reader picks bands by name, validity
still uses all six bands, so the runs are scored on the same pixels. Prithvi keeps the
pretrained Conv3d channel of each kept band (`select_patch_embed_weights`) and drops the others.
For another subset, set `bands` and take `mean`/`std` from `prithvi_normalization(bands)`.

## 4. Another area with its own labels

Write a config with the new `images` (sha256 from `python -c "from prithvi_unet.utils import sha256_file; print(sha256_file('f.tif'))"`),
`label` (one band, codes 1..K, 0 = no label), `class_map` (K classes), `expected_shape`,
`expected_crs`, a `split` band inside the area, `rare_codes`, and `acceptance`.

## 5. A new module

Each stage is one attribute of `PrithviUNet` (`model/prithvi_unet.py`): `encoder`,
`to_feature_map`, `pyramid`, `decoder`, `head`. Example: a small CNN stem on the raw image
gives real skips at 224² and 112² (UNETR-style):

```python
# model/stem.py
class ImageStem(nn.Module):
    def __init__(self, in_ch, ch=(32, 64)):
        super().__init__()
        self.full = ConvGNReLU(in_ch, ch[0])                      # 224 x 224
        self.half = nn.Sequential(nn.Conv2d(ch[0], ch[1], 3, stride=2, padding=1, bias=False),
                                  group_norm(ch[1]), nn.ReLU(inplace=True))   # 112 x 112
    def forward(self, image):                                     # image [B, C, T, H, W]
        x = image[:, :, -1]                                       # target date
        f224 = self.full(x)
        return f224, self.half(f224)
```

then in `PrithviUNet`: create `self.stem`, replace decoder block 4 with two blocks that
upsample 56 → 112 → 224 and concatenate the stem features, and drop the final bilinear
resize. Add a config flag for it, a shape test in `tests/test_model_parts.py`, and train a
new run. Keep the old path as default so existing checkpoints still load.
