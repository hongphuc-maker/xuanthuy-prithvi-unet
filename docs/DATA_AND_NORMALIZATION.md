# Data and normalisation

## Source

| Item | Value |
| --- | --- |
| Archive | Microsoft Planetary Computer STAC, collection `sentinel-2-l2a` |
| Selection | scene cloud cover < 10 %, footprint covers ≥ 99.9 % of the AOI; tile T48QXH, orbit R118 |
| Bands | B02 (490 nm, 10 m), B03 (560 nm, 10 m), B04 (665 nm, 10 m), B8A (865 nm, 20 m), B11 (1610 nm, 20 m), B12 (2190 nm, 20 m) |
| Grid | EPSG:32648, 10 m, 1,374 × 1,821; 20 m bands resampled bilinearly to 10 m (WarpedVRT) |
| Values | raw L2A DN, uint16, nodata 0; no offset removed and no scaling at export |
| Cloud mask | none (SCL not exported); 12 Aug has visible cloud, 30 Mar is unusually bright in the visible bands |
| Export code | `notebooks/00_sentinel2_export.ipynb` of the old repository |

## Offset

Since processing baseline 04.00 (25 January 2022) ESA shifts L2A values by
`BOA_ADD_OFFSET = −1000` ([ESA Level-2A products](https://sentinel.esa.int/web/sentinel/technical-guides/sentinel-2-msi/level-2a-algorithms-products)).
Planetary Computer keeps it in the DN and shows how to remove it
([baseline-change notebook](https://nbviewer.org/github/microsoft/PlanetaryComputerExamples/blob/main/datasets/sentinel-2-l2a/baseline-change.ipynb)).
Surface reflectance is therefore ρ = (DN − 1000) / 10000.

Images acquired before 25 January 2022 have no offset. Set `radiometry.dn_offset` to
`"auto"` (or pass `--dn-offset auto` to `scripts/predict.py`) to choose the offset per image
from the `boa_add_offset` / `s2_processing_baseline` tags written by
`scripts/export_sentinel2.py`, or from the acquisition date if the tags are missing.

Check in our data: water reflects almost nothing in SWIR. Over the ten 2026 images,
the 1st percentile of B11 and B12 DN is 1,017 and 1,022 (ρ ≈ 0.002). Without the
offset, water would have 10 % SWIR reflectance, which is not physical.

## Formula

```
ρ = (DN − 1000) / 10000                       reflectance, 0..1
z = (ρ − μ / 10000) / (σ / 10000)             Prithvi pretraining scale
  = (DN − 1000 − μ) / σ                       same number, shorter form
```

μ, σ (reflectance × 10000) from TerraTorch 1.2.12 `PRITHVI_V2_MEAN` / `PRITHVI_V2_STD`,
computed on the HLS pretraining data (HLS stores reflectance as int16 with scale 0.0001):

| Band | μ | σ |
| --- | ---: | ---: |
| B02 Blue | 1087 | 2248 |
| B03 Green | 1342 | 2179 |
| B04 Red | 1433 | 2178 |
| B8A NIR narrow | 2734 | 1850 |
| B11 SWIR 1 | 1958 | 1242 |
| B12 SWIR 2 | 1363 | 1049 |

"Divide by 10000" is step 1 only (without the offset it is also biased by +0.1).
The pretrained Conv3d was trained on step 2's scale, so step 2 is required.

## Band statistics (10 images, 9 Jan to 1 Oct 2026, 14,713,412 valid pixels)

`python scripts/band_statistics.py <images>`

| Band | DN p1 | DN median | DN p99 | DN < 1000 | mean ρ | z mean ± std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B02 | 924 | 1,490 | 3,101 | 2.6 % | 0.057 | −0.23 ± 0.22 |
| B03 | 1,274 | 1,802 | 3,280 | 0.04 % | 0.087 | −0.22 ± 0.21 |
| B04 | 1,043 | 1,747 | 3,418 | 0.5 % | 0.081 | −0.29 ± 0.25 |
| B8A | 945 | 2,206 | 5,846 | 2.5 % | 0.161 | −0.61 ± 0.80 |
| B11 | 1,017 | 1,709 | 4,198 | < 0.01 % | 0.094 | −0.82 ± 0.71 |
| B12 | 1,022 | 1,371 | 3,789 | < 0.01 % | 0.063 | −0.70 ± 0.63 |

z is not N(0, 1): the scene is darker (much water) than the global HLS sample. The
goal is the pretraining scale, not a locally standardised input. DN < 1000 means
slightly negative reflectance over dark water; values are not clipped.

## Masks and padding

* A pixel is valid when all six bands are non-zero on every listed image (P2 and P3
  use the same validity, so they are scored on the same pixels).
* Invalid pixels and the padding of edge crops are set to 0 after normalisation,
  which equals the Prithvi mean.
* Labels 1..11 become targets 0..10; 255 marks pixels ignored by the loss.
