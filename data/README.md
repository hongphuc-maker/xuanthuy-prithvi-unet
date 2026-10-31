# Data

Everything needed to train P2/P3, map Xuan Thuy and Thai Thuy, and score maps on the verified points
(192 MB). Used as `--data-root data/xuanthuy`; the configs list each file with its SHA-256, and the
loader refuses a file whose checksum differs.

```
data/
├── xuanthuy/                                   Xuan Thuy, grid EPSG:32648, 10 m, 1,374 x 1,821
│   ├── Sentinel2_2026-MM-DD_..._6bands_10m.tif   10 dates, 9 Jan to 1 Oct 2026
│   ├── rasterized_vector_categorized.tif         label raster, 11 classes (0 = no label)
│   └── verified_points_shapefiles.zip            1,037 verified points, one shapefile per class (7 classes)
└── thaithuy/                                   Thai Thuy, grid EPSG:32648, 10 m, 1,108 x 1,164
    ├── Sentinel2_2026-MM-DD_..._6bands_10m.tif   12 dates, 9 Jan to 13 Oct 2026 (no labels)
    └── verified_points_shapefiles.zip            616 verified points, one shapefile per class (6 classes)
```

## Sentinel-2 images

Six bands B02, B03, B04, B8A, B11, B12 of Sentinel-2 L2A from Microsoft Planetary Computer
(`sentinel-2-l2a`), raw DN (uint16, nodata 0) with the L2A offset of −1000 still in the values, band
names in the band descriptions and an `acquisition_date` tag. Export, offset and normalisation:
[`docs/DATA_AND_NORMALIZATION.md`](../docs/DATA_AND_NORMALIZATION.md).
Contains modified Copernicus Sentinel data 2026.

| Xuan Thuy date | Satellite | Used by | MB | SHA-256 |
| --- | --- | --- | ---: | --- |
| 2026-01-09 | S2B | P2 training | 12.8 | `50bc6049e2ceae1cd92f15d880c9d94d8c7727b4cb97c3a09fe5781651bcc8b0` |
| 2026-03-30 | S2B | P2 map of 1 Oct | 13.6 | `1040f13a4b12899e89f7fdc5d45a72913979dee9481b28ca06b6f96ba5245689` |
| 2026-05-24 | S2C | P2 map of 1 Oct | 13.9 | `3c67a06d88075b2dad060496b5ecb0e655a497695fc55ad780fcf1642913e2e5` |
| 2026-05-26 | S2A | P2 training | 13.9 | `9d4f4af0393b411759668173452cc5df658d895b5ccfbfb0b52c4eeff441f8b2` |
| 2026-07-18 | S2B | P2 training, P2 map of 1 Oct | 13.6 | `6fa20e2ec4f9dc73999d8b2e3ffd35ea670f4b7109507c355fb2e31f9cfdacab` |
| 2026-08-12 | S2C | P2 training (visible cloud) | 13.7 | `21a57a5432f684165b15fa8a6f52f8e7b103a9dc03a6a24692b15a03200b2617` |
| 2026-09-03 | S2A | P2 training, P2 map of 1 Oct | 13.3 | `df112425707fe4fcc301d295c07c307f416ee1cbbd196d88fb5ec8660cc16630` |
| 2026-09-21 | S2C | P2 and P3 training (target date) | 12.9 | `d924eb0e19175ba0c38d1fc058516783778bf39fb4b27257977111c1c3ecfeb3` |
| 2026-09-23 | S2A | P2 map of 1 Oct | 13.6 | `741e4affe19d562506f5104a9c877770963b413297cbbcaf2c0976d9022f3a23` |
| 2026-10-01 | S2C | maps of 1 Oct, all models (not used in training) | 13.1 | `f2d803bb8fadb3624d71898e3b215e565ef77e73fee07eadb08d946b75b46472` |

| Thai Thuy date | Satellite | MB | SHA-256 |
| --- | --- | ---: | --- |
| 2026-01-09 | S2B | 4.9 | `584607b61939235018ed8830f03f57a10ac3ebe7e0bafc7cf0c96cc210dcf7eb` |
| 2026-01-14 | S2C | 5.2 | `a1accfa89bfd2f2d7a04cb691fd5b8da04eda43c60b934fb24de3ea23fdad35b` |
| 2026-02-03 | S2C | 5.1 | `4be6066a82815ee1195f996b8c7ce3996a0db990ff56ba5a4bf979b13e9f5d09` |
| 2026-05-24 | S2C | 5.7 | `9b464faadd553ea67ce3382b1124debaf046295fce7995e0634923b938c83d30` |
| 2026-05-26 | S2A | 5.7 | `2479168ad7510a6ad8a6704875bf472f22d4d24e41dfefef3c147d6d3867b0b4` |
| 2026-07-18 | S2B | 5.7 | `b9f1ce6239f7f851e2af88fdef835db2dc71ec5393d33d118d0d8e78830387da` |
| 2026-08-12 | S2C | 5.1 | `a381cefacc1aa111e1407ebeecf4d2ecc9521d9bfc1fb7838e88287138dfbfb5` |
| 2026-08-17 | S2B | 5.3 | `05bab4041878a2d9e15082813cf7033aac9adaf71b8bab23dc8a9a2935833b85` |
| 2026-09-03 | S2A | 5.2 | `d633d862bd7d0ee8eb954226844bba5aa071e4aba3e2bcfd726a32dea76e88ea` |
| 2026-09-11 | S2C | 5.4 | `4011345cf50cf7940110ff0f32505bf4437a0ce468029fbe7c4ff46beb1360ee` |
| 2026-09-21 | S2C | 5.0 | `1e496846cfba3367ad8fab2a1e5966708bff8c4bd8c561f2af73c7190cd2320a` |
| 2026-10-13 | S2A | 5.1 | `aa194c7e68b0c1476828714ef439d1fcdc3575014ad69602379c6854979ab775` |

The Thai Thuy maps of the report use 21 Sep (P3) and 9 Jan, 26 May, 18 Jul, 12 Aug, 3 Sep, 21 Sep (P2).

## Labels

`xuanthuy/rasterized_vector_categorized.tif` (2.5 MB, SHA-256
`d02d44ac7d8249ed62ca1e4df6794ac9787accbc57ce73bf1df2c139217048d5`): class codes 1..11 on the
Xuan Thuy image grid, 0 = no label. Codes and names are the `class_map` of the configs
(Anm, Bgt, Dc, Dnn, DK, trống, Gt, Rnm, Stx, Vbn, Vcs); the names are kept as the project uses them.
The labels are historical: their reference date is not known, so they are not September 2026 ground truth.

## Verified points

One point shapefile per class (WGS 84 / UTM 48N), the file name is the class name.

| File | Points | Classes (points) | SHA-256 |
| --- | ---: | --- | --- |
| `xuanthuy/verified_points_shapefiles.zip` | 1,037 | Anm 161, Bgt 134, Dnn 240, Rnm 199, Stx 100, Vbn 103, Vcs 100 | `3159aac0e6e8001a8c83d684c66c633575cdbcc3d72a9a3b0964473d14780c21` |
| `thaithuy/verified_points_shapefiles.zip` | 616 | Anm 106, Bgt 103, Rnm 118, Stx 36, Vbn 108, Vcs 145 | `26feb2d9d90c4fdeed464b1d936bcff119086b8d69dba1df76010b20adccad18` |

Read them with `scripts/score_points.py` (or `prithvi_unet.inference.points`). Limits:

* The date of the points is not confirmed. A map that disagrees may be right about a change on the ground,
  so the score is **agreement**, not the accuracy of a dated map.
* The rare classes Dc, DK, trống and Gt have no points; the points cannot measure them.
* Only 151 of the 1,037 Xuan Thuy points lie in the validation band (columns 1122–1388); the other 886 lie
  in the training area. On the training image (21 Sep) a model has seen those pixels' labels, so the score
  is optimistic (P2: 0.938 outside the band, 0.827 inside). Score maps of a date not used in training
  (1 Oct), and never use the points to choose a checkpoint.

## Use

The label raster and the verified points are project reference data, included with the approval of the
project supervisor so that the results can be reproduced. They are not covered by the code licence
(`LICENSE`); ask the maintainers before using them for other work.
