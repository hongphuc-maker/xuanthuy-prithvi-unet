# Migration from `xuan-thuy-prithvi-segmentation`

## What is the same

* Configs: same schema (`prithvi-xtseg-v1`), same values. P2's radiometry is now marked
  confirmed (export code reviewed, SWIR water check), as P3's already was.
* Model function: identical. Old and new models with the same weights give the same
  logits (P3: max difference 0.0; P2: 7.5e-6, float32 rounding; argmax 100 % equal).
* Input crops: identical targets and coordinates; image values differ by at most 2.4e-7
  (reflectance step written separately).
* Patch sampling, batch schedule, validation windows, split: identical frames
  (`tests/test_sampling.py::test_identical_to_original_package`).
* Training procedure, selection rule, acceptance gate, map outputs.

## What changed

| Old | New |
| --- | --- |
| `EncoderDecoderFactory().build_model(...)` with necks given by name | `PrithviUNet` with each layer written in `model/` |
| `ReshapeTokensToImage` + `TargetTimeSlice` | `TokensToFeatureMap` |
| `LearnedInterpolateToPyramidal` + `replace_batch_norm` | `SimpleFeaturePyramid` with GroupNorm built in |
| TerraTorch `UNetDecoder` (duplicates the first feature for SMP) | `UNetDecoder` with explicit blocks and skips |
| `CheckpointBlock` wrapper (changes parameter names) | checkpointing inside `PrithviEncoder.forward` |
| dependency on `xuan-thuy-dmi-segmentation` | split, sampling, loss ported into this package |
| `(x + offset) * 10000 / scale`, then z-score | `dn_to_reflectance` then `standardize` |
| Protocol A/B inference manifests | `scripts/predict.py --images ...` |

Not ported: UperNet decoder option, spatial k-fold split, reference-point agreement,
plotting helpers, DMI/morphology experiments. They can be added as separate modules.

## Old checkpoints

```bash
python scripts/convert_legacy_checkpoint.py old_run/best_any.pt p3_best_any.pt --config configs/p3_single_date.json
```

or pass the old file directly to `scripts/predict.py`.

## Verification commands

```bash
pytest tests/test_legacy_equivalence.py     # needs terratorch + segmentation-models-pytorch
PYTHONPATH=<old helper repo>/src pytest tests/test_sampling.py
RUN_SLOW=1 pytest tests/test_end_to_end.py
```
