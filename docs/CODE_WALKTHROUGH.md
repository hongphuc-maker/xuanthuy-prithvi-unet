# Code walkthrough

One training step and one prediction, in the order the code runs.

## 1. Config — `src/prithvi_unet/config.py`

`load_config` reads `configs/p3_single_date.json` and refuses inconsistent settings:
band order must be the pretrained order, the target date must be the last active date
(the model reads out the last date), four readouts and four decoder blocks.

## 2. Scene — `src/prithvi_unet/data/scene.py`

`TrainingScene` reads the active-date GeoTIFFs (checksum, band names, date tag, grid),
the other listed images (validity only) and the label raster. It builds:

* `image_valid`: valid on every listed image;
* the split (`data/splits.py`): validation columns 1122–1388, 48-pixel guards;
* `train_class_counts` for the class weights.

`crop(row, col)` cuts a 224 × 224 window, converts DN with `data/radiometry.py`,
zeroes invalid pixels, pads edge crops, and computes the crop centre in latitude /
longitude for the location embedding.

## 3. Patches and batches — `src/prithvi_unet/data/sampling.py`

`sample_training_patches` picks 4,992 crops entirely inside the training domain:
416 for each rare class (codes 5, 6, 7), the rest uniformly. `build_batch_schedule`
groups them into 312 batches of 16, each with one anchor per rare class and every class
present. `build_validation_windows` covers the validation band with stride 112.
Written to `run/prepared/` with checksums; reused on resume.

## 4. Model — `src/prithvi_unet/model/`

`build_model(config)` creates `PrithviUNet` and loads the pinned Prithvi weights
(`encoder.load_pretrained`). One forward call (`prithvi_unet.py: forward_trace`):

1. `encoder.embed`: Conv3d patch embedding, + position, + date, + location, + CLS.
2. `encoder.forward`: 24 blocks, keep blocks 5, 11, 17, 23.
3. `TokensToFeatureMap`: drop CLS, tokens back to 14 × 14, keep the target date.
4. `SimpleFeaturePyramid`: 56², 28², 14², 7².
5. `UNetDecoder`: 7 → 14 → 28 → 56 with skips, then one block without skip.
6. `SegmentationHead`: Dropout + Conv 1×1 → 11 class scores at 56².
7. `F.interpolate(..., mode="bilinear")` → 224².

## 5. Loss and step — `src/prithvi_unet/training/`

`loss.py`: class weights `sqrt(median / count)`, clipped to [0.5, 5], mean 1; weighted
cross-entropy ignoring label 255.

`trainer.py: Trainer.train` for each step:

1. take the batch of 16 for this step (order reshuffled each pass);
2. run 16 micro-batches of one crop (GPU memory), each loss scaled by its share of the
   batch's class-weight mass, `backward()` accumulates gradients;
3. clip the gradient norm to 1, AdamW step (encoder lr 1e-5, new layers 1e-4);
4. every 156 steps validate (`inference/maps.py: validate`), save `best_any.pt` and
   `best_accepted.pt`, early-stop after 6 validations without +0.001 macro-F1;
5. save `latest_resume.pt` (model, optimiser, scaler, random state).

## 6. Prediction — `src/prithvi_unet/inference/`

`tiling.py`: windows of 224 with stride 112 over the scene; softmax of every window
added up per pixel and divided by the number of windows; `maps.py`: argmax → classes
1..11, max softmax, normalised entropy, written as GeoTIFFs with provenance tags.
`predict.py` does the same for new dates without labels and accepts old checkpoints.

## Where each operation comes from

| Code | Origin |
| --- | --- |
| Conv3d patch embedding, position/date/location encoders, 24 blocks | Prithvi-EO-2.0 (TerraTorch `prithvi_mae.PrithviViT`, timm `Block`), forward re-stated in `encoder.py` |
| Readout of blocks 5/11/17/23 | config; equal quarters of the encoder |
| Tokens → map, target-date slice | TerraTorch `ReshapeTokensToImage` + project `TargetTimeSlice` |
| Feature pyramid | ViTDet simple feature pyramid (TerraTorch `LearnedInterpolateToPyramidal`) |
| GroupNorm instead of BatchNorm | project (`replace_batch_norm` in the completed runs) |
| U-Net decoder blocks | segmentation_models_pytorch 0.5.0 `UnetDecoder` via TerraTorch `UNetDecoder` |
| Head, bilinear resize | TerraTorch `SegmentationHead`, `PixelWiseModel(rescale=True)` |
| Sampling, split, WCE | project (`xuan-thuy-dmi-segmentation` @ 0775770) |
