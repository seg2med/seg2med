# seg2med_train

Training and inference for the image-generation models of seg2med. Installation, data
preparation and the end-to-end steps are in the [main README](../README.md).

## Run

```bash
python -m seg2med_train.train --config <your config>.yaml
```

The YAML key `mode` selects what runs (`train`, `test`, `analyse`, `warmup`). Example configs:
[configs/test_ddpm_ct256.yaml](../configs/test_ddpm_ct256.yaml),
[configs/test_ddpm_mr256.yaml](../configs/test_ddpm_mr256.yaml) and the 512 configs in
[configs/workflow/](../configs/workflow/).

## Where things are

- `model_name` is looked up in `model_selection.MODEL_RUNNERS`: `ddpm2d_seg2med`,
  `ddpm2d_seg2med_multimodal`, `AttentionUnet`, `pix2pix`, `cycle_gan`.
- `dataset_name` is looked up in `dataloader_registry` (for example `image_prior_csv` and
  `semantic_prior_csv`, defined in `Dataloader.py`).
- All config keys and their defaults are in `utils/my_configs_yacs.py`; a YAML only needs
  the keys it changes.
- To add your own model, loader or config keys from another package, see "Extending seg2med"
  in the main README.

## Common dataset keys

- `dataset.normalize`: normalization method (`zscore`, `minmax`, ...).
- `dataset.pad`: padding mode used when resizing.
- `dataset.rotate`: rotate by 90 degrees during the transform.
- `dataset.train_number`, `dataset.val_number`: number of patients for training and for validation/test.
- `dataset.input_dim`: 2.0 for 2D, 3.0 for 3D, 3.5 for 3D patches.
- `dataset.resized_size`: volume shape after resizing.
- `dataset.patch_size`: patch shape; `(None, None, 1)` for 2D training.
- `dataset.zoom`: fixed zoom factor, used for low-resolution training.
- `dataset.tissue_min`, `tissue_max`, `bone_min`, `bone_max`: intensity thresholds used when a
  mask is built from the image.
