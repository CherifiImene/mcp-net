# API Reference

Every class and function in the `mcpnet` package and `scripts/run_ablation.py`, grouped by
module. For the *why* behind non-obvious choices (bug fixes, methodological caveats), see the
docstring in the source file itself — this reference focuses on *what each thing takes and
returns* so you can use the package without reading every file end to end.

## Contents

- [`mcpnet.model.architecture`](#mcpnetmodelarchitecture) — MCP-Net and its ablation variants
- [`mcpnet.model.layers`](#mcpnetmodellayers) — shared conv/upsampling building blocks
- [`mcpnet.data.data_loader`](#mcpnetdatadata_loader) — `DataLoader`
- [`mcpnet.data.augmentation`](#mcpnetdataaugmentation) — `Augment`
- [`mcpnet.data.preprocessing`](#mcpnetdatapreprocessing) — `standardize`
- [`mcpnet.data.acdc_utilities_stub`](#mcpnetdataacdc_utilities_stub) — fallback NIfTI I/O
- [`mcpnet.training.train`](#mcpnettrainingtrain) — `train_variant`
- [`mcpnet.training.losses`](#mcpnettraininglosses) — losses and Dice metrics
- [`mcpnet.training.callbacks`](#mcpnettrainingcallbacks) — `SnapshotEnsemble`
- [`mcpnet.inference.ensemble_predict`](#mcpnetinferenceensemble_predict) — single/ensemble inference
- [`mcpnet.evaluation.metrics`](#mcpnetevaluationmetrics) — per-case Dice/HD95
- [`mcpnet.evaluation.statistical_tests`](#mcpnetevaluationstatistical_tests) — Friedman/Wilcoxon
- [`mcpnet.utils.seeding`](#mcpnetutilsseeding) — `set_global_seed`
- [`scripts.run_ablation`](#scriptsrun_ablation) — CLI entry point / table runners

---

## `mcpnet.model.architecture`

Builds MCP-Net and every ablation variant. All builders share the signature
`(input_shape=(128, 128, 1), num_classes=4, name=<variant>)` and return a compiled-free
`tf.keras.Model` (call `.compile()` yourself, or use [`train_variant`](#mcpnettrainingtrain)).

### `input_pyramid(inputs, nb_scales)`

Builds the multi-resolution input pyramid by repeated 2×2 max-pooling.

| Param | Type | Description |
|---|---|---|
| `inputs` | `KerasTensor` | The raw input image tensor. |
| `nb_scales` | `int` | Number of pyramid levels to produce (including the original resolution). |

**Returns** `list[KerasTensor]`, length `nb_scales`, ordered finest → coarsest:
`[I, pool(I), pool²(I), ..., pool^(nb_scales-1)(I)]`.

### `down_pyramid(inputs, nb_scales, filters, kernels, nb_pyramid, groups=1, connections=None)`

One encoder depth: applies a conv block to each remaining scale, optionally fusing in a
"connection" tensor forwarded from the previous depth's finest-scale output.

| Param | Type | Description |
|---|---|---|
| `inputs` | `list[KerasTensor]` | Per-scale tensors for this depth (finest → coarsest), length `nb_scales`. |
| `nb_scales` | `int` | Number of scales processed at this depth. |
| `filters` | `int` | Conv filter count applied to every scale at this depth. |
| `kernels` | `int` | Conv kernel size applied to every scale at this depth. |
| `nb_pyramid` | `int` | Depth index, used only to make Keras layer names unique (`downsample_{i+1}_p{nb_pyramid}`, etc.). |
| `groups` | `int`, default `1` | Passed through to `Conv2D(groups=...)`. |
| `connections` | `KerasTensor` or `None`, default `None` | The previous depth's finest-scale output (`D_{d-1}[0]`). When `None`, no cross-scale reinjection happens — this is what gives `mcp_net_no_connections` and `vanilla_unet` their "connections off" behaviour for free. When set, it is max-pooled and concatenated onto scale `i`'s input, once per iteration, so it reaches each successive scale at progressively lower resolution. |

**Returns** `list[KerasTensor]`, length `nb_scales` — one conv output per scale, finest → coarsest.
`convs[0]` is always the tensor threaded forward as `connections` to the next depth.

### `up_pyramid(inputs, connections, nb_scales, filters, kernels, nb_pyramid, groups=1)`

One decoder depth: upsample, concatenate with the matching-resolution skip connection, then
two conv blocks.

| Param | Type | Description |
|---|---|---|
| `inputs` | `KerasTensor` | Tensor from the previous decoder stage (or the bottleneck on the first call). |
| `connections` | `list[KerasTensor]` | Skip tensors, ordered coarsest → finest resolution, e.g. `[p4[0], p3[0], p2[0], p1[0]]`. Must have `nb_scales` entries. |
| `nb_scales` | `int` | Number of decoder stages to run (one per entry in `connections`). |
| `filters` | `list[int]` | Filter count per stage, length `nb_scales`. |
| `kernels` | `list[int]` | Kernel size per stage, length `nb_scales`. |
| `nb_pyramid` | `int` | Used only for unique Keras layer names. |
| `groups` | `int`, default `1` | Passed through to `Conv2D(groups=...)`. |

Raises `AssertionError` if `len(connections)`, `len(filters)`, or `len(kernels)` don't match
`nb_scales`.

**Returns** `list[KerasTensor]`, length `nb_scales` — decoder output at each stage, coarsest →
finest. `convs[-1]` is the final, full-resolution decoder output (feed this into the 1×1 output
conv).

### `up_pyramid_additive(inputs, connections, nb_scales, filters, kernels, nb_pyramid, groups=1)`

Same signature and return shape as `up_pyramid`, but fuses each skip connection with `Add`
instead of `Concatenate` (projecting the skip through a 1×1 conv first if its channel count
doesn't already match). Used only by `build_full_mcp_net_additive_decoder` (Table D).

### `build_full_mcp_net(input_shape=(128, 128, 1), num_classes=4, name="full_mcp_net")`

The original submitted model: 5-depth multi-scale pyramid encoder with cross-scale
connections, concatenation-based decoder.

| Param | Type | Description |
|---|---|---|
| `input_shape` | `tuple[int, int, int]`, default `(128, 128, 1)` | Model input shape `(H, W, C)`. |
| `num_classes` | `int`, default `4` | Number of output segmentation classes (softmax channels). |
| `name` | `str`, default `"full_mcp_net"` | Keras model name. |

**Returns** `tf.keras.Model`, output shape `(H, W, num_classes)`, softmax-activated.

### `build_mcp_net_no_connections(input_shape=(128, 128, 1), num_classes=4, name="mcp_net_no_connections")`

Same encoder/decoder shape as `build_full_mcp_net`, but `connections` is never threaded through
`down_pyramid` — isolates the effect of the connections mechanism. Same params/returns as above.

### `build_vanilla_unet(input_shape=(128, 128, 1), num_classes=4, name="vanilla_unet")`

Single-path U-Net anchor: no multi-scale pyramid, no connections, standard skip connections
only. Matches MCP-Net's depth (5) and filter progression (`16, 32, 64, 64, 64`) for a
parameter-count-adjacent baseline. Same params/returns as above.

### `build_single_path_with_connections(input_shape=(128, 128, 1), num_classes=4, name="single_path_with_connections")`

Single-scale encoder (`nb_scales=1` at every depth) that still threads the connections
mechanism through — completes the pyramid × connections 2×2 factorial (Table C) together with
the three variants above. Same params/returns as above.

### `build_full_mcp_net_additive_decoder(input_shape=(128, 128, 1), num_classes=4, name="full_mcp_net_additive_decoder")`

Identical encoder to `build_full_mcp_net`; decoder uses `up_pyramid_additive` (Add-fusion)
instead of `up_pyramid` (Concatenate-fusion). Used for Table D (decoder fusion ablation).
Same params/returns as above.

### `get_model(variant_name, input_shape=(128, 128, 1), num_classes=4)`

Factory used by the ablation runner and training code — looks `variant_name` up in
`VARIANT_BUILDERS` and calls the matching builder.

| Param | Type | Description |
|---|---|---|
| `variant_name` | `str` | One of `"full_mcp_net"`, `"mcp_net_no_connections"`, `"vanilla_unet"`, `"single_path_with_connections"`, `"full_mcp_net_additive_decoder"`. |
| `input_shape` | `tuple[int, int, int]`, default `(128, 128, 1)` | Passed through to the builder. |
| `num_classes` | `int`, default `4` | Passed through to the builder. |

Raises `ValueError` if `variant_name` isn't a key in `VARIANT_BUILDERS`.

**Returns** `tf.keras.Model`.

### Module-level constants

- `DEFAULT_FILTERS = [16, 32, 64, 64, 64]` — per-depth filter counts (`vanilla_unet`'s encoder).
- `DEFAULT_KERNELS = [3, 3, 3, 3, 3]` — per-depth kernel sizes.
- `VARIANT_BUILDERS` — `dict[str, callable]` mapping variant name → builder function, used by `get_model`.

---

## `mcpnet.model.layers`

### `ActiveConv2D(inputs, filters, kernel, initializer="he_normal", name=None, dilation=1, activation="relu", groups=1, batch_norm=True, activation_first=False)`

The core conv block used everywhere in the model: `Conv2D → Activation → BatchNorm` by default.

| Param | Type | Description |
|---|---|---|
| `inputs` | `KerasTensor` | Input tensor. |
| `filters` | `int` | Number of output filters. |
| `kernel` | `int` | Conv kernel size. |
| `initializer` | `str`, default `"he_normal"` | Passed to `Conv2D(kernel_initializer=...)`. |
| `name` | `str` or `None`, default `None` | Layer name for the `Conv2D` op. |
| `dilation` | `int`, default `1` | Passed to `Conv2D(dilation_rate=...)`. |
| `activation` | `str` or `None`, default `"relu"` | Activation applied after (or before, see `activation_first`) the conv. `None`/falsy skips activation entirely. |
| `groups` | `int`, default `1` | Passed to `Conv2D(groups=...)` (grouped convolution). |
| `batch_norm` | `bool`, default `True` | Whether to apply `BatchNormalization` after activation. Set `False` for the final 1×1 output conv (see the model builders). |
| `activation_first` | `bool`, default `False` | If `True`, applies activation *before* the conv instead of after. Raises `AssertionError` if `True` with no `activation` given. |

**Returns** `KerasTensor`.

### `UpSamplingBlock(inputs, name, kernel=2, size=2, interpolation="nearest", filters="same")`

Nearest-neighbor upsample followed by one `ActiveConv2D`.

| Param | Type | Description |
|---|---|---|
| `inputs` | `KerasTensor` | Input tensor to upsample. |
| `name` | `str` | Base name, used to build the `upsampling_{name}` / `up_conv1_{name}` layer names. |
| `kernel` | `int`, default `2` | Kernel size for the post-upsample conv. |
| `size` | `int`, default `2` | Upsampling factor, passed to `UpSampling2D(size=...)`. |
| `interpolation` | `str`, default `"nearest"` | Passed to `UpSampling2D(interpolation=...)`. |
| `filters` | `int` or `"same"`, default `"same"` | Output filter count. `"same"` (default) keeps the input's channel count; pass an `int` to change it. Raises `AssertionError` if given something other than `"same"` or an `int`. |

**Returns** `KerasTensor`.

---

## `mcpnet.data.data_loader`

### `class DataLoader`

Loads ROI-cropped ACDC NIfTI volumes, splits them into train/val/test by patient, builds
augmented `tf.data` pipelines for train/val, and keeps a held-out, un-batched test set as numpy
arrays for per-case evaluation.

#### `DataLoader.__init__(self, data_path, gt_path, configs_path, buffer_size, train_batch_size, val_batch_size, augment=True, to_train=True, adjust_percents=True, classify=False, seed=42)`

| Param | Type | Description |
|---|---|---|
| `data_path` | `str` | Directory of input NIfTI images (trailing slash required — the loader does `data_path + filename`). |
| `gt_path` | `str` | Directory of ground-truth label NIfTI images. |
| `configs_path` | `str` | Path to a JSON file: `{"train": [...], "dev": [...], "test": [...]}` of filename stems (no extension), used to assign each file to a split. |
| `buffer_size` | `int` | `tf.data.Dataset.shuffle` buffer size for the training pipeline. |
| `train_batch_size` | `int` | Training batch size. |
| `val_batch_size` | `int` | Validation batch size. |
| `augment` | `bool`, default `True` | Whether the training pipeline applies `Augment.augmentation_pipeline`. |
| `to_train` | `bool`, default `True` | If `False`, the training pipeline is batched without shuffling/repeating (inference-style). |
| `adjust_percents` | `bool`, default `True` | If `True`, runs `adjust_labels_percent` over the training set to oversample small-foreground slices (see below). |
| `classify` | `bool`, default `False` | If `True`, loads classification labels from a `configs.csv` instead of segmentation masks (legacy path; not used by the ablation study). |
| `seed` | `int`, default `42` | Seeds Python/numpy/TF RNGs via `set_global_seed` (so `.shuffle(..., seed=seed)` and augmentation are reproducible) and is stored on `self.seed`. |

**Instance attributes set during `__init__`:**

- `self.X_train, self.Y_train, self.X_val, self.Y_val` — numpy arrays.
- `self.X_test, self.Y_test` — numpy arrays, held-out test set (empty arrays if no test files found).
- `self.train_dataset, self.val_dataset` — unbatched `tf.data.Dataset`s.
- `self.train_batches, self.val_batches` — batched/augmented `tf.data.Dataset`s, ready for `model.fit`.
- `self.train_step, self.val_step` — batches per epoch (ceil-divided), for `steps_per_epoch`/`validation_steps`.
- `self.test_voxel_spacing` — `dict[str, tuple]` mapping test filename stem → `(x_mm, y_mm, z_mm)` from the NIfTI header.

#### `DataLoader.load_dataset(self, data_path, labels_path, configs_path, classify=False)`

Walks `data_path`, loads and standardizes every NIfTI file, and assigns each file's slices to
train/val/test based on `configs_path`. **`dev` and `test` splits are kept fully separate**
(train/val/test), unlike the original code which merged them.

| Param | Type | Description |
|---|---|---|
| `data_path` | `str` | Image directory. |
| `labels_path` | `str` | Label directory. |
| `configs_path` | `str` | Path to the train/dev/test split JSON. |
| `classify` | `bool`, default `False` | See `__init__`. |

**Returns** `((X_train, Y_train), (X_val, Y_val))` — lists of per-slice numpy arrays. Test data is
stored on `self.X_test_raw` / `self.Y_test_raw` (consumed by `__init__`), not returned directly.

#### `DataLoader.adjust_labels_percent(self, training_data, training_gt)`

Runs `_adjust_label_percent` over every training slice and collects the crop-and-resize
augmented copies it produces.

| Param | Type | Description |
|---|---|---|
| `training_data` | `list[np.ndarray]` | Training images. |
| `training_gt` | `list[np.ndarray]` | Training ground-truth masks (one-hot). |

**Returns** `(adj_imgs, adj_gt)` — lists of additional (cropped+resized) training examples to
append to the originals.

#### `DataLoader.numpy(self, *args)`

Generator: converts each positional `list` argument to a `np.ndarray(dtype=np.float32)`.

| Param | Type | Description |
|---|---|---|
| `*args` | `list` | Any number of lists to convert. |

**Yields** one `np.ndarray` per input list, in order.

#### `DataLoader.get_dataset_stats(self, *datasets)`

Computes and writes the mean/std of the concatenation of the given datasets to
`../configurations/stats.txt`.

| Param | Type | Description |
|---|---|---|
| `*datasets` | `np.ndarray` | Arrays to concatenate along axis 0 before computing statistics. |

**Returns** `(mean, stdev)`.

#### `DataLoader.get_tf_dataset(self, *args)`

Generator: wraps each `(X, y)` tuple argument as a `tf.data.Dataset.from_tensor_slices`.

| Param | Type | Description |
|---|---|---|
| `*args` | `tuple[np.ndarray, np.ndarray]` | Any number of `(X, y)` pairs. |

**Yields** one `tf.data.Dataset` per input pair, in order.

#### `DataLoader.standardize_datasets(self)`

Applies `self._preprocess` (channel standardization using `self.mean`/`self.stdev`) to
`self.train_dataset` and `self.val_dataset` in place. Raises `Exception` if `self.mean`/
`self.stdev` haven't been set (via `get_dataset_stats`) first. Not called automatically —
opt-in, legacy path.

#### `DataLoader.batch_data(self, train_dataset, val_dataset, augment=True, to_train=True)`

Builds the final batched/prefetched `tf.data.Dataset`s used for training.

| Param | Type | Description |
|---|---|---|
| `train_dataset` | `tf.data.Dataset` | Unbatched training dataset. |
| `val_dataset` | `tf.data.Dataset` | Unbatched validation dataset. |
| `augment` | `bool`, default `True` | Applies `Augment.augmentation_pipeline` before batching if `True` and `to_train=True`. |
| `to_train` | `bool`, default `True` | If `False`, returns a plain (unshuffled, non-repeating) batched training dataset — used for inference/evaluation passes. |

**Returns** `(train_batches, val_batches)`.

---

## `mcpnet.data.augmentation`

### `class Augment`

Namespace of augmentation functions (called as `Augment.method_name(...)`, not instantiated).
Only `augmentation_pipeline` is on the active training path (wired into
`DataLoader.batch_data`); the rest are unused by the current pipeline but kept available.

| Function | Description |
|---|---|
| `crop_and_resize(image, bbx, size)` | Crops `image` to bounding box `bbx = [y1, x1, y2, x2]` and resizes to `size`. Used by `DataLoader._adjust_label_percent`. |
| `crop_and_resize_v2(image, mask)` | Probabilistically (30% chance, only when foreground is <7% of pixels) crops around the foreground region and resizes to 128×128. Not called by the active pipeline. |
| `random_mirroring(image, mask, mode="horizontal_and_vertical")` | Applies `RandomFlip` identically to image and mask (shared seed). |
| `random_rotation(image, mask, ratio=0.5)` | Applies `RandomRotation(0.5)` identically to image and mask (`ratio` is accepted but not actually used — the factor is hardcoded to `0.5`). |
| `random_translation(image, mask, ty, tx)` | Applies `RandomTranslation(ty, tx)` identically to image and mask. |
| `random_zoom(image, mask, w_ratio=0.02, h_ratio=0.02)` | Applies `RandomZoom(w_ratio, h_ratio)` identically to image and mask. |
| `random_affine_transformation(image, mask, tx=0.05, ty=0.05, r_ratio=0.75, zw_ratio=0.02, zh_ratio=0.02)` | Chains a random subset/order of translate/rotate/zoom (only when foreground ≤25% of pixels). |
| `augmentation_pipeline(image, mask, r_ratio=0.75)` | **The one actually used.** Runs an `albumentations.Compose` pipeline (`RandomRotate90`, `HorizontalFlip`, `VerticalFlip`, `Transpose`, `ShiftScaleRotate`, one of `GridDropout`/`Emboss`/`Affine`) via `tf.numpy_function`, applied identically to image and mask. `r_ratio` is accepted but currently unused inside the function body. |

All functions except `augmentation_pipeline` take `(image, mask)` (or `(image, bbx, size)` for
`crop_and_resize`) and return the transformed `(image, mask)` pair as TF tensors.

---

## `mcpnet.data.preprocessing`

### `standardize(image)`

Per-channel (z-dimension) mean/std standardization.

| Param | Type | Description |
|---|---|---|
| `image` | `np.ndarray`, shape `(H, W, D)` | Input volume. |

For each channel `c` along the last axis: subtracts the *global* image mean (not the per-channel
mean) from that channel, then divides by that channel's own std (skipped if std is 0).

**Returns** `np.ndarray`, same shape as `image`.

---

## `mcpnet.data.acdc_utilities_stub`

Fallback implementation of `acdc_utilities.load_nii`/`save_nii`, used automatically by
`data_loader.py` and `run_ablation.py` when the project's real `acdc_utilities.py` isn't on the
Python path. This is the ACDC challenge organizers' own `metrics.py` script (Clément Zotti,
2017) — bundled here for its NIfTI I/O helpers. **It does no resizing**, so if your real images
aren't already 128×128 on disk, you need the real module.

| Function | Description |
|---|---|
| `load_nii(img_path)` | Loads a `.nii`/`.nii.gz` file. Returns `(data, affine, header)` — `data` as `nimg.get_fdata()`, plus the affine matrix and header (used elsewhere for `header.get_zooms()`, i.e. voxel spacing). |
| `save_nii(img_path, data, affine, header)` | Saves `data` as a `.nii`/`.nii.gz` file at `img_path`, using the given affine and header. |
| `metrics(img_gt, img_pred, voxel_size)` | Computes `[Dice, Volume, VolumeErr]` for LV/RV/Myo between two label maps. Not used by `mcpnet.evaluation` (which uses `medpy` directly instead) — kept for parity with the organizers' original script. |
| `compute_metrics_on_files(path_gt, path_pred)` | CLI helper: prints `metrics()` output for one file pair. Not used by the ablation pipeline. |
| `compute_metrics_on_directories(dir_gt, dir_pred)` | CLI helper: writes a CSV of `metrics()` output for every file pair in two directories. Not used by the ablation pipeline. |
| `natural_order(sord)` / `conv_int(i)` | Sorting helpers so `"patient2" < "patient10"` instead of lexicographic order. |

This file is also runnable standalone as a CLI (`python acdc_utilities_stub.py GT_IMG PRED_IMG`),
mirroring the original ACDC organizer script — unrelated to the ablation pipeline, which imports
only `load_nii`.

---

## `mcpnet.training.train`

### `train_variant(variant_name, train_batches, val_batches, train_steps, val_steps, save_dir, input_shape=(128, 128, 1), num_classes=4, use_snapshot_ensemble=False, epochs=300, nb_cycles=10, max_lr=0.01, fixed_lr=1e-3, seed=42, run_name=None)`

Builds, compiles, and trains one architecture variant. Used for both Table A/C/D (fixed LR,
`use_snapshot_ensemble=False`) and Table B (cyclic LR, `use_snapshot_ensemble=True`).

| Param | Type | Description |
|---|---|---|
| `variant_name` | `str` | Must be a key in `mcpnet.model.architecture.VARIANT_BUILDERS` (e.g. `"full_mcp_net"`) — passed to `get_model()` to build the actual architecture. |
| `train_batches` | `tf.data.Dataset` | Batched, augmented training data (e.g. `data_loader.train_batches`). |
| `val_batches` | `tf.data.Dataset` | Batched validation data (e.g. `data_loader.val_batches`). |
| `train_steps` | `int` | Steps per epoch (e.g. `data_loader.train_step`). |
| `val_steps` | `int` | Validation steps (e.g. `data_loader.val_step`). |
| `save_dir` | `str` | Directory for checkpoints and completion markers. |
| `input_shape` | `tuple[int, int, int]`, default `(128, 128, 1)` | Passed to `get_model`. |
| `num_classes` | `int`, default `4` | Passed to `get_model`. |
| `use_snapshot_ensemble` | `bool`, default `False` | If `True`, trains with a cyclic-cosine LR schedule and periodic snapshotting (`SnapshotEnsemble` callback, peak LR = `max_lr`). If `False`, trains with a constant `fixed_lr` and a `ModelCheckpoint(save_best_only=True)` on the foreground Dice metric. |
| `epochs` | `int`, default `300` | Total training epochs. |
| `nb_cycles` | `int`, default `10` | Number of cosine-annealing cycles (only used if `use_snapshot_ensemble=True`). |
| `max_lr` | `float`, default `0.01` | Peak LR for the cyclic schedule (only used if `use_snapshot_ensemble=True`). |
| `fixed_lr` | `float`, default `1e-3` | Constant LR (only used if `use_snapshot_ensemble=False`). |
| `seed` | `int`, default `42` | Seeds weight initialization via `set_global_seed`. Use the same seed across variants for a fair architecture comparison. |
| `run_name` | `str` or `None`, default `None` | Checkpoint/snapshot filename label; defaults to `variant_name`. Set this to train the *same* architecture more than once under different labels (e.g. Table B's fixed-LR anchor row) without overwriting another run's checkpoint. |

**Returns** `(model, history, snapshot_paths)`:
- `model` — the trained `tf.keras.Model`.
- `history` — the `History` object from `model.fit`.
- `snapshot_paths` — `list[str]` of saved snapshot `.h5` paths (empty unless `use_snapshot_ensemble=True`).

**Side effect:** writes `{save_dir}/{run_name}_best.h5.complete` once `model.fit` finishes all
requested epochs — used by `scripts/run_ablation.py` to decide whether an on-disk checkpoint is
safe to reuse instead of retraining.

---

## `mcpnet.training.losses`

### `dc_per_class(y_pred, y_true, epsilon=0.00001)`

Dice coefficient between two single-class probability/label maps.

| Param | Type | Description |
|---|---|---|
| `y_pred` | tensor | Predicted values for one class, any shape. |
| `y_true` | tensor | Ground-truth values for one class, same shape as `y_pred`. |
| `epsilon` | `float`, default `0.00001` | Smoothing term to avoid division by zero. |

**Returns** scalar tensor, the Dice coefficient.

### `DiceIndex(y_pred, y_true, axis=(0, 1, 2), epsilon=0.00001)`

Unweighted mean Dice across all 4 classes **including background**. `axis` is accepted but
unused. Kept for exact reproducibility of the original training runs (this was the original
`ModelCheckpoint` monitor metric, before the `DiceIndexForeground` fix below).

**Returns** scalar tensor.

### `DiceIndexForeground(y_pred, y_true, epsilon=0.00001)`

Mean Dice over RV/Myo/LV only, excluding the (trivially easy, majority-pixel) background class.
Recommended checkpoint-selection metric going forward.

**Returns** scalar tensor.

### `class_dice_loss(y_pred, y_true, epsilon=0.000001)`

Soft Dice loss (`1 - Dice`) for one class, using squared terms in the denominator (as opposed
to `dc_per_class`'s unsquared sums).

**Returns** scalar tensor.

### `diceLoss(y_pred, y_true, epsilon=0.000001)`

The weighted multi-class Dice loss actually used for training: `0.36·RV + 0.34·Myo + 0.29·LV +
0.01·background`, each term from `class_dice_loss`. Weights are fixed per the ACDC challenge
organizers' recommendation, not configurable.

**Returns** scalar tensor.

### `loss(y_true, y_pred)`

The function actually passed to `model.compile(loss=...)`. Thin wrapper: `diceLoss(y_pred,
y_true)` — note the swapped argument order relative to `diceLoss`'s own signature; harmless
because Dice is symmetric in its two inputs.

### `TverskyIndex(y_true, y_pred)` / `TverskyLoss(targets, inputs, alpha=ALPHA, beta=BETA, smooth=1e-6)` / `FocalTverskyLoss(targets, inputs, alpha=ALPHA, beta=BETA, gamma=GAMMA, smooth=1e-6)`

Tversky/Focal-Tversky index and losses, carried over from the original `metrics.py` but **not
used** by the current ablation study (no config wires them in). Available if you want to add a
loss-function axis to the ablation later. `ALPHA=0.5`, `BETA=0.5`, `GAMMA=1` are module-level
defaults.

---

## `mcpnet.training.callbacks`

### `class SnapshotEnsemble(tf.keras.callbacks.Callback)`

Cyclic cosine-annealing LR schedule with periodic snapshotting, for Snapshot Ensembling
(Huang et al., 2017).

#### `SnapshotEnsemble.__init__(self, max_lr, total_epochs, batches_per_epoch, nb_cycles, save_dir, run_name="snapshot")`

| Param | Type | Description |
|---|---|---|
| `max_lr` | `float` | Peak learning rate at the start of every cycle. |
| `total_epochs` | `int` | Total training epochs across all cycles. |
| `batches_per_epoch` | `int` | Accepted but currently unused — the LR updates once per epoch, not per batch (a coarser schedule than the original Snapshot Ensembles paper). Kept for a possible future per-batch version. |
| `nb_cycles` | `int` | Number of cosine-annealing cycles; `total_epochs / nb_cycles` epochs per cycle. |
| `save_dir` | `str` | Directory to save snapshot `.h5` files into. |
| `run_name` | `str`, default `"snapshot"` | Filename prefix: `{save_dir}/{run_name}_cycle_{m}.h5`. |

#### `SnapshotEnsemble.cosine_annealing(self, max_lr, epoch, total_epochs, nb_cycles)`

Computes the LR for the given epoch: `(max_lr/2) * (cos(π · (epoch % epoch_per_cycle) /
epoch_per_cycle) + 1)`. Epoch 0 of every cycle trains at `max_lr`; LR decays to ~0 by the end
of each cycle.

#### `SnapshotEnsemble.on_epoch_begin(self, epoch, logs=None)`

Keras callback hook — sets the optimizer's LR via `cosine_annealing` at the start of each
epoch (handles both Keras 2's `optimizer.lr` and Keras 3's `optimizer.learning_rate`).

#### `SnapshotEnsemble.on_epoch_end(self, epoch, logs=None)`

Keras callback hook — at the end of each cycle (`(epoch+1) % epoch_per_cycle == 0`), saves the
current model to `{save_dir}/{run_name}_cycle_{m}.h5` and appends the path to
`self.snapshot_paths`.

**Instance attributes:** `self.lrates` (list of every LR actually set, in epoch order),
`self.snapshot_paths` (list of saved snapshot file paths, in cycle order — used by
`run_ablation.py`'s Table B).

---

## `mcpnet.inference.ensemble_predict`

### `load_snapshot_members(snapshot_paths, custom_objects)`

Loads a list of saved snapshot checkpoints as Keras models.

| Param | Type | Description |
|---|---|---|
| `snapshot_paths` | `list[str]` | Paths to `.h5` snapshot files (e.g. `SnapshotEnsemble.snapshot_paths`, or a subset of it). |
| `custom_objects` | `dict` | Must include the custom loss and metric functions (`loss`, `DiceIndex`, `DiceIndexForeground`) so Keras can deserialize the `.h5` files. |

**Returns** `list[tf.keras.Model]`.

### `predict_single(model, x)`

Single-model prediction (the `M=1` case).

| Param | Type | Description |
|---|---|---|
| `model` | `tf.keras.Model` | A trained model. |
| `x` | `np.ndarray`, shape `(batch, H, W, C)` | Input batch. |

**Returns** `(labels, probs)` — `labels`: `np.ndarray` shape `(batch, H, W)`, hard argmax class
per pixel. `probs`: `np.ndarray` shape `(batch, H, W, num_classes)`, raw softmax output.

### `ensemble_predict(members, x, return_probs=False)`

Soft-vote ensemble prediction across multiple snapshots (the `M>1` case).

| Param | Type | Description |
|---|---|---|
| `members` | `list[tf.keras.Model]` | Loaded models to ensemble — any subset of snapshots (this is what lets Table B sweep `M = 1, 3, 5, 10` from one training run without retraining). |
| `x` | `np.ndarray`, shape `(batch, H, W, C)` | Input batch. |
| `return_probs` | `bool`, default `False` | If `True`, also returns the averaged probability map. |

**Returns** `labels` (`np.ndarray`, shape `(batch, H, W)`) if `return_probs=False`; otherwise
`(labels, merged_probs)` where `merged_probs` has shape `(batch, H, W, num_classes)` and is the
mean (not sum) of each member's softmax output.

---

## `mcpnet.evaluation.metrics`

### `per_case_metrics(y_true, y_pred, case_id, variant_name, voxelspacing=None)`

Computes Dice and HD95 for RV/Myo/LV for a single case (slice or reconstructed volume).

| Param | Type | Description |
|---|---|---|
| `y_true` | `np.ndarray`, shape `(H, W)` or `(H, W, S)` | Ground-truth integer label map. |
| `y_pred` | `np.ndarray`, same shape as `y_true` | Predicted integer label map. |
| `case_id` | `str` | Identifier for this case (e.g. patient/volume filename), stored in the output row. |
| `variant_name` | `str` | Model/variant label, stored in the output row (e.g. `"full_mcp_net"`, `"M3_cyclic"`). |
| `voxelspacing` | `tuple` or `None`, default `None` | Real-world voxel spacing in mm, matching `y_true`'s axis order (e.g. `(z_mm, x_mm, y_mm)` for a `(S, H, W)` volume). If `None`, HD95 is computed in **pixel units** and a one-time warning is printed — almost never what you want to report. |

**Returns** `dict` with keys: `case_id`, `variant`, `dice_RV`, `hd95_RV`, `dice_Myo`, `hd95_Myo`,
`dice_LV`, `hd95_LV`, `mean_dice`, `voxelspacing_used`. Dice/HD95 are `NaN` when both
ground-truth and prediction are empty for that class; Dice is `0.0` (not NaN) when only the
prediction is empty.

### `evaluate_dataset(y_true_list, y_pred_list, case_ids, variant_name, voxelspacings=None)`

Runs `per_case_metrics` over a whole dataset.

| Param | Type | Description |
|---|---|---|
| `y_true_list` | `list[np.ndarray]` | Ground-truth label maps, one per case. |
| `y_pred_list` | `list[np.ndarray]` | Predicted label maps, one per case (same order). |
| `case_ids` | `list[str]` | Case identifiers, same order. |
| `variant_name` | `str` | Model/variant label applied to every row. |
| `voxelspacings` | `list[tuple]` or `None`, default `None` | One spacing tuple per case, same order. `None` per case (or the whole list) falls back to pixel-unit HD95. |

**Returns** `pd.DataFrame`, one row per case (as returned by `per_case_metrics`).

### `summarize(df, group_col="variant")`

Collapses per-case rows into mean ± std per variant — the actual paper table.

| Param | Type | Description |
|---|---|---|
| `df` | `pd.DataFrame` | Output of `evaluate_dataset` (or a concatenation of several calls). |
| `group_col` | `str`, default `"variant"` | Column to group by. |

**Returns** `pd.DataFrame` with a `["mean", "std"]` multi-index column level over every
`dice_*`/`hd95_*`/`mean_dice` column.

### Module constants

- `CLASS_NAMES = {1: "RV", 2: "Myo", 3: "LV"}` — label id → class name (0 = background, excluded from per-class reporting).

---

## `mcpnet.evaluation.statistical_tests`

### `friedman_omnibus(df, metric_col, variant_col, case_id_col="case_id")`

Omnibus test across 3+ variants at once — run this first, before pairwise comparisons.

| Param | Type | Description |
|---|---|---|
| `df` | `pd.DataFrame` | Per-case metrics (e.g. `evaluate_dataset`'s output). |
| `metric_col` | `str` | Column to test (e.g. `"mean_dice"`). |
| `variant_col` | `str` | Column identifying which variant each row belongs to. |
| `case_id_col` | `str`, default `"case_id"` | Column used to pair rows across variants. |

Raises `ValueError` if fewer than 3 variants are present (use `compare_variants` directly for 2).

**Returns** `dict` with `statistic`, `p_value`, `n_cases`, `variants`.

### `compare_variants(df, metric_col, variant_col, baseline_variant, case_id_col="case_id", correction=None)`

Pairwise Wilcoxon signed-rank test: `baseline_variant` vs. every other variant present.

| Param | Type | Description |
|---|---|---|
| `df` | `pd.DataFrame` | Per-case metrics. |
| `metric_col` | `str` | Column to test. |
| `variant_col` | `str` | Column identifying variants. |
| `baseline_variant` | `str` | The variant every other variant is compared against. Raises `ValueError` if not present in `df`. |
| `case_id_col` | `str`, default `"case_id"` | Column used to pair rows across variants. |
| `correction` | `None` or `"bonferroni"`, default `None` | If `"bonferroni"`, multiplies each p-value by the number of comparisons made (use when following up a significant Friedman result). |

**Returns** `pd.DataFrame`, one row per non-baseline variant, columns `variant`, `n_cases`,
`statistic`, `p_value`. Returns an empty (but correctly-columned) DataFrame with a printed
warning if there are no other variants to compare against.

### `annotate_significance(p_value, thresholds=((0.001, "***"), (0.01, "**"), (0.05, "*")))`

Converts a p-value to a significance-star string for a table footnote.

| Param | Type | Description |
|---|---|---|
| `p_value` | `float` or `None` | The p-value to annotate. `None` returns `"n/a"`. |
| `thresholds` | `tuple[tuple[float, str], ...]` | `(threshold, label)` pairs, checked in order; first threshold `p_value` is below wins. |

**Returns** `str` — one of the threshold labels, or `"ns"` (not significant) if no threshold matches.

---

## `mcpnet.utils.seeding`

### `set_global_seed(seed: int, deterministic_ops: bool = True)`

Fixes every RNG source that affects training reproducibility in one call: `PYTHONHASHSEED`,
Python's `random`, numpy, and TensorFlow's global RNGs.

| Param | Type | Description |
|---|---|---|
| `seed` | `int` | Seed value, applied to every RNG source. |
| `deterministic_ops` | `bool`, default `True` | If `True`, also calls `tf.config.experimental.enable_op_determinism()`, forcing deterministic GPU kernels. Confirmed necessary for cross-process/cross-runtime reproducibility (RNG seeding alone was empirically insufficient — see the module docstring for the specific HD95 discrepancy this fixes). May slow some ops down; set `False` only for rapid prototyping where exact reproducibility doesn't matter. |

Call once per run, before building the model or data pipeline (already wired into
`train_variant` and `DataLoader.__init__`).

---

## `scripts.run_ablation`

CLI entry point (`python scripts/run_ablation.py --config <path> --table {A,B,C,D,all}`) plus
the functions it's built from, importable for use in notebooks.

### `load_data_loader(config)`

Builds a `mcpnet.data.data_loader.DataLoader` from the parsed YAML config's `data:`/`training:`
sections (`augment=True, to_train=True, adjust_percents=True`).

| Param | Type | Description |
|---|---|---|
| `config` | `dict` | Parsed `configs/mcp_net_config.yaml`. |

**Returns** `DataLoader`.

### `load_test_cases(config)`

Generator yielding one held-out test case per patient volume (not per slice) — the unit both
HD95 and the paired significance tests need. Reads the same raw NIfTI files `DataLoader` does,
but keeps each patient's slices stacked as one volume instead of flattening them.

| Param | Type | Description |
|---|---|---|
| `config` | `dict` | Parsed config; uses `config["data"]["configs_path"]`, `data_path`, `gt_path`. |

**Yields** `(case_id, x_volume, y_true_volume, voxelspacing)`:
- `case_id` — `str`, filename stem.
- `x_volume` — `np.ndarray`, shape `(num_slices, H, W, 1)`.
- `y_true_volume` — `np.ndarray`, shape `(num_slices, H, W)`, integer labels (not one-hot).
- `voxelspacing` — `(z_mm, x_mm, y_mm)`, reordered from the NIfTI header's `(x, y, z)` zooms to match `x_volume`/`y_true_volume`'s `(S, H, W)` axis order. Requires the ROI files' headers to already carry correct spacing (see `fix_roi_spacing.py` note in the README).

### `_train_and_evaluate_variants(config, variant_names, data_loader, test_cases, table_label, force_retrain=False, trust_unmarked_checkpoints=None)`

Shared training + evaluation loop used by Tables A/C/D: for each variant, reuses an existing
completed checkpoint if one is found, otherwise trains it, then evaluates on `test_cases`.

| Param | Type | Description |
|---|---|---|
| `config` | `dict` | Parsed config. |
| `variant_names` | `list[str]` | Variants to train/evaluate, in order. |
| `data_loader` | `DataLoader` | Built via `load_data_loader`. |
| `test_cases` | `list` | Materialized output of `load_test_cases(config)`. |
| `table_label` | `str` | Used only in printed log messages (e.g. `"Table A"`). |
| `force_retrain` | `bool` or `list[str]`, default `False` | `True` retrains every variant regardless of existing checkpoints. A list/set/tuple of variant names retrains only those. |
| `trust_unmarked_checkpoints` | `list[str]` or `None`, default `None` | Variant names whose existing `.h5` checkpoint should be reused even without a `.complete` marker file (e.g. checkpoints trained before the completion-marker system existed). Use deliberately — this reintroduces the risk of silently reusing an interrupted/partial checkpoint for exactly those variants. |

A checkpoint at `{save_dir}/{variant_name}_best.h5` is only reused if its
`{...}.h5.complete` marker also exists (or the variant is in `trust_unmarked_checkpoints`) —
this prevents silently reusing a checkpoint left behind by an interrupted training run.

**Returns** `pd.DataFrame` — concatenated per-case metrics across all variants.

### `_save_and_report(full_df, results_dir, table_key, baseline_variant, run_friedman=False)`

Writes `{table_key}_per_case.csv` and `{table_key}_summary.csv`, runs significance testing, and
writes `{table_key}_significance.csv`.

| Param | Type | Description |
|---|---|---|
| `full_df` | `pd.DataFrame` | Per-case metrics across all variants (e.g. from `_train_and_evaluate_variants`). |
| `results_dir` | `str` | Output directory for the CSVs. |
| `table_key` | `str` | Filename prefix, e.g. `"table_a"`. |
| `baseline_variant` | `str` | Passed to `compare_variants` as the comparison baseline. |
| `run_friedman` | `bool`, default `False` | If `True` and 3+ variants are present, runs `friedman_omnibus` first and Bonferroni-corrects the pairwise p-values. |

**Returns** `(summary, sig)` — the summary and significance DataFrames.

### `run_table_a(config, force_retrain=False, trust_unmarked_checkpoints=None)`

Architecture ablation (3 variants: `vanilla_unet`, `mcp_net_no_connections`, `full_mcp_net`),
baseline `full_mcp_net`, with the Friedman omnibus test. **Returns** `(full_df, summary, sig)`.

### `run_table_c(config, force_retrain=False, trust_unmarked_checkpoints=None)`

2×2 factorial (pyramid × connections, 4 variants), reusing Table A's 3 shared checkpoints where
present — only `single_path_with_connections` actually trains. Also prints the four pairwise
readings of the 2×2 grid. **Returns** `(full_df, summary, sig)`.

### `run_table_d(config, force_retrain=False, trust_unmarked_checkpoints=None)`

Decoder fusion ablation (`full_mcp_net` vs. `full_mcp_net_additive_decoder`), reusing Table
A/C's `full_mcp_net` checkpoint where present. No Friedman test (only 2 variants).
**Returns** `(full_df, summary, sig)`.

### `run_table_b(config)`

Snapshot-ensemble ablation. Trains `full_mcp_net` once with a cyclic LR schedule to collect all
snapshots, optionally adds a fixed-LR single-model anchor row, then evaluates every
`M` in `config["ablation"]["table_b"]["snapshot_counts"]` by ensembling that many snapshots
(selected via `config["ablation"]["table_b"].get("snapshot_selection", "earliest")` — `"earliest"`
or `"latest"`). **Returns** `(full_df, summary, sig)`.

### Module constant

- `CUSTOM_OBJECTS = {"loss": loss, "DiceIndex": DiceIndex, "DiceIndexForeground": DiceIndexForeground}` — passed to `load_model(..., custom_objects=CUSTOM_OBJECTS)` wherever a checkpoint is deserialized.

### CLI arguments (`if __name__ == "__main__"`)

| Flag | Default | Description |
|---|---|---|
| `--config` | `configs/mcp_net_config.yaml` | Path to the YAML config. |
| `--table` | `all` | One of `A`, `B`, `C`, `D`, `all`. |
| `--force-retrain` | off | Retrain every variant even if a checkpoint exists. Only affects Tables A/C/D — Table B always trains its own cyclic-LR run. |
