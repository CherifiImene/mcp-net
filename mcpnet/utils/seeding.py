"""
Central reproducibility utility.

Independent sources of run-to-run non-determinism in this pipeline:

1. Conv2D `kernel_initializer="he_normal"` — draws from TF's global RNG.
2. `tf.data.Dataset.shuffle(BUFFER_SIZE)` in `DataLoader.batch_data`.
3. `np.random.normal()` / `np.random.choice()` / `np.random.uniform()` calls
   throughout `Augment` (crop probability, offset choice, per-call seeds for
   the Keras RandomFlip/Rotation/Translation/Zoom layers, affine transform
   ordering) — all draw from numpy's global RNG.
4. `albumentations.Compose` (used in `augmentation_pipeline`) — draws from
   Python's `random` module / numpy internally.

`set_global_seed()` fixes all four in one call. Call it once per training
run, before building the model or the data pipeline.

Seeding alone makes one run reproducible but doesn't measure how much of a
Dice difference between variants is seed-dependent training variance vs. a
genuine architectural effect — the rigorous version would train each
variant under 3-5 seeds and report cross-seed variance in addition to the
per-case variance `compare_variants`/`friedman_omnibus` already capture.
Not implemented here given the added compute cost (3-5x every training run
across Tables A/B/C/D).

Seeding Python/numpy/TF's RNGs is not sufficient for cross-process
reproducibility on GPU: cuDNN's default convolution algorithms are selected
non-deterministically across separate process/CUDA-context initializations
unless determinism is explicitly forced (`deterministic_ops=True`, the
default here). This can make training somewhat slower, since deterministic
cuDNN kernels are sometimes less optimized than the non-deterministic ones.
"""

import os
import random

import numpy as np
import tensorflow as tf


def set_global_seed(seed: int, deterministic_ops: bool = True):
    """Fixes Python's `random`, numpy, and TensorFlow's global RNGs.

    `deterministic_ops=True` (the default) additionally asks TensorFlow to
    use only deterministic GPU kernel implementations where available
    (`tf.config.experimental.enable_op_determinism()`) — this is what
    matters for cross-process reproducibility; seeding the RNGs alone is not
    sufficient. May make some ops slower; set to `False` only when exact
    reproducibility isn't required (e.g. rapid prototyping).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # tf.keras.utils.set_random_seed sets python/numpy/tf together (TF>=2.7);
    # calling the individual seeds above too is harmless redundancy and
    # keeps this working even if that helper isn't available on an older TF.
    try:
        tf.keras.utils.set_random_seed(seed)
    except AttributeError:
        tf.random.set_seed(seed)

    if deterministic_ops:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception as e:
            print(f"Warning: could not enable full op determinism: {e}")
