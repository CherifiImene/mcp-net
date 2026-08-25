import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # to make sure the script runs correctly from the command line


import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from mcpnet.model.heart_localizer_model import build_localizer
from mcpnet.data.localizer_augmentation import random_translate

# ============================== CONFIG ============================== #
DATASET_ROOT = "data/localizer_dataset"  # output of prepare_localizer_dataset.py
TRAIN_INPUT_SIZE = (256, 256)  # must match whatever prepare_localizer_dataset.py used
BATCH_SIZE = 16
EPOCHS = 100
CHECKPOINT_PATH = "checkpoints/heart_localizer_best.h5"
# ======================================================================= #


def load_npz(path):
    data = np.load(path.numpy().decode("utf-8"))
    return data["image"].astype(np.float32), data["bbox"].astype(np.float32)


def load_npz_tf(path):
    img, bbox = tf.py_function(load_npz, [path], [tf.float32, tf.float32])
    img.set_shape((*TRAIN_INPUT_SIZE, 1))
    bbox.set_shape((4,))
    return img, bbox


def make_tf_dataset(split_dir, batch_size, shuffle=True, augment=False):
    file_paths = sorted(glob.glob(os.path.join(split_dir, "*.npz")))
    print(f"{split_dir}: {len(file_paths)} files")

    ds = tf.data.Dataset.from_tensor_slices(file_paths)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(file_paths))
    ds = ds.map(load_npz_tf, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        # random translation, to make sure the model
        # doesn't get biased by the position of the heart
        ds = ds.map(lambda img, bbox: random_translate(img, bbox, max_shift_fraction=0.25),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def main():
    train_dir = os.path.join(DATASET_ROOT, "train")
    val_dir = os.path.join(DATASET_ROOT, "val")

    train_ds = make_tf_dataset(train_dir, BATCH_SIZE, shuffle=True, augment=True)
    val_ds = make_tf_dataset(val_dir, BATCH_SIZE, shuffle=False, augment=False)

    model = build_localizer((*TRAIN_INPUT_SIZE, 1))
    model.compile(optimizer=Adam(1e-3), loss=tf.keras.losses.Huber(), metrics=["mae"])

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)


if __name__ == "__main__":
    main()