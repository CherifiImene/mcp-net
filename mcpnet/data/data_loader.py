"""
DataLoader for ROI-cropped ACDC NIfTI volumes.
"""

import json
import os
from os import path, walk

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import Sequence, to_categorical

from mcpnet.data.augmentation import Augment
from mcpnet.data.preprocessing import standardize

try:
    from acdc_utilities import load_nii
except ImportError:
    from mcpnet.data.acdc_utilities_stub import load_nii  # bundled fallback

try:
    from constants import HOME_DIR
except ImportError:
    HOME_DIR = os.environ.get("MCP_HOME_DIR", ".")


class DataLoader:
    def __init__(
        self,
        data_path,
        gt_path,
        configs_path,
        buffer_size,
        train_batch_size,
        val_batch_size,
        augment=True,
        to_train=True,
        adjust_percents=True,
        classify=False,
        seed=42,
    ):
        # Makes .shuffle(BUFFER_SIZE, seed=...) below reproducible.
        from mcpnet.utils.seeding import set_global_seed
        set_global_seed(seed)
        self.seed = seed

        self.BUFFER_SIZE = buffer_size
        self.TR_BATCH_SIZE = train_batch_size
        self.VAL_BATCH_SIZE = val_batch_size
        self.gt_treshold = 0.07

        try:
            self.AUTOTUNE = tf.data.AUTOTUNE
        except AttributeError:
            self.AUTOTUNE = tf.data.experimental.AUTOTUNE

        (X_train, Y_train), (X_val, Y_val) = self.load_dataset(data_path, gt_path, configs_path, classify=classify)

        if to_train and adjust_percents:
            adjusted_X, adjusted_Y = self.adjust_labels_percent(X_train, Y_train)
            print(f"nb adj images: {len(adjusted_X)}")
            X_train.extend(adjusted_X)
            Y_train.extend(adjusted_Y)

        self.train_step = (len(X_train) // self.TR_BATCH_SIZE) + (1 if len(X_train) % self.TR_BATCH_SIZE != 0 else 0)
        self.val_step = len(X_val) // self.VAL_BATCH_SIZE + (1 if len(X_val) % self.VAL_BATCH_SIZE != 0 else 0)

        self.X_train, self.Y_train, self.X_val, self.Y_val = self.numpy(X_train, Y_train, X_val, Y_val)

        self.train_dataset, self.val_dataset = self.get_tf_dataset((self.X_train, self.Y_train), (self.X_val, self.Y_val))

        self.train_batches, self.val_batches = self.batch_data(self.train_dataset, self.val_dataset, augment, to_train=to_train)

        # Held-out test set (kept fully separate from val — see load_dataset).
        # Not batched/augmented: run_ablation.py evaluates it case-by-case.
        if len(self.X_test_raw) > 0:
            self.X_test = np.array(self.X_test_raw, dtype=np.float32)
            self.Y_test = np.array(self.Y_test_raw, dtype=np.float32)
        else:
            self.X_test, self.Y_test = np.array([]), np.array([])

    def adjust_labels_percent(self, training_data, training_gt):
        assert isinstance(training_data, list), f"Expected training data to be of type list, got {type(training_data)}"
        assert isinstance(training_gt, list), f"Expected training ground truths to be of type list, got {type(training_gt)}"

        adj_imgs, adj_gt = [], []
        for i in range(len(training_data)):
            m_img, m_gt = self._adjust_label_percent(training_data[i], training_gt[i])
            if m_img is not None and m_gt is not None:
                adj_imgs.append(m_img)
                adj_gt.append(m_gt)
        return adj_imgs, adj_gt

    def load_dataset(self, data_path, labels_path, configs_path, classify=False):
        """
        `dev` and `test` splits from configs.json are kept fully SEPARATE
        (X_val/Y_val vs X_test/Y_test). Merging them would let checkpoint
        selection (`ModelCheckpoint(monitor="val_DiceIndex")`) see
        test-patient data, contaminating every downstream "test" number.
        """
        X_train, Y_train = [], []
        X_val, Y_val = [], []
        X_test, Y_test = [], []

        with open(configs_path) as json_file:
            data = json.load(json_file)

        if classify:
            p = HOME_DIR + "/Train/configs.csv"
            df = pd.read_csv(p)
            df["Group"].replace(["DCM", "HCM", "MINF", "NOR", "RV"], [0, 1, 2, 3, 4], inplace=True)

        train, val, test = data["train"], data["dev"], data["test"]
        self.test_voxel_spacing = {}  # filename (no ext) -> (x_mm, y_mm, z_mm), from header.get_zooms()

        for (_, _, filenames) in walk(data_path):
            for filename in filenames:
                img_data, img_affine, img_header = load_nii(data_path + filename)
                img_data = standardize(img_data)
                img_data = np.expand_dims(img_data, axis=-1)

                if not classify:
                    img_label, _, _ = load_nii(labels_path + filename)

                def _labels_for(img_label):
                    if classify:
                        return [df.iloc[int(filename[7:10]) - 1]["Group"] for _ in range(img_data.shape[2])]
                    return [to_categorical(img_label[:, :, i], num_classes=4) for i in range(img_label.shape[2])]

                if filename[:-3] in train:
                    X_train.extend([img_data[:, :, i, :] for i in range(img_data.shape[2])])
                    Y_train.extend(_labels_for(img_label if not classify else None))

                elif filename[:-3] in val:
                    X_val.extend([img_data[:, :, i, :] for i in range(img_data.shape[2])])
                    Y_val.extend(_labels_for(img_label if not classify else None))

                elif filename[:-3] in test:
                    X_test.extend([img_data[:, :, i, :] for i in range(img_data.shape[2])])
                    Y_test.extend(_labels_for(img_label if not classify else None))
                    # zooms is (x, y, z) in mm; used downstream for HD95 (see
                    # mcpnet.evaluation.metrics) — must be passed explicitly
                    # or HD95 defaults to pixel units.
                    self.test_voxel_spacing[filename[:-3]] = img_header.get_zooms()

        print(f"Loaded: {len(X_train)} train / {len(X_val)} val / {len(X_test)} test slices")

        self.X_test_raw, self.Y_test_raw = X_test, Y_test

        return (X_train, Y_train), (X_val, Y_val)

    def numpy(self, *args):
        for arg in args:
            assert isinstance(arg, list), f"expected arguments to be of type list got : {type(arg)}"
            yield np.array(arg, dtype=np.float32)

    def get_dataset_stats(self, *datasets):
        concat_dataset = np.concatenate(datasets, axis=0)
        mean = np.mean(concat_dataset)
        stdev = np.std(concat_dataset)

        if not os.path.exists(HOME_DIR + "/configurations/"):
            os.makedirs("../configurations/")
        with open("../configurations/stats.txt", "w") as f:
            f.write(f"\nmean: {mean} std: {stdev}")
        return mean, stdev

    def get_tf_dataset(self, *args):
        for arg in args:
            X, y = arg
            dataset = tf.data.Dataset.from_tensor_slices((X, y))
            yield dataset

    def standardize_datasets(self):
        if self.mean is not None and self.stdev is not None:
            self.train_dataset = self.train_dataset.map(self._preprocess, num_parallel_calls=self.AUTOTUNE)
            self.val_dataset = self.val_dataset.map(self._preprocess, num_parallel_calls=self.AUTOTUNE)
        else:
            raise Exception("Mean/stdev not set — call 'get_dataset_stats' first")

    def _preprocess(self, image, mask):
        image = tf.cast((image - self.mean) / self.stdev, dtype=tf.float32)
        image = tf.expand_dims(image, axis=-1)
        return image, mask

    def _adjust_label_percent(self, image, mask):
        if image.ndim == 2:
            image = tf.expand_dims(image, axis=-1)

        gt = tf.argmax(mask, axis=-1)
        label_percent = tf.cast(tf.math.count_nonzero(gt), tf.int32) / tf.cast(tf.size(gt), tf.int32)

        if label_percent < 0.07 and label_percent != 0.0:
            result = np.asarray(gt != 0).nonzero()

            y1, x1 = result[0].min(), result[1].min()
            y2, x2 = result[0].max(), result[1].max()

            offset = 15
            if label_percent <= 0.03:
                offset = 5

            y1 = y1 - offset if offset < y1 else y1
            x1 = x1 - offset if offset < x1 else x1
            y2 = y2 + offset if (y2 + offset) < 128 else 128
            x2 = x2 + offset if (x2 + offset) < 128 else 128

            boxe = [y1, x1, y2, x2]
            boxe = tf.convert_to_tensor(boxe, dtype=tf.int32)

            gt = tf.expand_dims(gt, axis=-1)
            cropped_img = Augment.crop_and_resize(image=image, bbx=boxe, size=image.shape[:-1])

            cropped_gt = Augment.crop_and_resize(image=mask, bbx=boxe, size=image.shape[:-1])
            cropped_gt = tf.squeeze(cropped_gt)
            return cropped_img, cropped_gt
        return None, None

    def _set_shapes(self, image, mask):
        image.set_shape((128, 128, 1))
        mask.set_shape((128, 128, 4))
        return image, mask

    def batch_data(self, train_dataset, val_dataset, augment=True, to_train=True):
        if augment and to_train:
            train_batches = (
                train_dataset.cache()
                .shuffle(self.BUFFER_SIZE, seed=self.seed, reshuffle_each_iteration=True)
                .map(lambda image, mask: Augment.augmentation_pipeline(image, mask), num_parallel_calls=self.AUTOTUNE)
                .map(self._set_shapes, num_parallel_calls=self.AUTOTUNE)
                .batch(self.TR_BATCH_SIZE)
                .repeat()
                .prefetch(buffer_size=self.AUTOTUNE)
            )
        elif to_train:
            train_batches = (
                train_dataset.cache()
                .shuffle(self.BUFFER_SIZE, seed=self.seed, reshuffle_each_iteration=True)
                .batch(self.TR_BATCH_SIZE)
                .repeat()
                .prefetch(buffer_size=self.AUTOTUNE)
            )
        else:
            train_batches = train_dataset.batch(self.TR_BATCH_SIZE)

        val_batches = val_dataset.batch(self.VAL_BATCH_SIZE)
        return train_batches, val_batches