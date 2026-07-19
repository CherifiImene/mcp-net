"""
Data augmentation utilities.
"""

import numpy as np
import tensorflow as tf
import albumentations as A


class Augment:
    def crop_and_resize(image, bbx, size):
        cropped = tf.image.crop_to_bounding_box(
            image=image,
            offset_height=bbx[0],
            offset_width=bbx[1],
            target_height=bbx[2] - bbx[0],
            target_width=bbx[3] - bbx[1],
        )
        cropped = tf.expand_dims(cropped, axis=0)
        resized = tf.image.resize(images=cropped, size=size)
        return tf.squeeze(input=resized, axis=0)

    def crop_and_resize_v2(image, mask):
        label_percent = tf.cast(tf.math.count_nonzero(mask[1:]), tf.int32) / tf.cast(tf.size(mask), tf.int32)
        p = np.random.normal()
        if p >= 0.7 and label_percent <= 0.07:
            offset = np.random.choice([5, 10, 20, 25, 30])
            result = tf.where(mask[1:] != 0)

            y1, x1 = tf.reduce_min(result[0]), tf.reduce_min(result[1])
            y2, x2 = tf.reduce_max(result[0]), tf.reduce_max(result[1])

            boxe = [y1 - offset, x1 - offset, y2 + offset, x2 + offset]
            boxe = tf.convert_to_tensor(boxe, dtype=tf.int32)
            aug_img = tf.image.crop_to_bounding_box(
                image=image,
                offset_height=boxe[0],
                offset_width=boxe[1],
                target_height=boxe[2] - boxe[0],
                target_width=boxe[3] - boxe[1],
            )
            aug_img = tf.image.resize(images=aug_img, size=(128, 128))

            aug_gt = tf.image.crop_to_bounding_box(
                image=mask,
                offset_height=boxe[0],
                offset_width=boxe[1],
                target_height=boxe[2] - boxe[0],
                target_width=boxe[3] - boxe[1],
            )
            aug_gt = tf.image.resize(images=aug_gt, size=(128, 128))

            return aug_img, aug_gt
        return image, mask

    def random_mirroring(image, mask, mode="horizontal_and_vertical"):
        seed = np.random.uniform(low=0, high=10000)
        mirror_img = tf.keras.layers.RandomFlip(mode=mode, seed=seed)(image)
        mirror_gt = tf.keras.layers.RandomFlip(mode=mode, seed=seed)(mask)
        return mirror_img, mirror_gt

    def random_rotation(image, mask, ratio=0.5):
        seed = np.random.uniform(low=0, high=10000)
        rotated_img = tf.keras.layers.RandomRotation(0.5, seed=seed)(image)
        rotated_gt = tf.keras.layers.RandomRotation(0.5, seed=seed)(mask)
        return rotated_img, rotated_gt

    def random_translation(image, mask, ty, tx):
        seed = np.random.uniform(low=0, high=10000)
        trans_img = tf.keras.layers.RandomTranslation(ty, tx, seed=seed)(image)
        trans_gt = tf.keras.layers.RandomTranslation(ty, tx, seed=seed)(mask)
        return trans_img, trans_gt

    def random_zoom(image, mask, w_ratio=0.02, h_ratio=0.02):
        seed = np.random.uniform(low=0, high=10000)
        zoom_img = tf.keras.layers.RandomZoom(w_ratio, h_ratio, seed=seed)(image)
        zoom_gt = tf.keras.layers.RandomZoom(w_ratio, h_ratio, seed=seed)(mask)
        return zoom_img, zoom_gt

    def random_affine_transformation(image, mask, tx=0.05, ty=0.05, r_ratio=0.75, zw_ratio=0.02, zh_ratio=0.02):
        label_percent = tf.cast(tf.math.count_nonzero(mask[1:]), tf.int32) / tf.cast(tf.size(mask), tf.int32)
        if label_percent <= 0.25:
            possibilities = ("TRZ", "TZR", "RTZ", "RZT", "ZRT", "ZTR")
            choice = np.random.choice(possibilities)
            prev_img, prev_gt = image, mask
            for car in choice:
                if car == "T":
                    prev_img, prev_gt = Augment.random_translation(prev_img, prev_gt, ty=ty, tx=tx)
                elif car == "R":
                    prev_img, prev_gt = Augment.random_rotation(prev_img, prev_gt, ratio=r_ratio)
                else:
                    prev_img, prev_gt = Augment.random_zoom(prev_img, prev_gt, w_ratio=zw_ratio, h_ratio=zh_ratio)
            return prev_img, prev_gt
        return image, mask

    def augmentation_pipeline(image, mask, r_ratio=0.75):
        def aug_fn(image, mask):
            aug = A.Compose(
                [
                    A.RandomRotate90(),
                    # A.Flip() was removed in albumentations>=2.0; HorizontalFlip
                    # + VerticalFlip together are equivalent.
                    A.HorizontalFlip(),
                    A.VerticalFlip(),
                    A.Transpose(),
                    A.ShiftScaleRotate(shift_limit=0.04, scale_limit=0.06, rotate_limit=45, p=0.4),
                    A.OneOf(
                        [
                            A.GridDropout(p=0.35),
                            A.Emboss(p=0.1),
                            A.Affine(p=0.55, shear=(-5, 5), rotate=(-80, 90)),
                        ],
                        p=0.5,
                    ),
                ]
            )
            augmented = aug(image=image, mask=mask)
            return augmented["image"], augmented["mask"]

        aug_img, aug_mask = tf.numpy_function(func=aug_fn, inp=[image, mask], Tout=(tf.float32, tf.float32))
        return aug_img, aug_mask