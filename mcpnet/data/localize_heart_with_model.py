import numpy as np
import tensorflow as tf

from mcpnet.utils.bbox_from_mask import denormalize_bbox
from mcpnet.data.localize_heart_heuristic import (
    crop_to_bbox_with_margin, resize_preserve_aspect_then_pad,
    MARGIN_PX, MODEL_INPUT_SIZE
)
from mcpnet.data.preprocessing import standardize

LOCALIZER_INPUT_SIZE = (256, 256)  # must match TRAIN_INPUT_SIZE used at training time


def resize_and_normalize_for_localizer(image_slice, target_size):
    img = tf.convert_to_tensor(image_slice[..., None], dtype=tf.float32)
    img = tf.image.resize(img, target_size).numpy()
    img_min, img_max = img.min(), img.max()
    return (img - img_min) / (img_max - img_min + 1e-8)


def predict_bbox(localizer_model, image_slice):
    # one bbox prediction per (patient, frame), applied to the whole slice
    # stack 
    resized = resize_and_normalize_for_localizer(image_slice, LOCALIZER_INPUT_SIZE)
    norm_bbox = localizer_model.predict(resized[None, ...], verbose=0)[0]
    return denormalize_bbox(norm_bbox, image_slice.shape)


def localize_heart_with_model(img_4D, frame, localizer_model, gt=None,
                               margin_px=MARGIN_PX, target_size=MODEL_INPUT_SIZE, verbose=False):
    frame_img = img_4D[:, :, :, frame - 1]

    mid_slice = frame_img.shape[2] // 2
    bbox = predict_bbox(localizer_model, frame_img[:, :, mid_slice])

    cropped_img = crop_to_bbox_with_margin(frame_img, bbox, margin_px)
    resized_img = resize_preserve_aspect_then_pad(cropped_img, target_size, is_label=False)
    final_img = standardize(resized_img)

    final_gt = None
    if isinstance(gt, np.ndarray):
        cropped_gt = crop_to_bbox_with_margin(gt, bbox, margin_px)
        final_gt = resize_preserve_aspect_then_pad(cropped_gt, target_size, is_label=True)

    cropped_h, cropped_w = cropped_img.shape[0], cropped_img.shape[1]
    scale_factor = target_size / max(cropped_h, cropped_w)

    # used_fallback/method_used kept in the return signature for drop-in
    # compatibility with localize_heart_robust's callers,
    # ( used_fallback is always False here )
    used_fallback = False
    method_used = "learned_localizer"

    if verbose:
        print(f"bbox={bbox}, method={method_used}, output shape={final_img.shape}, "
              f"scale_factor={scale_factor:.4f}, value range=[{final_img.min():.3f}, {final_img.max():.3f}]")

    return final_img, final_gt, bbox, used_fallback, method_used, scale_factor


# example usage:
# from tensorflow.keras.models import load_model
# localizer_model = load_model("heart_localizer_best.h5")
# final_img, final_gt, bbox, used_fallback, method, scale = localize_heart_with_model(
#     img_4D, frame, localizer_model, gt=gt_frame
# )