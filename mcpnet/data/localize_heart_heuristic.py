import numpy as np
import tensorflow as tf
from scipy import ndimage
import cv2
from skimage.filters import farid, threshold_li, threshold_otsu, threshold_triangle

MODEL_INPUT_SIZE = 128
MARGIN_PX = 10


# def standardize_single_phase(image):
#     standardized_image = np.zeros(image.shape)
#     for c in range(image.shape[2]):
#         image_slice = image[:, :, c]
#         centered = image_slice - np.mean(image)
#         if np.std(centered) != 0:
#             standardized_image[:, :, c] = centered / np.std(centered)
#         else:
#             standardized_image[:, :, c] = centered
#     return standardized_image


def detect_motion(img):
    # motion = mean absolute frame-to-frame difference across the FULL
    # periodic cardiac cycle, including the wrap-around from the last
    # frame back to the first (the heart is periodic, so this transition
    # is a real part of the cycle too, not just T-1 of T pairs)
    cardiac_cycle = img.shape[3]
    results = np.empty((img.shape[0], img.shape[1], img.shape[2], cardiac_cycle))
    for i in range(cardiac_cycle):
        next_i = (i + 1) % cardiac_cycle
        results[:, :, :, i] = np.abs(img[:, :, :, i] - img[:, :, :, next_i])
    return np.sqrt(results.mean(axis=3))


def detect_roi_boundaries(motion):
    f = np.zeros(motion.shape)
    for channel in range(motion.shape[2]):
        f[:, :, channel] = farid(motion[:, :, channel])
    return f


def get_largest_component_2d(binary_2d):
    if binary_2d.sum() == 0:
        return binary_2d
    s = ndimage.generate_binary_structure(2, 2)
    labeled_array, numpatches = ndimage.label(binary_2d, s)
    if numpatches == 0:
        return np.zeros_like(binary_2d, dtype=np.uint8)
    sizes = ndimage.sum(binary_2d, labeled_array, range(1, numpatches + 1))
    max_label = np.argmax(sizes) + 1
    return (labeled_array == max_label).astype(np.uint8)


def _try_threshold_methods(heart_bd):
    for method_name, method_fn in [("triangle", threshold_triangle),
                                    ("otsu", threshold_otsu),
                                    ("li", threshold_li)]:
        try:
            mid_frame_idx = heart_bd.shape[-1] // 2
            mid_slice = heart_bd[:, :, mid_frame_idx]
            total_px = mid_slice.shape[0] * mid_slice.shape[1]

            t = method_fn(mid_slice)
            binary_slice = (mid_slice >= t).astype(np.uint8)
            lcc = get_largest_component_2d(binary_slice)
            component_px = lcc.sum()
            
            print(f"method: {method_name}, percentage: {component_px/total_px}")
            if 0.1 * total_px < component_px < 0.7 * total_px:
                
                return lcc, method_name
        except Exception:
            continue
    return None, None


def _bbox_from_component(lcc_2d):
    # returns (y1, x1, y2, x2), RELATIVE to whatever image lcc_2d was computed on
    coords = np.where(lcc_2d == 1)
    if len(coords[0]) == 0:
        return None
    y1, y2 = int(coords[0].min()), int(coords[0].max())
    x1, x2 = int(coords[1].min()), int(coords[1].max())
    return y1, x1, y2, x2


def _center_crop_bbox(image_shape, size_px=MODEL_INPUT_SIZE):
    h, w = image_shape[0], image_shape[1]
    half = size_px // 2
    cy, cx = h // 2, w // 2
    return max(0, cy - half), max(0, cx - half), min(h, cy + half), min(w, cx + half)


def get_margin_adjusted_bbox(image_shape, bbox, margin_px=MARGIN_PX):
    h, w = image_shape[0], image_shape[1]
    y1, x1, y2, x2 = bbox
    y1 = max(0, y1 - margin_px)
    x1 = max(0, x1 - margin_px)
    y2 = min(h, y2 + margin_px)
    x2 = min(w, x2 + margin_px)
    return (y1, x1, y2, x2)


def crop_to_bbox_with_margin(image, bbox, margin_px=MARGIN_PX):
    y1, x1, y2, x2 = get_margin_adjusted_bbox(image.shape, bbox, margin_px)
    return image[y1:y2, x1:x2, ...]


def resize_preserve_aspect_then_pad(cropped, target_size=MODEL_INPUT_SIZE, is_label=False):
    h, w = cropped.shape[0], cropped.shape[1]
    if h == 0 or w == 0:
        raise ValueError(f"Cropped region is empty (shape {cropped.shape}) — check bbox/margin.")

    cropped = cropped.astype("float32")
    method = "nearest" if is_label else "bilinear"

    resized = tf.image.resize(
        cropped, (target_size, target_size),
        method=method, preserve_aspect_ratio=True,
    ).numpy()

    if is_label:
        resized = np.round(resized)

    new_h, new_w = resized.shape[0], resized.shape[1]
    pad_h, pad_w = target_size - new_h, target_size - new_w
    pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
    pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
    pad_width = [(pad_top, pad_bottom), (pad_left, pad_right)] + [(0, 0)] * (resized.ndim - 2)
    padded = np.pad(resized, pad_width, mode="constant", constant_values=0)

    assert padded.shape[0] == target_size and padded.shape[1] == target_size, \
        f"Padding failed to reach target size: got {padded.shape}"
    return padded


def crop_center_percentage(image_input, target_area_fraction=0.70):
    # returns (x_start, y_start, x_end, y_end) -- NOTE: x,y order, different
    # from _bbox_from_component's (y1,x1,y2,x2) -- must be translated
    # carefully when combining the two, see localize_heart_robust below
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise ValueError(f"Could not load image from: {image_input}")
    else:
        img = image_input

    height, width = img.shape[0], img.shape[1]
    scale_factor = target_area_fraction ** 0.5
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    x_start = (width - new_width) // 2
    y_start = (height - new_height) // 2
    x_end = x_start + new_width
    y_end = y_start + new_height

    cropped_img = img[y_start:y_end, x_start:x_end]
    return cropped_img, (x_start, y_start, x_end, y_end)


def localize_heart_robust(img_4D, frame, gt=None, margin_px=MARGIN_PX,
                           target_size=MODEL_INPUT_SIZE, verbose=False):
    used_fallback = False
    method_used = None

    img_4D_center, crop_box = crop_center_percentage(img_4D, 0.7)
    x_start, y_start, x_end, y_end = crop_box

    try:
        motion = detect_motion(img_4D_center)
        heart_bd = detect_roi_boundaries(motion)
        lcc, method_used = _try_threshold_methods(heart_bd)
    except Exception as e:
        if verbose:
            print(f"Motion-based detection raised an exception: {e}")
        lcc = None

    frame_img = img_4D[:, :, :, frame - 1]  # full, UNCROPPED frame -- crop_box translation below maps back to this

    if lcc is not None:
        relative_bbox = _bbox_from_component(lcc)  # (y1_rel, x1_rel, y2_rel, x2_rel), relative to the center-cropped region
        y1_rel, x1_rel, y2_rel, x2_rel = relative_bbox

        y1 = y1_rel + y_start
        x1 = x1_rel + x_start
        y2 = y2_rel + y_start
        x2 = x2_rel + x_start

        bbox = (y1, x1, y2, x2)
    else:
        used_fallback = True
        bbox = _center_crop_bbox(frame_img.shape, target_size)
        if verbose:
            print("Falling back to center crop (motion-based detection failed or degenerate).")

    crop_margin = margin_px if used_fallback else 5
    cropped_img = crop_to_bbox_with_margin(frame_img, bbox, crop_margin)
    resized_img = resize_preserve_aspect_then_pad(cropped_img, target_size, is_label=False)
    final_img = resized_img #standardize_single_phase(resized_img)

    final_gt = None
    if isinstance(gt, np.ndarray):
        cropped_gt = crop_to_bbox_with_margin(gt, bbox, crop_margin)
        final_gt = resize_preserve_aspect_then_pad(cropped_gt, target_size, is_label=True)

    cropped_h, cropped_w = cropped_img.shape[0], cropped_img.shape[1]
    scale_factor = target_size / max(cropped_h, cropped_w)

    if verbose:
        print(f"bbox={bbox}, method={method_used}, used_fallback={used_fallback}, "
              f"output shape={final_img.shape}, scale_factor={scale_factor:.4f}, "
              f"value range=[{final_img.min():.3f}, {final_img.max():.3f}]")

    return final_img, final_gt, bbox, used_fallback, method_used, scale_factor