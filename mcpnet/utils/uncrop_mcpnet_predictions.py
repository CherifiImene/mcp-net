"""
Pastes full_mcp_net's 128x128 ROI-space predictions back into
ORIGINAL, full-resolution image coordinates.

TWO CASES, TWO FUNCTIONS:
  - uncrop_prediction(): ACDC's fixed-128x128 crop (confirmed from the
    original localize_heart() source: height/width were hardcoded
    constants, only the crop's start coordinate ever adjusted — no resize
    was ever involved) — a direct paste, no resize-reversal needed.
  - uncrop_with_resize_reversal(): M&Ms's variable-size crop (via
    localize_heart_robust: crop -> resize preserving aspect ratio -> pad)
    — needs the FULL inverse: un-pad -> un-resize -> paste. A previous
    version of this handled only the resize, assumed a square crop, and
    produced a real, reproduced bug on non-square crops (the normal case
    for a motion-detected bounding box).

Background (class 0) is used to fill everything OUTSIDE the original crop
region — the model made no prediction there, and treating it as background
is the standard, defensible convention (also matches how the ORIGINAL
ground truth looks outside the heart region).
"""

import numpy as np

from mcpnet.data.localize_heart_heuristic import get_margin_adjusted_bbox, MODEL_INPUT_SIZE


def uncrop_prediction(pred_crop, bbox, orig_size):
    """
    pred_crop: (crop_h, crop_w, S) — a prediction already at the SAME
          pixel scale as the bbox region (for ACDC: always 128x128, since
          no resize was ever involved; for M&Ms: whatever size the
          resize-reversal produced — see compare_baselines_mms.py).
    bbox: (y1, x1, y2, x2) — in ORIGINAL image pixel coordinates.
    orig_size: (H, W, S) — the original image's full size.

    Returns: (H, W, S) array, background (0) everywhere outside the crop,
    pred_crop's values pasted at the bbox location.

    SAFETY CHECK GENERALIZED (was previously hardcoded to require exactly
    128x128 — correct for ACDC's fixed-size crops, but wrong for M&Ms,
    where localize_heart_robust's crop size legitimately varies per case
    before being resized to 128 for the model). Now checks that pred_crop's
    shape matches the bbox's OWN size — still catches a genuine mismatch
    (e.g. passing the wrong bbox for a given prediction) for BOTH ACDC and
    M&Ms usage, without assuming a specific fixed size.
    """
    y1, x1, y2, x2 = bbox
    H, W, S = orig_size

    bbox_h, bbox_w = y2 - y1, x2 - x1
    assert pred_crop.shape[0] == bbox_h and pred_crop.shape[1] == bbox_w, (
        f"pred_crop shape {pred_crop.shape[:2]} doesn't match bbox size "
        f"({bbox_h}x{bbox_w}) — these must match for a valid, non-distorted paste."
    )
    assert pred_crop.shape[2] == S, (
        f"Slice count mismatch: prediction has {pred_crop.shape[2]} slices, "
        f"orig_size says {S} — these must match for a valid un-crop."
    )

    canvas = np.zeros((H, W, S), dtype=pred_crop.dtype)
    canvas[y1:y2, x1:x2, :] = pred_crop
    return canvas


def uncrop_with_resize_reversal(pred_128, raw_bbox, orig_size, margin_px):
    """
    Inverse of the FIXED resize_preserve_aspect_then_pad (pad-to-square
    FIRST, then uniform resize to target_size). The forward pass is now:
    crop (+margin) -> pad to square -> resize uniformly to 128x128. The
    correct inverse is: resize DIRECTLY back to the square size -> un-pad
    -> paste.

    CONFIRMED FIX: the previous version (aspect-preserving resize then
    pad, inverted as un-pad then un-resize) forced the crop's SHORTER
    side through an intermediate, non-128 rounded size during the forward
    pass (e.g. 95, not 128) — round-tripping through that odd size via
    nearest-neighbor is not perfectly invertible. Confirmed via direct
    testing: recovered only 720/852 and 744/852 pixels of a synthetic
    thin ring for non-square bboxes (132 and 108 lost respectively),
    while square bboxes recovered perfectly (why this wasn't caught until
    non-square cases were specifically tested). This version matches the
    new forward transform exactly, eliminating that intermediate size.

    pred_128: (128, 128, S) — the model's raw output.
    raw_bbox: (y1, x1, y2, x2) — the RAW detected bbox (BEFORE margin).
    orig_size: (H, W, S) — the original, un-cropped image's full size.
    margin_px: must match whatever margin_px was used during extraction.
    """
    import tensorflow as tf

    H, W, S = orig_size
    y1, x1, y2, x2 = get_margin_adjusted_bbox((H, W), raw_bbox, margin_px)
    cropped_h, cropped_w = y2 - y1, x2 - x1

    square_size = max(cropped_h, cropped_w)

    # STEP 1: resize the 128x128 prediction DIRECTLY to square_size, using
    # the same one-hot-based resize as the forward pass (confirmed to
    # reduce boundary round-trip loss vs plain nearest-neighbor)
    from scripts.localize_heart_robust import resize_label_via_onehot
    resized_square = resize_label_via_onehot(pred_128.astype(np.int32), square_size)

    # STEP 2: un-pad -- crop back to the actual (cropped_h, cropped_w)
    # rectangle, using the SAME pad_top/pad_left split the forward pass used
    pad_h_total, pad_w_total = square_size - cropped_h, square_size - cropped_w
    pad_top, pad_left = pad_h_total // 2, pad_w_total // 2
    unpadded = resized_square[pad_top: pad_top + cropped_h, pad_left: pad_left + cropped_w, :]

    # STEP 3: paste at the margin-adjusted bbox location in a full-size canvas
    return uncrop_prediction(unpadded, (y1, x1, y2, x2), orig_size)


def uncrop_probabilities(pred_probs_128, raw_bbox, orig_size, margin_px):
    # CONFIRMED IMPROVEMENT over the discretize-then-one-hot approach,
    # specifically for PREDICTIONS (which only ever go through a SINGLE
    # upscale: 128 -> original crop size -- unlike ground truth extraction,
    # which goes through a full down-then-up round trip). Tested across 5
    # synthetic configurations against a genuine high-resolution reference
    # boundary: keeping raw softmax probabilities and deferring argmax
    # until AFTER upscaling consistently and substantially outperformed
    # discretizing first then one-hot-encoding an already-discrete result
    # (e.g. 0.878 -> 0.967 Dice in one config). The current pipeline's
    # discretize-then-onehot approach throws away real confidence
    # information at the FIRST argmax, then treats the result as if it
    # were 100% certain everywhere -- this function avoids that entirely.
    #
    # pred_probs_128: (128, 128, S, num_classes) -- RAW SOFTMAX OUTPUT,
    #   NOT yet argmax'd. This requires the prediction-saving step to
    #   save probabilities instead of (or alongside) discrete labels.
    # raw_bbox, orig_size, margin_px: same as uncrop_with_resize_reversal.
    import tensorflow as tf

    H, W, S = orig_size
    y1, x1, y2, x2 = get_margin_adjusted_bbox((H, W), raw_bbox, margin_px)
    cropped_h, cropped_w = y2 - y1, x2 - x1
    square_size = max(cropped_h, cropped_w)

    # resize probabilities directly to square_size -- bilinear, since these
    # are already smooth [0,1] values, no one-hot step needed at all
    num_classes = pred_probs_128.shape[-1]
    probs_shwc = np.transpose(pred_probs_128, (2, 0, 1, 3))  # (S,128,128,C)
    resized_square = tf.image.resize(probs_shwc, (square_size, square_size), method="bilinear").numpy()

    pad_h_total, pad_w_total = square_size - cropped_h, square_size - cropped_w
    pad_top, pad_left = pad_h_total // 2, pad_w_total // 2
    unpadded_probs = resized_square[:, pad_top: pad_top + cropped_h, pad_left: pad_left + cropped_w, :]

    # argmax ONLY NOW, after upscaling -- this is the key difference
    unpadded_labels = np.argmax(unpadded_probs, axis=-1).astype(np.int32)  # (S, cropped_h, cropped_w)
    unpadded_labels_hws = np.transpose(unpadded_labels, (1, 2, 0))  # (cropped_h, cropped_w, S)

    return uncrop_prediction(unpadded_labels_hws, (y1, x1, y2, x2), orig_size)