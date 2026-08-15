import numpy as np


def extract_bbox_from_mask(mask, min_size_fraction=None, margin_fraction=None):
    # mask: (H, W), any nonzero value = foreground. Returns normalized
    # (y1, x1, y2, x2) in [0,1], resolution-independent so the same target
    # works regardless of the original image's size
    H, W = mask.shape
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return None  # no foreground -- caller should skip this sample

    y1, y2 = coords[0].min(), coords[0].max()
    x1, x2 = coords[1].min(), coords[1].max()

    if min_size_fraction is not None:
        # resolution-relative minimum: computed SEPARATELY for height and
        # width, each as a fraction of THAT dimension's own size -- so a
        # non-square image gets a correctly different pixel minimum for H
        # vs W, rather than one shared absolute value
        min_h = min_size_fraction * H
        min_w = min_size_fraction * W
        h, w = y2 - y1, x2 - x1
        if h < min_h:
            cy = (y1 + y2) // 2
            y1 = max(0, int(cy - min_h // 2))
            y2 = min(H, int(cy + min_h // 2))
        if w < min_w:
            cx = (x1 + x2) // 2
            x1 = max(0, int(cx - min_w // 2))
            x2 = min(W, int(cx + min_w // 2))

    if margin_fraction is not None:
        # UNCONDITIONAL expansion, applied to every box regardless of
        # size (unlike min_size_fraction above, which only kicks in for
        # boxes smaller than the threshold) -- so even an
        # already-adequately-sized but tightly-fitted box gets some
        # buffer, baked directly into the training target rather than
        # relying on a margin added later at crop time
        margin_h = margin_fraction * H
        margin_w = margin_fraction * W
        y1 = max(0, int(y1 - margin_h))
        y2 = min(H, int(y2 + margin_h))
        x1 = max(0, int(x1 - margin_w))
        x2 = min(W, int(x2 + margin_w))

    return np.array([y1 / H, x1 / W, y2 / H, x2 / W], dtype=np.float32)


def denormalize_bbox(norm_bbox, image_shape):
    # inverse of the above, for turning a predicted normalized box back
    # into actual pixel coordinates for a given image size
    H, W = image_shape[0], image_shape[1]
    y1, x1, y2, x2 = norm_bbox
    return (int(y1 * H), int(x1 * W), int(y2 * H), int(x2 * W))