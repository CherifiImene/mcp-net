import numpy as np
from medpy.metric import dc


def compute_mean_dice(pred_shw, gt_shw, classes=(1, 2, 3)):
    # pred_shw, gt_shw: (S, H, W), integer class labels, ACDC convention
    dices = []
    for c in classes:
        pred_bin = (pred_shw == c)
        gt_bin = (gt_shw == c)
        d = dc(pred_bin, gt_bin) if gt_bin.sum() > 0 or pred_bin.sum() > 0 else 1.0
        dices.append(d)
    return np.mean(dices)


def apply_corruption(raw_image, corruption_fn, severity):
    # raw_image: whatever shape the caller's own pipeline uses -- corruption
    # functions work on any shape via broadcasting, no assumptions here
    if severity == 0.0 or corruption_fn is None:
        return raw_image
    return corruption_fn(raw_image, severity)