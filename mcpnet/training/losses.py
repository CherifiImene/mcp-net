"""
Loss / metric definitions: the ACDC-challenge weighted Dice loss and metrics.

1. `DiceIndex` averages Dice across all 4 classes (background + RV + Myo +
   LV) with equal weight. Background Dice is close to 1.0 almost regardless
   of foreground segmentation quality, since background covers the vast
   majority of pixels in a 128x128 slice — so `val_DiceIndex` (used for
   `ModelCheckpoint(monitor="val_DiceIndex")`) is dominated by a class that
   isn't clinically meaningful. See `DiceIndexForeground` below.
"""

import tensorflow.keras.backend as K
import tensorflow as tf

# Tversky/Focal-Tversky constants — unused by the ablation study itself,
# kept for a possible future loss-function ablation axis.
ALPHA = 0.5
BETA = 0.5
GAMMA = 1


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def dc_per_class(y_pred, y_true, epsilon=0.00001):
    ref = K.flatten(y_true)
    pred = K.flatten(y_pred)

    dc_num = 2 * K.sum(ref * pred) + epsilon
    dc_denom = K.sum(pred) + K.sum(ref) + epsilon

    dc = K.mean(dc_num / dc_denom)
    return dc


def DiceIndex(y_pred, y_true, axis=(0, 1, 2), epsilon=0.00001):
    """Unweighted mean Dice over all 4 classes, including background."""
    dc_bg = dc_per_class(y_pred[:, :, :, 0], y_true[:, :, :, 0], epsilon=epsilon)
    dc_vd = dc_per_class(y_pred[:, :, :, 1], y_true[:, :, :, 1], epsilon=epsilon)
    dc_myo = dc_per_class(y_pred[:, :, :, 2], y_true[:, :, :, 2], epsilon=epsilon)
    dc_vg = dc_per_class(y_pred[:, :, :, 3], y_true[:, :, :, 3], epsilon=epsilon)

    dc = (dc_vd + dc_myo + dc_vg + dc_bg) / 4
    return dc


def DiceIndexForeground(y_pred, y_true, epsilon=0.00001):
    """Mean Dice over RV/Myo/LV only, excluding background — used as the
    `ModelCheckpoint(monitor=...)` metric since it reflects the clinically
    relevant classes rather than being dominated by trivially-easy
    background Dice.
    """
    dc_vd = dc_per_class(y_pred[:, :, :, 1], y_true[:, :, :, 1], epsilon=epsilon)
    dc_myo = dc_per_class(y_pred[:, :, :, 2], y_true[:, :, :, 2], epsilon=epsilon)
    dc_vg = dc_per_class(y_pred[:, :, :, 3], y_true[:, :, :, 3], epsilon=epsilon)
    return (dc_vd + dc_myo + dc_vg) / 3


def TverskyIndex(y_true, y_pred):
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float64)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float64)
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)

    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)

    alpha = 0.3
    smooth = K.epsilon()
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos)


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def loss(y_true, y_pred):
    return diceLoss(y_pred, y_true)


def class_dice_loss(y_pred, y_true, epsilon=0.000001):
    ref = K.flatten(y_true)
    pred = K.flatten(y_pred)

    dc_num = 2 * K.sum(ref * pred) + epsilon
    dc_denom = K.sum(pred ** 2) + K.sum(ref ** 2) + epsilon

    dc_loss = 1 - K.mean(dc_num / dc_denom)
    return dc_loss


def diceLoss(y_pred, y_true, epsilon=0.000001):
    """Weighted multi-class Dice loss, weights per the ACDC-challenge
    organizers' recommendation (0.36 RV / 0.34 Myo / 0.29 LV / 0.01 bg).
    """
    y_t_bg = y_true[:, :, :, 0]
    y_pred_bg = y_pred[:, :, :, 0]

    y_t_rv = y_true[:, :, :, 1]
    y_pred_rv = y_pred[:, :, :, 1]

    y_t_myo = y_true[:, :, :, 2]
    y_pred_myo = y_pred[:, :, :, 2]

    y_t_lv = y_true[:, :, :, 3]
    y_pred_lv = y_pred[:, :, :, 3]

    bg_loss = class_dice_loss(y_true=y_t_bg, y_pred=y_pred_bg, epsilon=epsilon)
    rv_loss = class_dice_loss(y_true=y_t_rv, y_pred=y_pred_rv, epsilon=epsilon)
    myo_loss = class_dice_loss(y_true=y_t_myo, y_pred=y_pred_myo, epsilon=epsilon)
    lv_loss = class_dice_loss(y_true=y_t_lv, y_pred=y_pred_lv, epsilon=epsilon)

    weighted_dc_loss = 0.36 * rv_loss + 0.34 * myo_loss + 0.29 * lv_loss + 0.01 * bg_loss
    return weighted_dc_loss


def TverskyLoss(targets, inputs, alpha=ALPHA, beta=BETA, smooth=1e-6):
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    TP = K.sum((inputs * targets))
    FP = K.sum(((1 - targets) * inputs))
    FN = K.sum((targets * (1 - inputs)))

    Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    return 1 - Tversky


def FocalTverskyLoss(targets, inputs, alpha=ALPHA, beta=BETA, gamma=GAMMA, smooth=1e-6):
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    TP = K.sum((inputs * targets))
    FP = K.sum(((1 - targets) * inputs))
    FN = K.sum((targets * (1 - inputs)))

    Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    FocalTversky = K.pow((1 - Tversky), gamma)
    return FocalTversky
