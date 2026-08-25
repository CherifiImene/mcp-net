import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # mcpnet root folder

import nibabel as nib
import numpy as np
import pandas as pd
from medpy.metric import dc

from mcpnet.utils.image_corruptions import add_gaussian_noise, add_rician_noise, add_bias_field, add_motion_blur

# ============================== CONFIG ============================== #
# already-cropped (128x128) ACDC files
ACDC_CROPPED_DATA_DIR = "./data/mcpnet/cropped_images"
ACDC_CROPPED_LABELS_DIR = "./data/mcpnet/cropped_labels"
CONFIGS_JSON = "data/configv2.json"
CHECKPOINT_PATH = "checkpoints/full_mcp_net_best.h5"

RESULTS_DIR = "results/robustness_testing"

CORRUPTIONS = {
    "gaussian_noise": (add_gaussian_noise, [0.0, 0.05, 0.10, 0.15, 0.20]),
    "rician_noise": (add_rician_noise, [0.0, 0.05, 0.10, 0.15, 0.20]),
    "bias_field": (add_bias_field, [0.0, 0.10, 0.20, 0.30, 0.40]),
    "motion_blur": (add_motion_blur, [0.0, 0.20, 0.40, 0.60, 0.80]),
}
# ======================================================================= #


def standardize_single_phase(image):
    standardized = np.zeros(image.shape)
    for c in range(image.shape[2]):
        image_slice = image[:, :, c]
        centered = image_slice - np.mean(image)
        if np.std(centered) != 0:
            standardized[:, :, c] = centered / np.std(centered)
        else:
            standardized[:, :, c] = centered
    return standardized


def load_test_case_filenames():
    with open(CONFIGS_JSON) as f:
        split = json.load(f)
    # test list entries reference the pre-cropped files directly
    return split["test"]


def compute_dice_per_case(model, raw_image, gt, corruption_fn, severity):
    corrupted = raw_image if severity == 0.0 and corruption_fn is None else corruption_fn(raw_image, severity)
    standardized = standardize_single_phase(corrupted)
    x = np.transpose(standardized, (2, 0, 1))[..., np.newaxis]  # (S,128,128,1)
    probs = model.predict(x, verbose=0)
    pred = np.argmax(probs, axis=-1)  # (S,128,128)
    gt_shw = np.transpose(gt, (2, 0, 1))

    dices = []
    for c in [1, 2, 3]:
        pred_bin = (pred == c)
        gt_bin = (gt_shw == c)
        d = dc(pred_bin, gt_bin) if gt_bin.sum() > 0 or pred_bin.sum() > 0 else 1.0
        dices.append(d)
    return np.mean(dices)


def main():
    from tensorflow.keras.models import load_model
    from mcpnet.training.losses import DiceIndex, DiceIndexForeground, loss

    custom_objects = {"loss": loss, "DiceIndex": DiceIndex, "DiceIndexForeground": DiceIndexForeground}
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    model = load_model(CHECKPOINT_PATH, custom_objects=custom_objects)

    test_filenames = load_test_case_filenames()
    print(f"{len(test_filenames)} test cases (already-cropped files)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []

    for corruption_name, (corruption_fn, severities) in CORRUPTIONS.items():
        print(f"\n=== {corruption_name} ===")
        for severity in severities:
            case_dices = []
            for fname in test_filenames:
                img_path = os.path.join(ACDC_CROPPED_DATA_DIR, fname + '.gz')
                gt_path = os.path.join(ACDC_CROPPED_LABELS_DIR, fname + '.gz')
                if not (os.path.exists(img_path) and os.path.exists(gt_path)):
                    print(f"  [SKIP] {fname}: not found in cropped {img_path} dirs")
                    continue

                raw_image = nib.load(img_path).get_fdata()  # already (128,128,S)
                gt = nib.load(gt_path).get_fdata()

                mean_dice = compute_dice_per_case(
                    model, raw_image, gt,
                    corruption_fn if severity > 0 else None, severity
                )
                case_dices.append(mean_dice)
                rows.append({"corruption": corruption_name, "severity": severity,
                             "case": fname, "mean_dice": mean_dice})

            print(f"  severity={severity}: mean_dice={np.mean(case_dices):.4f} (n={len(case_dices)})")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "robustness_per_case.csv"), index=False)

    summary = df.groupby(["corruption", "severity"])["mean_dice"].agg(["mean", "std"])
    summary.to_csv(os.path.join(RESULTS_DIR, "robustness_summary.csv"))
    print(f"\nSaved to {RESULTS_DIR}/")
    print(summary)


if __name__ == "__main__":
    main()