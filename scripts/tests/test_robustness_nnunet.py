import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # mcpnet root

import numpy as np
import pandas as pd
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

from mcpnet.utils.image_corruptions import add_gaussian_noise, add_rician_noise, add_bias_field, add_motion_blur
from mcpnet.utils.robustness_core import compute_mean_dice, apply_corruption

# ============================== CONFIG ============================== #
ACDC_RAW_DIR = "data/ACDC_Segmentation"
CONFIGS_JSON = "data/configv2.json"
MODEL_TRAINING_OUTPUT_DIR = "data/nnUnet/Dataset027_ACDC/nnUNetTrainerSeeded_250epochs__nnUNetPlans__2d"

RESULTS_DIR = "results/robustness_testing_nnunet"
CORRUPTIONS = {
    "gaussian_noise": (add_gaussian_noise, [0.0, 0.05, 0.10, 0.15, 0.20]),
    "rician_noise": (add_rician_noise, [0.0, 0.05, 0.10, 0.15, 0.20]),
    "bias_field": (add_bias_field, [0.0, 0.10, 0.20, 0.30, 0.40]),
    "motion_blur": (add_motion_blur, [0.0, 0.20, 0.40, 0.60, 0.80]),
}
# ======================================================================= #


def build_predictor():
    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=MODEL_TRAINING_OUTPUT_DIR,
        use_folds=(0,), checkpoint_name="checkpoint_final.pth",
    )
    return predictor


def load_test_patients():
    import json
    with open(CONFIGS_JSON) as f:
        split = json.load(f)
    return sorted({n.split("_")[0].replace(".nii", "") for n in split["test"]})


def find_patient_dir(patient_id):
    for subfolder in ("training", "testing"):
        candidate = os.path.join(ACDC_RAW_DIR, subfolder, patient_id)
        if os.path.isdir(candidate):
            return candidate
    return None


def parse_info_cfg(patient_dir):
    ed, es = None, None
    with open(os.path.join(patient_dir, "Info.cfg")) as f:
        for line in f:
            key, _, value = line.strip().partition(":")
            if key.strip().upper() == "ED":
                ed = int(value.strip())
            elif key.strip().upper() == "ES":
                es = int(value.strip())
    return ed, es


def main():
    predictor = build_predictor()
    test_patients = load_test_patients()
    print(f"{len(test_patients)} test patients")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    reader = SimpleITKIO()

    for corruption_name, (corruption_fn, severities) in CORRUPTIONS.items():
        print(f"\n=== {corruption_name} ===")
        for severity in severities:
            case_dices = []
            for patient_id in test_patients:
                patient_dir = find_patient_dir(patient_id)
                if patient_dir is None:
                    continue
                ed_frame, es_frame = parse_info_cfg(patient_dir)

                for phase, frame in [("ed", ed_frame), ("es", es_frame)]:
                    img_path = os.path.join(patient_dir, f"{patient_id}_frame{frame:02d}.nii.gz")
                    gt_path = os.path.join(patient_dir, f"{patient_id}_frame{frame:02d}_gt.nii.gz")
                    if not (os.path.exists(img_path) and os.path.exists(gt_path)):
                        continue

                    # CONFIRMED real API (nnunetv2/inference/examples.py): read
                    # via SimpleITKIO for correctly-formatted image + properties,
                    # then predict_single_npy_array handles nnU-Net's own
                    # internal (plans-based) preprocessing automatically --
                    # apply corruption to the RAW array BEFORE this call, same
                    # principle as the other two scripts (corrupt the raw
                    # acquisition, let the model's own real preprocessing run
                    # on top of that, exactly as it would with a genuinely
                    # noisier scan)
                    img_arr, props = reader.read_images([img_path])  # (1, S, H, W)
                    gt_arr, _ = reader.read_images([gt_path])

                    # nnU-Net's reader returns (channel, S, H, W) -- completely different
                    # axis order from the (H,W,S) convention the shared
                    # corruption functions expect.
                    # Transpose to (H,W,S) for corruption, then
                    # back, rather than assume shape[0]/shape[1] are H,W.
                    img_hws = np.transpose(img_arr[0], (1, 2, 0))  # (H,W,S)
                    corrupted_hws = apply_corruption(img_hws, corruption_fn, severity)
                    corrupted = np.transpose(corrupted_hws, (2, 0, 1))[np.newaxis, ...]  # back to (1,S,H,W)

                    pred = predictor.predict_single_npy_array(corrupted, props, None, None, False)

                    mean_dice = compute_mean_dice(np.squeeze(pred), np.squeeze(gt_arr))
                    case_dices.append(mean_dice)
                    rows.append({"corruption": corruption_name, "severity": severity,
                                 "patient_id": patient_id, "phase": phase, "mean_dice": mean_dice})

            print(f"  severity={severity}: mean_dice={np.mean(case_dices):.4f} (n={len(case_dices)})")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "robustness_per_case_nnunet.csv"), index=False)
    summary = df.groupby(["corruption", "severity"])["mean_dice"].agg(["mean", "std"])
    summary.to_csv(os.path.join(RESULTS_DIR, "robustness_summary_nnunet.csv"))
    print(f"\nSaved to {RESULTS_DIR}/")
    print(summary)


if __name__ == "__main__":
    main()