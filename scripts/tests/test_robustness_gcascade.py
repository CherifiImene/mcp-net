import sys
sys.path.insert(0, ".")  # G-CASCADE repo root

import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from mcpnet.utils.image_corruptions import add_gaussian_noise, add_rician_noise, add_bias_field, add_motion_blur
from mcpnet.utils.robustness_core import compute_mean_dice, apply_corruption

# ============================== CONFIG ============================== #
ACDC_RAW_DIR = "../data/ACDC_Segmentation"
CONFIGS_JSON = "../data/configv2.json"

ENCODER = "PVT"
NUM_CLASSES = 4
IMG_SIZE = 224
SKIP_AGGREGATION = "additive"
CHECKPOINT_PATH = "model_pth/ACDC/PVT_GCASCADE_MUTATION_w3_7_Run1_224/PVT_GCASCADE_MUTATION_w3_7_Run1_pretrain_epo400_bs12_lr0.0001_224_s5/best.pth"

RESULTS_DIR = "../results/robustness_testing_gcascade"
CORRUPTIONS = {
    "gaussian_noise": (add_gaussian_noise, [0.0, 0.05, 0.10, 0.15, 0.20]),
    "rician_noise": (add_rician_noise, [0.0, 0.05, 0.10, 0.15, 0.20]),
    "bias_field": (add_bias_field, [0.0, 0.10, 0.20, 0.30, 0.40]),
    "motion_blur": (add_motion_blur, [0.0, 0.20, 0.40, 0.60, 0.80]),
}
# ======================================================================= #


def build_model():
    from lib.networks import PVT_GCASCADE
    model = PVT_GCASCADE(n_class=NUM_CLASSES, img_size=IMG_SIZE, k=11, padding=5,
                          conv="mr", gcb_act="gelu", skip_aggregation=SKIP_AGGREGATION)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


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


def predict_one_case(model, raw_image, device):
    # per-slice min-max [0,1], matching this project's established
    # G-CASCADE/TransUNet-family preprocessing convention
    S = raw_image.shape[2]
    preds = np.zeros((S, raw_image.shape[0], raw_image.shape[1]), dtype=np.int32)
    for s in range(S):
        img_slice = raw_image[:, :, s]
        img_min, img_max = img_slice.min(), img_slice.max()
        normalized = (img_slice - img_min) / (img_max - img_min + 1e-8)

        H, W = normalized.shape
        x = torch.from_numpy(normalized).float().unsqueeze(0).unsqueeze(0)
        x_resized = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        x_resized = x_resized.to(device)

        with torch.no_grad():
            output = model(x_resized)
            # CONFIRMED from G-CASCADE's own utils/utils.py test_single_volume:
            # `outputs = 0.0; for idx in range(len(P)): outputs += P[idx]` --
            # sums ALL FOUR multi-scale outputs, not just the first one.
            # An earlier version of this script incorrectly used only
            # output[0], which would silently discard p2/p3/p4 and diverge
            # from G-CASCADE's actual, real inference behavior.
            if isinstance(output, (list, tuple)):
                summed = sum(output)
            else:
                summed = output
            pred_224 = torch.argmax(torch.softmax(summed, dim=1), dim=1).squeeze(0)

        pred_resized = F.interpolate(
            pred_224.unsqueeze(0).unsqueeze(0).float(), size=(H, W), mode="nearest"
        ).squeeze().cpu().numpy().astype(np.int32)
        preds[s] = pred_resized
    return preds


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)

    test_patients = load_test_patients()
    print(f"{len(test_patients)} test patients")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []

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

                    raw_image = nib.load(img_path).get_fdata()
                    gt = nib.load(gt_path).get_fdata()

                    corrupted = apply_corruption(raw_image, corruption_fn, severity)
                    pred_shw = predict_one_case(model, corrupted, device)
                    gt_shw = np.transpose(gt, (2, 0, 1))

                    mean_dice = compute_mean_dice(pred_shw, gt_shw)
                    case_dices.append(mean_dice)
                    rows.append({"corruption": corruption_name, "severity": severity,
                                 "patient_id": patient_id, "phase": phase, "mean_dice": mean_dice})

            print(f"  severity={severity}: mean_dice={np.mean(case_dices):.4f} (n={len(case_dices)})")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "robustness_per_case_gcascade.csv"), index=False)
    summary = df.groupby(["corruption", "severity"])["mean_dice"].agg(["mean", "std"])
    summary.to_csv(os.path.join(RESULTS_DIR, "robustness_summary_gcascade.csv"))
    print(f"\nSaved to {RESULTS_DIR}/")
    print(summary)


if __name__ == "__main__":
    main()