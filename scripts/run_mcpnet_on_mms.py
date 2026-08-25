"""
run_mcpnet_on_mms.py — saves BOTH the cropped (128x128) prediction AND
the original-size (uncropped) prediction for each case.

The original-size prediction is computed using raw softmax probabilities
(via uncrop_probabilities). Probabilities are
used ONLY transiently, in-memory, during this run -- nothing extra gets
saved to disk beyond the two prediction files per case, same footprint
as before plus one more discrete-label file.

Needs bbox_info.json (written by extract_mms_rois.py) for each case's
bbox/orig_size/margin_px, to know where and how large to paste the
original-size prediction back.

USAGE:
    python run_mcpnet_on_mms.py --config configs/mcp_net_config.yaml \
        --checkpoint checkpoints/full_mcp_net_best.h5 \
        --mms-data-dir /content/mms_extracted_rois/data \
        --bbox-info /content/mms_extracted_rois/bbox_info.json \
        --output-dir-cropped /content/mms_predictions_cropped \
        --output-dir-original /content/mms_predictions_original
Run from the repo root so `mcpnet` is importable.
"""

import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nibabel as nib
import numpy as np

from mcpnet.utils.uncrop_mcpnet_predictions import uncrop_probabilities
from mcpnet.data.preprocessing import standardize
# original, raw M&Ms folders -- used ONLY to fetch each case's true
# spacing/affine for the original-size prediction file
MMS_ORIGINAL_DIRS = ["data/M&Ms/validation", "data/M&Ms/testing"]
IMAGE_SUFFIX = "_sa.nii.gz"


def find_original_image_header(case_id):
    # case_id is "{patient_id}_{phase}" -- strip the phase suffix to get
    # the patient folder name, then search both split dirs for it
    patient_id = case_id.rsplit("_", 1)[0]
    for split_dir in MMS_ORIGINAL_DIRS:
        patient_dir = os.path.join(split_dir, patient_id)
        if not os.path.isdir(patient_dir):
            continue
        image_file = next((f for f in os.listdir(patient_dir) if f.endswith(IMAGE_SUFFIX)), None)
        if image_file is not None:
            return nib.load(os.path.join(patient_dir, image_file)).affine
    return None  # caller falls back to identity affine, with a warning


def run_inference_and_save(config_path, checkpoint_path, mms_data_dir, bbox_info_path,
                            output_dir_cropped, output_dir_original):
    import yaml
    from tensorflow.keras.models import load_model
    from mcpnet.training.losses import DiceIndex, DiceIndexForeground, loss

    custom_objects = {"loss": loss, "DiceIndex": DiceIndex, "DiceIndexForeground": DiceIndexForeground}

    with open(config_path) as f:
        _ = yaml.safe_load(f)  # not strictly needed here, kept for consistency with other scripts

    print(f"Loading checkpoint: {checkpoint_path}")
    model = load_model(checkpoint_path, custom_objects=custom_objects)

    with open(bbox_info_path) as f:
        bbox_info = json.load(f)

    os.makedirs(output_dir_cropped, exist_ok=True)
    os.makedirs(output_dir_original, exist_ok=True)

    case_files = sorted([f for f in os.listdir(mms_data_dir) if f.endswith(".nii.gz")])
    print(f"Found {len(case_files)} M&Ms ROI-extracted cases in {mms_data_dir}")

    n_saved, n_skipped = 0, 0
    for fname in case_files:
        case_id = fname[: -len(".nii.gz")]
        img_path = os.path.join(mms_data_dir, fname)

        if case_id not in bbox_info:
            print(f"  [SKIP] {case_id}: not found in bbox_info.json")
            n_skipped += 1
            continue

        img_nii = nib.load(img_path)
        img_data = img_nii.get_fdata().astype(np.float32)  # (H, W, S)
        spacing = img_nii.header.get_zooms()

        # standardize the image
        img_data = standardize(img_data)
        # (S, H, W, 1) -- one batch element per slice, matching the model's
        # expected input shape
        x = np.transpose(img_data, (2, 0, 1))[..., np.newaxis]

        # RAW probabilities, direct from the model, since we specifically need the
        # pre-argmax softmax values for uncrop_probabilities
        probs = model.predict(x, verbose=0)  # (S, 128, 128, num_classes)
        labels_shw = np.argmax(probs, axis=-1)  # (S, 128, 128)

        # --- save CROPPED (128x128) prediction, same as before ---
        pred_hws = np.transpose(labels_shw, (1, 2, 0)).astype(np.float32)  # (128,128,S)
        affine = np.diag([spacing[0], spacing[1], spacing[2] if len(spacing) > 2 else 1.0, 1.0]).astype(np.float64)
        cropped_out_path = os.path.join(output_dir_cropped, f"{case_id}.nii.gz")
        nib.save(nib.Nifti1Image(pred_hws, affine), cropped_out_path)

        # --- save ORIGINAL-SIZE prediction, using raw probabilities ---
        info = bbox_info[case_id]
        bbox = tuple(info["bbox"])
        orig_size = tuple(info["orig_size"])
        margin_px = info["margin_px"]

        probs_hwsc = np.transpose(probs, (1, 2, 0, 3))  # (128,128,S,num_classes) -- matches
                                                            # uncrop_probabilities' expected input
        original_size_pred = uncrop_probabilities(probs_hwsc, bbox, orig_size, margin_px)

        # spacing for the original-size file: the TRUE original M&Ms
        # image's own affine, looked up directly -- falls back to identity
        # (with a warning, not a crash) if that specific patient's raw
        # file can't be found, so one missing case doesn't halt the batch
        original_affine = find_original_image_header(case_id)
        if original_affine is None:
            print(f"  [WARNING] {case_id}: could not find original M&Ms file for correct spacing, "
                  f"using identity affine instead")
            original_affine = np.eye(4)

        original_out_path = os.path.join(output_dir_original, f"{case_id}.nii.gz")
        nib.save(nib.Nifti1Image(original_size_pred.astype(np.float32), original_affine), original_out_path)

        n_saved += 1
        print(f"  {case_id}: cropped -> {cropped_out_path}, original-size -> {original_out_path}")

    print(f"\nDone. Saved {n_saved} cases (both cropped and original-size), skipped {n_skipped}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mcp_net_config.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/full_mcp_net_best.h5")
    parser.add_argument("--mms-data-dir", type=str, required=True)
    parser.add_argument("--bbox-info", type=str, required=True)
    parser.add_argument("--output-dir-cropped", type=str, default="results/mms_predictions_cropped")
    parser.add_argument("--output-dir-original", type=str, default="results/mms_predictions_original")
    args = parser.parse_args()

    run_inference_and_save(args.config, args.checkpoint, args.mms_data_dir, args.bbox_info,
                            args.output_dir_cropped, args.output_dir_original)