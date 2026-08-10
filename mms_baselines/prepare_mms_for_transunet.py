"""
prepare_mms_for_transunet.py — standalone script (NOT part of the mcpnet
repo). Converts M&Ms images into TransUNet's expected test-volume format
(.npy.h5, matching dataset_acdc.py's non-train-split loading path), at
FULL RESOLUTION — same reasoning as prepare_mms_for_nnunet.py: TransUNet
was trained on full-resolution ACDC, so it needs full-resolution M&Ms for
a fair generalization test, not MCP-Net's ROI-cropped space.

NORMALIZATION: per-slice min-max to [0,1], matching
prepare_acdc_for_transunet.py's convention exactly — consistent with what
TransUNet actually learned to expect during ACDC training.

OUTPUT:
    {OUTPUT_DIR}/test_vol_h5/{case}.npy.h5   — 'image'/'label' keys, (S,H,W)
    {OUTPUT_DIR}/lists_ACDC/test_vol.txt     — list file test.py reads

USES THE SAME M&Ms METADATA CSV as extract_mms_rois.py / prepare_mms_for_nnunet.py.

USAGE (Colab-style, no argparse):
    Edit CONFIG below, then run. Follow with TransUNet's test.py, pointing
    --volume_path / --list_dir at this script's output, and add an 'MMS'
    entry to test.py's dataset_config (same pattern as the 'ACDC' entry).
"""

import os

import h5py
import nibabel as nib
import numpy as np
import pandas as pd

# ============================== CONFIG — EDIT THESE ============================== #
MMS_PATIENT_DIRS = ["data/M&Ms/validation",
                    "data/M&Ms/testing"]
METADATA_CSV_PATH = "data/M&Ms/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"

PATIENT_ID_COL = "External code"
ED_FRAME_COL = "ED"
ES_FRAME_COL = "ES"

IMAGE_SUFFIX = "_sa.nii.gz"
LABEL_SUFFIX = "_sa_gt.nii.gz"

OUTPUT_DIR = "data/transUnet/mms_for_transunet"
# =================================================================================== #


def normalize_slice(img_slice):
    """Per-slice min-max to [0,1] — matches prepare_acdc_for_transunet.py exactly."""
    lo, hi = img_slice.min(), img_slice.max()
    if hi - lo < 1e-8:
        return np.zeros_like(img_slice, dtype=np.float32)
    return ((img_slice - lo) / (hi - lo)).astype(np.float32)


def load_metadata(csv_path):
    df = pd.read_csv(csv_path)
    print(f"Loaded metadata CSV: {csv_path}")
    print(f"Columns found: {list(df.columns)}")

    for col in (PATIENT_ID_COL, ED_FRAME_COL, ES_FRAME_COL):
        if col not in df.columns:
            raise ValueError(f"Configured column '{col}' not found. Actual columns: {list(df.columns)}")

    metadata = {}
    for _, row in df.iterrows():
        pid = str(row[PATIENT_ID_COL]).strip()
        try:
            metadata[pid] = (int(row[ED_FRAME_COL]), int(row[ES_FRAME_COL]))
        except (ValueError, TypeError):
            continue
    return metadata


def main():
    metadata = load_metadata(METADATA_CSV_PATH)

    vol_dir = os.path.join(OUTPUT_DIR, "test_vol_h5")
    list_dir = os.path.join(OUTPUT_DIR, "lists_ACDC")
    os.makedirs(vol_dir, exist_ok=True)
    os.makedirs(list_dir, exist_ok=True)

    vol_names = []
    for split_dir in MMS_PATIENT_DIRS:
        if not os.path.exists(split_dir):
            print(f"WARNING: split dir does not exist, skipping: {split_dir}")
            continue

        patient_ids = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
        for patient_id in patient_ids:
            if patient_id not in metadata:
                print(f"  [SKIP] {patient_id}: not in metadata CSV")
                continue

            ed_frame, es_frame = metadata[patient_id]
            patient_dir = os.path.join(split_dir, patient_id)
            image_file = next((f for f in os.listdir(patient_dir) if f.endswith(IMAGE_SUFFIX)), None)
            label_file = next((f for f in os.listdir(patient_dir) if f.endswith(LABEL_SUFFIX)), None)
            if image_file is None or label_file is None:
                print(f"  [SKIP] {patient_id}: missing image or label file")
                continue

            img4d = nib.load(os.path.join(patient_dir, image_file)).get_fdata()
            gt4d = nib.load(os.path.join(patient_dir, label_file)).get_fdata()

            for phase, frame in [("ed", ed_frame), ("es", es_frame)]:
                img_vol = img4d[:, :, :, frame]  # (H, W, S)
                gt_vol = gt4d[:, :, :, frame]

                img_norm = np.stack([normalize_slice(img_vol[:, :, s]) for s in range(img_vol.shape[2])], axis=0)
                gt_stack = np.stack([gt_vol[:, :, s] for s in range(gt_vol.shape[2])], axis=0).astype(np.uint8)

                vol_name = f"{patient_id}_{phase}"
                with h5py.File(os.path.join(vol_dir, f"{vol_name}.npy.h5"), "w") as hf:
                    hf.create_dataset("image", data=img_norm)
                    hf.create_dataset("label", data=gt_stack)
                vol_names.append(vol_name)

    with open(os.path.join(list_dir, "test_vol.txt"), "w") as f:
        f.write("\n".join(vol_names) + "\n")

    print(f"\nSaved {len(vol_names)} volume(s) to {vol_dir}")
    print(f"List file: {os.path.join(list_dir, 'test_vol.txt')}")
    print(f"\nAdd to test.py's dataset_config:")
    print(f"  'MMS': {{'Dataset': ACDC_dataset, 'volume_path': '{vol_dir}', "
          f"'list_dir': '{list_dir}', 'num_classes': 4, 'z_spacing': 1}}")

