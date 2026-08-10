import os

import nibabel as nib
import numpy as np
import pandas as pd

# ============================== CONFIG — EDIT THESE ============================== #
MMS_PATIENT_DIRS = ["data/M&Ms/validation",
                    "data/M&Ms/testing"]  # storage-limited scope

METADATA_CSV_PATH = "data/M&Ms/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"

# CONFIRM against your real CSV's printed columns (see extract_mms_rois.py's
# earlier dry-run output) before trusting these.
PATIENT_ID_COL = "External code"
ED_FRAME_COL = "ED"
ES_FRAME_COL = "ES"

IMAGE_SUFFIX = "_sa.nii.gz"
LABEL_SUFFIX = "_sa_gt.nii.gz"

OUTPUT_IMAGES_DIR = "data/nnUnet/mms_for_nnunet/images"
OUTPUT_LABELS_DIR = "data/nnUnet/mms_for_nnunet/labels"
# =================================================================================== #

def load_metadata(csv_path):
    df = pd.read_csv(csv_path)
    print(f"Loaded metadata CSV: {csv_path}")
    print(f"Columns found: {list(df.columns)}")

    for col in (PATIENT_ID_COL, ED_FRAME_COL, ES_FRAME_COL):
        if col not in df.columns:
            raise ValueError(
                f"Configured column '{col}' not found in CSV. Actual columns: {list(df.columns)}"
            )

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
    os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

    n_saved, n_skipped = 0, 0
    for split_dir in MMS_PATIENT_DIRS:
        if not os.path.exists(split_dir):
            print(f"WARNING: split dir does not exist, skipping: {split_dir}")
            continue

        patient_ids = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
        for patient_id in patient_ids:
            if patient_id not in metadata:
                print(f"  [SKIP] {patient_id}: not in metadata CSV")
                n_skipped += 1
                continue

            ed_frame, es_frame = metadata[patient_id]
            patient_dir = os.path.join(split_dir, patient_id)
            image_file = next((f for f in os.listdir(patient_dir) if f.endswith(IMAGE_SUFFIX)), None)
            label_file = next((f for f in os.listdir(patient_dir) if f.endswith(LABEL_SUFFIX)), None)
            if image_file is None or label_file is None:
                print(f"  [SKIP] {patient_id}: missing image or label file")
                n_skipped += 1
                continue

            img_nii = nib.load(os.path.join(patient_dir, image_file))
            img4d = img_nii.get_fdata()  # (H, W, S, T)

            # CONFIRMED BUG FIX: M&Ms's multi-vendor NIfTI files can have
            # non-orthonormal affine matrices (some vendors' acquisition
            # geometry produces small shear/oblique components). nibabel
            # tolerates this fine (confirmed — this is why it never showed
            # up as a problem anywhere else in this pipeline), but nnU-Net's
            # internal reader uses SimpleITK, which is STRICT and crashes
            # with "ITK ERROR: ITK only supports orthonormal direction
            # cosines" — confirmed by reproducing this exact error with a
            # deliberately non-orthonormal affine. Fixed by rebuilding a
            # clean, orthonormal DIAGONAL affine from spacing alone
            # (header.get_zooms() works regardless of the original affine's
            # orthonormality) — safe because this project's entire
            # evaluation pipeline only ever reads spacing scalars, never
            # orientation/rotation, so nothing relevant is lost.
            spacing = img_nii.header.get_zooms()
            affine = np.diag([spacing[0], spacing[1], spacing[2] if len(spacing) >= 3 else 1.0, 1.0]).astype(np.float64)

            gt_nii = nib.load(os.path.join(patient_dir, label_file))
            gt4d = gt_nii.get_fdata()

            for phase, frame in [("ed", ed_frame), ("es", es_frame)]:
                img_vol = img4d[:, :, :, frame - 1].astype(np.float32)  # (H, W, S), RAW intensities, no normalization
                gt_vol = gt4d[:, :, :, frame - 1].astype(np.float32)

                case_id = f"{patient_id}_{phase}"
                # nnU-Net's REQUIRED naming: {case}_0000.nii.gz (4-digit channel suffix)
                nib.save(nib.Nifti1Image(img_vol, affine),
                          os.path.join(OUTPUT_IMAGES_DIR, f"{case_id}_0000.nii.gz"))
                nib.save(nib.Nifti1Image(gt_vol, affine),
                          os.path.join(OUTPUT_LABELS_DIR, f"{case_id}.nii.gz"))
                n_saved += 1

    print(f"\nSaved {n_saved} case(s), skipped {n_skipped}.")
    print(f"Images (for nnUNetv2_predict -i): {OUTPUT_IMAGES_DIR}")
    print(f"Labels (for evaluation afterward): {OUTPUT_LABELS_DIR}")
    print(f"\nRun: nnUNetv2_predict -i {OUTPUT_IMAGES_DIR} -o <predictions_dir> -d 27 -c 2d -f 0 -tr nnUNetTrainerSeeded")

