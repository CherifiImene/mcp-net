import json
import os
from tensorflow.keras.models import load_model

import nibabel as nib
import numpy as np
import pandas as pd

from mcpnet.data.acdc_utilities_stub import save_nii
from mcpnet.data.localize_heart_with_model import localize_heart_with_model
# ============================== CONFIG ============================== #
MMS_SPLIT_DIRS = ["data/M&Ms/validation", "data/M&Ms/testing"]
METADATA_CSV_PATH = "data/M&Ms/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"


PATIENT_ID_COL = "External code"
ED_FRAME_COL = "ED"
ES_FRAME_COL = "ES"

IMAGE_SUFFIX = "_sa.nii.gz"
LABEL_SUFFIX = "_sa_gt.nii.gz"

OUTPUT_ROOT = "data/M&Ms/mms_extracted_rois"
BBOX_JSON_PATH = "data/M&Ms/mms_extracted_rois/bbox_info.json"
MARGIN_PX = 15
# M&Ms CSV frame indices are direct array indices, no "-1" needed
# ======================================================================= #

model_path = "checkpoints/heart_localizer_best.h5"
localizer_model = load_model(model_path)

def load_metadata(csv_path):
    df = pd.read_csv(csv_path)
    metadata = {}
    for _, row in df.iterrows():
        pid = str(row[PATIENT_ID_COL]).strip()
        try:
            metadata[pid] = (int(row[ED_FRAME_COL]), int(row[ES_FRAME_COL]))
        except (ValueError, TypeError):
            continue
    return metadata


def process_one_patient(patient_dir, patient_id, ed_frame, es_frame, output_root, bbox_info):
    image_file = next((f for f in os.listdir(patient_dir) if f.endswith(IMAGE_SUFFIX)), None)
    label_file = next((f for f in os.listdir(patient_dir) if f.endswith(LABEL_SUFFIX)), None)
    if image_file is None:
        return {"patient_id": patient_id, "status": "missing_image_file"}

    img_nii = nib.load(os.path.join(patient_dir, image_file))
    img4d = img_nii.get_fdata()
    original_header = img_nii.header
    original_spacing = img_nii.header.get_zooms()[:2]

    gt4d = None
    if label_file is not None:
        gt4d = nib.load(os.path.join(patient_dir, label_file)).get_fdata()

    results = []
    for phase, frame in [("ed", ed_frame), ("es", es_frame)]:
        gt_frame = gt4d[:, :, :, frame] if gt4d is not None else None

        try:
            # final_img, final_gt, bbox, used_fallback, method, scale = localize_heart_robust(
            #     img4d, frame+1, gt=gt_frame, verbose=False
            # )
            final_img, final_gt, bbox, used_fallback, method, scale = localize_heart_with_model(
                      img4d, frame+1, localizer_model, gt=gt_frame
                  )


        except Exception as e:
            results.append({"phase": phase, "status": f"error: {e}"})
            continue

        # new effective spacing, since crop+resize changes it per image
        new_spacing_x = original_spacing[0] / scale
        new_spacing_y = original_spacing[1] / scale
        new_affine = np.diag([new_spacing_x, new_spacing_y, 1.0, 1.0]).astype(np.float64)

        img_out_path = os.path.join(output_root, "data", f"{patient_id}_{phase}.nii.gz")
        os.makedirs(os.path.dirname(img_out_path), exist_ok=True)
        save_nii(img_out_path, final_img.astype(np.float32), new_affine, original_header)

        if final_gt is not None:
            gt_out_path = os.path.join(output_root, "labels", f"{patient_id}_{phase}.nii.gz")
            os.makedirs(os.path.dirname(gt_out_path), exist_ok=True)
            save_nii(gt_out_path, final_gt.astype(np.float32), new_affine, original_header)

        # save everything needed to un-crop later WITHOUT re-running
        # localize_heart_robust
        case_id = f"{patient_id}_{phase}"
        bbox_info[case_id] = {
            "bbox": list(bbox),  # (y1, x1, y2, x2), pre-margin, as returned by localize_heart_robust
            "scale_factor": float(scale),
            "used_fallback": bool(used_fallback),
            "method": method,
            "orig_size": list(img4d.shape[:3]),  # (H, W, S) -- the full canvas to paste back into
            # Saved explicitly so un-cropping never
            # has to guess or hardcode this itself.
            "margin_px": MARGIN_PX,
        }

        results.append({
            "phase": phase, "status": "ok", "used_fallback": used_fallback,
            "method": method, "new_spacing": (new_spacing_x, new_spacing_y),
        })
    print(f"Finished processing patient: {patient_id}")
    return {"patient_id": patient_id, "results": results}


def main():
    metadata = load_metadata(METADATA_CSV_PATH)
    print(f"Loaded ED/ES frame info for {len(metadata)} patients.")

    n_processed, n_skipped, n_fallback = 0, 0, 0
    bbox_info = {}

    for split_dir in MMS_SPLIT_DIRS:
        if not os.path.exists(split_dir):
            print(f"WARNING: split dir does not exist, skipping: {split_dir}")
            continue

        patient_ids = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
        print(f"Processing {len(patient_ids)} patients in {split_dir}...")

        for patient_id in patient_ids:
            if patient_id not in metadata:
                print(f"  [SKIP] {patient_id}: not found in metadata CSV")
                n_skipped += 1
                continue

            ed_frame, es_frame = metadata[patient_id]
            patient_dir = os.path.join(split_dir, patient_id)
            log = process_one_patient(patient_dir, patient_id, ed_frame, es_frame, OUTPUT_ROOT, bbox_info)

            if "results" in log:
                n_processed += 1
                for r in log["results"]:
                    if r.get("used_fallback"):
                        n_fallback += 1

    os.makedirs(os.path.dirname(BBOX_JSON_PATH), exist_ok=True)
    with open(BBOX_JSON_PATH, "w") as f:
        json.dump(bbox_info, f, indent=2)

    print(f"\nPatients processed: {n_processed}, skipped: {n_skipped}, "
          f"frames using fallback: {n_fallback}")
    print(f"Bbox info for {len(bbox_info)} cases written to {BBOX_JSON_PATH}")
