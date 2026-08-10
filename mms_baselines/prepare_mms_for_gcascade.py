import os

import nibabel as nib
import numpy as np
import pandas as pd

# ============================== CONFIG ============================== #
MMS_SPLIT_DIRS = ["data/M&Ms/validation", 'data/M&Ms/testing']
METADATA_CSV_PATH = "data/M&Ms/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"

PATIENT_ID_COL, ED_FRAME_COL, ES_FRAME_COL = "External code", "ED", "ES"
IMAGE_SUFFIX, LABEL_SUFFIX = "_sa.nii.gz", "_sa_gt.nii.gz"

#
VOLUME_PATH_OUTPUT = "data/GCASCADE/mms_for_gcascade_test"
LIST_DIR_OUTPUT = "data/GCASCADE/mms_for_gcascade_test/lists_ACDC"
# =================================================================================== #


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


def main():
    metadata = load_metadata(METADATA_CSV_PATH)
    os.makedirs(VOLUME_PATH_OUTPUT, exist_ok=True)
    os.makedirs(LIST_DIR_OUTPUT, exist_ok=True)

    vol_names = []
    for split_dir in MMS_SPLIT_DIRS:
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

            # M&Ms CSV frame indices are DIRECT array indices, no "-1"
            # (confirmed earlier in this project -- different convention
            # from ACDC's Info.cfg)
            for phase, frame in [("ed", ed_frame), ("es", es_frame)]:
                img_vol = img4d[:, :, :, frame]  # (H, W, S)
                gt_vol = np.round(gt4d[:, :, :, frame])  # round -- some M&Ms gt
                                                            # files have near-integer
                                                            # floating point drift
                                                            # (confirmed earlier),
                                                            # would corrupt exact
                                                            # class handling downstream

                # G-CASCADE's confirmed format: 'img'/'label' keys (NOT
                # 'image'/'label' like TransUNet), full volume as (S,H,W)
                # to match how ACDCdataset's test split loads it
                img_shw = np.transpose(img_vol, (2, 0, 1)).astype(np.float32)
                gt_shw = np.transpose(gt_vol, (2, 0, 1)).astype(np.uint8)

                vol_name = f"{patient_id}_{phase}"
                np.savez(os.path.join(VOLUME_PATH_OUTPUT, f"{vol_name}.npz"),
                          img=img_shw, label=gt_shw)
                # list entry needs the extension -- ACDCdataset uses list
                # entries DIRECTLY as filenames, no extension appended
                # (confirmed bug/fix from the earlier ACDC-side G-CASCADE prep)
                vol_names.append(f"{vol_name}.npz")

    with open(os.path.join(LIST_DIR_OUTPUT, "test.txt"), "w") as f:
        f.write("\n".join(vol_names) + "\n")

    print(f"\nSaved {len(vol_names)} volume(s) to {VOLUME_PATH_OUTPUT}")
    print(f"List file: {os.path.join(LIST_DIR_OUTPUT, 'test.txt')}")
    print(f"\nRun test_ACDC.py with:")
    print(f"  --volume_path {VOLUME_PATH_OUTPUT} --list_dir {LIST_DIR_OUTPUT}")
    print("(match every other flag -- encoder, batch_size, lr, max_epochs, img_size, "
          "seed, save_path -- EXACTLY to your training command, same as the ACDC run)")
