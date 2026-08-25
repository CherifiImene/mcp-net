import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import nibabel as nib
import numpy as np
import pandas as pd
import tensorflow as tf

from mcpnet.utils.bbox_from_mask import extract_bbox_from_mask

# ============================== CONFIG ============================== #
ACDC_RAW_DIR = "data/ACDC/raw"
MMS_TRAIN_DIR = "data/M&Ms/training"
MMS_METADATA_CSV = "data/M&Ms/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"
MMS_PATIENT_ID_COL, MMS_ED_COL, MMS_ES_COL = "External code", "ED", "ES"
MMS_IMAGE_SUFFIX, MMS_LABEL_SUFFIX = "_sa.nii.gz", "_sa_gt.nii.gz"

TRAIN_INPUT_SIZE = (256, 256)  # baked in at prep time -- change here + rerun
                                 # this script if you ever want a different size

SPLIT_SEED = 42
VAL_FRACTION = 0.15
OUTPUT_ROOT = "data/localizer_dataset/"  # writes train/ and val/ subfolders here
SPLIT_INFO_PATH = "configs/heart_localizer_split.json"
MIN_BBOX_SIZE_FRACTION = 0.25  # minimum bbox height/width, as a fraction of
                                  # the ORIGINAL image's own H/W (computed
                                  # separately per dimension) -- enforced on
                                  # the training TARGETS directly, so the
                                  # model learns to never predict a box
                                  # smaller than this, rather than relying
                                  # on a fixed margin added after the fact
MARGIN_FRACTION = 0.1  # unconditional buffer added to EVERY box (not just
                          # small ones) -- so even an adequately-sized but
                          # tightly-fitted box has slack, meaning a
                          # slightly-off prediction still captures the
                          # whole heart

# ======================================================================= #

def find_acdc_patient_dirs():
    dirs = []
    for subfolder in ("training", "testing"):
        root = os.path.join(ACDC_RAW_DIR, subfolder)
        if not os.path.isdir(root):
            continue
        for pid in sorted(os.listdir(root)):
            candidate = os.path.join(root, pid)
            if os.path.isdir(candidate):
                dirs.append((pid, candidate))
    return dirs


def parse_acdc_info_cfg(patient_dir):
    ed, es = None, None
    with open(os.path.join(patient_dir, "Info.cfg")) as f:
        for line in f:
            key, _, value = line.strip().partition(":")
            if key.strip().upper() == "ED":
                ed = int(value.strip())
            elif key.strip().upper() == "ES":
                es = int(value.strip())
    return ed, es


def build_dataset_from_acdc():
    entries = []
    for patient_id, patient_dir in find_acdc_patient_dirs():
        ed_frame, es_frame = parse_acdc_info_cfg(patient_dir)
        for phase, frame in [("ed", ed_frame), ("es", es_frame)]:
            img_path = os.path.join(patient_dir, f"{patient_id}_frame{frame:02d}.nii.gz")
            gt_path = os.path.join(patient_dir, f"{patient_id}_frame{frame:02d}_gt.nii.gz")
            if not (os.path.exists(img_path) and os.path.exists(gt_path)):
                continue
            n_slices = nib.load(img_path).shape[2]
            for s in range(n_slices):
                entries.append({
                    "source": "acdc", "img_path": img_path, "gt_path": gt_path,
                    "frame_4d_index": None, "slice_idx": s,
                    "source_id": f"ACDC_{patient_id}_{phase}_s{s}",
                })
    return entries


def load_mms_metadata():
    df = pd.read_csv(MMS_METADATA_CSV)
    metadata = {}
    for _, row in df.iterrows():
        pid = str(row[MMS_PATIENT_ID_COL]).strip()
        try:
            metadata[pid] = (int(row[MMS_ED_COL]), int(row[MMS_ES_COL]))
        except (ValueError, TypeError):
            continue
    return metadata


def build_dataset_from_mms():
    if not os.path.isdir(MMS_TRAIN_DIR):
        print(f"WARNING: {MMS_TRAIN_DIR} not found, skipping M&Ms data.")
        return []

    metadata = load_mms_metadata()
    entries = []
    patient_ids = sorted([d for d in os.listdir(MMS_TRAIN_DIR) if os.path.isdir(os.path.join(MMS_TRAIN_DIR, d))])

    for patient_id in patient_ids:
        if patient_id not in metadata:
            continue
        ed_frame, es_frame = metadata[patient_id]
        patient_dir = os.path.join(MMS_TRAIN_DIR, patient_id)
        image_file = next((f for f in os.listdir(patient_dir) if f.endswith(MMS_IMAGE_SUFFIX)), None)
        label_file = next((f for f in os.listdir(patient_dir) if f.endswith(MMS_LABEL_SUFFIX)), None)
        if image_file is None or label_file is None:
            continue

        img_path = os.path.join(patient_dir, image_file)
        gt_path = os.path.join(patient_dir, label_file)
        n_slices = nib.load(img_path).shape[2]

        for phase, frame in [("ed", ed_frame), ("es", es_frame)]:
            for s in range(n_slices):
                entries.append({
                    "source": "mms", "img_path": img_path, "gt_path": gt_path,
                    "frame_4d_index": frame, "slice_idx": s,
                    "source_id": f"MMS_{patient_id}_{phase}_s{s}",
                })
    return entries


def resize_and_normalize(image_slice, target_size):
    img = tf.convert_to_tensor(image_slice[..., None], dtype=tf.float32)
    img = tf.image.resize(img, target_size).numpy()
    img_min, img_max = img.min(), img.max()
    return (img - img_min) / (img_max - img_min + 1e-8)


def process_and_save(entries, split_name, output_root):
    split_dir = os.path.join(output_root, split_name)
    os.makedirs(split_dir, exist_ok=True)

    by_source = {}
    for entry in entries:
        key = (entry["img_path"], entry["gt_path"], entry["frame_4d_index"])
        by_source.setdefault(key, []).append(entry)

    n_saved, n_skipped_empty = 0, 0
    for i, (key, slice_entries) in enumerate(by_source.items()):
        if (i + 1) % 20 == 0:
            print(f"  [{split_name}] {i+1}/{len(by_source)} source files processed...", flush=True)

        img_path, gt_path, frame_4d_index = key
        img_data = nib.load(img_path).get_fdata()
        gt_data = nib.load(gt_path).get_fdata()
        if frame_4d_index is not None:
            img_data = img_data[:, :, :, frame_4d_index]
            gt_data = np.round(gt_data[:, :, :, frame_4d_index])

        # FIXED: previously each slice used its OWN, locally-computed bbox
        # as its training target -- but inference only looks at ONE
        # representative (middle) slice and applies that single prediction
        # to the WHOLE stack. Since heart cross-section size varies a lot
        # by slice level (base slices are typically largest, apex slices
        # can be tiny or miss the RV entirely), a small/local bbox target
        # trains the model to under-predict relative to what's actually
        # needed to cover the whole stack. Fixed: find the slice with the
        # LARGEST foreground area in this (patient, phase), and use ITS
        # bbox as the shared target for EVERY slice sample from this same
        # phase -- so regardless of which slice the model happens to see,
        # it learns to predict the whole-phase-covering box.
        best_bbox, best_area = None, -1
        for entry in slice_entries:
            s = entry["slice_idx"]
            gt_slice = gt_data[:, :, s]
            area = (gt_slice > 0).sum()
            if area > best_area:
                bbox = extract_bbox_from_mask(gt_slice, min_size_fraction=MIN_BBOX_SIZE_FRACTION,
                                               margin_fraction=MARGIN_FRACTION)
                if bbox is not None:
                    best_area = area
                    best_bbox = bbox

        if best_bbox is None:
            n_skipped_empty += len(slice_entries)
            continue  # this entire (patient, phase) has no foreground anywhere, skip all its slices

        for entry in slice_entries:
            s = entry["slice_idx"]
            img_slice = resize_and_normalize(img_data[:, :, s], TRAIN_INPUT_SIZE)
            out_path = os.path.join(split_dir, f"{entry['source_id']}.npz")
            np.savez(out_path, image=img_slice.astype(np.float32), bbox=best_bbox.astype(np.float32))
            n_saved += 1

    print(f"{split_name}: saved {n_saved} files, skipped {n_skipped_empty} slices from all-empty phases")
    return n_saved


def main():
    acdc_entries = build_dataset_from_acdc()
    mms_entries = build_dataset_from_mms()
    all_entries = acdc_entries + mms_entries
    print(f"Gathered metadata: {len(acdc_entries)} ACDC + {len(mms_entries)} M&Ms = {len(all_entries)} total entries")

    
    group_keys = {}
    for entry in all_entries:
        key = (entry["img_path"], entry["gt_path"], entry["frame_4d_index"])
        group_keys.setdefault(key, []).append(entry)

    keys_list = list(group_keys.keys())
    rng = np.random.RandomState(SPLIT_SEED)
    indices = np.arange(len(keys_list))
    rng.shuffle(indices)
    n_val_groups = int(len(indices) * VAL_FRACTION)
    val_keys = {keys_list[i] for i in indices[:n_val_groups]}

    train_entries, val_entries = [], []
    for key, group_entries in group_keys.items():
        if key in val_keys:
            val_entries.extend(group_entries)
        else:
            train_entries.extend(group_entries)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    split_info = {
        "seed": SPLIT_SEED, "val_fraction": VAL_FRACTION,
        "train_ids": [e["source_id"] for e in train_entries],
        "val_ids": [e["source_id"] for e in val_entries],
    }
    with open(SPLIT_INFO_PATH, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"Split (seed={SPLIT_SEED}): {len(keys_list)-n_val_groups} train groups / {n_val_groups} val groups "
          f"({len(train_entries)} train slices, {len(val_entries)} val slices)")

    process_and_save(train_entries, "train", OUTPUT_ROOT)
    process_and_save(val_entries, "val", OUTPUT_ROOT)

    print(f"\nDone. Prepared dataset saved to {OUTPUT_ROOT}/train and {OUTPUT_ROOT}/val")
    print("Run train_heart_localizer.py next -- it just reads these files directly, "
          "no need to re-scan ACDC/M&Ms or recompute anything.")


if __name__ == "__main__":
    main()