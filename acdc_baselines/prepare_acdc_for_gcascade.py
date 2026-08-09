import json
import os

import numpy as np

try:
    from acdc_utilities import load_nii
except ImportError:
    import nibabel as nib
    def load_nii(path):
        img = nib.load(path)
        return img.get_fdata(), img.affine, img.header

# ============================== CONFIG — EDIT THESE ============================== #
ACDC_RAW_DIR = "data/ACDC/raw"          # original 'training'/'testing' folder structure
CONFIGS_JSON = "data/acdc_split.json"  # same split as everywhere else

ROOT_DIR_OUTPUT = "data/GCASCADE/ACDC_gcascade"       # train/ + valid/ subfolders go here
VOLUME_PATH_OUTPUT = "data/GCASCADE/ACDC_gcascade_test"  # test volumes go here (SEPARATE tree)
LIST_DIR_OUTPUT = "data/GCASCADE/ACDC_gcascade/lists_ACDC"
# =================================================================================== #


def normalize_slice(img_slice):
    """Per-slice min-max to [0,1] — matches prepare_acdc_for_transunet.py's
    convention, kept consistent across all baseline-preparation scripts in
    this project rather than introducing a third normalization scheme.
    """
    lo, hi = img_slice.min(), img_slice.max()
    if hi - lo < 1e-8:
        return np.zeros_like(img_slice, dtype=np.float32)
    return ((img_slice - lo) / (hi - lo)).astype(np.float32)


def load_patient_split(configs_path):
    with open(configs_path) as f:
        split = json.load(f)

    def patient_ids(names):
        return sorted({n.split("_")[0].replace(".nii", "")[: len("patient000")] for n in names})

    return {k: patient_ids(v) for k, v in split.items()}


def find_patient_dir(patient_id):
    for subfolder in ("training", "testing"):
        candidate = os.path.join(ACDC_RAW_DIR, subfolder, patient_id)
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(f"Could not find {patient_id} under {ACDC_RAW_DIR}/training or /testing")


def parse_info_cfg(patient_dir):
    ed, es = None, None
    with open(os.path.join(patient_dir, "Info.cfg")) as f:
        for line in f:
            key, _, value = line.strip().partition(":")
            if key.strip().upper() == "ED":
                ed = int(value.strip())
            elif key.strip().upper() == "ES":
                es = int(value.strip())
    if ed is None or es is None:
        raise ValueError(f"Could not parse ED/ES from {patient_dir}/Info.cfg")
    return ed, es


def find_frame_files(patient_dir, patient_id, frame_num):
    for ext in (".nii.gz", ".nii"):
        img_path = os.path.join(patient_dir, f"{patient_id}_frame{frame_num:02d}{ext}")
        gt_path = os.path.join(patient_dir, f"{patient_id}_frame{frame_num:02d}_gt{ext}")
        if os.path.exists(img_path):
            return img_path, (gt_path if os.path.exists(gt_path) else None)
    raise FileNotFoundError(f"Could not find frame {frame_num:02d} for {patient_id} in {patient_dir}")


def write_slices(patients, out_dir, list_path):
    """Used for BOTH train and valid — G-CASCADE treats both as per-slice
    datasets, confirmed from the real dataset_ACDC.py source (see module docstring).
    """
    os.makedirs(out_dir, exist_ok=True)
    slice_names = []
    for patient_id in patients:
        patient_dir = find_patient_dir(patient_id)
        ed, es = parse_info_cfg(patient_dir)
        for phase, frame_num in [("ed", ed), ("es", es)]:
            img_path, gt_path = find_frame_files(patient_dir, patient_id, frame_num)
            if gt_path is None:
                print(f"WARNING: no ground truth for {patient_id}_{phase} — skipping.")
                continue
            img_vol, _, _ = load_nii(img_path)
            gt_vol, _, _ = load_nii(gt_path)

            for s in range(img_vol.shape[2]):
                img_slice = normalize_slice(img_vol[:, :, s])
                gt_slice = gt_vol[:, :, s].astype(np.uint8)
                slice_name = f"{patient_id}_{phase}_slice{s:03d}"
                # CONFIRMED KEY NAMES: 'img'/'label', not TransUNet's 'image'/'label'
                np.savez(os.path.join(out_dir, f"{slice_name}.npz"), img=img_slice, label=gt_slice)
                # BUG FOUND VIA TESTING: ACDCdataset uses the list-file entry
                # DIRECTLY as the filename (os.path.join(data_dir, split,
                # slice_name)) — it does NOT append ".npz" itself. The list
                # entry must include the extension, or np.load() gets a
                # path with no extension and fails with FileNotFoundError.
                slice_names.append(f"{slice_name}.npz")

    with open(list_path, "w") as f:
        f.write("\n".join(slice_names) + "\n")
    print(f"Wrote {len(slice_names)} slices to {out_dir}, list: {list_path}")
    return len(slice_names)


def write_volumes(patients, out_dir, list_path):
    """Used ONLY for the test split — full 3D volumes, one .npz per case,
    written to a SEPARATE directory tree (volume_path), confirmed from
    train_ACDC.py's usage (see module docstring).
    """
    os.makedirs(out_dir, exist_ok=True)
    vol_names = []
    for patient_id in patients:
        patient_dir = find_patient_dir(patient_id)
        ed, es = parse_info_cfg(patient_dir)
        for phase, frame_num in [("ed", ed), ("es", es)]:
            img_path, gt_path = find_frame_files(patient_dir, patient_id, frame_num)
            if gt_path is None:
                print(f"WARNING: no ground truth for {patient_id}_{phase} — skipping.")
                continue
            img_vol, _, _ = load_nii(img_path)   # (H, W, S)
            gt_vol, _, _ = load_nii(gt_path)

            img_norm = np.stack([normalize_slice(img_vol[:, :, s]) for s in range(img_vol.shape[2])], axis=0)
            gt_stack = np.stack([gt_vol[:, :, s] for s in range(gt_vol.shape[2])], axis=0).astype(np.uint8)

            vol_name = f"{patient_id}_{phase}"
            np.savez(os.path.join(out_dir, f"{vol_name}.npz"), img=img_norm, label=gt_stack)
            # Same fix as write_slices() — list entry needs the extension.
            vol_names.append(f"{vol_name}.npz")

    with open(list_path, "w") as f:
        f.write("\n".join(vol_names) + "\n")
    print(f"Wrote {len(vol_names)} volumes to {out_dir}, list: {list_path}")
    return len(vol_names)


def main():
    split = load_patient_split(CONFIGS_JSON)
    print(f"Split loaded: {len(split['train'])} train / {len(split['dev'])} dev / "
          f"{len(split['test'])} test patients")

    os.makedirs(LIST_DIR_OUTPUT, exist_ok=True)

    write_slices(split["train"], os.path.join(ROOT_DIR_OUTPUT, "train"),
                 os.path.join(LIST_DIR_OUTPUT, "train.txt"))
    write_slices(split["dev"], os.path.join(ROOT_DIR_OUTPUT, "valid"),
                 os.path.join(LIST_DIR_OUTPUT, "valid.txt"))
    write_volumes(split["test"], VOLUME_PATH_OUTPUT,
                  os.path.join(LIST_DIR_OUTPUT, "test.txt"))

    print(f"\nDone. Point train_ACDC.py at:")
    print(f"  --root_dir {ROOT_DIR_OUTPUT}")
    print(f"  --volume_path {VOLUME_PATH_OUTPUT}")
    print(f"  --list_dir {LIST_DIR_OUTPUT}")
