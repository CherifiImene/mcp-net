import json
import os

import h5py
import numpy as np

try:
    from acdc_utilities import load_nii
except ImportError:
    import nibabel as nib
    def load_nii(path):
        img = nib.load(path)
        return img.get_fdata(), img.affine, img.header

# ============================== CONFIG — EDIT THESE ============================== #
ACDC_RAW_DIR = "data/ACDC/raw"          # must contain 'training/' and 'testing/' — the
                                          # SAME original ACDC download used everywhere else
CONFIGS_JSON = "data/acdc_split.json"  # SAME split file as nnU-Net prep
OUTPUT_DIR = "data/transUnet"       # output root — point TransUNet's --root_path /
                                          # --list_dir here once written (see guide)
IMG_SIZE = 224                           # TransUNet's default Synapse img_size; ACDC papers
                                          # commonly also use 224 — resized at LOAD time by
                                          # TransUNet's own RandomGenerator
# =================================================================================== #


def normalize_slice(img_slice):
    """Per-slice min-max to [0,1]. See module docstring's normalization caveat."""
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


def write_train_slices(patients, out_dir, list_path):
    os.makedirs(out_dir, exist_ok=True)
    slice_names = []
    for patient_id in patients:
        patient_dir = find_patient_dir(patient_id)
        ed, es = parse_info_cfg(patient_dir)
        for phase, frame_num in [("ed", ed), ("es", es)]:
            img_path, gt_path = find_frame_files(patient_dir, patient_id, frame_num)
            if gt_path is None:
                print(f"WARNING: no ground truth for {patient_id}_{phase} — skipping "
                      f"(likely one of ACDC's blind test-set patients; those can't be "
                      f"used for training regardless of your split file).")
                continue
            img_vol, _, _ = load_nii(img_path)   # (H, W, S)
            gt_vol, _, _ = load_nii(gt_path)

            for s in range(img_vol.shape[2]):
                img_slice = normalize_slice(img_vol[:, :, s])
                gt_slice = gt_vol[:, :, s].astype(np.uint8)
                slice_name = f"{patient_id}_{phase}_slice{s:03d}"
                np.savez(os.path.join(out_dir, f"{slice_name}.npz"),
                         image=img_slice, label=gt_slice)
                slice_names.append(slice_name)

    with open(list_path, "w") as f:
        f.write("\n".join(slice_names) + "\n")
    print(f"Wrote {len(slice_names)} training slices to {out_dir}, list: {list_path}")
    return len(slice_names)


def write_volumes(patients, out_dir, list_path, require_gt=True):
    os.makedirs(out_dir, exist_ok=True)
    vol_names = []
    for patient_id in patients:
        patient_dir = find_patient_dir(patient_id)
        ed, es = parse_info_cfg(patient_dir)
        for phase, frame_num in [("ed", ed), ("es", es)]:
            img_path, gt_path = find_frame_files(patient_dir, patient_id, frame_num)
            if gt_path is None:
                if require_gt:
                    print(f"WARNING: no ground truth for {patient_id}_{phase} — skipping "
                          f"(can't compute Dice/HD95 without it).")
                    continue
                gt_vol = None
            img_vol, _, _ = load_nii(img_path)  # (H, W, S)
            img_norm = np.stack([normalize_slice(img_vol[:, :, s]) for s in range(img_vol.shape[2])], axis=0)  # (S,H,W)

            vol_name = f"{patient_id}_{phase}"
            out_path = os.path.join(out_dir, f"{vol_name}.npy.h5")
            with h5py.File(out_path, "w") as hf:
                hf.create_dataset("image", data=img_norm)
                if gt_path is not None:
                    gt_vol, _, _ = load_nii(gt_path)
                    gt_norm = np.stack([gt_vol[:, :, s] for s in range(gt_vol.shape[2])], axis=0).astype(np.uint8)
                    hf.create_dataset("label", data=gt_norm)
            vol_names.append(vol_name)

    with open(list_path, "w") as f:
        f.write("\n".join(vol_names) + "\n")
    print(f"Wrote {len(vol_names)} volumes to {out_dir}, list: {list_path}")
    return len(vol_names)


def main():
    split = load_patient_split(CONFIGS_JSON)
    print(f"Split loaded: {len(split['train'])} train / {len(split['dev'])} dev / "
          f"{len(split['test'])} test patients")

    lists_dir = os.path.join(OUTPUT_DIR, "lists_ACDC")
    os.makedirs(lists_dir, exist_ok=True)

    write_train_slices(
        split["train"],
        out_dir=os.path.join(OUTPUT_DIR, "train_npz"),
        list_path=os.path.join(lists_dir, "train.txt"),
    )
    write_volumes(
        split["dev"],
        out_dir=os.path.join(OUTPUT_DIR, "dev_vol_h5"),
        list_path=os.path.join(lists_dir, "dev_vol.txt"),
    )
    write_volumes(
        split["test"],
        out_dir=os.path.join(OUTPUT_DIR, "test_vol_h5"),
        list_path=os.path.join(lists_dir, "test_vol.txt"),
    )

    print(f"\nDone. In TransUNet's train.py, add an 'ACDC' entry to dataset_config pointing at:")
    print(f"  root_path: {os.path.join(OUTPUT_DIR, 'train_npz')}")
    print(f"  list_dir:  {lists_dir}")
    print(f"  num_classes: 4")
    print("See the accompanying guide for the dataset_acdc.py / trainer_acdc drop-in files.")


main()