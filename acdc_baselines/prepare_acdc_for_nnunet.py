"""
prepare_acdc_for_nnunet.py — standalone script
to prepare ACDC data for nnU-Net v2 and align its train/val/test split with
MCP-Net split, so the eventual statistical comparison is apples-to-apples.

This script wraps `nnunetv2.dataset_conversion.Dataset027_ACDC` (installed as part of the
`nnunetv2` pip package) rather than reimplementing ACDC conversion from scratch.

  - It expects your ACDC download's standard 'training' and 'testing'
    subfolders (per-patient folders, each with Info.cfg, patientXXX_4d.nii.gz,
    two frame files + their _gt.nii.gz, exactly the standard ACDC challenge
    layout).
  - It labels classes as background=0, RV=1, MLV=2 (myocardium), LVC=3 (LV
    cavity).
  - By default it creates its OWN random 5-fold split (seed=1234).

USAGE :
    1. pip install nnunetv2
    2. Set the three nnU-Net environment variables (see CONFIG below)
    3. Edit CONFIG, run this script
    4. Then run :
         nnUNetv2_plan_and_preprocess -d <DATASET_ID> --verify_dataset_integrity
         nnUNetv2_train <DATASET_ID> 2d 0
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ============================== CONFIG — EDIT THESE ============================== #
ACDC_RAW_DIR = 'data/ACDC/raw' # must contain 'training/' and 'testing/' subfolders
                                          # (the standard, unmodified ACDC challenge download)
CONFIGS_JSON = 'data/acdc_split.json' # existing MCP-Net split file
                                          # ({"train": [...], "dev": [...], "test": [...]})
DATASET_ID = 27  # nnU-Net dataset ID (27 is the ID the official ACDC
                                          # script defaults to; keep it unless it collides with
                                          # another dataset you already have)
TASK_NAME = "ACDC"

# nnU-Net's three required environment variables — MUST be set before nnunetv2
# is imported anywhere (including by this script), so set them first.
NNUNET_RAW = "data/nnUnet/nnUNet_raw"
NNUNET_PREPROCESSED = "data/nnUnet/nnUNet_preprocessed"
NNUNET_RESULTS = "data/nnUnet/nnUNet_results"
# =================================================================================== #

os.environ["nnUNet_raw"] = NNUNET_RAW
os.environ["nnUNet_preprocessed"] = NNUNET_PREPROCESSED
os.environ["nnUNet_results"] = NNUNET_RESULTS

for d in (NNUNET_RAW, NNUNET_PREPROCESSED, NNUNET_RESULTS):
    os.makedirs(d, exist_ok=True)


def run_official_converter():
    """Runs the REAL nnunetv2.dataset_conversion.Dataset027_ACDC converter
    in-process, using Isensee et al.'s actual, tested conversion logic
    rather than a reimplementation.
    """
    print("Running the official nnU-Net ACDC converter ...")
    import nnunetv2.dataset_conversion.Dataset027_ACDC as acdc_converter
    import nnunetv2.paths as nnunet_paths

    acdc_converter.nnUNet_raw = str(nnunet_paths.nnUNet_raw)  
    acdc_converter.convert_acdc(ACDC_RAW_DIR, DATASET_ID)
    print("Official converter finished (default split will be overridden next).")


def load_mcp_net_split():
    with open(CONFIGS_JSON) as f:
        split = json.load(f)
    # acdc_split.json entries look like "patient001.nii" 
    # normalize down to just the patient ID for matching
    # against nnU-Net's "patientXXX" case naming.
    def patient_ids(names):
        return sorted({n.split("_")[0].replace(".nii", "")[:len("patient000")] for n in names})

    return {
        "train": patient_ids(split["train"]),
        "dev": patient_ids(split["dev"]),
        "test": patient_ids(split["test"]),
    }


def enforce_mcp_net_split(mcp_split):
    """Moves any imagesTr/labelsTr case belonging to MCP-Net TEST
    patients OUT of the training set entirely, into a clearly-separated
    held-out folder. Also rewrites splits_final.json so nnU-Net's internal
    5-fold CV uses your train/dev patients specifically, not its own
    seed-1234 random split.
    """
    dataset_name = f"Dataset{DATASET_ID:03d}_{TASK_NAME}"
    dataset_dir = Path(NNUNET_RAW) / dataset_name
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    held_out_dir = dataset_dir / "mcpnet_test_heldout"  # NOT imagesTs — nnU-Net never touches this
    held_out_dir.mkdir(exist_ok=True)

    test_patients = set(mcp_split["test"])
    moved = []
    for img_file in list(images_tr.glob("*.nii.gz")):
        patient_id = img_file.name[: len("patient000")]
        if patient_id in test_patients:
            case_id = img_file.name.replace("_0000.nii.gz", "")
            shutil.move(str(img_file), str(held_out_dir / img_file.name))
            label_file = labels_tr / f"{case_id}.nii.gz"
            if label_file.exists():
                shutil.move(str(label_file), str(held_out_dir / label_file.name))
            moved.append(case_id)

    print(f"Moved {len(moved)} case(s) belonging to your MCP-Net TEST patients out of "
          f"imagesTr/labelsTr into {held_out_dir} (nnU-Net will never train/preprocess-plan on these).")

    
    dataset_json_path = dataset_dir / "dataset.json"
    with open(dataset_json_path) as f:
        dataset_json = json.load(f)
    dataset_json["numTraining"] = len(list(labels_tr.glob("*.nii.gz")))
    with open(dataset_json_path, "w") as f:
        json.dump(dataset_json, f, indent=4)

    # Rebuild splits_final.json using MCP-Net train/dev patients specifically
    remaining_cases = [f.stem.replace(".nii", "") for f in labels_tr.glob("*.nii.gz")]
    train_patients = set(mcp_split["train"])
    dev_patients = set(mcp_split["dev"])

    train_cases = [c for c in remaining_cases if c[: len("patient000")] in train_patients]
    val_cases = [c for c in remaining_cases if c[: len("patient000")] in dev_patients]

    unmatched = [c for c in remaining_cases if c not in train_cases and c not in val_cases]
    if unmatched:
        print(f"WARNING: {len(unmatched)} case(s) in imagesTr didn't match your train or dev "
              f"patient lists — check for patient-ID formatting mismatches between configs.json "
              f"and nnU-Net's naming: {unmatched[:5]}{'...' if len(unmatched) > 5 else ''}")

    # nnU-Net expects a list of dicts, one per fold. Since we're not doing our
    # own 5-fold CV (we want ONE split matching MCP-Net's)
    # use the same train/val split for every fold entry nnU-Net's default 5-fold config
    # expects
    # this makes "fold 0" (what we'll actually train) use our exact split
    # folds 1-4 are identical copies (unused in our case)
    single_split = {"train": train_cases, "val": val_cases}
    splits = [single_split for _ in range(5)]

    preprocessed_dir = Path(NNUNET_PREPROCESSED) / dataset_name
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    with open(preprocessed_dir / "splits_final.json", "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Wrote splits_final.json: {len(train_cases)} train / {len(val_cases)} val cases "
          f"(matching your MCP-Net split). Train nnU-Net with fold 0 specifically:\n"
          f"  nnUNetv2_train {DATASET_ID} 2d 0")
    print(f"\nHeld-out test patients ({len(test_patients)}) are in: {held_out_dir}")
    print("Run inference on these separately after training — do NOT let nnU-Net's own "
          "planning/preprocessing/training pipeline touch this folder.")


def main():
    run_official_converter()
    mcp_split = load_mcp_net_split()
    print(f"\nMCP-Net split loaded: {len(mcp_split['train'])} train / "
          f"{len(mcp_split['dev'])} dev / {len(mcp_split['test'])} test patients")
    enforce_mcp_net_split(mcp_split)
    print("\nDone. Next steps:")
    print(f"  nnUNetv2_plan_and_preprocess -d {DATASET_ID} --verify_dataset_integrity")
    print(f"  nnUNetv2_train {DATASET_ID} 2d 0")


main()