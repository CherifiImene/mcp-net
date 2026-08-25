"""
Four-model (full_mcp_net, nnU-Net, TransUNet, G-Cascade) comparison on M&Ms
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import os

import numpy as np
import pandas as pd

from mcpnet.evaluation.metrics import evaluate_dataset, summarize
from mcpnet.evaluation.statistical_tests import annotate_significance, compare_variants, friedman_omnibus


from mcpnet.data.acdc_utilities_stub import load_nii


# ============================== CONFIG — EDIT THESE ============================== #
MMS_ORIGINAL_DIRS = ["./data/M&Ms/validation",
                    "./data/M&Ms/testing"]

METADATA_CSV_PATH =  "./data/M&Ms/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"
PATIENT_ID_COL, ED_FRAME_COL, ES_FRAME_COL = "External code", "ED", "ES"
IMAGE_SUFFIX, LABEL_SUFFIX = "_sa.nii.gz", "_sa_gt.nii.gz"

MCPNET_ORIGINAL_PREDICTIONS_DIR = "./data/full_mcpnet/full_mcp_net_mms_original"              # from run_mcpnet_on_mms.py, 128x128 ROI space
NNUNET_PREDICTIONS_DIR = "./data/nnUnet/nnunet_mms"       # from nnUNetv2_predict, original resolution
TRANSUNET_PREDICTIONS_DIR = "./data/TransUnet/transunet_mms/TU_ACDC224/TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224_s5"  # from TransUNet test.py, original resolution
GCASCADE_PREDICTIONS_DIR = "./data/GCascade/gcascade_mms/"

RESULTS_DIR = "results/table_mms_comparison"
BASELINE_VARIANT = "full_mcp_net"
BBOX_JSON_PATH = "data/M&Ms/mms_extracted_rois/bbox_info.json"

# ============================== LABEL CONVENTION — VERIFY FIRST ============================== #
# M&Ms uses BG=0, LV=1, MYO=2, RV=3 — RV and LV are
# SWAPPED relative to ACDC's BG=0, RV=1, MYO=2, LV=3. This remap corrects
# M&Ms ground truth into ACDC's convention before any metric is computed.
MMS_LABEL_REMAP = {0: 0, 1: 3, 2: 2, 3: 1}  # M&Ms class -> ACDC-convention class
# =================================================================================== #


def remap_labels(label_array, remap_dict):
    """Applies MMS_LABEL_REMAP to a label array
    """
    remapped = np.zeros_like(label_array)
    for old_class, new_class in remap_dict.items():
        remapped[label_array == old_class] = new_class
    return remapped


def canonical_case_id(name):
    while True:
        for ext in (".nii.gz", ".nii", ".gz"):
            if name.endswith(ext):
                name = name[: -len(ext)]
                break
        else:
            break
    for suffix in ("_pred",):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    if name.endswith(".npz"):  # G-CASCADE-specific -- strip embedded .npz left over after _pred removal
        name = name[: -len(".npz")]
    return name


def find_file_by_canonical_id(directory, target_canonical_id):
    for f in os.listdir(directory):
        if canonical_case_id(f) == target_canonical_id:
            return os.path.join(directory, f)
    return None


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


def load_mcpnet_prediction_original(case_id):
    pred_path = find_file_by_canonical_id(MCPNET_ORIGINAL_PREDICTIONS_DIR, case_id)
    if pred_path is None:
        raise FileNotFoundError(f"No MCP-Net original-size prediction found for {case_id}")
    pred_data, _, _ = load_nii(pred_path)
    return np.transpose(pred_data, (2, 0, 1)).astype(np.int32)


def list_test_cases():
    
    metadata = load_metadata(METADATA_CSV_PATH)
    cases = []
    for split_dir in MMS_ORIGINAL_DIRS:
        if not os.path.exists(split_dir):
            continue
        for patient_id in sorted(os.listdir(split_dir)):
            if patient_id not in metadata:
                continue
            patient_dir = os.path.join(split_dir, patient_id)
            image_file = next((f for f in os.listdir(patient_dir) if f.endswith(IMAGE_SUFFIX)), None)
            label_file = next((f for f in os.listdir(patient_dir) if f.endswith(LABEL_SUFFIX)), None)
            if image_file is None or label_file is None:
                continue

            ed_frame, es_frame = metadata[patient_id]
            for phase, frame in [("ed", ed_frame), ("es", es_frame)]:
                case_id = f"{patient_id}_{phase}"
                cases.append((case_id, patient_id, phase, frame, patient_dir, image_file, label_file))
    return cases


def load_ground_truth_and_spacing(patient_dir, label_file, frame):
   
    gt4d, _, gt_header = load_nii(os.path.join(patient_dir, label_file))
    
    gt_vol = gt4d[:, :, :, frame]
    # CONFIRMED BUG: some M&Ms GT files store near-integer floating point
    # values (e.g. 1.000000019557774) instead of exact integers -- likely
    # from resampling/interpolation during the dataset's own creation.
    # Ground truth should always be discrete integer classes, 
    # so rounding here is always correct.
    gt_vol = np.round(gt_vol)
    gt_remapped = remap_labels(gt_vol, MMS_LABEL_REMAP)
    y_true = np.transpose(gt_remapped, (2, 0, 1)).astype(np.int32)
    zooms = gt_header.get_zooms()
    voxelspacing = (zooms[2], zooms[0], zooms[1])
    return y_true, voxelspacing


def load_nnunet_prediction(patient_id, phase):
    filename = f"{patient_id}_{phase}.nii.gz"
    filepath = os.path.join(NNUNET_PREDICTIONS_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No nnU-Net prediction found at {filepath}")
    pred_data, _, _ = load_nii(filepath)
    return np.transpose(pred_data, (2, 0, 1)).astype(np.int32)


def load_transunet_prediction(case_id):
    pred_path = find_file_by_canonical_id(TRANSUNET_PREDICTIONS_DIR, case_id)
    if pred_path is None:
        raise FileNotFoundError(f"No TransUNet prediction found for {case_id}")
    pred_data, _, _ = load_nii(pred_path)
    return np.transpose(pred_data, (2, 1, 0)).astype(np.int32)


def load_gcascade_prediction(case_id):
    
    pred_path = find_file_by_canonical_id(GCASCADE_PREDICTIONS_DIR, case_id)
    if pred_path is None:
        raise FileNotFoundError(f"No G-CASCADE prediction found for {case_id}")
    pred_data, _, _ = load_nii(pred_path)
    return np.transpose(pred_data, (2, 1, 0)).astype(np.int32)


def evaluate_full_mcp_net(cases):
    y_true_list, y_pred_list, case_ids, voxelspacings = [], [], [], []

    for i, (case_id, patient_id, phase, frame, patient_dir, image_file, label_file) in enumerate(cases):
        if (i+1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{len(cases)}] {case_id}...", flush=True)
        y_true, voxelspacing = load_ground_truth_and_spacing(patient_dir, label_file, frame)
        pred = load_mcpnet_prediction_original(case_id)

        assert pred.shape == y_true.shape, f"full_mcp_net shape mismatch {case_id}: {pred.shape} vs {y_true.shape}"
        y_true_list.append(y_true)
        y_pred_list.append(pred)
        case_ids.append(case_id)
        voxelspacings.append(voxelspacing)

    return evaluate_dataset(y_true_list, y_pred_list, case_ids, "full_mcp_net", voxelspacings=voxelspacings)


def evaluate_nnunet(cases):
    """Never loads img4d — nnU-Net's predictions are already in
    original-resolution space, no un-cropping needed.
    """
    y_true_list, y_pred_list, case_ids, voxelspacings = [], [], [], []

    for i, (case_id, patient_id, phase, frame, patient_dir, image_file, label_file) in enumerate(cases):
        if (i+1) % 20 == 0 or i == 0:
            print(f'  [{i+1}/{len(cases)}] {case_id}...', flush=True)
        y_true, voxelspacing = load_ground_truth_and_spacing(patient_dir, label_file, frame)
        pred = load_nnunet_prediction(patient_id, phase)

        assert pred.shape == y_true.shape, f"nnU-Net shape mismatch {case_id}: {pred.shape} vs {y_true.shape}"
        y_true_list.append(y_true)
        y_pred_list.append(pred)
        case_ids.append(case_id)
        voxelspacings.append(voxelspacing)

    return evaluate_dataset(y_true_list, y_pred_list, case_ids, "nnUNet", voxelspacings=voxelspacings)


def evaluate_transunet(cases):
    """Never loads img4d — since predictions are already in original-resolution space"""
    y_true_list, y_pred_list, case_ids, voxelspacings = [], [], [], []

    for i, (case_id, patient_id, phase, frame, patient_dir, image_file, label_file) in enumerate(cases):
        if (i+1) % 20 == 0 or i == 0:
            print(f'  [{i+1}/{len(cases)}] {case_id}...', flush=True)
        y_true, voxelspacing = load_ground_truth_and_spacing(patient_dir, label_file, frame)
        pred = load_transunet_prediction(case_id)

        assert pred.shape == y_true.shape, f"TransUNet shape mismatch {case_id}: {pred.shape} vs {y_true.shape}"
        y_true_list.append(y_true)
        y_pred_list.append(pred)
        case_ids.append(case_id)
        voxelspacings.append(voxelspacing)

    return evaluate_dataset(y_true_list, y_pred_list, case_ids, "TransUNet", voxelspacings=voxelspacings)


def evaluate_gcascade(cases):
    """Never loads img4d — since predictions are already in original-resolution space."""
    y_true_list, y_pred_list, case_ids, voxelspacings = [], [], [], []

    for i, (case_id, patient_id, phase, frame, patient_dir, image_file, label_file) in enumerate(cases):
        if (i+1) % 20 == 0 or i == 0:
            print(f'  [{i+1}/{len(cases)}] {case_id}...', flush=True)
        y_true, voxelspacing = load_ground_truth_and_spacing(patient_dir, label_file, frame)
        pred = load_gcascade_prediction(case_id)

        assert pred.shape == y_true.shape, f"G-CASCADE shape mismatch {case_id}: {pred.shape} vs {y_true.shape}"
        y_true_list.append(y_true)
        y_pred_list.append(pred)
        case_ids.append(case_id)
        voxelspacings.append(voxelspacing)

    return evaluate_dataset(y_true_list, y_pred_list, case_ids, "GCASCADE", voxelspacings=voxelspacings)


def main():
    if MMS_LABEL_REMAP == {0: 0, 1: 1, 2: 2, 3: 3}:
        print("WARNING: MMS_LABEL_REMAP is still the IDENTITY default. If you have not yet "
              "run check_mms_label_convention.py and confirmed this is correct, STOP and do "
              "that first — see this script's module docstring.")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    cases = list_test_cases()  # cheap — no image data loaded yet
    print(f"Found {len(cases)} test cases.")

    # Evaluated ONE MODEL AT A TIME 
    # each call below loads only what that specific model's pass needs, 
    # and its accumulated lists go out of scope (and become eligible for garbage collection) 
    # as soon as the function returns, before the next model's pass begins. 
    # Peak memory is roughly one model's worth of data, not three held simultaneously.
    print("\n=== Evaluating full_mcp_net ===")
    df_mcpnet = evaluate_full_mcp_net(cases)

    print("\n=== Evaluating nnUNet ===")
    df_nnunet = evaluate_nnunet(cases)

    print("\n=== Evaluating TransUNet ===")
    df_transunet = evaluate_transunet(cases)

    print("\n=== Evaluating G-CASCADE ===")
    df_gcascade = evaluate_gcascade(cases)

    full_df = pd.concat([df_mcpnet, df_nnunet, df_transunet, df_gcascade], ignore_index=True)
    full_df.to_csv(os.path.join(RESULTS_DIR, "per_case.csv"), index=False)

    summary = summarize(full_df)
    summary.to_csv(os.path.join(RESULTS_DIR, "summary.csv"))
    print("\nM&Ms comparison summary:\n", summary)

    omnibus = friedman_omnibus(full_df, metric_col="mean_dice", variant_col="variant")
    print(f"\nFriedman omnibus: statistic={omnibus['statistic']:.4f}, p={omnibus['p_value']:.4g}")

    sig = compare_variants(full_df, metric_col="mean_dice", variant_col="variant",
                            baseline_variant=BASELINE_VARIANT, correction="bonferroni")
    sig["significance"] = sig["p_value"].apply(annotate_significance)
    sig.to_csv(os.path.join(RESULTS_DIR, "significance.csv"), index=False)
    print(f"\nSignificance (baseline={BASELINE_VARIANT}):\n", sig)

    return full_df, summary, sig


if __name__ == "__main__":
    main()