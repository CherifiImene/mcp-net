"""
Compares full_mcp_net, nnU-Net, G-CASCADE and TransUNet in a
COMMON evaluation space (original, full-resolution ACDC image coordinates)

WHAT THIS DOES:
  1. For each test-set case (from configs.json), finds the original patient
     folder + Info.cfg to get ED/ES frame numbers.
  2. Loads the ORIGINAL, full-resolution ground truth for that frame —
     this is the y_true for ALL three models in this comparison (not the
     ROI-cropped ground truth load_test_cases() uses for Tables A-D).
  3. Loads full_mcp_net's 128x128 prediction and UN-CROPS it back into
     original coordinates using the saved bbox+size JSON (uncrop_prediction,
     verified separately with a known-pattern test).
  4. Loads nnU-Net's prediction — filename is the case identifier directly, 
     matched via the same Info.cfg-derived frame number as ground truth.
  5. Loads TransUNet's prediction — filename is "{case}_pred.nii.gz", 
     matched via canonical_case_id.
  6. Loads G-CASCADE's prediction — filename is "{case}_pred.nii.gz", 
     matched via canonical_case_id.
  7. Runs the SAME evaluate_dataset/compare_variants/friedman_omnibus
     pipeline as every other table in this project, now with all three
     models on equal footing.

"""

import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import numpy as np
import pandas as pd

from mcpnet.evaluation.metrics import evaluate_dataset, summarize
from mcpnet.evaluation.statistical_tests import annotate_significance, compare_variants, friedman_omnibus

from scripts.uncrop_mcpnet_predictions import uncrop_prediction

try:
    from acdc_utilities import load_nii
except ImportError:
    import nibabel as nib
    def load_nii(path):
        img = nib.load(path)
        return img.get_fdata(), img.affine, img.header

# ============================== CONFIG — EDIT THESE ============================== #
ACDC_RAW_DIR = "/content/drive/MyDrive/PFE_CHERIFI_Livrables/Code/Training_and_data_preparation/ACDC_Datasets/ACDC_Segmentation"                    # original training/testing folders + Info.cfg
CONFIGS_JSON = "data/configv2.json" 
BBOX_JSON = "data/bboxes.json"  # your saved {"patient001": {"size":..., "bboxe":...}}

MCPNET_PREDICTIONS_DIR = "results/mcpnet_predictions"    # {case_id}_ed.nii.gz / _es.nii.gz, 128x128xS
NNUNET_PREDICTIONS_DIR = "data/nnUnet/nnUNet_results"    # {patient}_frame{NN}.nii.gz (nnU-Net's own naming)
TRANSUNET_PREDICTIONS_DIR = "data/transUnet/predictions/TU_ACDC224/TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224_s5"  # {case}_pred.nii.gz
GCASCADE_PREDICTIONS_DIR = "results/gcascade_predictions"    # {case}_pred.nii.gz 

RESULTS_DIR = "results/table_e_original_resolution"
BASELINE_VARIANT = "full_mcp_net"
# =================================================================================== #


def canonical_case_id(name):
    """Strips extensions and known output-tool suffixes:
      - TransUNet: '{case}_pred.nii.gz' 
      - G-CASCADE: '{case}.npz_pred.nii.gz'
    """
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
    if name.endswith(".npz"):  # G-CASCADE-specific — strip the embedded .npz left over after _pred removal
        name = name[: -len(".npz")]
    return name


def load_patient_split(configs_path):
    with open(configs_path) as f:
        split = json.load(f)

    def patient_ids(names):
        return sorted({n.split("_")[0].replace(".nii", "")[: len("patient000")] for n in names})

    return patient_ids(split["test"])


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


def find_original_frame_file(patient_dir, patient_id, frame_num, suffix=""):
    for ext in (".nii.gz", ".nii"):
        candidate = os.path.join(patient_dir, f"{patient_id}_frame{frame_num:02d}{suffix}{ext}")
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Could not find frame {frame_num:02d}{suffix} for {patient_id} in {patient_dir}")


def find_file_by_canonical_id(directory, target_canonical_id):
    for f in os.listdir(directory):
        if canonical_case_id(f) == target_canonical_id:
            return os.path.join(directory, f)
    return None


def build_test_cases(bbox_data):
    """Yields (case_id, patient_id, phase, frame_num, y_true_original,
    voxelspacing, patient_dir) for every ED/ES phase of every test patient.
    """
    test_patients = load_patient_split(CONFIGS_JSON)
    for patient_id in test_patients:
        patient_dir = find_patient_dir(patient_id)
        ed_frame, es_frame = parse_info_cfg(patient_dir)

        for phase, frame_num in [("ed", ed_frame), ("es", es_frame)]:
            gt_path = find_original_frame_file(patient_dir, patient_id, frame_num, suffix="_gt")
            gt_data, _, gt_header = load_nii(gt_path)  # (H, W, S), ORIGINAL resolution
            zooms = gt_header.get_zooms()
            voxelspacing = (zooms[2], zooms[0], zooms[1])  # (z,x,y), matching (S,H,W) convention

            y_true = np.transpose(gt_data, (2, 0, 1)).astype(np.int32)  # (S, H, W)

            case_id = f"{patient_id}_{phase}"
            yield case_id, patient_id, phase, frame_num, y_true, voxelspacing, patient_dir


def load_mcpnet_prediction(case_id, patient_id, bbox_entry, orig_hw):
    pred_path = find_file_by_canonical_id(MCPNET_PREDICTIONS_DIR, case_id)
    if pred_path is None:
        raise FileNotFoundError(f"No MCP-Net prediction found for {case_id} in {MCPNET_PREDICTIONS_DIR}")
    pred_data, _, _ = load_nii(pred_path)  # (128, 128, S)

    bbox = tuple(bbox_entry["bboxe"])
    S = pred_data.shape[2]
    orig_size = (orig_hw[0], orig_hw[1], S)

    pred_hws = pred_data.astype(np.int32)
    uncropped = uncrop_prediction(pred_hws, bbox, orig_size)  # (H, W, S)
    return np.transpose(uncropped, (2, 0, 1))  # (S, H, W), matching y_true convention


def load_nnunet_prediction(patient_id, frame_num):
    filename = f"{patient_id}_frame{frame_num:02d}.nii.gz"
    filepath = os.path.join(NNUNET_PREDICTIONS_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No nnU-Net prediction found at {filepath}")
    pred_data, _, _ = load_nii(filepath)  # (H, W, S), original resolution already
    return np.transpose(pred_data, (2, 0, 1)).astype(np.int32)


def load_transunet_prediction(case_id):
    
    pred_path = find_file_by_canonical_id(TRANSUNET_PREDICTIONS_DIR, case_id)
    if pred_path is None:
        raise FileNotFoundError(f"No TransUNet prediction found for {case_id} in {TRANSUNET_PREDICTIONS_DIR}")
    pred_data, _, _ = load_nii(pred_path)  # (W, H, S) — confirmed via direct test, see docstring
    return np.transpose(pred_data, (2, 1, 0)).astype(np.int32)  # -> (S, H, W), matching y_true convention


def load_gcascade_prediction(case_id):
    
    pred_path = find_file_by_canonical_id(GCASCADE_PREDICTIONS_DIR, case_id)
    if pred_path is None:
        raise FileNotFoundError(f"No G-CASCADE prediction found for {case_id} in {GCASCADE_PREDICTIONS_DIR}")
    pred_data, _, _ = load_nii(pred_path)  # (W, H, S) — confirmed via direct test, see docstring
    return np.transpose(pred_data, (2, 1, 0)).astype(np.int32)  # -> (S, H, W), matching y_true convention


def main():
    with open(BBOX_JSON) as f:
        bbox_data = json.load(f)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_dfs = []
    results = {"full_mcp_net": [], "nnUNet": [], "TransUNet": [], "GCASCADE": []}

    for case_id, patient_id, phase, frame_num, y_true, voxelspacing, patient_dir in build_test_cases(bbox_data):
        orig_hw = (y_true.shape[1], y_true.shape[2])  # (H, W) from (S,H,W)

        mcpnet_pred = load_mcpnet_prediction(case_id, patient_id, bbox_data[patient_id], orig_hw)
        nnunet_pred = load_nnunet_prediction(patient_id, frame_num)
        transunet_pred = load_transunet_prediction(case_id)
        gcascade_pred = load_gcascade_prediction(case_id)

        assert mcpnet_pred.shape == y_true.shape, f"MCP-Net shape mismatch for {case_id}: {mcpnet_pred.shape} vs {y_true.shape}"
        assert nnunet_pred.shape == y_true.shape, f"nnU-Net shape mismatch for {case_id}: {nnunet_pred.shape} vs {y_true.shape}"
        assert transunet_pred.shape == y_true.shape, f"TransUNet shape mismatch for {case_id}: {transunet_pred.shape} vs {y_true.shape}"
        assert gcascade_pred.shape == y_true.shape, f"G-CASCADE shape mismatch for {case_id}: {gcascade_pred.shape} vs {y_true.shape}"

        results["full_mcp_net"].append((case_id, y_true, mcpnet_pred, voxelspacing))
        results["nnUNet"].append((case_id, y_true, nnunet_pred, voxelspacing))
        results["TransUNet"].append((case_id, y_true, transunet_pred, voxelspacing))
        results["GCASCADE"].append((case_id, y_true, gcascade_pred, voxelspacing))

    for model_name, entries in results.items():
        case_ids = [e[0] for e in entries]
        y_true_list = [e[1] for e in entries]
        y_pred_list = [e[2] for e in entries]
        voxelspacings = [e[3] for e in entries]
        df = evaluate_dataset(y_true_list, y_pred_list, case_ids, model_name, voxelspacings=voxelspacings)
        all_dfs.append(df)

    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df.to_csv(os.path.join(RESULTS_DIR, "per_case.csv"), index=False)

    summary = summarize(full_df)
    summary.to_csv(os.path.join(RESULTS_DIR, "summary.csv"))
    print("\nSummary (original-resolution, common space):\n", summary)

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