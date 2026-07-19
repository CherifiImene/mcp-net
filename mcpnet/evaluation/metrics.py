"""
Per-case evaluation metrics.

Computes metrics PER CASE because both ablation
tables and the Wilcoxon significance tests need paired, per-case values —
that's the raw material `statistical_tests.py` consumes.
"""

import warnings

from medpy.metric import dc, hd95
import numpy as np
import pandas as pd

CLASS_NAMES = {1: "RV", 2: "Myo", 3: "LV"}  # 0 assumed background, excluded from per-class reporting

_spacing_warning_shown = False  # print the spacing warning once per run, not once per case


def per_case_metrics(y_true, y_pred, case_id, variant_name, voxelspacing=None):
    """y_true, y_pred: integer label maps, shape (H, W) for a single slice or
    (H, W, S) for a reconstructed volume — matches whatever `load_test_cases`
    yields. Volume-level cases are the right unit for clinically meaningful
    HD95.

    voxelspacing: real-world spacing in mm, matching y_true/y_pred's rank
    (e.g. (x_mm, y_mm, z_mm) for a 3D volume case, from
    `load_nii(path)[2].get_zooms()`). If None, HD95 is computed in pixel
    units and a warning is printed.
    """
    global _spacing_warning_shown
    if voxelspacing is None and not _spacing_warning_shown:
        warnings.warn(
            "hd95 computed WITHOUT voxelspacing — results are in PIXEL units, "
            "not millimeters. Pass voxelspacing=header.get_zooms() from "
            "load_nii() for clinically meaningful, cross-paper-comparable HD95. "
            "(This warning prints once per run.)",
            stacklevel=2,
        )
        _spacing_warning_shown = True

    row = {"case_id": case_id, "variant": variant_name}

    for class_id, class_name in CLASS_NAMES.items():
        true_mask = (y_true == class_id).astype(np.uint8)
        pred_mask = (y_pred == class_id).astype(np.uint8)

        if true_mask.sum() == 0 and pred_mask.sum() == 0:
            row[f"dice_{class_name}"] = np.nan
            row[f"hd95_{class_name}"] = np.nan
            continue

        row[f"dice_{class_name}"] = dc(pred_mask, true_mask) if pred_mask.sum() > 0 else 0.0

        if pred_mask.sum() > 0 and true_mask.sum() > 0:
            try:
                row[f"hd95_{class_name}"] = hd95(pred_mask, true_mask, voxelspacing=voxelspacing)
            except Exception:
                row[f"hd95_{class_name}"] = np.nan
        else:
            row[f"hd95_{class_name}"] = np.nan

    dice_cols = [f"dice_{name}" for name in CLASS_NAMES.values()]
    row["mean_dice"] = np.nanmean([row[c] for c in dice_cols])
    row["voxelspacing_used"] = str(voxelspacing)  # auditable per-row, not just a global assumption
    return row


def evaluate_dataset(y_true_list, y_pred_list, case_ids, variant_name, voxelspacings=None):
    """Runs per_case_metrics over a whole dataset, returns a tidy DataFrame —
    one row per case, ready to append to results/ablation_tables/*.csv.

    voxelspacings: optional list, same length/order as case_ids, one spacing
    tuple per case.
    """
    if voxelspacings is None:
        voxelspacings = [None] * len(case_ids)

    rows = [
        per_case_metrics(yt, yp, cid, variant_name, voxelspacing=vs)
        for yt, yp, cid, vs in zip(y_true_list, y_pred_list, case_ids, voxelspacings)
    ]
    return pd.DataFrame(rows)


def summarize(df, group_col="variant"):
    """Collapses per-case rows into the mean +/- std table for the paper
    (Table A / Table B columns)."""
    metric_cols = [c for c in df.columns if c.startswith("dice_") or c.startswith("hd95_") or c == "mean_dice"]
    summary = df.groupby(group_col)[metric_cols].agg(["mean", "std"])
    return summary
