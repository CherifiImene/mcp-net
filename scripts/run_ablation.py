"""
Reproduces Tables A-D (architecture ablation, snapshot ensemble, pyramid x
connections factorial, decoder fusion) end-to-end, using the ROI already
produced by the image-processing pipeline (a learned localizer is out of
scope here).

Usage:
    python scripts/run_ablation.py --config configs/mcp_net_config.yaml --table A
    python scripts/run_ablation.py --config configs/mcp_net_config.yaml --table B
    python scripts/run_ablation.py --config configs/mcp_net_config.yaml --table all
"""

import argparse
import os

import numpy as np
import pandas as pd
import yaml

from mcpnet.evaluation.metrics import evaluate_dataset, summarize
from mcpnet.evaluation.statistical_tests import annotate_significance, compare_variants
from mcpnet.inference.ensemble_predict import ensemble_predict, load_snapshot_members, predict_single
from mcpnet.training.losses import DiceIndex, DiceIndexForeground, loss
from mcpnet.training.train import train_variant


# --------------------------------------------------------------------------- #
def load_data_loader(config):
    """Builds a DataLoader (mcpnet.data.data_loader.DataLoader) from config
    values.
    """
    from mcpnet.data.data_loader import DataLoader

    d = config["data"]
    t = config["training"]
    return DataLoader(
        data_path=d["data_path"],
        gt_path=d["gt_path"],
        configs_path=d["configs_path"],
        buffer_size=d["buffer_size"],
        train_batch_size=t["train_batch_size"],
        val_batch_size=t["val_batch_size"],
        augment=True,
        to_train=True,
        adjust_percents=True,
    )


def load_test_cases(config):
    """Yields (case_id, x_volume, y_true_volume, voxelspacing) per patient
    (not per slice).

    Doesn't reuse DataLoader directly: DataLoader.load_dataset flattens every
    patient's slices into one long list and discards patient identity — fine
    for training, but per-case evaluation (Table A/B, the Wilcoxon tests, and
    clinically meaningful HD95) needs each patient's slices kept together as
    one volume. This reads the same raw files DataLoader does, but keeps each
    patient's slices stacked as one volume/case, using the "test" split names
    from configs.json.

    x_volume     : shape (num_slices, H, W, 1)
    y_true_volume: shape (num_slices, H, W) — integer labels, not one-hot
    voxelspacing : (x_mm, y_mm, z_mm) from the NIfTI header
                   (`load_nii(...)[2].get_zooms()`), threaded through to
                   `evaluate_dataset` so `medpy.metric.hd95` gets real-world
                   spacing instead of defaulting to pixel units.

    Voxel spacing is read directly from the ROI file's own header
    (`d["data_path"]`). This assumes the ROI files' headers already carry
    correct spacing (see `fix_roi_spacing.py`) — ROI extraction used padding,
    not resizing, so spacing itself was never altered by cropping, only the
    header metadata needed correcting.
    """
    import json
    from os import walk

    import numpy as np

    from mcpnet.data.preprocessing import standardize

    try:
        from acdc_utilities import load_nii
    except ImportError:
        from mcpnet.data.acdc_utilities_stub import load_nii

    d = config["data"]
    with open(d["configs_path"]) as f:
        split = json.load(f)
    test_names = set(split["test"])

    for (_, _, filenames) in walk(d["data_path"]):
        for filename in filenames:
            if filename[:-3] not in test_names:
                continue

            img_data, _, img_header = load_nii(d["data_path"] + filename)
            img_data = standardize(img_data)
            img_data = np.expand_dims(img_data, axis=-1)  # (H, W, num_slices, 1)

            label_data, _, _ = load_nii(d["gt_path"] + filename)  # (H, W, num_slices)

            x_volume = np.transpose(img_data, (2, 0, 1, 3))  # (num_slices, H, W, 1)
            y_true_volume = np.transpose(label_data, (2, 0, 1)).astype(np.int32)  # (num_slices, H, W)

            zooms = img_header.get_zooms()  # (x_mm, y_mm, z_mm)

            case_id = filename[:-3]
            # x_volume/y_true_volume are transposed to (S, H, W, ...) above, so
            # voxelspacing must be reordered the same way to stay aligned with hd95's array.
            voxelspacing = (zooms[2], zooms[0], zooms[1])  # -> (z_mm, x_mm, y_mm), matching (S, H, W)
            yield case_id, x_volume, y_true_volume, voxelspacing


CUSTOM_OBJECTS = {"loss": loss, "DiceIndex": DiceIndex, "DiceIndexForeground": DiceIndexForeground}


# --------------------------------------------------------------------------- #
# Shared training + evaluation logic for any variant-comparison table
# --------------------------------------------------------------------------- #
def _train_and_evaluate_variants(config, variant_names, data_loader, test_cases, table_label, force_retrain=False, trust_unmarked_checkpoints=None):
    """For each variant: reuses an existing checkpoint from `save_dir` if one
    is already there instead of retraining from scratch. This is what makes
    Table C/D cheap when they share variants with Table A — only the
    variant(s) new to that table actually train.

    Looks for BOTH `{save_dir}/{variant_name}_best.h5` AND its completion
    marker `{save_dir}/{variant_name}_best.h5.complete` before reusing.
    `ModelCheckpoint(save_best_only=True)` writes the .h5 file the moment val
    Dice first improves, which can happen early in training, so an
    interrupted run can leave a .h5 file that looks, by filename alone,
    identical to a fully-trained checkpoint. The marker is written by
    `train_variant` only after `model.fit()` completes all requested epochs
    (see training/train.py), so an interrupted run's checkpoint is correctly
    treated as not safe to reuse.

    `force_retrain=True` retrains every variant regardless of what's on
    disk. Pass a list of variant names instead to retrain selectively.

    A checkpoint trained before the completion-marker system existed will
    never have a `.complete` marker and will always look incomplete here,
    even if it genuinely finished. For such a checkpoint, either create its
    marker file manually (see README) or pass
    `trust_unmarked_checkpoints=[variant_name, ...]` to treat its existing
    .h5 file as trusted without a marker — use deliberately, since it
    reintroduces the interrupted-checkpoint risk for exactly those variants.
    """
    from tensorflow.keras.models import load_model

    save_dir = config["paths"]["save_dir"]
    if isinstance(force_retrain, (list, set, tuple)):
        force_retrain_set = set(force_retrain)
        force_retrain_all = False
    else:
        force_retrain_set = set()
        force_retrain_all = bool(force_retrain)

    trusted_unmarked = set(trust_unmarked_checkpoints or [])

    all_rows = []
    for variant_name in variant_names:
        checkpoint_path = os.path.join(save_dir, f"{variant_name}_best.h5")
        marker_path = checkpoint_path + ".complete"
        must_retrain = force_retrain_all or (variant_name in force_retrain_set)
        marker_ok = os.path.exists(marker_path) or (variant_name in trusted_unmarked)

        if not must_retrain and os.path.exists(checkpoint_path) and marker_ok:
            reuse_reason = "trusted without marker (explicitly requested)" if variant_name in trusted_unmarked and not os.path.exists(marker_path) else "complete marker found"
            print(f"\n=== {table_label}: REUSING existing checkpoint for '{variant_name}' "
                  f"({checkpoint_path}, {reuse_reason}) — skipping training ===")
            model = load_model(checkpoint_path, custom_objects=CUSTOM_OBJECTS)
        else:
            if os.path.exists(checkpoint_path) and not os.path.exists(marker_path) and not must_retrain:
                print(f"\n=== {table_label}: found INCOMPLETE checkpoint for '{variant_name}' "
                      f"(no completion marker — likely from an interrupted run). Retraining from scratch. ===")
            print(f"\n=== {table_label}: training '{variant_name}' (fixed LR, no ensemble) ===")
            model, _, _ = train_variant(
                variant_name=variant_name,
                train_batches=data_loader.train_batches,
                val_batches=data_loader.val_batches,
                train_steps=data_loader.train_step,
                val_steps=data_loader.val_step,
                save_dir=save_dir,
                input_shape=tuple(config["model"]["input_shape"]),
                num_classes=config["model"]["num_classes"],
                use_snapshot_ensemble=False,
                epochs=config["training"]["epochs"],
                fixed_lr=config["training"]["fixed_lr"],
                seed=config["training"].get("seed", 42),
            )

        y_true_list, y_pred_list, case_ids, voxelspacings = [], [], [], []
        for case_id, x, y_true, voxelspacing in test_cases:
            labels, _ = predict_single(model, x)
            y_true_list.append(y_true)
            y_pred_list.append(np.squeeze(labels))
            case_ids.append(case_id)
            voxelspacings.append(voxelspacing)

        df = evaluate_dataset(y_true_list, y_pred_list, case_ids, variant_name, voxelspacings=voxelspacings)
        all_rows.append(df)

    return pd.concat(all_rows, ignore_index=True)


def _save_and_report(full_df, results_dir, table_key, baseline_variant, run_friedman=False):
    os.makedirs(results_dir, exist_ok=True)
    full_df.to_csv(os.path.join(results_dir, f"{table_key}_per_case.csv"), index=False)

    summary = summarize(full_df)
    summary.to_csv(os.path.join(results_dir, f"{table_key}_summary.csv"))
    print(f"\n{table_key} summary:\n", summary)

    n_variants = full_df["variant"].nunique()
    if run_friedman and n_variants >= 3:
        from mcpnet.evaluation.statistical_tests import friedman_omnibus
        omnibus = friedman_omnibus(full_df, metric_col="mean_dice", variant_col="variant")
        print(f"\n{table_key} Friedman omnibus: statistic={omnibus['statistic']:.4f}, "
              f"p={omnibus['p_value']:.4g}, n_cases={omnibus['n_cases']}")
        if omnibus["p_value"] >= 0.05:
            print(f"  -> NOT significant (p>=0.05): pairwise comparisons below are exploratory, "
                  f"not confirmatory, since the omnibus test found no overall difference.")
        correction = "bonferroni" if n_variants > 2 else None
    else:
        correction = None

    sig = compare_variants(full_df, metric_col="mean_dice", variant_col="variant",
                            baseline_variant=baseline_variant, correction=correction)
    sig["significance"] = sig["p_value"].apply(annotate_significance)
    sig.to_csv(os.path.join(results_dir, f"{table_key}_significance.csv"), index=False)
    print(f"\n{table_key} significance (baseline = {baseline_variant}"
          f"{', Bonferroni-corrected' if correction else ''}):\n", sig)

    return summary, sig


# --------------------------------------------------------------------------- #
# Table A — architecture ablation (3 variants)
# --------------------------------------------------------------------------- #
def run_table_a(config, force_retrain=False, trust_unmarked_checkpoints=None):
    data_loader = load_data_loader(config)
    test_cases = list(load_test_cases(config))

    full_df = _train_and_evaluate_variants(
        config, config["ablation"]["table_a"]["variants"], data_loader, test_cases, "Table A",
        force_retrain=force_retrain, trust_unmarked_checkpoints=trust_unmarked_checkpoints,
    )
    summary, sig = _save_and_report(
        full_df, config["paths"]["results_dir"], "table_a", baseline_variant="full_mcp_net", run_friedman=True
    )
    return full_df, summary, sig


# --------------------------------------------------------------------------- #
# Table C — 2x2 factorial: multi-scale pyramid x connections
# --------------------------------------------------------------------------- #
def run_table_c(config, force_retrain=False, trust_unmarked_checkpoints=None):
    data_loader = load_data_loader(config)
    test_cases = list(load_test_cases(config))

    variants = config["ablation"]["table_c"]["variants"]
    # 3 of these 4 variants are already trained by Table A —
    # _train_and_evaluate_variants reuses those checkpoints automatically,
    # only single_path_with_connections actually trains here.
    full_df = _train_and_evaluate_variants(
        config, variants, data_loader, test_cases, "Table C",
        force_retrain=force_retrain, trust_unmarked_checkpoints=trust_unmarked_checkpoints,
    )
    summary, sig = _save_and_report(
        full_df, config["paths"]["results_dir"], "table_c", baseline_variant="full_mcp_net", run_friedman=True
    )

    # Report the 2x2 explicitly: pyramid effect with/without connections held constant
    print("\nTable C — reading the 2x2 factorial:")
    print("  Pyramid effect (no connections):  mcp_net_no_connections  vs  vanilla_unet")
    print("  Pyramid effect (with connections): full_mcp_net            vs  single_path_with_connections")
    print("  Connections effect (single-path):  single_path_with_connections vs vanilla_unet")
    print("  Connections effect (pyramid):      full_mcp_net            vs  mcp_net_no_connections")
    return full_df, summary, sig


# --------------------------------------------------------------------------- #
# Table D — decoder fusion ablation (concat vs. additive skip fusion)
# --------------------------------------------------------------------------- #
def run_table_d(config, force_retrain=False, trust_unmarked_checkpoints=None):
    data_loader = load_data_loader(config)
    test_cases = list(load_test_cases(config))

    variants = config["ablation"]["table_d"]["variants"]
    baseline = config["ablation"]["table_d"]["baseline"]
    # full_mcp_net is already trained by Table A/C — only
    # full_mcp_net_additive_decoder actually trains here, unless force_retrain=True.
    full_df = _train_and_evaluate_variants(
        config, variants, data_loader, test_cases, "Table D",
        force_retrain=force_retrain, trust_unmarked_checkpoints=trust_unmarked_checkpoints,
    )
    summary, sig = _save_and_report(
        full_df, config["paths"]["results_dir"], "table_d", baseline_variant=baseline, run_friedman=False
    )
    return full_df, summary, sig




# --------------------------------------------------------------------------- #
# Table B
# --------------------------------------------------------------------------- #
def run_table_b(config):
    results_dir = config["paths"]["results_dir"]
    save_dir = config["paths"]["save_dir"]
    os.makedirs(results_dir, exist_ok=True)

    data_loader = load_data_loader(config)
    test_cases = list(load_test_cases(config))
    variant_name = config["ablation"]["table_b"]["architecture"]

    # one cyclic training run produces all snapshots needed
    print(f"\n=== Table B: training '{variant_name}' with cyclic LR / snapshot ensemble ===")
    _, _, snapshot_paths = train_variant(
        variant_name=variant_name,
        train_batches=data_loader.train_batches,
        val_batches=data_loader.val_batches,
        train_steps=data_loader.train_step,
        val_steps=data_loader.val_step,
        save_dir=save_dir,
        input_shape=tuple(config["model"]["input_shape"]),
        num_classes=config["model"]["num_classes"],
        use_snapshot_ensemble=True,
        epochs=config["training"]["epochs"],
        nb_cycles=config["training"]["nb_cycles"],
        max_lr=config["training"]["max_lr"],
        seed=config["training"].get("seed", 42),
    )
    print(f"Collected {len(snapshot_paths)} snapshots: {snapshot_paths}")

    all_rows = []

    # optional anchor row: conventional fixed-LR single model
    if config["ablation"]["table_b"].get("include_single_fixed_lr_baseline", True):
        print("\n--- Row: single model, fixed LR, no cyclic schedule ---")
        fixed_model, _, _ = train_variant(
            variant_name=variant_name,
            run_name=f"{variant_name}_fixedlr_anchor",  # distinct checkpoint label so it doesn't overwrite Table A's
            train_batches=data_loader.train_batches,
            val_batches=data_loader.val_batches,
            train_steps=data_loader.train_step,
            val_steps=data_loader.val_step,
            save_dir=save_dir,
            input_shape=tuple(config["model"]["input_shape"]),
            num_classes=config["model"]["num_classes"],
            use_snapshot_ensemble=False,
            epochs=config["training"]["epochs"],
            fixed_lr=config["training"]["fixed_lr"],
            seed=config["training"].get("seed", 42),
        )
        y_true_list, y_pred_list, case_ids, voxelspacings = [], [], [], []
        for case_id, x, y_true, voxelspacing in test_cases:
            labels, _ = predict_single(fixed_model, x)
            y_true_list.append(y_true)
            y_pred_list.append(np.squeeze(labels))
            case_ids.append(case_id)
            voxelspacings.append(voxelspacing)
        df = evaluate_dataset(y_true_list, y_pred_list, case_ids, "M1_fixed_lr", voxelspacings=voxelspacings)
        all_rows.append(df)

    # M=1, M=3, M=5, M=10 rows — all reuse the SAME trained snapshots, just
    # varying how many are ensembled at inference.
    #
    # Selecting the earliest m cycles (rather than the most recent) matters
    # under warm-restart cyclic LR: each cycle continues from the previous
    # cycle's converged weights, so later cycles tend to converge toward
    # increasingly similar solutions, while earlier cycles (closer to the
    # original random init) are more likely to differ from each other.
    # Taking the most recent m as m grows mostly adds redundant, similar
    # late-cycle snapshots. Configurable via table_b.snapshot_selection to
    # compare both orderings.
    snapshot_selection = config["ablation"]["table_b"].get("snapshot_selection", "earliest")

    for m in config["ablation"]["table_b"]["snapshot_counts"]:
        m = min(m, len(snapshot_paths))
        label = f"M{m}_cyclic"
        print(f"\n--- Row: {label} ({m} snapshot(s), selection='{snapshot_selection}') ---")
        selected_paths = snapshot_paths[:m] if snapshot_selection == "earliest" else snapshot_paths[-m:]
        members = load_snapshot_members(selected_paths, custom_objects=CUSTOM_OBJECTS)

        y_true_list, y_pred_list, case_ids, voxelspacings = [], [], [], []
        for case_id, x, y_true, voxelspacing in test_cases:
            if m == 1:
                labels, _ = predict_single(members[0], x)
            else:
                labels = ensemble_predict(members, x)
            y_true_list.append(y_true)
            y_pred_list.append(np.squeeze(labels))
            case_ids.append(case_id)
            voxelspacings.append(voxelspacing)

        df = evaluate_dataset(y_true_list, y_pred_list, case_ids, label, voxelspacings=voxelspacings)
        all_rows.append(df)

    full_df = pd.concat(all_rows, ignore_index=True)
    full_df.to_csv(os.path.join(results_dir, "table_b_per_case.csv"), index=False)

    summary = summarize(full_df)
    summary.to_csv(os.path.join(results_dir, "table_b_summary.csv"))
    print("\nTable B summary:\n", summary)

    max_m_label = f"M{min(config['ablation']['table_b']['snapshot_counts'][-1], len(snapshot_paths))}_cyclic"
    sig = compare_variants(full_df, metric_col="mean_dice", variant_col="variant", baseline_variant=max_m_label)
    sig["significance"] = sig["p_value"].apply(annotate_significance)
    sig.to_csv(os.path.join(results_dir, "table_b_significance.csv"), index=False)
    print(f"\nTable B significance (baseline = {max_m_label}):\n", sig)

    return full_df, summary, sig


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mcp_net_config.yaml")
    parser.add_argument("--table", type=str, choices=["A", "B", "C", "D", "all"], default="all")
    parser.add_argument("--force-retrain", action="store_true",
                         help="Retrain every variant even if a checkpoint already exists in save_dir "
                              "(e.g. from a previous Table A run). Only affects Tables A/C/D — Table B "
                              "always trains its own cyclic-LR snapshots regardless.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.table in ("A", "all"):
        run_table_a(cfg, force_retrain=args.force_retrain)
    if args.table in ("B", "all"):
        run_table_b(cfg)
    if args.table in ("C", "all"):
        run_table_c(cfg, force_retrain=args.force_retrain)
    if args.table in ("D", "all"):
        run_table_d(cfg, force_retrain=args.force_retrain)
