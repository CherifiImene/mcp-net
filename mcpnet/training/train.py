"""
Generic training loop used for both ablation tables.

  * Table A (architecture ablation): call with `use_snapshot_ensemble=False`
    and a fixed (non-cyclic) learning rate, for each of the 3 architecture
    variants in mcpnet.model.architecture.VARIANT_BUILDERS.

  * Table B (ensemble ablation): call with `use_snapshot_ensemble=True` and
    `variant_name="full_mcp_net"` only — the ensemble sweep doesn't need to
    be redone for every architecture variant, just the final one.

Takes `train_batches`, `val_batches`, `train_steps`, `val_steps` directly
rather than building a data loader internally — wired in from
`scripts/run_ablation.py`.
"""

import os

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

from mcpnet.model.architecture import get_model
from mcpnet.training.callbacks import SnapshotEnsemble
from mcpnet.training.losses import DiceIndexForeground, loss
from mcpnet.utils.seeding import set_global_seed

# Keras 3 auto-names a metric passed as a plain function by converting its
# __name__ to snake_case: `DiceIndex` -> logged key 'val_dice_index', not
# 'val_DiceIndex'. Monitoring the wrong key makes ModelCheckpoint save every
# epoch regardless of whether the metric improved.
MONITOR_METRIC_NAME = "val_dice_index_foreground"


def train_variant(
    variant_name,
    train_batches,
    val_batches,
    train_steps,
    val_steps,
    save_dir,
    input_shape=(128, 128, 1),
    num_classes=4,
    use_snapshot_ensemble=False,
    epochs=300,
    nb_cycles=10,
    max_lr=0.01,
    fixed_lr=1e-3,
    seed=42,
    run_name=None,
):
    """Trains one architecture variant and returns (model, history, snapshot_paths).

    `snapshot_paths` is empty unless `use_snapshot_ensemble=True`.

    `variant_name` must be a key in VARIANT_BUILDERS (e.g. "full_mcp_net") —
    passed to `get_model()` to build the architecture.

    `run_name` defaults to `variant_name`; it labels checkpoint filenames and
    SnapshotEnsemble's internal naming, letting the same architecture be
    trained more than once under different labels without overwriting an
    existing checkpoint for that architecture.

    `seed` fixes weight initialization; combined with seeding the data
    loader's shuffle/augmentation with the same value, the full run becomes
    reproducible. Uses the same seed for every variant by default so only
    architecture differs between them, not initialization.
    """
    set_global_seed(seed)
    os.makedirs(save_dir, exist_ok=True)
    run_name = run_name or variant_name

    model = get_model(variant_name, input_shape=input_shape, num_classes=num_classes)
    optimizer = tf.keras.optimizers.Adam(learning_rate=(max_lr if use_snapshot_ensemble else fixed_lr))
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy", DiceIndexForeground])

    callbacks = []
    snapshot_cb = None
    if use_snapshot_ensemble:
        snapshot_cb = SnapshotEnsemble(
            max_lr=max_lr,
            total_epochs=epochs,
            batches_per_epoch=train_steps,
            nb_cycles=nb_cycles,
            save_dir=save_dir,
            run_name=run_name,
        )
        callbacks.append(snapshot_cb)
    else:
        # fixed-LR baseline: keep best checkpoint by foreground val Dice
        # (RV/Myo/LV only, excluding background — see MONITOR_METRIC_NAME)
        callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(save_dir, f"{run_name}_best.h5"),
                monitor=MONITOR_METRIC_NAME,
                mode="max",
                save_best_only=True,
                verbose=1,
            )
        )

    history = model.fit(
        train_batches,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_data=val_batches,
        validation_steps=val_steps,
        callbacks=callbacks,
    )

    # Completion marker, written only once fit() has run all requested
    # epochs — ModelCheckpoint(save_best_only=True) writes {run_name}_best.h5
    # as soon as val Dice first improves, so an interrupted run can leave a
    # checkpoint indistinguishable by filename from a finished one.
    # run_ablation.py checks for this marker before reusing a checkpoint.
    epochs_completed = len(history.history.get("loss", []))
    if epochs_completed >= epochs:
        marker_path = os.path.join(save_dir, f"{run_name}_best.h5.complete")
        with open(marker_path, "w") as f:
            f.write(f"epochs_completed={epochs_completed}\nrequested_epochs={epochs}\nseed={seed}\n")
    else:
        print(f"WARNING: '{run_name}' only completed {epochs_completed}/{epochs} epochs "
              f"(likely interrupted) — no completion marker written. A checkpoint .h5 file may "
              f"still exist from mid-training saves, but will NOT be reused by run_ablation.py "
              f"until this variant is retrained to completion.")

    snapshot_paths = snapshot_cb.snapshot_paths if snapshot_cb is not None else []
    return model, history, snapshot_paths
