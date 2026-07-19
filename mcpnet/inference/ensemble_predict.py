"""
Inference utilities for both single-model and snapshot-ensemble prediction.
"""

from tensorflow.keras.models import load_model

import numpy as np


def load_snapshot_members(snapshot_paths, custom_objects):
    """`custom_objects` must include the loss + DiceIndex (and any other
    custom layers/metrics) so `load_model` can deserialize the .h5 files.
    """
    return [load_model(p, custom_objects=custom_objects) for p in snapshot_paths]


def predict_single(model, x):
    """M=1 case: plain single-model softmax -> argmax."""
    probs = model.predict(x)
    return np.argmax(probs, axis=-1), probs


def ensemble_predict(members, x, return_probs=False):
    """M>1 case: soft-vote across snapshot members.

    members : list of loaded Keras models (a subset or all snapshots — lets
              Table B sweep M = 1, 3, 5, 10 from one training run without
              retraining, by varying how many snapshots are passed in).
    x        : input batch, shape (batch, H, W, C)

    Returns: hard labels (batch, H, W), and optionally the averaged
    probability map (batch, H, W, num_classes).
    """
    predictions = [m.predict(x) for m in members]  # each: (batch, H, W, num_classes)
    predictions = np.array(predictions)  # (M, batch, H, W, num_classes)

    merged_probs = np.sum(predictions, axis=0)
    merged_labels = np.argmax(merged_probs, axis=-1)

    if return_probs:
        return merged_labels, merged_probs
    return merged_labels
