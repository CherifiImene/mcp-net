"""
Snapshot Ensemble callback (cyclic cosine annealing + periodic checkpointing).

The cosine schedule uses `epoch % epoch_per_cycle` so epoch 0 (and the start
of every cycle) trains at `max_lr`, matching the standard formulation
(Loshchilov & Hutter 2017 / Huang et al. 2017).

The LR updates once per epoch, not per batch — a coarser-grained cyclic
schedule than the original Snapshot Ensembles paper, which anneals
per-iteration.
"""

import tensorflow as tf
from math import pi


class SnapshotEnsemble(tf.keras.callbacks.Callback):
    def __init__(self, max_lr, total_epochs, batches_per_epoch, nb_cycles, save_dir, run_name="snapshot"):
        super().__init__()
        self.max_lr = max_lr
        self.nb_epochs = total_epochs
        self.batches_per_epoch = batches_per_epoch  # kept for reference / future per-batch version
        self.nb_cycles = nb_cycles
        self.save_dir = save_dir
        self.run_name = run_name
        self.lrates = []
        self.snapshot_paths = []

    def cosine_annealing(self, max_lr, epoch, total_epochs, nb_cycles):
        epoch_per_cycle = tf.floor(total_epochs / nb_cycles)
        lr = (max_lr / 2) * (tf.math.cos(pi * (epoch % epoch_per_cycle) / epoch_per_cycle) + 1)
        return lr

    def on_epoch_begin(self, epoch, logs=None):
        lr = self.cosine_annealing(self.max_lr, epoch, self.nb_epochs, self.nb_cycles)
        # Keras 3 renamed optimizer.lr -> optimizer.learning_rate.
        try:
            self.model.optimizer.learning_rate.assign(lr)
        except AttributeError:
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        self.lrates.append(float(lr))

    def on_epoch_end(self, epoch, logs=None):
        epoch_per_cycle = tf.floor(self.nb_epochs / self.nb_cycles)
        if epoch != 0 and (epoch + 1) % epoch_per_cycle == 0:
            m = (epoch + 1) // epoch_per_cycle
            filepath = f"{self.save_dir}/{self.run_name}_cycle_{int(m)}.h5"
            self.model.save(filepath)
            self.snapshot_paths.append(filepath)
            print(f">saved snapshot {filepath}, epoch: {epoch}")
