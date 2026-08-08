"""
nnUNetTrainerSeeded_250epochs.py — combines the two custom trainers
(seeding + reduced epoch budget) into one.

INSTALLATION: copy to:
    <nnunetv2 install path>/training/nnUNetTrainer/variants/nnUNetTrainerSeeded_250epochs.py

USAGE:
    nnUNetv2_train 27 2d 0 -tr nnUNetTrainerSeeded_250epochs --c
"""

import random

import numpy as np
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerSeeded_250epochs(nnUNetTrainer):
    SEED = 5  # match mcpnet project's seed

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        random.seed(self.SEED)
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)
        torch.cuda.manual_seed(self.SEED)
        torch.cuda.manual_seed_all(self.SEED)
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.num_epochs = 250
        self.print_to_log_file(
            f"nnUNetTrainerSeeded_250epochs: seeded with SEED={self.SEED}, num_epochs=250"
        )