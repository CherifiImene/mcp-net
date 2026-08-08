"""
patch_nnunet_determinism.py — standalone, one-time script to fix a confirmed reproducibility gap in the installed
nnunetv2 package.

`nnunetv2/run/run_training.py` hardcodes, in BOTH its single-GPU and DDP
training paths:
    if torch.cuda.is_available():
        cudnn.deterministic = False
        cudnn.benchmark = True
nnU-Net prioritizes speed over reproducibility by default. Since this line lives 
INSIDE the function the `nnUNetv2_train` CLI command calls, monkeypatching from an external script
run before the CLI command has no effect — nnU-Net's own line executes and
overwrites any prior setting every time. The only reliable fix is editing
the installed source directly, which is what this script does.

WHAT THIS DOES NOT FIX:
nnU-Net's default data augmentation pipeline uses
`NonDetMultiThreadedAugmenter` (nnunetv2/training/nnUNetTrainer/
nnUNetTrainer.py) a genuinely non-deterministic multi-process augmenter,
by explicit design (its name says so), used for training throughput. Its
deterministic sibling class (`MultiThreadedAugmenter`, imported in the same
file) exists but swapping it in is a more invasive change to nnU-Net's own
trainer logic than the cudnn flags below, with real risk of subtly breaking
something in a framework we don't maintain. NOT patched here

USAGE:
    Run this ONCE after `pip install nnunetv2`, before training. Safe to
    run multiple times (idempotent — checks current state before editing).
"""

import os
import nnunetv2

RUN_TRAINING_PATH = os.path.join(os.path.dirname(nnunetv2.__file__), "run", "run_training.py")

OLD_PATTERN = "cudnn.deterministic = False\n        cudnn.benchmark = True"
NEW_PATTERN = "cudnn.deterministic = True\n        cudnn.benchmark = False"

# The DDP path has different indentation (12 spaces vs 8) — handle both explicitly
OLD_PATTERN_DDP = "cudnn.deterministic = False\n            cudnn.benchmark = True"
NEW_PATTERN_DDP = "cudnn.deterministic = True\n            cudnn.benchmark = False"


def main():
    print(f"Patching: {RUN_TRAINING_PATH}")

    if not os.path.exists(RUN_TRAINING_PATH):
        raise FileNotFoundError(
            f"Could not find run_training.py at {RUN_TRAINING_PATH} — is nnunetv2 installed? "
            f"(pip install nnunetv2)"
        )

    with open(RUN_TRAINING_PATH) as f:
        content = f.read()

    already_patched = "cudnn.deterministic = True" in content
    if already_patched:
        print("Already patched (found 'cudnn.deterministic = True') — nothing to do.")
        return

    n_occurrences_single = content.count(OLD_PATTERN)
    n_occurrences_ddp = content.count(OLD_PATTERN_DDP)

    if n_occurrences_single == 0 and n_occurrences_ddp == 0:
        raise RuntimeError(
            "Could not find the expected 'cudnn.deterministic = False' pattern in "
            "run_training.py — nnunetv2's internals may have changed since this script "
            "was written. Open the file manually and check for cudnn.deterministic / "
            "cudnn.benchmark lines near the training entry points (run_training / run_ddp)."
        )

    content = content.replace(OLD_PATTERN_DDP, NEW_PATTERN_DDP)
    content = content.replace(OLD_PATTERN, NEW_PATTERN)

    with open(RUN_TRAINING_PATH, "w") as f:
        f.write(content)

    print(f"Patched {n_occurrences_single + n_occurrences_ddp} occurrence(s): "
          f"cudnn.deterministic False->True, cudnn.benchmark True->False")
    print("\nNote: cudnn.benchmark=False may make training somewhat slower (it disables "
          "cuDNN's runtime algorithm autotuning, which is itself a source of "
          "non-determinism) — the same speed/reproducibility tradeoff already accepted "
          "for MCP-Net's own training via deterministic_ops=True.")
    print("\nStill NOT fixed (see this script's docstring): nnU-Net's data augmentation "
          "pipeline uses NonDetMultiThreadedAugmenter by design. Disclose this as a "
          "residual limitation in your methods section.")


if __name__ == "__main__":
    main()