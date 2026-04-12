"""
train_stardist.py — Train StarDist2D on auto-annotated proppant particle masks.

Prerequisites:
    1. python -m annotation.generate_masks   ← builds dataset/stardist/
    2. This script                            ← trains the model

GPU usage:
    Training automatically uses GPU if TF detects one.
    On Windows with CUDA: ensure CUDA 11.x + cuDNN 8.x are installed.
    On Jetson: use NVIDIA's TF L4T wheel (already has GPU support).

Output:
    runs/stardist_proppant/   ← model directory
      config.json
      thresholds.json
      weights_best.h5         ← best validation loss
      weights_last.h5

Usage:
    python train_stardist.py
    python train_stardist.py --epochs 400 --batch 2   # override defaults
    python train_stardist.py --resume                 # continue from last checkpoint
"""
import argparse
import os
import sys
import numpy as np
from pathlib import Path

# ── GPU setup — must happen BEFORE importing TensorFlow ──────────────────────

def _configure_gpu(use_gpu: bool):
    """Set TF GPU memory growth and visible devices."""
    import tensorflow as tf

    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("[GPU] Disabled — running on CPU")
        return False

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("[GPU] No GPU detected — falling back to CPU")
        return False

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    print(f"[GPU] Using {len(gpus)} GPU(s): {[g.name for g in gpus]}")
    return True


# ── Augmentation ──────────────────────────────────────────────────────────────

def _make_augmenter():
    """
    Return a function (x, y) → (x_aug, y_aug) with random flips and rotations.
    StarDist applies this per-patch during training.
    """
    from stardist.models import StarDist2D  # noqa
    import tensorflow as tf

    def augment(x, y):
        # Random horizontal flip
        if np.random.rand() > 0.5:
            x = x[:, ::-1]
            y = y[:, ::-1]
        # Random vertical flip
        if np.random.rand() > 0.5:
            x = x[::-1, :]
            y = y[::-1, :]
        # Random 90° rotation
        k = np.random.randint(0, 4)
        x = np.rot90(x, k)
        y = np.rot90(y, k)
        return x, y

    return augment


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_split(split_dir: Path) -> "tuple[list, list]":
    """Load all X/Y pairs from a split directory. Returns (X_list, Y_list)."""
    img_dir  = split_dir / "images"
    mask_dir = split_dir / "masks"

    if not img_dir.exists() or not mask_dir.exists():
        return [], []

    X, Y = [], []
    for img_path in sorted(img_dir.glob("*.npy")):
        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            continue
        x = np.load(str(img_path)).astype(np.float32)
        y = np.load(str(mask_path)).astype(np.int32)
        X.append(x)
        Y.append(y)

    return X, Y


# ── Main training ─────────────────────────────────────────────────────────────

def train(
    epochs: int,
    batch_size: int,
    patch_size: tuple,
    n_rays: int,
    use_gpu: bool,
    resume: bool,
):
    from config import STARDIST_MODEL_DIR, STARDIST_DATA_DIR

    import tensorflow as tf
    _configure_gpu(use_gpu)

    # Lazy import after GPU setup
    from stardist import fill_label_holes
    from stardist.models import Config2D, StarDist2D
    from csbdeep.utils import normalize

    print(f"\n── Loading dataset from {STARDIST_DATA_DIR} ──")
    X_train, Y_train = _load_split(STARDIST_DATA_DIR / "train")
    X_val,   Y_val   = _load_split(STARDIST_DATA_DIR / "val")

    if not X_train:
        print("ERROR: No training data found.")
        print("Run first:  python -m annotation.generate_masks")
        sys.exit(1)

    print(f"  Train: {len(X_train)} images")
    print(f"  Val:   {len(X_val)} images")

    # ── Fill label holes (StarDist best practice) ─────────────────────────────
    Y_train = [fill_label_holes(y) for y in Y_train]
    Y_val   = [fill_label_holes(y) for y in Y_val]

    # ── Verify patch size vs label completeness ───────────────────────────────
    # StarDist requires label extents < half patch size
    # (particles must fit within a patch)
    n_channel = 1  # grayscale

    # ── Model config ──────────────────────────────────────────────────────────
    model_dir  = STARDIST_MODEL_DIR.parent
    model_name = STARDIST_MODEL_DIR.name

    if resume and STARDIST_MODEL_DIR.exists():
        print(f"[Resume] Loading existing model from {STARDIST_MODEL_DIR}")
        model = StarDist2D(None, name=model_name, basedir=str(model_dir))
    else:
        conf = Config2D(
            n_rays          = n_rays,
            grid            = (2, 2),      # 2× downsampled output — efficient
            n_channel_in    = n_channel,
            train_patch_size= patch_size,
            train_batch_size= batch_size,
            train_epochs    = epochs,
            train_learning_rate = 3e-4,
            # Validation
            train_tensorboard_loss_smoothing = 0.5,
        )
        print(f"\n── Model config ──")
        print(f"  n_rays       : {n_rays}")
        print(f"  grid         : (2, 2)")
        print(f"  patch_size   : {patch_size}")
        print(f"  batch_size   : {batch_size}")
        print(f"  epochs       : {epochs}")

        model = StarDist2D(conf, name=model_name, basedir=str(model_dir))

    # ── Median size check (verifies dataset particle scale) ───────────────────
    from stardist import calculate_extents
    median_size = calculate_extents(Y_train)
    fov = np.array(model.config.train_patch_size)
    print(f"\n[Check] Median particle extent : {median_size}")
    print(f"[Check] Field of view (patch)  : {fov}")
    if any(median_size > fov / 2):
        print("WARNING: Particles may be too large for patch size — consider increasing STARDIST_PATCH_SIZE")

    # ── Augmentation ──────────────────────────────────────────────────────────
    augmenter = _make_augmenter()

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n── Starting training ────────────────────────────")
    model.train(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        augmenter=augmenter,
        epochs=epochs,
        steps_per_epoch=max(10, len(X_train) // batch_size),
        seed=42,
    )

    # ── Threshold optimisation ────────────────────────────────────────────────
    print("\n── Optimising detection thresholds on validation set ──")
    model.optimize_thresholds(X_val, Y_val)

    print(f"\nModel saved to: {STARDIST_MODEL_DIR}")
    print("Next: update config.py STARDIST_MODEL_DIR if needed, then run app.py")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    from config import (
        STARDIST_TRAIN_EPOCHS, STARDIST_BATCH_SIZE,
        STARDIST_PATCH_SIZE, STARDIST_N_RAYS, STARDIST_USE_GPU,
    )

    parser = argparse.ArgumentParser(description="Train StarDist proppant model")
    parser.add_argument("--epochs",  type=int,   default=STARDIST_TRAIN_EPOCHS)
    parser.add_argument("--batch",   type=int,   default=STARDIST_BATCH_SIZE)
    parser.add_argument("--patch",   type=int,   default=STARDIST_PATCH_SIZE[0],
                        help="Patch size (square, default from config)")
    parser.add_argument("--rays",    type=int,   default=STARDIST_N_RAYS)
    parser.add_argument("--cpu",     action="store_true",
                        help="Force CPU even if GPU available")
    parser.add_argument("--resume",  action="store_true",
                        help="Continue training from existing checkpoint")
    args = parser.parse_args()

    train(
        epochs     = args.epochs,
        batch_size = args.batch,
        patch_size = (args.patch, args.patch),
        n_rays     = args.rays,
        use_gpu    = STARDIST_USE_GPU and not args.cpu,
        resume     = args.resume,
    )


if __name__ == "__main__":
    main()
