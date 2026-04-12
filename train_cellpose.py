"""
train_cellpose.py -- Fine-tune CellPose on auto-annotated proppant particle masks.

Starts from the pretrained 'cpsam' model and fine-tunes on your 705-image
particle dataset (auto-annotated into 512x512 patches).

Prerequisites:
    python -m annotation.generate_masks    <- builds dataset/stardist/

GPU:
    CellPose uses PyTorch -- GPU is enabled automatically if CUDA is available.
    RTX 3070 will make training ~10-20x faster than CPU.

Output:
    runs/cellpose_proppant/models/cellpose_proppant
    (update CELLPOSE_MODEL_PATH in config.py after training)

Usage:
    python train_cellpose.py
    python train_cellpose.py --epochs 200 --batch 16
    python train_cellpose.py --cpu          # force CPU
    python train_cellpose.py --max-samples 500   # reduce if RAM is tight
"""
import argparse
import random
import sys
import numpy as np
from pathlib import Path

# Max patches to load into RAM.
# Memory estimate per patch (512x512):
#   image  : 1 MB  (float32)
#   mask   : 0.5 MB (uint16)
#   flows  : 4 MB  (4ch float32, computed by CellPose)
# 1500 patches -> ~8.3 GB total -- safe on 16 GB RAM
DEFAULT_MAX_TRAIN = 1500
DEFAULT_MAX_VAL   = 300


def _load_split(split_dir: Path, max_samples: int, seed: int = 42):
    """Load X/Y npy pairs, randomly subsampling if the split is too large."""
    img_dir  = split_dir / "images"
    mask_dir = split_dir / "masks"

    if not img_dir.exists():
        return [], []

    pairs = []
    for img_path in sorted(img_dir.glob("*.npy")):
        mask_path = mask_dir / img_path.name
        if mask_path.exists():
            pairs.append((img_path, mask_path))

    if len(pairs) > max_samples:
        rng = random.Random(seed)
        pairs = rng.sample(pairs, max_samples)
        pairs.sort()
        print(f"  (subsampled to {max_samples} patches to stay within RAM)")

    X, Y = [], []
    for img_path, mask_path in pairs:
        x = np.load(str(img_path)).astype(np.float32)
        y = np.load(str(mask_path)).astype(np.uint16)
        X.append(x)
        Y.append(y)

    return X, Y


def _estimate_diameter(Y_sample):
    """Estimate median particle diameter from a list of mask arrays."""
    all_diameters = []
    for y in Y_sample[:50]:
        counts = np.bincount(y.ravel())[1:]
        counts = counts[counts > 0]
        all_diameters.extend((2.0 * np.sqrt(counts / np.pi)).tolist())
    if all_diameters:
        return float(np.median(all_diameters))
    return None


def train(epochs: int, batch_size: int, learning_rate: float, use_gpu: bool,
          max_train: int, max_val: int):
    from config import STARDIST_DATA_DIR, CELLPOSE_MODEL_PATH, CELLPOSE_PRETRAINED

    import torch
    gpu_available = torch.cuda.is_available() and use_gpu
    if gpu_available:
        print(f"[GPU] {torch.cuda.get_device_name(0)} -- training on GPU")
    else:
        print("[CPU] No GPU / CPU mode forced")

    from cellpose import models, train as cp_train, io as cp_io
    cp_io.logger_setup()

    # -- Load dataset ----------------------------------------------------------
    print(f"\n-- Loading dataset from {STARDIST_DATA_DIR} --")
    X_train, Y_train = _load_split(STARDIST_DATA_DIR / "train", max_train)
    X_val,   Y_val   = _load_split(STARDIST_DATA_DIR / "val",   max_val)

    if not X_train:
        print("ERROR: No training data found.")
        print("Run first:  python -m annotation.generate_masks")
        sys.exit(1)

    print(f"  Train: {len(X_train)} patches loaded")
    print(f"  Val:   {len(X_val)}   patches loaded")

    median_d = _estimate_diameter(Y_train)
    if median_d:
        print(f"  Median particle diameter: {median_d:.1f} px")

    # -- Load base model -------------------------------------------------------
    print(f"\n-- Fine-tuning from pretrained '{CELLPOSE_PRETRAINED}' --")
    model = models.CellposeModel(
        gpu=gpu_available,
        model_type=CELLPOSE_PRETRAINED,
    )

    # -- Training --------------------------------------------------------------
    save_dir = CELLPOSE_MODEL_PATH.parent.parent   # runs/cellpose_proppant/
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"  epochs       : {epochs}")
    print(f"  batch_size   : {batch_size}")
    print(f"  learning_rate: {learning_rate}")
    print(f"  save_path    : {save_dir}")
    print()

    model_path = cp_train.train_seg(
        model.net,
        train_data      = X_train,
        train_labels    = Y_train,
        test_data       = X_val   if X_val   else None,
        test_labels     = Y_val   if Y_val   else None,
        normalize       = True,
        save_path       = str(save_dir),
        model_name      = "cellpose_proppant",
        n_epochs        = epochs,
        learning_rate   = learning_rate,
        weight_decay    = 1e-5,
        batch_size      = batch_size,
        min_train_masks = 1,
        nimg_per_epoch  = min(200, len(X_train)),
    )

    print(f"\n-- Training complete ---------------------------------")
    print(f"Model saved to: {model_path}")
    print()
    print("Next steps:")
    print(f"  1. Update config.py: CELLPOSE_MODEL_PATH = Path(r'{model_path}')")
    print(f"  2. Run: python app.py --windowed")


def main():
    from config import CELLPOSE_TRAIN_EPOCHS, CELLPOSE_BATCH_SIZE, CELLPOSE_USE_GPU

    parser = argparse.ArgumentParser(description="Fine-tune CellPose on proppant particles")
    parser.add_argument("--epochs",      type=int,   default=CELLPOSE_TRAIN_EPOCHS)
    parser.add_argument("--batch",       type=int,   default=CELLPOSE_BATCH_SIZE)
    parser.add_argument("--lr",          type=float, default=2e-4)
    parser.add_argument("--cpu",         action="store_true", help="Force CPU")
    parser.add_argument("--max-samples", type=int,   default=DEFAULT_MAX_TRAIN,
                        help=f"Max training patches to load (default {DEFAULT_MAX_TRAIN}). "
                             "Reduce to 500 if you hit OOM.")
    args = parser.parse_args()

    max_val = max(50, args.max_samples // 5)

    train(
        epochs        = args.epochs,
        batch_size    = args.batch,
        learning_rate = args.lr,
        use_gpu       = CELLPOSE_USE_GPU and not args.cpu,
        max_train     = args.max_samples,
        max_val       = max_val,
    )


if __name__ == "__main__":
    main()
