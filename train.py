"""
train.py — YOLOv8 Instance Segmentation Training for Proppant QC.

Usage:
    python train.py                          # Train with defaults
    python train.py --epochs 200 --imgsz 1280
    python train.py --model yolov8s-seg.pt   # Use small model instead of nano
    python train.py --resume                 # Resume interrupted training
    python train.py --device 0               # Force GPU 0
"""
import argparse
import sys
from ultralytics import YOLO
from config import (
    DATASET_YAML, RUNS_DIR, MODEL_BASE,
    TRAIN_EPOCHS, TRAIN_IMGSZ, TRAIN_BATCH,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8-seg for proppant QC")
    p.add_argument("--model", default=MODEL_BASE,
                   help="Base model or checkpoint path (default: %(default)s)")
    p.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    p.add_argument("--imgsz", type=int, default=TRAIN_IMGSZ)
    p.add_argument("--batch", type=int, default=TRAIN_BATCH)
    p.add_argument("--device", default="0",
                   help="Device: '0' = GPU 0 (default), 'cpu' = CPU")
    p.add_argument("--resume", action="store_true",
                   help="Resume training from last checkpoint")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  Proppant QC — YOLOv8-seg Training")
    print("=" * 60)
    print(f"  Model:   {args.model}")
    print(f"  Epochs:  {args.epochs}")
    print(f"  ImgSize: {args.imgsz}")
    print(f"  Batch:   {args.batch}")
    print(f"  Device:  {args.device or 'auto'}")
    print(f"  Dataset: {DATASET_YAML}")
    print("=" * 60)

    # Verify dataset YAML exists
    if not DATASET_YAML.exists():
        print(f"\nERROR: Dataset YAML not found at {DATASET_YAML}")
        print("Run prepare_dataset.py first, then annotate your images.")
        sys.exit(1)

    model = YOLO(args.model)

    # ── Train ──────────────────────────────────────────────────────────
    model.train(
        data=str(DATASET_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device if args.device else None,
        project=str(RUNS_DIR),
        name="proppant_seg",
        exist_ok=True,
        resume=args.resume,

        # Augmentation — tuned for dense overlapping particle segmentation
        augment=True,
        hsv_h=0.02,        # Slight hue shift
        hsv_s=0.5,         # More saturation variation
        hsv_v=0.5,         # More brightness variation (helps with shadows)
        degrees=180.0,     # Full rotation (particles are round — all angles valid)
        translate=0.15,
        scale=0.5,         # More scale jitter for size variety
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,        # Always mosaic for dense particle training
        mixup=0.15,
        copy_paste=0.4,    # Copy-paste teaches overlapping instances
        close_mosaic=50,   # Disable mosaic last 50 epochs for fine-tuning
        overlap_mask=False, # Each instance keeps its own mask (critical for segmentation)

        # Training hyperparameters
        lr0=0.01,
        lrf=0.005,
        warmup_epochs=5,
        patience=80,
        val=True,
        plots=True,
    )

    # ── Post-training validation ───────────────────────────────────────
    best_path = RUNS_DIR / "proppant_seg" / "weights" / "best.pt"
    if best_path.exists():
        print("\n" + "=" * 60)
        print("  Post-Training Validation")
        print("=" * 60)
        best_model = YOLO(str(best_path))
        metrics = best_model.val(data=str(DATASET_YAML))
        print(f"  mAP50  (box):  {metrics.box.map50:.4f}")
        print(f"  mAP50  (mask): {metrics.seg.map50:.4f}")
        print(f"  mAP50-95 (mask): {metrics.seg.map:.4f}")
        print("=" * 60)
        print(f"\n  Best model saved to: {best_path}")
    else:
        print("\nWARNING: best.pt not found — training may have failed.")

    print("\nDone. To run inference: python app.py")


if __name__ == "__main__":
    main()
