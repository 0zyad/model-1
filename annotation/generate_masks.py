"""
annotation/generate_masks.py — Auto-generate per-particle instance masks for CellPose training.

FAST approach: extract 512×512 patches from raw images FIRST, then run
watershed annotation on each small patch.

Why this is faster:
    Full 20MP image (3648×5472) watershed: ~10s per image → 705 images = 2 hours
    Patch-level (512×512) watershed:        ~0.1s per patch → 14000 patches = ~23 min

Output directory:
    dataset/stardist/
        train/images/*.npy  (uint8 grayscale, 512×512)
        train/masks/*.npy   (int32 instance labels, 512×512)
        val/images/*.npy
        val/masks/*.npy

Usage:
    python -m annotation.generate_masks           # full run
    python -m annotation.generate_masks --check   # dry-run stats only
    python -m annotation.generate_masks --clean   # wipe and rebuild
"""
import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

try:
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    _SKIMAGE_OK = True
except ImportError:
    _SKIMAGE_OK = False
    print("WARNING: scikit-image not found. Install: pip install scikit-image")

from config import (
    SOURCE_DIRS,
    STARDIST_DATA_DIR,
    ANNOTATE_MIN_AREA,
    ANNOTATE_MAX_AREA,
    ANNOTATE_MIN_CIRC,
    ANNOTATE_PEAK_MIN_DIST,
    ANNOTATE_PATCH_SIZE,
    ANNOTATE_MAX_PATCHES,
    ANNOTATE_MIN_PARTICLES,
    STARDIST_VAL_SPLIT,
)

# Scale-adjusted thresholds for patch-level annotation
# Patches are 512×512 cut from 3648×5472 images (scale ~0.14)
# Particle areas scale as (512/3648)² ≈ 0.0197 of original
_SCALE  = (ANNOTATE_PATCH_SIZE / 3648.0) ** 2
_MIN_A  = max(30,  int(ANNOTATE_MIN_AREA  * _SCALE))
_MAX_A  = min(80000, int(ANNOTATE_MAX_AREA * _SCALE))
_PEAK_D = max(5,   int(ANNOTATE_PEAK_MIN_DIST * (ANNOTATE_PATCH_SIZE / 3648.0)))

# ── Image processing helpers ──────────────────────────────────────────────────

def _clahe(gray: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def _binary_particles(gray: np.ndarray) -> np.ndarray:
    """Dark particles on light background — mirrors inference.py logic."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bin_otsu = cv2.threshold(blurred, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, bin_fix  = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.bitwise_or(bin_otsu, bin_fix)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    return binary


def _circularity(contour) -> float:
    area  = cv2.contourArea(contour)
    perim = cv2.arcLength(contour, True)
    return 4.0 * np.pi * area / (perim * perim) if perim > 0 else 0.0


# ── Core: annotate one 512×512 patch ────────────────────────────────────────

def annotate_patch(patch_gray: np.ndarray) -> np.ndarray:
    """
    Watershed annotation on a single grayscale patch.
    Returns int32 label array (0=bg, 1..N=instances).
    """
    if not _SKIMAGE_OK:
        raise RuntimeError("scikit-image required: pip install scikit-image")

    h, w = patch_gray.shape
    enhanced = _clahe(patch_gray)
    binary   = _binary_particles(enhanced)

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5).astype(np.float32)

    coords = peak_local_max(dist, min_distance=_PEAK_D, labels=binary)
    if len(coords) == 0:
        return np.zeros((h, w), dtype=np.int32)

    markers = np.zeros((h, w), dtype=np.int32)
    for idx, (r, c) in enumerate(coords, start=1):
        markers[r, c] = idx

    ws = watershed(-dist, markers, mask=binary)

    final  = np.zeros((h, w), dtype=np.int32)
    new_id = 1
    for lab in range(1, int(ws.max()) + 1):
        region = ws == lab
        area   = float(region.sum())
        if area < _MIN_A or area > _MAX_A:
            continue
        region_u8 = region.astype(np.uint8) * 255
        cnts, _   = cv2.findContours(region_u8, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
        if not cnts or _circularity(cnts[0]) < ANNOTATE_MIN_CIRC:
            continue
        final[region] = new_id
        new_id += 1

    return final


# ── Dataset builder ───────────────────────────────────────────────────────────

def _collect_image_paths() -> list:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    paths = []
    for folder_list in SOURCE_DIRS.values():
        for folder in folder_list:
            folder = Path(folder)
            if not folder.exists():
                continue
            for f in sorted(folder.iterdir()):
                if f.suffix.lower() in exts:
                    paths.append(f)
    return paths


def _random_patches(gray: np.ndarray, n: int) -> list:
    """Return up to n random (512×512) patch arrays from a full image."""
    h, w  = gray.shape
    ps    = ANNOTATE_PATCH_SIZE
    if h < ps or w < ps:
        return [gray]  # image smaller than patch — use as-is

    stride     = ps // 2
    candidates = [
        (r, c)
        for r in range(0, h - ps + 1, stride)
        for c in range(0, w - ps + 1, stride)
    ]
    random.shuffle(candidates)
    patches = []
    for r, c in candidates[:n * 3]:          # sample more than needed, filter later
        patches.append((r, c, gray[r:r+ps, c:c+ps]))
    return patches


def build_dataset(dry_run: bool = False) -> dict:
    paths = _collect_image_paths()
    if not paths:
        print("ERROR: No source images found. Check config.py SOURCE_DIRS.")
        return {}

    print(f"Found {len(paths)} source images")
    print(f"Patch size: {ANNOTATE_PATCH_SIZE}×{ANNOTATE_PATCH_SIZE}  "
          f"Max patches/image: {ANNOTATE_MAX_PATCHES}")
    print(f"Particle area range (patch-scale): {_MIN_A}–{_MAX_A} px²  "
          f"Peak dist: {_PEAK_D} px\n")

    random.seed(42)
    random.shuffle(paths)
    n_val  = max(1, int(len(paths) * STARDIST_VAL_SPLIT))
    splits = {"val": paths[:n_val], "train": paths[n_val:]}

    stats = {
        "total_images": len(paths),
        "train_patches": 0, "val_patches": 0,
        "skipped_images": 0, "total_particles": 0,
    }

    for split, split_paths in splits.items():
        img_dir  = STARDIST_DATA_DIR / split / "images"
        mask_dir = STARDIST_DATA_DIR / split / "masks"
        if not dry_run:
            img_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)

        for img_idx, img_path in enumerate(split_paths):
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                stats["skipped_images"] += 1
                continue

            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            patch_candidates = _random_patches(gray, ANNOTATE_MAX_PATCHES)

            saved = 0
            for item in patch_candidates:
                if saved >= ANNOTATE_MAX_PATCHES:
                    break

                if isinstance(item, tuple):
                    r, c, patch = item
                else:
                    patch = item   # full image (small image case)

                labels = annotate_patch(patch)
                n_inst = int(labels.max())

                if n_inst < ANNOTATE_MIN_PARTICLES:
                    continue

                if not dry_run:
                    tag = f"{img_path.stem}_{saved:03d}"
                    np.save(str(img_dir  / f"{tag}.npy"), patch.astype(np.uint8))
                    np.save(str(mask_dir / f"{tag}.npy"), labels.astype(np.int32))

                saved += 1
                stats["total_particles"] += n_inst

            if saved == 0:
                stats["skipped_images"] += 1
            else:
                stats[f"{split}_patches"] += saved

            # Progress print every 50 images
            done = img_idx + 1
            if done % 50 == 0 or done == len(split_paths):
                print(f"  [{split}] {done}/{len(split_paths)} images — "
                      f"{stats[f'{split}_patches']} patches so far")

    stats["avg_per_patch"] = round(
        stats["total_particles"] / max(stats["train_patches"] + stats["val_patches"], 1), 1
    )
    return stats


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate CellPose training patches")
    parser.add_argument("--check", action="store_true",
                        help="Dry-run: show stats without saving")
    parser.add_argument("--clean", action="store_true",
                        help="Delete existing dataset before rebuild")
    args = parser.parse_args()

    if args.clean and STARDIST_DATA_DIR.exists():
        shutil.rmtree(str(STARDIST_DATA_DIR))
        print(f"Removed: {STARDIST_DATA_DIR}\n")

    stats = build_dataset(dry_run=args.check)

    print("\n── Annotation Summary ──────────────────────────────")
    for k, v in stats.items():
        print(f"  {k:<22}: {v}")

    if not args.check:
        total = stats.get("train_patches", 0) + stats.get("val_patches", 0)
        print(f"\nTotal patches saved: {total}")
        print(f"Dataset location:    {STARDIST_DATA_DIR}")
        print("\nNext: python train_cellpose.py")


if __name__ == "__main__":
    main()
