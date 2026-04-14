"""
prepare_dataset.py — Organize raw images into YOLO dataset structure.

Copies images from download folders into raw_images/ with sanitized names,
then creates an 80/20 stratified train/val split under dataset/.

Usage:
    python prepare_dataset.py
"""
import shutil
import random
from pathlib import Path
from config import SOURCE_DIRS, RAW_IMAGES_DIR, DATASET_DIR

TRAIN_RATIO = 0.80
SEED = 42


def sanitize_and_copy(source_dirs, dest_dir: Path, prefix: str) -> list[Path]:
    """Copy all images from one or more source dirs into dest_dir
    with sanitized names: PREFIX_001.jpg, PREFIX_002.jpg, ...
    source_dirs can be a single Path or a list of Paths."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    if isinstance(source_dirs, Path):
        source_dirs = [source_dirs]

    # Collect all images across all source parts
    source_images = []
    for src_dir in source_dirs:
        if not src_dir.exists():
            continue
        source_images.extend(sorted(
            [f for f in src_dir.iterdir() if f.suffix.lower() in extensions]
        ))

    copied = []
    for idx, src in enumerate(source_images, start=1):
        new_name = f"{prefix}_{idx:03d}{src.suffix.lower()}"
        dst = dest_dir / new_name
        shutil.copy2(src, dst)
        copied.append(dst)

    return copied


def create_split(all_files: dict[str, list[Path]]):
    """Create train/val split and copy into dataset/images/{train,val}/."""
    random.seed(SEED)

    splits = {"train": [], "val": []}

    for category, files in all_files.items():
        shuffled = files.copy()
        random.shuffle(shuffled)
        n_train = max(1, int(len(shuffled) * TRAIN_RATIO))
        splits["train"].extend(shuffled[:n_train])
        splits["val"].extend(shuffled[n_train:])

    for split_name, file_list in splits.items():
        img_dir = DATASET_DIR / "images" / split_name
        lbl_dir = DATASET_DIR / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for f in file_list:
            shutil.copy2(f, img_dir / f.name)

    return splits


def main():
    print("=" * 60)
    print("  Proppant QC — Dataset Preparation")
    print("=" * 60)

    # Step 1: Copy and rename from source folders
    all_files: dict[str, list[Path]] = {}
    category_map = {
        "P4070": "P4070_only",
        "P2040": "P2040_only",
        "MIX2":  "Mix_4070_2040",
    }

    for prefix, source_dirs_entry in SOURCE_DIRS.items():
        cat_name = category_map[prefix]
        dest = RAW_IMAGES_DIR / cat_name

        # source_dirs_entry may be a single Path or a list of Paths
        dirs = source_dirs_entry if isinstance(source_dirs_entry, list) else [source_dirs_entry]
        found = [d for d in dirs if d.exists()]
        if not found:
            print(f"  WARNING: No source folders found for {prefix}")
            continue

        copied = sanitize_and_copy(found, dest, prefix)
        all_files[prefix] = copied
        print(f"  [{cat_name}] Copied {len(copied)} images -> {dest}")

    total = sum(len(v) for v in all_files.values())
    print(f"\n  Total raw images: {total}")

    # Step 2: Create train/val split
    print("\n  Creating train/val split (80/20)...")
    splits = create_split(all_files)

    print(f"  Train: {len(splits['train'])} images")
    print(f"  Val:   {len(splits['val'])} images")
    print(f"\n  Dataset structure created at: {DATASET_DIR}")
    print(f"  Images: {DATASET_DIR / 'images'}")
    print(f"  Labels: {DATASET_DIR / 'labels'}  (empty — add annotations here)")

    print("\n" + "=" * 60)
    print("  NEXT STEP: Annotate images using Roboflow or CVAT.")
    print("  See LABELING_GUIDE.md for instructions.")
    print("  Place exported .txt label files into dataset/labels/train/")
    print("  and dataset/labels/val/ with matching filenames.")
    print("=" * 60)


if __name__ == "__main__":
    main()
