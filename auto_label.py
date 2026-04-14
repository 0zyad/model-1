"""
auto_label.py — Automatic annotation using OpenCV for Proppant QC.

Detects particles via thresholding and contour extraction, classifies them
based on filename prefix and size, and generates YOLO segmentation label files.

Strategy:
  - P4070_xxx images: all dark particles → proppant_40_70 (class 0)
  - P2040_xxx images: all dark particles → proppant_20_40 (class 1)
  - MIX2_xxx images:  dark particles, classify by size (large=class1, small=class0)

Usage:
    python auto_label.py                     # Label all images
    python auto_label.py --preview           # Show preview overlays (press any key to advance)
    python auto_label.py --split train       # Label only train split
"""
import argparse
import cv2
import numpy as np
from pathlib import Path
from config import DATASET_DIR

# ── YOLO class IDs ────────────────────────────────────────────────────
CLS_4070 = 0   # proppant_40_70
CLS_2040 = 1   # proppant_20_40

# ── Detection parameters ──────────────────────────────────────────────
MIN_CONTOUR_AREA = 15          # Lower to catch smaller particles
MAX_CONTOUR_AREA_RATIO = 0.06  # Skip contours larger than 6% of image (artifacts)
MIN_POLYGON_POINTS = 4         # Minimum vertices for YOLO polygon
MIN_CIRCULARITY = 0.20         # Lower to catch more irregular/overlapping particles


def detect_dark_particles(img, gray):
    """Detect dark (proppant) particles using adaptive thresholding.
    Uses aggressive watershed to separate tightly packed particles."""
    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu's threshold — works well for bimodal (dark particles on white bg)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Also try a fixed threshold to catch lighter particles
    _, binary2 = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.bitwise_or(binary, binary2)

    # Adaptive threshold to catch particles in uneven lighting
    adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 51, 10
    )
    binary = cv2.bitwise_or(binary, adaptive)

    # Morphological operations to clean up noise
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Aggressive watershed to separate densely packed particles
    binary = separate_touching_aggressive(img, binary)

    return binary


def detect_sand_particles(img, dark_mask):
    """Detect sand particles (light, amber/yellowish) in MIXS images."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Sand is amber/yellowish/translucent: Hue ~15-40, moderate saturation, high value
    lower_sand = np.array([10, 20, 120])
    upper_sand = np.array([45, 180, 255])
    sand_mask = cv2.inRange(hsv, lower_sand, upper_sand)

    # Also detect lighter/whitish sand grains
    lower_light = np.array([0, 5, 170])
    upper_light = np.array([50, 80, 255])
    light_mask = cv2.inRange(hsv, lower_light, upper_light)

    # Combine sand masks
    combined = cv2.bitwise_or(sand_mask, light_mask)

    # Remove anything that overlaps with dark particles
    combined = cv2.bitwise_and(combined, cv2.bitwise_not(dark_mask))

    # Remove pure white background: background is very bright and low saturation
    bg_lower = np.array([0, 0, 220])
    bg_upper = np.array([180, 30, 255])
    bg_mask = cv2.inRange(hsv, bg_lower, bg_upper)
    combined = cv2.bitwise_and(combined, cv2.bitwise_not(bg_mask))

    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Separate touching sand grains
    combined = separate_touching(combined)

    return combined


def separate_touching(binary):
    """Use distance transform + watershed to separate touching particles."""
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    max_val = dist.max()
    if max_val == 0:
        return binary
    _, sure_fg = cv2.threshold(dist, 0.45 * max_val, 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    sure_bg = cv2.dilate(binary, None, iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)

    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    if len(binary.shape) == 2:
        color_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    else:
        color_img = binary.copy()

    cv2.watershed(color_img, markers)

    result = np.zeros_like(binary)
    result[markers > 1] = 255
    result[markers == -1] = 0

    return result


def separate_touching_aggressive(img, binary):
    """Aggressively separate densely packed particles using local maxima
    of the distance transform as seeds for watershed."""
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    max_val = dist.max()
    if max_val == 0:
        return binary

    nonzero_dist = dist[dist > 0]
    if len(nonzero_dist) == 0:
        return binary

    # Estimate typical particle radius from distance transform
    typical_radius = np.percentile(nonzero_dist, 25)  # Lower percentile for denser packing
    # Use a kernel slightly larger than a particle to find centers
    kernel_size = max(3, int(typical_radius))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = min(kernel_size, 51)

    # Find local maxima of distance transform
    local_max = cv2.dilate(dist, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    ))
    # A pixel is a local maximum if it equals the dilated value
    min_dist = max(2.0, typical_radius * 0.3)  # Lower threshold to find more seeds
    sure_fg = ((dist == local_max) & (dist >= min_dist)).astype(np.uint8) * 255

    # Dilate the single-pixel seeds slightly so they survive as markers
    sure_fg = cv2.dilate(sure_fg, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    sure_bg = cv2.dilate(binary, None, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Use the actual color image for better watershed boundaries
    if len(img.shape) == 2:
        color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        color_img = img.copy()

    cv2.watershed(color_img, markers)

    result = np.zeros_like(binary)
    result[markers > 1] = 255
    result[markers == -1] = 0

    return result


def extract_contours(binary_mask, img_h, img_w):
    """Extract contours and filter by area and circularity."""
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    max_area = img_h * img_w * MAX_CONTOUR_AREA_RATIO
    valid = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_CONTOUR_AREA or area > max_area or len(c) < 3:
            continue
        # Circularity filter: reject elongated artifacts (edges, scratches)
        perimeter = cv2.arcLength(c, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < MIN_CIRCULARITY:
                continue
        valid.append(c)
    return valid


def contour_to_yolo_polygon(contour, img_h, img_w):
    """Convert OpenCV contour to YOLO normalized polygon coordinates.
    Returns list of (x, y) normalized floats, or None if too few points."""
    # Simplify contour to reduce vertices
    epsilon = 0.015 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) < MIN_POLYGON_POINTS:
        # If simplified too much, use original but subsample
        approx = contour

    points = approx.reshape(-1, 2)

    # Subsample if too many points (YOLO files get very large)
    if len(points) > 50:
        indices = np.linspace(0, len(points) - 1, 50, dtype=int)
        points = points[indices]

    if len(points) < MIN_POLYGON_POINTS:
        return None

    # Normalize to 0-1
    normalized = []
    for x, y in points:
        nx = round(float(x) / img_w, 6)
        ny = round(float(y) / img_h, 6)
        # Clamp
        nx = max(0.0, min(1.0, nx))
        ny = max(0.0, min(1.0, ny))
        normalized.append((nx, ny))

    return normalized


def classify_by_size(contours, img_area):
    """Classify contours into large (20/40) and small (40/70) using
    K-means clustering on equivalent diameters for a cleaner split."""
    if not contours:
        return [], []

    areas = np.array([cv2.contourArea(c) for c in contours])
    # Use equivalent diameter (sqrt of area) — more linear scale for clustering
    diameters = np.sqrt(areas)

    if len(diameters) < 4:
        # Too few particles to cluster — fall back to median split
        median_d = np.median(diameters)
        large = [c for c, d in zip(contours, diameters) if d > median_d * 1.3]
        small = [c for c, d in zip(contours, diameters) if d <= median_d * 1.3]
        return large, small

    # K-means with k=2 to find the natural size split
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    data = diameters.reshape(-1, 1).astype(np.float32)
    _, labels, centers = cv2.kmeans(data, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    # Determine which cluster is the larger particles
    large_cluster = 0 if centers[0][0] > centers[1][0] else 1

    large = []  # proppant_20_40
    small = []  # proppant_40_70
    for i, c in enumerate(contours):
        if labels[i][0] == large_cluster:
            large.append(c)
        else:
            small.append(c)

    return large, small


def process_image(img_path: Path, label_dir: Path, preview: bool = False):
    """Process a single image: detect particles, classify, write YOLO label."""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  WARNING: Could not read {img_path.name}")
        return 0

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    prefix = img_path.stem.split("_")[0]  # P4070, P2040, MIX2, MIXS

    lines = []  # YOLO label lines

    if prefix == "P4070":
        # All particles are proppant_40_70
        dark_mask = detect_dark_particles(img, gray)
        contours = extract_contours(dark_mask, h, w)
        for c in contours:
            poly = contour_to_yolo_polygon(c, h, w)
            if poly:
                coords = " ".join(f"{x} {y}" for x, y in poly)
                lines.append(f"{CLS_4070} {coords}")

    elif prefix == "P2040":
        # All particles are proppant_20_40
        dark_mask = detect_dark_particles(img, gray)
        contours = extract_contours(dark_mask, h, w)
        for c in contours:
            poly = contour_to_yolo_polygon(c, h, w)
            if poly:
                coords = " ".join(f"{x} {y}" for x, y in poly)
                lines.append(f"{CLS_2040} {coords}")

    elif prefix == "MIX2":
        # Mixed 40/70 + 20/40 — classify by size
        dark_mask = detect_dark_particles(img, gray)
        contours = extract_contours(dark_mask, h, w)
        large, small = classify_by_size(contours, h * w)
        for c in large:
            poly = contour_to_yolo_polygon(c, h, w)
            if poly:
                coords = " ".join(f"{x} {y}" for x, y in poly)
                lines.append(f"{CLS_2040} {coords}")
        for c in small:
            poly = contour_to_yolo_polygon(c, h, w)
            if poly:
                coords = " ".join(f"{x} {y}" for x, y in poly)
                lines.append(f"{CLS_4070} {coords}")

    else:
        print(f"  WARNING: Unknown prefix '{prefix}' for {img_path.name}")
        return 0

    # Write label file
    label_path = label_dir / f"{img_path.stem}.txt"
    with open(label_path, "w") as f:
        f.write("\n".join(lines))

    # Preview
    if preview and lines:
        preview_img = img.copy()
        colors = {CLS_4070: (0, 255, 0), CLS_2040: (255, 165, 0)}
        for line in lines:
            parts = line.split()
            cls_id = int(parts[0])
            coords = [(float(parts[i]) * w, float(parts[i + 1]) * h)
                       for i in range(1, len(parts), 2)]
            pts = np.array(coords, dtype=np.int32)
            cv2.drawContours(preview_img, [pts], -1, colors[cls_id], 2)

        # Resize for display
        scale = min(1200 / w, 800 / h)
        display = cv2.resize(preview_img, (int(w * scale), int(h * scale)))
        cv2.imshow(f"{img_path.name} - {len(lines)} particles", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return len(lines)


def main():
    parser = argparse.ArgumentParser(description="Auto-label proppant images")
    parser.add_argument("--preview", action="store_true",
                        help="Show preview of each labeled image")
    parser.add_argument("--split", choices=["train", "val"],
                        help="Only process a specific split (default: both)")
    args = parser.parse_args()

    splits = [args.split] if args.split else ["train", "val"]

    print("=" * 60)
    print("  Proppant QC — Auto Labeling")
    print("=" * 60)

    total_images = 0
    total_particles = 0

    for split in splits:
        img_dir = DATASET_DIR / "images" / split
        lbl_dir = DATASET_DIR / "labels" / split
        lbl_dir.mkdir(parents=True, exist_ok=True)

        if not img_dir.exists():
            print(f"  WARNING: {img_dir} not found, skipping.")
            continue

        images = sorted(
            [f for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp")
             for f in img_dir.glob(ext)]
        )
        print(f"\n  [{split}] Processing {len(images)} images...")

        for img_path in images:
            count = process_image(img_path, lbl_dir, preview=args.preview)
            print(f"    {img_path.name}: {count} particles labeled")
            total_images += 1
            total_particles += count

    print(f"\n" + "=" * 60)
    print(f"  Done! Labeled {total_images} images, {total_particles} total particles.")
    print(f"  Labels saved to: {DATASET_DIR / 'labels'}")
    print(f"\n  NEXT STEP: python train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
