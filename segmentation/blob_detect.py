"""
segmentation/blob_detect.py — Blob detection helper for missed-particle recovery.

Role in the pipeline:
    StarDist (primary) → Watershed (split merges) → Blob detection (gap-fill)

Blob detection runs ONLY on image regions not already covered by StarDist masks.
It uses cv2.SimpleBlobDetector with circularity + inertia filters — much more
precise at finding round objects than the generic Otsu + contour approach.

Two detectors are used:
    detector_40_70  — tuned to smaller particle size range (40/70 mesh)
    detector_20_40  — tuned to larger particle size range (20/40 mesh)

If PIXELS_PER_MM is calibrated, detector radii are set from physical mm ranges.
If not calibrated, radii are derived from the median diameter of StarDist detections.

Returns a list of (mask, class_id, confidence) tuples for uncovered blobs.
"""
import numpy as np
import cv2
from typing import List, Tuple, Optional

from config import (
    CLASS_NAMES,
    PIXELS_PER_MM,
    EXPECTED_SIZE_MM,
)

# Confidence score assigned to blob-detected particles
BLOB_CONFIDENCE = 0.38


def _make_detector(
    min_radius_px: float,
    max_radius_px: float,
    min_circularity: float = 0.50,
    min_inertia: float = 0.50,
) -> cv2.SimpleBlobDetector:
    """
    Build a SimpleBlobDetector tuned for a specific particle size band.

    SimpleBlobDetector uses AREA (px²), not radius directly, so we convert.
    """
    params = cv2.SimpleBlobDetector_Params()

    # Area (px²) from radius bounds
    params.filterByArea     = True
    params.minArea          = float(np.pi * min_radius_px ** 2)
    params.maxArea          = float(np.pi * max_radius_px ** 2) * 1.5   # generous upper

    # Circularity — proppant particles are close to perfect circles
    params.filterByCircularity = True
    params.minCircularity      = min_circularity

    # Inertia — ratio of minor/major axis; 1.0 = perfect circle
    params.filterByInertia  = True
    params.minInertiaRatio  = min_inertia

    # Convexity — proppant is convex
    params.filterByConvexity = True
    params.minConvexity      = 0.80

    # Color — detect dark blobs on light background (same convention as rest of system)
    params.filterByColor = True
    params.blobColor     = 0   # 0 = dark blobs

    # Threshold sweep
    params.minThreshold = 10
    params.maxThreshold = 220
    params.thresholdStep = 10

    return cv2.SimpleBlobDetector_create(params)


def _build_covered_mask(
    h: int,
    w: int,
    existing_masks: List[np.ndarray],
    dilation_iters: int = 8,
) -> np.ndarray:
    """Build a dilated union of all existing particle masks (StarDist + watershed output)."""
    covered = np.zeros((h, w), dtype=np.uint8)
    for m in existing_masks:
        covered[m > 0.5] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(covered, kernel, iterations=dilation_iters)


def _keypoint_to_mask(kp: cv2.KeyPoint, h: int, w: int) -> np.ndarray:
    """Convert a keypoint (centre + diameter) to a filled circular boolean mask."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cx = int(round(kp.pt[0]))
    cy = int(round(kp.pt[1]))
    r  = max(2, int(round(kp.size / 2)))
    cv2.circle(mask, (cx, cy), r, 255, -1)
    return mask.astype(bool)


def _estimate_radius_bounds(
    existing_masks: List[np.ndarray],
    class_id: int,
) -> Tuple[float, float]:
    """
    Estimate pixel-space radius bounds for one class.

    Priority:
        1. PIXELS_PER_MM calibration (most accurate)
        2. Median diameter of StarDist detections for that class
        3. Hardcoded fallback ratios
    """
    mm_range = EXPECTED_SIZE_MM.get(CLASS_NAMES.get(class_id, ""), None)

    if PIXELS_PER_MM is not None and mm_range is not None:
        r_min = (mm_range[0] / 2) * PIXELS_PER_MM
        r_max = (mm_range[1] / 2) * PIXELS_PER_MM
        return r_min, r_max

    # Derive from existing detections
    if existing_masks:
        areas = [float(m.sum()) for m in existing_masks]
        if areas:
            median_area = float(np.median(areas))
            median_r    = np.sqrt(median_area / np.pi)
            if class_id == 0:                       # 40/70 — small
                return median_r * 0.3, median_r * 0.9
            else:                                    # 20/40 — large
                return median_r * 0.7, median_r * 2.0

    # Final fallback: generic reasonable range
    return 5.0, 80.0


# ── Public API ────────────────────────────────────────────────────────────────

def detect_missed_particles(
    img: np.ndarray,
    existing_masks: List[np.ndarray],
    existing_class_ids: Optional[List[int]] = None,
) -> List[Tuple[np.ndarray, int, float]]:
    """
    Find particles missed by StarDist + watershed in uncovered image regions.

    Args:
        img:               Full BGR image
        existing_masks:    Float32 or bool (H,W) arrays from StarDist/watershed
        existing_class_ids: Class IDs for existing masks (used to calibrate radii).
                            May be None before classification.

    Returns:
        List of (mask_bool, class_id, confidence) for each newly found particle.
        class_id is 0 (40/70) or 1 (20/40) assigned by size.
    """
    h, w = img.shape[:2]
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Build covered region so we don't re-detect what StarDist already found
    covered = _build_covered_mask(h, w, existing_masks, dilation_iters=8)

    # Mask out already-covered areas before running blob detector
    gray_uncovered = gray.copy()
    # Replace covered pixels with a mid-grey (won't trigger threshold)
    gray_uncovered[covered > 0] = 128

    # ── Per-class blob detectors ──────────────────────────────────────────────
    found: List[Tuple[np.ndarray, int, float]] = []

    for class_id in [0, 1]:   # 0 = 40/70 (small), 1 = 20/40 (large)
        # Separate existing masks by class for better radius estimation
        cls_masks = (
            [m for m, c in zip(existing_masks, existing_class_ids) if c == class_id]
            if existing_class_ids is not None
            else existing_masks
        )

        r_min, r_max = _estimate_radius_bounds(cls_masks, class_id)

        if r_min <= 0 or r_max <= r_min:
            continue

        detector = _make_detector(
            min_radius_px   = r_min,
            max_radius_px   = r_max,
            min_circularity = 0.45,
            min_inertia     = 0.45,
        )

        keypoints = detector.detect(gray_uncovered)

        for kp in keypoints:
            # Skip if centre is already in a covered region (belt-and-suspenders)
            cx, cy = int(round(kp.pt[0])), int(round(kp.pt[1]))
            if cy < h and cx < w and covered[cy, cx] > 0:
                continue

            # Size validation
            radius   = kp.size / 2
            area_px  = np.pi * radius ** 2
            if area_px < 30:
                continue

            mask = _keypoint_to_mask(kp, h, w)
            found.append((mask, class_id, BLOB_CONFIDENCE))

    if found:
        print(f"  Blob detector: added {len(found)} missed particles")

    return found
