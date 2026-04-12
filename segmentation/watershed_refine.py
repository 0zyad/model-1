"""
segmentation/watershed_refine.py — Watershed splitting of merged particle blobs.

Used as a post-processing step after StarDist when a single labeled instance
has an area significantly larger than the expected per-particle area, suggesting
it covers 2+ touching particles that StarDist merged into one.

Algorithm:
    1. Euclidean distance transform on the binary blob mask
    2. Detect local maxima in the distance map (one per particle center)
    3. Watershed on inverted distance with those maxima as seeds
    4. Validate sub-particles by minimum area cutoff

Dependencies: scipy, scikit-image (both standard in any CV environment)
"""
import numpy as np
import cv2

try:
    from scipy import ndimage as ndi
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max
    _SKIMAGE_OK = True
except ImportError:
    _SKIMAGE_OK = False


def split_merged_particles(
    image: np.ndarray,
    merged_mask: np.ndarray,
    expected_area_px: float,
    min_distance: "int | None" = None,
    min_sub_area_ratio: float = 0.15,
) -> "list[np.ndarray]":
    """
    Try to split a merged blob into individual particle masks.

    Args:
        image:              Full image (used for context only; not processed here)
        merged_mask:        (H, W) bool — the single merged instance
        expected_area_px:   Typical single-particle area in pixels
        min_distance:       Min px between watershed seed peaks (auto if None)
        min_sub_area_ratio: Sub-particle is kept only if area >=
                            expected_area_px × this ratio (rejects tiny fragments)

    Returns:
        List of (H, W) bool masks for individual particles,
        or [] if split was not possible (caller keeps original mask).
    """
    if not _SKIMAGE_OK:
        # scikit-image not available — return nothing so caller keeps original
        return []

    mask_u8 = merged_mask.astype(np.uint8)

    # Expected diameter → auto min_distance (35% of expected radius)
    expected_diameter = 2.0 * np.sqrt(expected_area_px / np.pi)
    if min_distance is None:
        min_distance = max(5, int(expected_diameter * 0.35))

    # Distance transform — peaks correspond to particle centers
    distance = ndi.distance_transform_edt(mask_u8)

    # Seed detection via local maxima
    coords = peak_local_max(
        distance,
        min_distance=min_distance,
        labels=mask_u8,
    )

    if len(coords) < 2:
        # Only one peak → can't split meaningfully
        return []

    # Build marker image — each seed gets a unique integer label
    markers = np.zeros(distance.shape, dtype=np.int32)
    for idx, (r, c) in enumerate(coords, start=1):
        markers[r, c] = idx

    # Watershed on inverted distance (basins at peaks)
    labels = watershed(-distance, markers, mask=mask_u8)

    # Extract sub-masks; filter by minimum area
    min_area = expected_area_px * min_sub_area_ratio
    sub_masks = []
    for label_id in range(1, len(coords) + 1):
        sub = labels == label_id
        if sub.sum() >= min_area:
            sub_masks.append(sub)

    # Only accept the split if we got at least 2 valid sub-particles
    return sub_masks if len(sub_masks) >= 2 else []


def estimate_expected_area(masks: "list[np.ndarray]") -> "float | None":
    """
    Estimate the typical single-particle area from a list of already-detected masks.

    Uses the lower quartile of areas to avoid bias from merged blobs.
    Returns None if fewer than 5 masks are provided.
    """
    if len(masks) < 5:
        return None
    areas = sorted(float(m.sum()) for m in masks)
    # Lower quartile — merged blobs are large outliers, small particles real
    q25_idx = len(areas) // 4
    return float(np.median(areas[:max(1, q25_idx * 2)]))
