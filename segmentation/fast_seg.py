"""
segmentation/fast_seg.py — Fast classical CV segmentation for proppant particles.

Uses CLAHE + Otsu threshold + distance transform + watershed.
~100x faster than CellPose SAM for high-contrast proppant images.
Processes a full 20MP image in ~2-5 seconds on CPU.

Same interface as CellposeSegmentor.segment():
    masks, probs = segmentor.segment(image)
"""
import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


# Particle size limits in pixels (for FULL image, not patches).
# These are calibrated for ~3648-5472px wide images.
# Will be scaled automatically per image based on estimated particle size.
_MIN_CIRC      = 0.30   # minimum circularity (0=line, 1=perfect circle)
_MIN_AREA_FRAC = 0.002  # min particle area as fraction of estimated median
_MAX_AREA_FRAC = 8.0    # max particle area as fraction of estimated median
_OTSU_WEIGHT   = 0.85   # blend Otsu threshold toward darker side


def _clahe(gray: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def _binary_particles(gray: np.ndarray) -> np.ndarray:
    """Dark particles on light background — Otsu + fixed threshold OR."""
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


class FastSegmentor:
    """
    Classical CV segmentor — no GPU, no neural network.
    Accurate for high-contrast proppant images.
    Processes a 20MP image in ~2-5 seconds.
    """

    def segment(
        self,
        image: np.ndarray,
        prob_thresh: float = 0.50,   # unused, kept for API compatibility
        nms_thresh:  float = 0.40,   # unused, kept for API compatibility
    ):
        """
        Segment all particles in a BGR or grayscale image.

        Returns:
            masks : list of (H, W) bool arrays, one per particle
            probs : per-instance circularity score (0-1, higher = rounder)
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        h, w = gray.shape

        # 1. Enhance contrast + threshold
        enhanced = _clahe(gray)
        binary   = _binary_particles(enhanced)

        # 2. Distance transform — peaks = particle centers
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5).astype(np.float32)

        # Estimate typical particle radius from dist map (75th percentile of peaks)
        peak_vals = dist[dist > dist.max() * 0.3]
        median_r  = float(np.median(peak_vals)) if len(peak_vals) > 0 else 8.0
        min_dist  = max(4, int(median_r * 0.7))

        # 3. Watershed segmentation
        coords  = peak_local_max(dist, min_distance=min_dist, labels=binary)
        if len(coords) == 0:
            return [], []

        markers = np.zeros((h, w), dtype=np.int32)
        for idx, (r, c) in enumerate(coords, start=1):
            markers[r, c] = idx

        ws = watershed(-dist, markers, mask=binary)

        # 4. Filter by area and circularity
        # Estimate size limits from median particle radius
        expected_area = np.pi * median_r ** 2
        min_area = max(20, expected_area * _MIN_AREA_FRAC)
        max_area = expected_area * _MAX_AREA_FRAC

        masks = []
        probs = []

        for lab in range(1, int(ws.max()) + 1):
            region = ws == lab
            area   = float(region.sum())
            if area < min_area or area > max_area:
                continue

            region_u8 = region.astype(np.uint8) * 255
            cnts, _   = cv2.findContours(region_u8, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue

            circ = _circularity(cnts[0])
            if circ < _MIN_CIRC:
                continue

            masks.append(region)
            probs.append(circ)   # circularity as confidence proxy

        print(f"  [FastSeg] {len(coords)} seeds -> {len(masks)} particles")
        return masks, probs
