"""
segmentation/stardist_seg.py — StarDist inference wrapper for particle segmentation.

StarDist2D predicts star-convex polygon instances, which is ideal for dense
fields of roughly circular particles (proppant). It handles touching/overlapping
particles far better than bounding-box-based models.

Returns a list of (H, W) bool masks — one per detected particle — compatible
with the existing ProppantAnalyzer pipeline.

Dependencies:
    pip install stardist csbdeep tensorflow
    # Jetson: use NVIDIA's TF wheel, not pip's tensorflow
"""
import numpy as np
import cv2
from pathlib import Path

# ── Lazy TF import: don't crash at module load if stardist not installed ─────

_stardist_cls   = None
_normalize_fn   = None


def _load_stardist():
    global _stardist_cls, _normalize_fn
    if _stardist_cls is None:
        try:
            import tensorflow as tf

            # GPU memory growth — prevents TF from grabbing all VRAM at start.
            # Critical on Jetson (shared CPU/GPU memory) and multi-process setups.
            gpus = tf.config.list_physical_devices("GPU")
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError:
                    pass  # already initialized

            from stardist.models import StarDist2D
            from csbdeep.utils import normalize
            _stardist_cls = StarDist2D
            _normalize_fn = normalize
        except ImportError as exc:
            raise ImportError(
                f"StarDist not installed: {exc}\n"
                "  pip install stardist csbdeep tensorflow\n"
                "  Jetson: use NVIDIA's TF L4T wheel instead of pip tensorflow"
            ) from exc
    return _stardist_cls, _normalize_fn


# ── Main segmentor class ──────────────────────────────────────────────────────

class StarDistSegmentor:
    """
    Loads a trained StarDist2D model and runs per-image instance segmentation.

    model_path: directory produced by train_stardist.py
                (contains config.json, thresholds.json, weights_best.h5)
    """

    def __init__(self, model_path: "str | Path"):
        model_path = Path(model_path)
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"StarDist model directory not found: {model_path}\n"
                "Run train_stardist.py first."
            )

        StarDist2D, normalize = _load_stardist()
        self._normalize = normalize

        # StarDist loads from (basedir, name) — basedir = parent, name = last dir
        self.model = StarDist2D(
            None,
            name=model_path.name,
            basedir=str(model_path.parent),
        )
        print(f"[StarDist] Model loaded: {model_path}")

    # ── Core inference ────────────────────────────────────────────────────────

    def segment(
        self,
        image: np.ndarray,
        prob_thresh: float = 0.50,
        nms_thresh:  float = 0.40,
        n_tiles: "tuple | None" = None,
    ) -> "tuple[list[np.ndarray], list[float]]":
        """
        Run StarDist on a BGR or grayscale uint8 image.

        n_tiles: tile grid for large images, e.g. (2, 2) for 1920×1080.
                 None = auto-detect based on image size.

        Returns:
            masks: list of (H, W) bool arrays
            probs: per-instance probability scores (same length as masks)
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Percentile normalization recommended by StarDist authors.
        # pmin=1, pmax=99.8 handles outlier bright/dark pixels robustly.
        img_norm = self._normalize(
            gray.astype(np.float32), pmin=1, pmax=99.8, axis=None
        )

        # Auto-tile for large images (>= 1024 px in any dim) to avoid OOM
        if n_tiles is None:
            h, w = gray.shape
            th = max(1, (h + 511) // 512)
            tw = max(1, (w + 511) // 512)
            n_tiles = (th, tw) if (th > 1 or tw > 1) else None

        labels, details = self.model.predict_instances(
            img_norm,
            prob_thresh=prob_thresh,
            nms_thresh=nms_thresh,
            n_tiles=n_tiles,
            verbose=False,
        )

        # labels: (H, W) int32 — 0=background, 1..N=instances
        probs = list(details.get("prob", []))
        masks = []
        kept_probs = []

        for inst_id in range(1, int(labels.max()) + 1):
            mask = labels == inst_id
            if mask.sum() < 30:     # discard noise blobs
                continue
            masks.append(mask)
            kept_probs.append(float(probs[inst_id - 1]) if inst_id - 1 < len(probs) else 0.7)

        return masks, kept_probs

    def segment_with_refinement(
        self,
        image: np.ndarray,
        expected_area_px: "float | None" = None,
        prob_thresh: float = 0.50,
        nms_thresh:  float = 0.40,
    ) -> "tuple[list[np.ndarray], list[float]]":
        """
        StarDist segmentation + watershed split for merged blobs.

        expected_area_px: typical single-particle area; any instance with
                          area > 2.5× this is sent to watershed for splitting.

        Returns:
            (masks, probs) with merged blobs split where possible.
        """
        from segmentation.watershed_refine import split_merged_particles

        masks, probs = self.segment(image, prob_thresh=prob_thresh, nms_thresh=nms_thresh)

        if expected_area_px is None or len(masks) == 0:
            return masks, probs

        refined_masks  = []
        refined_probs  = []

        for mask, prob in zip(masks, probs):
            area = float(mask.sum())
            if area > expected_area_px * 2.5:
                sub_masks = split_merged_particles(image, mask, expected_area_px)
                if sub_masks:
                    refined_masks.extend(sub_masks)
                    # Assign the same probability to all sub-particles
                    refined_probs.extend([prob] * len(sub_masks))
                    continue
            refined_masks.append(mask)
            refined_probs.append(prob)

        return refined_masks, refined_probs
