"""
segmentation/cellpose_seg.py — CellPose instance segmentation for proppant particles.

CellPose uses a gradient-flow approach to separate touching/overlapping round objects.
It is PyTorch-native (no TensorFlow) and works on Python 3.13 with existing CUDA stack.

Memory strategy for 20MP images:
  - Downsample to ~2MP before inference (process_scale = 0.33)
  - Masks stay at downsampled resolution — NOT upscaled (avoids OOM)
  - self.last_scale is set so the caller can adjust pixel measurements
  - The overlay is drawn on the downsampled image then upscaled at the end
"""
import numpy as np
import cv2
from pathlib import Path


class CellposeSegmentor:
    """
    Loads a CellPose model (pretrained or fine-tuned) and runs instance segmentation.

    model_path: path to a fine-tuned model file (from train_cellpose.py).
                If None or file doesn't exist, falls back to pretrained from config.
    use_gpu:    use RTX 3070 for inference (strongly recommended).
    diameter:   expected particle diameter in pixels (at processed resolution).
                None = CellPose auto-estimates.
    """

    # Scale applied to images before inference — exposed so caller can adjust measurements
    last_scale: float = 1.0

    def __init__(
        self,
        model_path: "str | Path | None" = None,
        use_gpu: bool = True,
        diameter: "float | None" = None,
    ):
        from cellpose import models

        model_path = Path(model_path) if model_path else None

        if model_path is not None and model_path.exists():
            self.model = models.CellposeModel(
                gpu=use_gpu,
                pretrained_model=str(model_path),
            )
            print(f"[CellPose] Fine-tuned model loaded: {model_path}")
        else:
            from config import CELLPOSE_PRETRAINED
            self.model = models.CellposeModel(gpu=use_gpu, model_type=CELLPOSE_PRETRAINED)
            if model_path is not None:
                print(f"[CellPose] Fine-tuned model not found at {model_path}")
            print(f"[CellPose] Using pretrained '{CELLPOSE_PRETRAINED}' model")

        self.diameter = diameter

    # ── Core inference ────────────────────────────────────────────────────────

    def segment(
        self,
        image: np.ndarray,
        prob_thresh: float = 0.50,
        nms_thresh:  float = 0.40,
        progress_fn=None,
    ) -> "tuple[list[np.ndarray], list[float]]":
        """
        Segment all particles in a BGR or grayscale image.

        For large images (>2MP), downsamples to ~2MP before inference.
        Masks are returned at the DOWNSAMPLED resolution — NOT upscaled.
        Call self.last_scale after this to get the scale factor used.
        Caller must divide diameter_px by last_scale to get original-resolution values.

        Returns:
            masks: list of (H_small, W_small) bool arrays, one per particle
            probs: per-instance confidence score (0-1)
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        h_orig, w_orig = gray.shape

        # ── Resize large images for speed + memory ────────────────────────────
        # 20MP at full res = 901 masks × 76MB each = OOM. Downsample first.
        # Masks stay small — caller scales measurements, draws overlay on small img.
        self.last_scale = 1.0
        if gray.size > 2_000_000:
            self.last_scale = 0.33
            nh = int(h_orig * self.last_scale)
            nw = int(w_orig * self.last_scale)
            gray = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
            print(f"  [CellPose] Resized {w_orig}x{h_orig} -> {nw}x{nh} (scale={self.last_scale})")

        # CLAHE: boosts dark-particle contrast on bright background
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        if progress_fn:
            progress_fn(0.1, "Running CellPose segmentation...")

        # ── CellPose inference — built-in tiling, no manual loop ─────────────
        labels, flows, _ = self.model.eval(
            gray,
            diameter=self.diameter,
            flow_threshold=nms_thresh,
            cellprob_threshold=0.5,    # raised from -1.0 — cuts false positives
            normalize=True,
        )

        if progress_fn:
            progress_fn(0.85, "Extracting instances...")

        # ── Extract per-instance bool masks at downsampled resolution ─────────
        cpm = flows[2] if (hasattr(flows, '__len__') and len(flows) > 2) else None
        n_instances = int(labels.max())

        all_masks: list = []
        all_probs: list = []

        for inst_id in range(1, n_instances + 1):
            m = (labels == inst_id)
            if int(m.sum()) < 20:
                continue

            conf = 0.70
            if cpm is not None:
                try:
                    conf = float(np.mean(cpm[m]))
                    conf = max(0.0, min(1.0, conf))
                except Exception:
                    pass

            all_masks.append(m)   # bool, at downsampled resolution — NO upscale
            all_probs.append(conf)

        if progress_fn:
            progress_fn(1.0, f"Found {len(all_masks)} particles")

        print(f"  [CellPose] {len(all_masks)} particles (from {n_instances} raw instances)")
        return all_masks, all_probs

    def segment_with_refinement(
        self,
        image: np.ndarray,
        expected_area_px: "float | None" = None,
        prob_thresh: float = 0.50,
        nms_thresh:  float = 0.40,
    ) -> "tuple[list[np.ndarray], list[float]]":
        """CellPose + watershed split for merged blobs."""
        from segmentation.watershed_refine import split_merged_particles

        masks, probs = self.segment(image, prob_thresh=prob_thresh, nms_thresh=nms_thresh)

        if expected_area_px is None or len(masks) == 0:
            return masks, probs

        # expected_area_px is in original resolution — scale it down
        expected_area_small = expected_area_px * (self.last_scale ** 2)

        refined_masks = []
        refined_probs = []
        for mask, prob in zip(masks, probs):
            area = float(mask.sum())
            if area > expected_area_small * 2.5:
                sub = split_merged_particles(image, mask, expected_area_small)
                if sub:
                    refined_masks.extend(sub)
                    refined_probs.extend([prob] * len(sub))
                    continue
            refined_masks.append(mask)
            refined_probs.append(prob)

        return refined_masks, refined_probs
