"""
config.py — Central configuration for the Proppant QC Vision System.
All tunable parameters, paths, and class definitions live here.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset"
DATASET_YAML = DATASET_DIR / "proppant.yaml"
RAW_IMAGES_DIR = PROJECT_ROOT / "raw_images"
LOG_DIR = PROJECT_ROOT / "logs"
RUNS_DIR = PROJECT_ROOT / "runs"

# ── Source image folders (new 705-image dataset, split across 6 downloads) ─
_NEW_DATA_ROOTS = [
    Path(r"C:\Users\ZIYAD ABDUL ATIF\Downloads") /
    f"New_Data_Project-20260408T160630Z-3-00{i}" / "New_Data_Project"
    for i in range(1, 7)
]
SOURCE_DIRS = {
    "P4070": [r / "40_70" for r in _NEW_DATA_ROOTS],
    "P2040": [r / "20_40" for r in _NEW_DATA_ROOTS],
    "MIX2":  [r / "mix"   for r in _NEW_DATA_ROOTS],
}

# ── Class Definitions ──────────────────────────────────────────────────
CLASS_NAMES = {0: "proppant_40_70", 1: "proppant_20_40"}
CLASS_COLORS = {
    0: (0, 255, 0),      # Green  — proppant 40/70  (BGR)
    1: (0, 165, 255),    # Orange — proppant 20/40  (BGR)
}
NUM_CLASSES = 2

# ── PASS / FAIL Thresholds ────────────────────────────────────────────
PURITY_THRESHOLD = 0.90           # >=90% of one proppant type → PASS, else FAIL (mix)
MIN_CLASSIFIED_RATIO = 0.90       # >=90% particles must be classified
MAX_SIZE_ERROR = 0.10             # Spec 8: ±10 wt% vs laboratory sieve analysis
MAX_PROCESSING_TIME = 20.0        # Seconds

# ── Training Defaults ─────────────────────────────────────────────────
MODEL_BASE = "yolov8l-seg.pt"     # Large model — better recall on dense particles
TRAIN_EPOCHS = 500
TRAIN_IMGSZ = 1024                # Balanced res — fits yolov8l in GPU memory
TRAIN_BATCH = 2                   # RTX 3070 8.6GB can handle batch=2 at 1024px
CONFIDENCE_THRESHOLD = 0.10       # Low threshold to catch more overlapping particles
IOU_THRESHOLD = 0.20              # Lower NMS to keep more dense/overlapping particles

# ── Expected Particle Sizes (mm) ─────────────────────────────────────
# 20/40 mesh ≈ 0.42–0.84 mm,  40/70 mesh ≈ 0.21–0.42 mm
EXPECTED_SIZE_MM = {
    "proppant_40_70": (0.21, 0.42),
    "proppant_20_40": (0.42, 0.84),
}

# ── Calibration ───────────────────────────────────────────────────────
# Set after measuring with a reference object; None = use relative ratios
PIXELS_PER_MM = None

# ── Sieve Reference (Spec 8 validation) ───────────────────────────
# Lab sieve Excel file used to validate model mass-fraction accuracy
SIEVE_EXCEL_PATH = Path(r"C:\Users\ZIYAD ABDUL ATIF\Downloads\Sieve manual .xlsx")

# ── CellPose Model ────────────────────────────────────────────────
# Using pretrained cyto3 (3-5x faster than cpsam, equivalent accuracy on round particles).
# Set CELLPOSE_MODEL_PATH = None to use pretrained model directly.
CELLPOSE_MODEL_PATH  = None
CELLPOSE_PRETRAINED  = "cyto3"   # cyto3 is 3-5x faster than cpsam; equivalent accuracy on round particles
CELLPOSE_DIAMETER    = 65        # px — at 1/3 scale: 40/70≈45px, 20/40≈90px; 65 is midpoint so both fit within CellPose 0.5x-2x range without fragmentation
CELLPOSE_PROB_THRESH = 0.50
CELLPOSE_NMS_THRESH  = 0.40      # maps to flow_threshold (0.4 = best for proppant)
CELLPOSE_USE_GPU     = True

# ── CellPose Training ─────────────────────────────────────────────
STARDIST_DATA_DIR    = PROJECT_ROOT / "dataset" / "stardist"   # shared annotation dir
STARDIST_VAL_SPLIT   = 0.15
CELLPOSE_TRAIN_EPOCHS = 100     # fine-tuning from cyto3 needs fewer epochs than from scratch
CELLPOSE_BATCH_SIZE   = 2      # cpsam (SAM-based) is large; batch=2 fits in 8GB VRAM

# ── Annotation (shared between CellPose and StarDist) ─────────────
ANNOTATE_MIN_AREA      = 500     # px² — images are 20MP (3648×5472); particles are large
ANNOTATE_MAX_AREA      = 100000  # px² — covers 20/40 particles (~250px diam) at this res
ANNOTATE_MIN_CIRC      = 0.30    # circularity = 4π·area/perimeter² threshold
ANNOTATE_PEAK_MIN_DIST = 20      # px — min seed distance; larger for high-res images

# ── Patch extraction (annotation → training data) ─────────────────
ANNOTATE_PATCH_SIZE    = 512     # px — extract NxN patches from annotated images
ANNOTATE_MAX_PATCHES   = 20      # max patches per source image (keeps dataset ~1-2 GB)
ANNOTATE_MIN_PARTICLES = 3       # skip patches with fewer particles than this

# ── Camera ────────────────────────────────────────────────────────
CAMERA_INDEX = 0              # Default USB camera
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080

# ── UI ────────────────────────────────────────────────────────────
APP_TITLE = "Proppant QC System v2.0"
FULLSCREEN = True             # Kiosk mode on Jetson
TOUCH_BUTTON_HEIGHT = 50      # Touch buttons sized for 800x480 touchscreen
FONT_SIZE_NORMAL = 12
FONT_SIZE_LARGE = 15
FONT_SIZE_TITLE = 19

# Colors (ISA-101 industrial dark theme)
BG_COLOR = "#1a1a1e"
CARD_COLOR = "#2d2d30"
ACCENT_BLUE = "#0d6efd"
ACCENT_GREEN = "#00dd00"
ACCENT_RED = "#ff3333"
TEXT_COLOR = "#ffffff"
MUTED_COLOR = "#a0a0a0"
