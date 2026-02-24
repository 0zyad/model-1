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

# ── Source image folders (raw downloads) ───────────────────────────────
SOURCE_DIRS = {
    "P4070": Path(r"c:\Users\Ziyad\Downloads\drive-download-20260215T164453Z-1-001\40-70"),
    "P2040": Path(r"c:\Users\Ziyad\Downloads\drive-download-20260215T164453Z-1-001\20-40"),
    "MIX2": Path(r"c:\Users\Ziyad\Downloads\drive-download-20260215T164453Z-1-001\20-40 + 40-70"),
    "MIXS": Path(r"c:\Users\Ziyad\Downloads\drive-download-20260215T164453Z-1-001\20-40 + 40-70 + sand"),
}

# ── Class Definitions ──────────────────────────────────────────────────
CLASS_NAMES = {0: "proppant_40_70", 1: "proppant_20_40", 2: "sand"}
CLASS_COLORS = {
    0: (0, 255, 0),      # Green  — proppant 40/70  (BGR)
    1: (0, 165, 255),    # Orange — proppant 20/40  (BGR)
    2: (0, 0, 255),      # Red    — sand            (BGR)
}
NUM_CLASSES = 3

# ── PASS / FAIL Thresholds ────────────────────────────────────────────
PURITY_THRESHOLD = 0.90           # >=90% of one proppant type → PASS
SAND_FAIL_THRESHOLD = 0.10        # >10% sand → FAIL
MIN_CLASSIFIED_RATIO = 0.90       # >=90% particles must be classified
MAX_SIZE_ERROR = 0.10             # Spec 8: ±10 wt% vs laboratory sieve analysis
MAX_PROCESSING_TIME = 20.0        # Seconds

# ── Training Defaults ─────────────────────────────────────────────────
MODEL_BASE = "yolov8l-seg.pt"     # Large model — better recall on dense particles
TRAIN_EPOCHS = 500
TRAIN_IMGSZ = 1024                # Balanced res — fits yolov8l in GPU memory
TRAIN_BATCH = 1                   # yolov8l needs batch=1 at 1024px
CONFIDENCE_THRESHOLD = 0.10       # Low threshold to catch more overlapping particles
IOU_THRESHOLD = 0.20              # Lower NMS to keep more dense/overlapping particles

# ── Expected Particle Sizes (mm) ─────────────────────────────────────
# 20/40 mesh ≈ 0.42–0.84 mm,  40/70 mesh ≈ 0.21–0.42 mm
EXPECTED_SIZE_MM = {
    "proppant_40_70": (0.21, 0.42),
    "proppant_20_40": (0.42, 0.84),
    "sand": (0.05, 0.50),
}

# ── Calibration ───────────────────────────────────────────────────────
# Set after measuring with a reference object; None = use relative ratios
PIXELS_PER_MM = None

# ── Sieve Reference (Spec 8 validation) ───────────────────────────
# Lab sieve Excel file used to validate model mass-fraction accuracy
SIEVE_EXCEL_PATH = Path(r"C:\Users\Ziyad\Downloads\Sieve manual .xlsx")

# ── Camera ────────────────────────────────────────────────────────
CAMERA_INDEX = 0              # Default USB camera
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080

# ── UI ────────────────────────────────────────────────────────────
APP_TITLE = "Proppant QC System v2.0"
FULLSCREEN = True             # Kiosk mode on Jetson
TOUCH_BUTTON_HEIGHT = 40      # Touch buttons sized for 800x480
FONT_SIZE_NORMAL = 11
FONT_SIZE_LARGE = 13
FONT_SIZE_TITLE = 16

# Colors (dark theme)
BG_COLOR = "#1e1e1e"
CARD_COLOR = "#252526"
ACCENT_BLUE = "#0e639c"
ACCENT_GREEN = "#4ec9b0"
ACCENT_RED = "#f44747"
TEXT_COLOR = "#d4d4d4"
MUTED_COLOR = "#808080"
