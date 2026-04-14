# Proppant QC System v2.0

AI-powered proppant quality control — detects and classifies 40/70 and 20/40 mesh proppant particles from a single image. Runs as a touch-screen kiosk app on NVIDIA Jetson or any PC.

---

## What it does

- Segments every particle in the image using **CellPose** (deep learning, GPU-accelerated)
- Classifies particles as **40/70 mesh** or **20/40 mesh** by size
- Reports PASS / FAIL verdict with composition percentages
- Validates against **lab sieve reference data** (Spec 8, ±10 wt%)
- Shows results in a 5-card touch-friendly UI with no scrolling

---

## Quick start — Jetson

### 1. Clone the repo

```bash
git clone https://github.com/0zyad/model-1.git
cd model-1
```

### 2. Install dependencies

```bash
bash setup_jetson.sh
```

This installs everything needed: PyQt5, CellPose, OpenCV, scipy, scikit-image, matplotlib, openpyxl.

> **Note:** CellPose requires PyTorch. On Jetson, install the NVIDIA-provided PyTorch wheel **before** running setup:
> ```bash
> # Jetson JetPack 5.x (JP51) — PyTorch 2.x for ARM
> pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v51 \
>     torch torchvision
> ```
> Check your JetPack version with: `cat /etc/nv_tegra_release`

### 3. Add your sieve reference file

Copy your lab sieve Excel file to the project folder and update `config.py`:

```python
SIEVE_EXCEL_PATH = Path("/path/to/your/Sieve manual .xlsx")
```

The Excel file should have two tables (20/40 and 40/70) with columns:
`Mesh size | weight empty (g) | weight filled (g) | proppant weight (g) | weight %`

### 4. Run

```bash
# Fullscreen kiosk mode (for Jetson touchscreen)
./run.sh

# Windowed mode (for testing on PC)
python3 app.py --windowed
```

### 5. Auto-start on boot (optional)

```bash
bash deploy_jetson.sh
```

The app will launch automatically every time the Jetson boots.

---

## Quick start — Windows PC

```bash
git clone https://github.com/0zyad/model-1.git
cd model-1
pip install -r requirements.txt
pip install cellpose scikit-image scipy
python app.py --windowed
```

---

## How to use

1. **Dashboard** — Enter a batch ID (optional), then press **Start New Test** (camera) or **Upload Image**
2. **Running** — Progress bar while AI analyzes the image (~8–15 sec on Jetson with GPU)
3. **Results** — 5 cards, navigate with `‹` `›` arrows:
   - **Card 1** — Segmentation overlay (green = 40/70, orange = 20/40)
   - **Card 2** — Verdict (PASS/FAIL) + composition percentages
   - **Card 3** — Spec compliance checks
   - **Card 4** — Sieve distribution chart (model vs lab)
   - **Card 5** — Summary + export buttons

---

## Project structure

```
model-1/
├── app.py                  # Main application entry point
├── config.py               # All tunable parameters and paths
├── inference_stardist.py   # Core analysis pipeline (CellPose + classifier)
├── logger.py               # Saves JSON/CSV/overlay to logs/
├── segmentation/
│   ├── cellpose_seg.py     # CellPose wrapper
│   └── watershed_refine.py # Watershed post-processing
├── widgets/
│   ├── common.py           # Shared UI components + dark theme
│   ├── dashboard.py        # Home screen
│   ├── place_tray.py       # Camera preview screen
│   ├── running.py          # Analysis progress screen
│   └── results.py          # 5-card results viewer
├── setup_jetson.sh         # Jetson dependency installer
├── deploy_jetson.sh        # Jetson auto-start setup
├── run.sh                  # Launch script
└── logs/                   # Auto-created, stores all test results
```

---

## Configuration (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `SIEVE_EXCEL_PATH` | — | Path to your lab sieve Excel file |
| `CELLPOSE_PRETRAINED` | `"cyto3"` | CellPose model — cyto3 is fastest |
| `CELLPOSE_DIAMETER` | `65` | Expected particle diameter in pixels (at 0.33x scale) |
| `CELLPOSE_USE_GPU` | `True` | Set False if no GPU |
| `PIXELS_PER_MM` | `None` | Set after calibrating with a reference object |
| `PURITY_THRESHOLD` | `0.90` | >= 90% of one type → PASS |
| `MAX_SIZE_ERROR` | `0.10` | Spec 8: ±10 wt% vs lab sieve |
| `CAMERA_INDEX` | `0` | USB camera index |

---

## Classification logic

Particles are classified by size ratio — no per-particle model needed:

- **Pure 40/70**: < 5% of particles are large (≥ 118px diameter) → all classified as 40/70
- **Pure 20/40**: > 40% of particles are large → all classified as 20/40
- **Mixed**: 5–40% large → split at 113px boundary

Pixel sizes are in **original resolution space** (measured at this camera setup: 40/70 ≈ 90–110px, 20/40 ≈ 115–175px).

---

## Keyboard shortcuts

| Key | Action |
|---|---|
| `F11` | Toggle fullscreen |
| `Escape` | Exit fullscreen |

---

## Requirements

- Python 3.10+
- PyTorch 2.0+ (CUDA recommended)
- CellPose 3.x+
- PyQt5
- OpenCV, NumPy, scipy, scikit-image, matplotlib, openpyxl

---

## Hardware tested

- **Development**: Windows 11, RTX 3070, Python 3.13
- **Deployment target**: NVIDIA Jetson Orin / AGX with JetPack 5.x
