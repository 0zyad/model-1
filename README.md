# Proppant QC System v2.0

AI-powered proppant quality control — detects and classifies 40/70 and 20/40 mesh proppant particles from a single image. Runs as a fullscreen kiosk app on NVIDIA Jetson or any Windows PC.

---

## What changed — Apr 18 2026

- **Sieve distribution chart** now shows both model (blue) and lab (amber) lines. Place `sieve_reference.xlsx` in the project folder to enable the amber line.
- **Sieve chart auto-calibrates** from observed particle sizes when `PIXELS_PER_MM` is not set — chart is now a close estimate before calibration, exact after calibration.
- **Mixed sample chart fixed** — for FAIL results, the chart now shows only the dominant class on its correct mesh range instead of plotting all particles together.
- **Calibration reloads per-analysis** — saving calibration in the dialog takes effect immediately on the next run, no restart needed.
- **Scanned image support** — uploaded images with scanner DPI metadata (≥200 DPI) auto-compute their own `pixels_per_mm`.

---

## What it does

- Segments every particle using **CellPose** (deep learning, GPU-accelerated)
- Classifies particles as **40/70 mesh** or **20/40 mesh** by size
- Reports **PASS / FAIL** verdict with composition percentages
- Validates against lab sieve reference data (Spec 8, ±10 wt%)
- Touch-friendly 5-card results UI, no scrolling needed

---

## Setup on Jetson (start here)

### Step 1 — Install PyTorch for Jetson

This must be done **before** anything else. Check your JetPack version first:

```bash
cat /etc/nv_tegra_release
```

Then install the matching PyTorch wheel from NVIDIA:

```bash
# JetPack 5.x (JP51/JP52) — PyTorch 2.x for ARM64
pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v51 \
    torch torchvision
```

Verify it worked:

```bash
python3 -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

You should see `CUDA: True`. If not, stop and fix PyTorch before continuing.

---

### Step 2 — Clone the repo

```bash
git clone https://github.com/0zyad/model-1.git
cd model-1
```

---

### Step 3 — Install all dependencies

```bash
bash setup_jetson.sh
```

This installs: PyQt5, CellPose, OpenCV, scipy, scikit-image, matplotlib, openpyxl.

---

### Step 4 — Copy your sieve reference file

Copy your lab sieve Excel file into the project folder and name it exactly:

```
sieve_reference.xlsx
```

Example:

```bash
cp /path/to/your/"Sieve manual .xlsx" ~/model-1/sieve_reference.xlsx
```

The Excel file needs two tables (one for 20/40, one for 40/70) with columns:
`Mesh size | weight empty (g) | weight filled (g) | proppant weight (g) | weight %`

> If you skip this step the app still works — it just skips the sieve comparison check.

---

### Step 5 — Run the app

```bash
# Fullscreen kiosk mode (normal use)
./run.sh

# Windowed mode (for testing)
python3 app.py --windowed
```

---

### Step 6 — Auto-start on boot (optional)

```bash
bash deploy_jetson.sh
```

The app will launch automatically every time the Jetson boots.

---

## Setup on Windows PC

```bash
git clone https://github.com/0zyad/model-1.git
cd model-1
pip install -r requirements.txt
pip install cellpose scikit-image scipy
python app.py --windowed
```

Copy your sieve Excel file to the project folder as `sieve_reference.xlsx`.

---

## How to use

1. **Dashboard** — Enter a batch ID (optional), then press **Start New Test** (live camera) or **Upload Image** (from file)
2. **Running** — Progress bar while AI analyzes the image (~8–15 sec on Jetson with GPU)
3. **Results** — 5 cards, swipe or use `‹` `›` arrows:
   - **Card 1** — Segmentation overlay (green = 40/70, orange = 20/40)
   - **Card 2** — Verdict (PASS/FAIL) + composition percentages
   - **Card 3** — Spec compliance checks
   - **Card 4** — Sieve distribution chart (model vs lab)
   - **Card 5** — Summary + export buttons (JSON / CSV / overlay image)

---

## Camera

The system was trained on images captured at **5472 × 3648** resolution. The camera must be set to this resolution for correct particle sizing. This is already configured in `config.py` — do not change `CAMERA_WIDTH` / `CAMERA_HEIGHT`.

If your camera does not support 5472×3648, contact the project owner before changing anything — all classification thresholds depend on this resolution.

---

## Camera Calibration (required for accurate sieve chart)

Calibration tells the app how many pixels equal 1 mm in your specific camera setup. Without it, PASS/FAIL and composition still work — but the sieve distribution chart uses an estimate.

### How to calibrate on Jetson

**Step 1 — Run a known pure sample first**

Use a pure 40/70 sample that has already been lab-sieve tested. Run it through the app and note the result batch ID (shown top-right, e.g. `2026-04-18_001`).

**Step 2 — Get the median particle pixel size from the log**

```bash
cd ~/model-1
python3 -c "
import json, glob, numpy as np
logs = sorted(glob.glob('logs/*.json'))
data = json.load(open(logs[-1]))
px = [p['diameter_px'] for p in data['particles']]
print(f'Median particle: {np.median(px):.1f} px')
print(f'Total particles: {len(px)}')
"
```

**Step 3 — Calculate pixels per mm**

For **40/70** proppant, the physical midpoint is **0.315 mm**:
```
pixels_per_mm = median_px / 0.315
```

For **20/40** proppant, the physical midpoint is **0.630 mm**:
```
pixels_per_mm = median_px / 0.630
```

Example: if median = 74 px → `74 / 0.315 = 235 px/mm`

**Step 4 — Enter in the app**

Click **Calibrate Camera** on the dashboard:
- **Pixels field**: enter the median_px value from Step 2
- **mm field**: enter `0.315` (for 40/70) or `0.630` (for 20/40)
- Click **Save**

Calibration is saved permanently to `calibration.json`. The next analysis uses it immediately — no restart needed.

**Step 5 — Verify**

Run the same sample again. The sieve distribution chart (Card 4) blue model line should now closely match the amber lab line.

### Alternative: use a physical ruler

If you have a ruler or coin under the camera:
1. Capture the image with the app
2. Open it on the Jetson: `eog logs/latest_overlay.jpg`
3. Zoom in and count how many pixels span a known mm length
4. Enter those values in the Calibrate Camera dialog directly

---

## Configuration (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `SIEVE_EXCEL_PATH` | `sieve_reference.xlsx` | Lab sieve file in project folder |
| `CELLPOSE_PRETRAINED` | `cyto3` | CellPose model — cyto3 is fastest |
| `CELLPOSE_DIAMETER` | `65` | Particle diameter at 0.33x scale (do not change) |
| `CELLPOSE_USE_GPU` | `True` | Set False only if no GPU |
| `PIXELS_PER_MM` | `None` | Set after calibrating with a reference object |
| `PURITY_THRESHOLD` | `0.90` | ≥90% one type → PASS |
| `MAX_SIZE_ERROR` | `0.10` | Spec 8: ±10 wt% vs lab sieve |
| `CAMERA_INDEX` | `0` | USB camera index |
| `CAMERA_WIDTH` | `5472` | Must match training resolution |
| `CAMERA_HEIGHT` | `3648` | Must match training resolution |

---

## Project structure

```
model-1/
├── app.py                    # Main entry point
├── config.py                 # All parameters and paths
├── inference_stardist.py     # Core analysis pipeline (CellPose + classifier)
├── camera.py                 # Camera capture module
├── logger.py                 # Saves JSON/CSV/overlay to logs/
├── segmentation/
│   ├── cellpose_seg.py       # CellPose wrapper
│   └── watershed_refine.py   # Watershed post-processing
├── widgets/
│   ├── common.py             # Shared UI components + dark theme
│   ├── dashboard.py          # Home screen
│   ├── place_tray.py         # Camera preview screen
│   ├── running.py            # Analysis progress screen
│   └── results.py            # 5-card results viewer
├── setup_jetson.sh           # Jetson dependency installer
├── deploy_jetson.sh          # Jetson auto-start setup
├── run.sh                    # Launch script
├── sieve_reference.xlsx      # ← copy your lab Excel file here
└── logs/                     # Auto-created, stores all test results
```

---

## Classification logic

Particles are classified by size (no per-particle model needed):

- **Pure 40/70**: < 15% of particles are large (≥118px) → all classified as 40/70
- **Pure 20/40**: > 40% of particles are large → all classified as 20/40
- **Mixed**: 15–40% large → each particle split at 113px boundary

Pixel sizes are in original resolution space (5472×3648 camera: 40/70 ≈ 90–110px, 20/40 ≈ 115–175px).

---

## Keyboard shortcuts

| Key | Action |
|---|---|
| `F11` | Toggle fullscreen |
| `Escape` | Exit fullscreen |

---

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA (install from NVIDIA for Jetson)
- CellPose 3.x+
- PyQt5
- OpenCV, NumPy, scipy, scikit-image, matplotlib, openpyxl

---

## Hardware tested

- **Development**: Windows 11, RTX 3070, Python 3.13
- **Deployment**: NVIDIA Jetson with JetPack 5.x

---

## Troubleshooting

**Segmentation looks wrong (huge blobs grouping many particles)**
→ Camera resolution is not 5472×3648. Check `CAMERA_WIDTH`/`CAMERA_HEIGHT` in `config.py`.

**App shows 100% wrong class on a pure sample**
→ Almost always caused by wrong camera resolution (see above).

**`CUDA: False` on Jetson**
→ PyTorch was not installed from the NVIDIA wheel. Reinstall following Step 1.

**Sieve chart is empty / no compliance check**
→ `sieve_reference.xlsx` is missing from the project folder. See Step 4.

**App crashes on import**
→ Run `bash setup_jetson.sh` again — a package likely failed to install silently.

**Sieve chart blue and amber lines are far apart**
→ Camera calibration is not set. Follow the Camera Calibration steps above to set `PIXELS_PER_MM`. The chart uses an estimate until calibrated.

**Calibration saved but chart didn't change**
→ The calibration takes effect on the NEXT analysis run — re-run the image after saving.
