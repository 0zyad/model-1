# Proppant Vision — AI Quality Control System

> Senior Capstone Project · KFUPM × Industry · 2025–2026

An automated quality control system for oil & gas proppant samples. Replaces manual lab sieve inspection with a deep learning computer vision pipeline running in real time on edge hardware.

**Deployed as a touchscreen kiosk on NVIDIA Jetson Orin.**

---

## What it does

A technician places a proppant sample tray in front of a camera and presses Capture. The system:

1. Segments every particle using **CellPose** (GPU-accelerated instance segmentation)
2. Classifies particles as **40/70 mesh** or **20/40 mesh** by size distribution
3. Generates a **PASS / FAIL** verdict against API/ISO industry specifications
4. Compares results against lab sieve reference data (Spec 8, ±10 wt% tolerance)
5. Exports results as JSON, CSV, and annotated overlay image for lab records

Full analysis runs in **8–15 seconds** on Jetson GPU.

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Segmentation | CellPose (PyTorch) — `cyto3` pretrained, fine-tuned on proppant data |
| Computer Vision | OpenCV, scikit-image, scipy |
| UI | PyQt5 — 5-screen kiosk app, touch-friendly, fullscreen |
| Hardware | NVIDIA Jetson Orin (deployment), RTX 3070 (development) |
| Language | Python 3.10+ |

---

## Results

- Detects **400–900 particles per image** at 5472×3648 resolution
- **90–97% classification rate** on pure samples
- Size measurement error: **5–8%** vs manual sieve
- Runs at full resolution via automatic 0.33× downscale + mask reprojection (avoids OOM on Jetson)

---

## App screens

| Screen | Description |
|--------|-------------|
| Dashboard | Batch ID entry, camera or upload mode, test history |
| Place Tray | Live camera preview, capture button |
| Running | Progress bar with live step labels while AI runs |
| Results | 5-card swipeable view — overlay, verdict, spec checks, sieve chart, export |

---

## Project structure

```
proppant-vision/
├── app.py                    # Entry point
├── config.py                 # All parameters and thresholds
├── inference_stardist.py     # Core pipeline (CellPose → classifier → verdict)
├── camera.py                 # Camera capture
├── logger.py                 # Saves JSON/CSV/overlay to logs/
├── segmentation/
│   ├── cellpose_seg.py       # CellPose wrapper (GPU, CLAHE, auto-scale)
│   └── watershed_refine.py   # Watershed post-processing
├── widgets/
│   ├── dashboard.py          # Home screen
│   ├── place_tray.py         # Camera preview
│   ├── running.py            # Analysis progress
│   └── results.py            # 5-card results viewer
├── setup_jetson.sh           # Jetson dependency installer
├── deploy_jetson.sh          # Jetson auto-start on boot
└── run.sh                    # Launch script
```

---

## Setup — Windows (dev)

```bash
git clone https://github.com/0zyad/proppant-vision.git
cd proppant-vision
pip install -r requirements.txt
pip install cellpose scikit-image scipy
python app.py --windowed
```

## Setup — Jetson (deployment)

**Step 1 — Install PyTorch for Jetson first** (must match your JetPack version):

```bash
# Check JetPack version
cat /etc/nv_tegra_release

# Install PyTorch from NVIDIA wheel (JetPack 5.x)
pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v51 \
    torch torchvision

# Verify GPU is available
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

**Step 2 — Clone and install:**

```bash
git clone https://github.com/0zyad/proppant-vision.git
cd proppant-vision
bash setup_jetson.sh
```

**Step 3 — Add your lab sieve file** (optional but recommended):

```bash
cp /path/to/your/sieve_file.xlsx ./sieve_reference.xlsx
```

**Step 4 — Run:**

```bash
./run.sh                    # Fullscreen kiosk
python3 app.py --windowed   # Windowed (testing)
bash deploy_jetson.sh       # Auto-start on boot
```

---

## Configuration (`config.py`)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `CELLPOSE_PRETRAINED` | `cyto3` | Pretrained model — do not change |
| `CELLPOSE_DIAMETER` | `65` | At 0.33× scale — do not change |
| `CAMERA_WIDTH/HEIGHT` | `5472×3648` | Must match training resolution |
| `PURITY_THRESHOLD` | `0.90` | ≥90% one type → PASS |
| `MAX_SIZE_ERROR` | `0.10` | Spec 8: ±10 wt% vs lab |
| `PIXELS_PER_MM` | `None` | Set after camera calibration |

> The camera **must** be set to 5472×3648. All classification thresholds depend on this resolution.

---

## Classification logic

No per-particle model is needed — classification uses size distribution:

- **Pure 40/70**: < 15% of particles are large (≥118 px) → all classified as 40/70
- **Pure 20/40**: > 40% large → all classified as 20/40  
- **Mixed**: 15–40% large → each particle split at 113 px boundary

Pixel sizes are in original-resolution space (40/70 ≈ 90–110 px, 20/40 ≈ 115–175 px at 5472×3648).

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Segmentation groups many particles into blobs | Camera resolution ≠ 5472×3648 — check config |
| 100% wrong class on a pure sample | Same — wrong camera resolution |
| `CUDA: False` on Jetson | PyTorch not installed from NVIDIA wheel — redo Step 1 |
| Sieve chart empty | `sieve_reference.xlsx` missing from project folder |
| App crashes on import | Run `bash setup_jetson.sh` again |

---

## Hardware tested

- **Development:** Windows 11, RTX 3070, Python 3.13
- **Deployment:** NVIDIA Jetson Orin, JetPack 5.x
