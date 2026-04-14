#!/bin/bash
# setup_jetson.sh — Install all dependencies for Proppant QC on Jetson
# Run ONCE after cloning the repo.
#
# Prerequisites:
#   - JetPack 5.x installed
#   - PyTorch wheel already installed (see README)
#
# Usage:
#   bash setup_jetson.sh

set -e

echo "========================================"
echo "  Proppant QC — Jetson Setup"
echo "========================================"
echo ""

# ── System packages ───────────────────────────────────────────────────
echo "[1/4] Installing system packages..."
sudo apt-get update -q
sudo apt-get install -y \
    python3-pip \
    python3-pyqt5 \
    python3-pyqt5.qtsvg \
    libopencv-dev \
    python3-opencv \
    libhdf5-dev \
    pkg-config \
    libfreetype6-dev

# ── Python packages ───────────────────────────────────────────────────
echo ""
echo "[2/4] Installing Python packages..."
pip3 install --upgrade pip

# Core vision
pip3 install \
    cellpose \
    opencv-python-headless \
    scikit-image \
    scipy

# Data + plotting
pip3 install \
    numpy \
    matplotlib \
    openpyxl \
    Pillow

# ── Verify PyTorch + CUDA ─────────────────────────────────────────────
echo ""
echo "[3/4] Checking PyTorch + CUDA..."
python3 -c "
import torch
print(f'  PyTorch version : {torch.__version__}')
print(f'  CUDA available  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU             : {torch.cuda.get_device_name(0)}')
else:
    print('  WARNING: CUDA not available — app will run on CPU (slower)')
"

# ── Create required directories ───────────────────────────────────────
echo ""
echo "[4/4] Creating directories..."
mkdir -p logs runs

# ── Verify app can be imported ────────────────────────────────────────
echo ""
echo "Verifying app imports..."
python3 -c "
from PyQt5.QtWidgets import QApplication
import cellpose
import cv2
import numpy
import matplotlib
import openpyxl
print('  All imports OK')
"

echo ""
echo "========================================"
echo "  Setup complete!"
echo ""
echo "  Run the app:"
echo "    ./run.sh              (fullscreen kiosk)"
echo "    python3 app.py --windowed   (windowed)"
echo ""
echo "  Auto-start on boot:"
echo "    bash deploy_jetson.sh"
echo "========================================"
