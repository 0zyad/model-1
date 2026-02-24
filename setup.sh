#!/bin/bash
# setup.sh — Install dependencies for Proppant QC System
# Works on both PC (Windows/Linux) and Jetson

set -e

echo "=== Proppant QC System — Setup ==="

# Detect platform
if [ -f /etc/nv_tegra_release ]; then
    echo "Jetson detected"
    sudo apt update
    sudo apt install -y python3-pyqt5 python3-matplotlib python3-opencv
    pip3 install ultralytics numpy Pillow
else
    echo "PC detected (Windows/Linux)"
    pip install -r requirements.txt
fi

# Create logs directory
mkdir -p logs

echo ""
echo "=== Setup complete! ==="
echo "Run the app with:  python app.py --windowed"
echo "Or fullscreen:     python app.py"
