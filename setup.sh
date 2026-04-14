#!/bin/bash
# setup.sh — Install dependencies for Proppant QC System
# For Jetson: use setup_jetson.sh instead (handles GPU + apt packages)
# For PC (Windows/Linux):

set -e

echo "=== Proppant QC System — PC Setup ==="

pip install -r requirements.txt
pip install cellpose scikit-image scipy

mkdir -p logs runs

echo ""
echo "=== Setup complete! ==="
echo "Run the app with:  python app.py --windowed"
