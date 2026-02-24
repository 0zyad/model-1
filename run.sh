#!/bin/bash
# run.sh — Launch the Proppant QC System
# Usage:
#   ./run.sh              # fullscreen (kiosk mode)
#   ./run.sh --windowed   # windowed mode for testing

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

python3 app.py "$@"
