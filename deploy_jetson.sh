#!/bin/bash
# deploy_jetson.sh — Install + set up auto-start kiosk on Jetson
# Run once after copying model-1/ to the Jetson.

set -e

APP_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Proppant QC — Jetson Deployment ==="
echo "App directory: $APP_DIR"

# 1. Install dependencies
echo ""
echo "--- Installing dependencies ---"
./setup.sh

# 2. Update desktop entry with actual path
echo ""
echo "--- Setting up auto-start ---"
sed "s|/path/to/model-1|$APP_DIR|g" "$APP_DIR/proppant-qc.desktop" > /tmp/proppant-qc.desktop

# 3. Copy to autostart
mkdir -p ~/.config/autostart
cp /tmp/proppant-qc.desktop ~/.config/autostart/proppant-qc.desktop
chmod +x ~/.config/autostart/proppant-qc.desktop

# 4. Desktop shortcut
cp /tmp/proppant-qc.desktop ~/Desktop/proppant-qc.desktop 2>/dev/null || true
chmod +x ~/Desktop/proppant-qc.desktop 2>/dev/null || true

echo ""
echo "=== Deployment complete! ==="
echo "The app will auto-start on next boot."
echo "To run now:  ./run.sh"
