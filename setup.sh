#!/usr/bin/env bash
# One-time setup: install deps + download the MediaPipe hand model.
set -euo pipefail

echo "[1/2] Installing Python deps (use a Python 3.9–3.12 venv)…"
pip install -r requirements.txt

echo "[2/2] Downloading MediaPipe hand_landmarker model → models/…"
mkdir -p models
curl -sSL -o models/hand_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

echo "Done. Model: $(du -h models/hand_landmarker.task | cut -f1)"
