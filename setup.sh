#!/usr/bin/env bash
set -euo pipefail

echo "==========================="
echo "Setting up environment..."
echo "==========================="

# Create venv if missing (use .venv on Unix)
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

# Activate venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "Upgrading pip..."
python -m pip install --upgrade pip

# On macOS, install default PyTorch wheel (CPU + Apple Silicon MPS supported)
echo "Installing packages..."
pip install torch transformers pillow tqdm

echo "==========================="
echo "Setup complete!"
echo "==========================="
echo "To activate later: source .venv/bin/activate"
