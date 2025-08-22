#!/usr/bin/env bash
set -euo pipefail

# Jump to the folder this script lives in
cd "$(dirname "$0")"

echo "==========================="
echo "Running Alt Tag Generator"
echo "==========================="

# Ensure virtual env exists
if [ ! -f ".venv/bin/activate" ]; then
  echo "[!] Virtual environment not found."
  echo "    Please run:  ./setup.sh"
  exit 1
fi

# Activate venv
# shellcheck disable=SC1091
source ".venv/bin/activate"

# Run with unbuffered output so tqdm renders nicely
python3 -u alt_tag_generator.py
status=$?

if [ $status -ne 0 ]; then
  echo "[!] Script exited with code $status."
else
  echo "[OK] Finished successfully."
fi
