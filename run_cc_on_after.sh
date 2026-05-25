#!/bin/bash
set -eo pipefail
cd "/Users/atlaie/prepilot/cc-benchbox"
source .venv/bin/activate
echo "[$(date)] waiting for CC-off PID 96044..."
while kill -0 96044 2>/dev/null; do sleep 60; done
echo "[$(date)] CC-off finished. Starting CC-on..."
python phase3_sweep_max_tokens.py \
    --matrix phase3-matrix.yaml \
    --cc-state on \
    --max-tokens 32 128 512 1024 2048 \
    --n-requests 100 100 50 20 10 \
    --out-dir runs/phase3
echo "[$(date)] CC-on finished. Exit code: $?"
