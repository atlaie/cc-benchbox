#!/bin/bash
# Wait on the C3-off smoke test, then sanity-check the Tier 1B YAML edits,
# then launch the two attestation cells.
set -o pipefail
cd "/Users/atlaie/prepilot/cc-benchbox"
source .venv/bin/activate

echo "[$(date)] waiting for C3-off matrix PID 6401..."
while kill -0 6401 2>/dev/null; do sleep 30; done
echo "[$(date)] C3-off matrix finished."

# Pre-flight: confirm the YAML has the Tier 1B cells before chewing
# through a 30-min deploy and finding out it doesn't.
for cell in C1-off-attest C1-on-attest; do
    if ! grep -q "cell_id: $cell" configs/phase3-matrix.yaml; then
        echo "[$(date)] FATAL: $cell not found in configs/phase3-matrix.yaml" >&2
        exit 3
    fi
done
echo "[$(date)] YAML pre-flight ok. Launching Tier 1B."

python phase3_run_matrix.py \
    --matrix configs/phase3-matrix.yaml \
    --only-cells C1-off-attest,C1-on-attest \
    --out-dir runs/phase3
EXIT_CODE=$?
echo "[$(date)] Tier 1B finished. Exit code: $EXIT_CODE"
exit $EXIT_CODE
