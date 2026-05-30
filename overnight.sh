#!/usr/bin/env bash
set -uo pipefail
cd ~/cc-benchbox

BASE_URL=https://cc-deep-eval-egress.debug.pour-demain.containers.tinfoil.dev
DIGEST=sha256:31f411f1d2afbb6e92192901fd6ba3e54c456e04655041032f82db1b3945959d

# === 1. Token-output sweep ===
# tok=128 at N=50, tok=512 at N=25 (compute scales linearly with output length,
# so reduce N to keep total wall ~3 hours).
mkdir -p runs/phase3/token_sweep

for spec in "128 50" "512 25"; do
  TOK=$(echo $spec | awk '{print $1}')
  NREQ=$(echo $spec | awk '{print $2}')
  echo "=== Sweep: max_new_tokens=$TOK, N=$NREQ ==="

  # Pass-through baseline (raw payload returned)
  python phase3_vllm_driver.py \
    --condition repe_bundle \
    --base-url "$BASE_URL/v1" --api-key "$VLLM_API_KEY" \
    --pairs-json data/pairs.json --n-requests $NREQ --req-rate 0.2 \
    --max-new-tokens $TOK \
    --out-dir runs/phase3/token_sweep/C2-on-tok${TOK} \
    --cell-id C2-on-tok${TOK} --cc-state on \
    --model glm-5-1 --image-digest "$DIGEST" --skip-health \
    2>&1 | tee runs/phase3/token_sweep/C2-tok${TOK}.log

  # E3 full pipeline (in-TEE encoder)
  python phase3_egress_driver_v2.py \
    --condition repe_bundle \
    --base-url "$BASE_URL" --api-key "$VLLM_API_KEY" \
    --pairs-json data/pairs.json --n-requests $NREQ --req-rate 0.2 \
    --max-new-tokens $TOK \
    --out-dir runs/phase3/token_sweep/E3-on-tok${TOK} \
    --cell-id E3-on-tok${TOK} --cc-state on \
    --egress-stages "aggregate,plot,bundle,ledger" \
    --model glm-5-1 --image-digest "$DIGEST" --skip-health \
    2>&1 | tee runs/phase3/token_sweep/E3-tok${TOK}.log
done

echo "=== Token sweep complete ==="