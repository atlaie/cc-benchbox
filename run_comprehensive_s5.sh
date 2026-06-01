#!/usr/bin/env bash
# run_comprehensive_s5.sh — All-in-one §5 measurement run.
#
# Preconditions:
#   1. CC-on pysyft deploy is up (--debug, NO --disable-cc-mode)
#   2. capture_residual_stream_v2.py and capture_routing_v2.py in cc-benchbox/
#   3. phase3_pysyft_driver.py is patched (patch_driver_auditor_role.py applied)
#   4. cc-deep-eval repo on PYTHONPATH
#
# Usage:
#   cd /Users/atlaie/prepilot/cc-benchbox
#   export PYTHONPATH=/Users/atlaie/prepilot/cc-deep-eval
#   bash run_comprehensive_s5.sh <DATASITE_URL>
#
# Example:
#   bash run_comprehensive_s5.sh https://pysyft-m1.debug.pour-demain.containers.tinfoil.dev
#
# Total: 4 cells × 50 prompts × ~14s/req ≈ 47 min/auditor ≈ 94 min total
# (plus ~5 min setup overhead)

set -euo pipefail

DATASITE_URL="${1:?Usage: bash run_comprehensive_s5.sh <DATASITE_URL>}"
OUTBASE="runs/phase3_pysyft/S5-comprehensive"
PAIRS="data/pairs.json"
N_PER_CELL=50
RATE=0.15
MAX_TOK=32

AUD1_EMAIL="auditor1@aisi.gov.uk"
AUD1_PASS="aud1tor_pa55!"
AUD2_EMAIL="auditor2@aisi.gov.uk"
AUD2_PASS="aud2tor_pa55!"
ADMIN_EMAIL="info@openmined.org"
ADMIN_PASS="changethis"

echo "============================================================"
echo "Comprehensive §5 measurement run"
echo "Datasite: ${DATASITE_URL}"
echo "Output:   ${OUTBASE}"
echo "============================================================"
echo ""

# ---- STEP 0: Setup (accounts + v2 endpoints) --------------------------------
echo "[SETUP] Creating auditor accounts and registering v2 endpoints..."

python3 -c "
import syft as sy
import sys; sys.path.insert(0, '.')
from capture_residual_stream_v2 import build_endpoint as build_rs
from capture_routing_v2 import build_endpoint as build_rt

admin = sy.login(url='${DATASITE_URL}', email='${ADMIN_EMAIL}', password='${ADMIN_PASS}')

# Create auditor accounts (idempotent)
for email, name, pw in [
    ('${AUD1_EMAIL}', 'Auditor One',  '${AUD1_PASS}'),
    ('${AUD2_EMAIL}', 'Auditor Two',  '${AUD2_PASS}'),
]:
    existing = [u for u in admin.users if u.email == email]
    if not existing:
        admin.register(email=email, name=name, password=pw, password_verify=pw)
        print(f'  Created {email}')
    else:
        print(f'  {email} already exists')

# Delete old endpoints and register v2 (identity-gated)
for path in ['prepilot.capture_residual_stream', 'prepilot.capture_routing']:
    try:
        admin.custom_api.delete(path)
        print(f'  Deleted old {path}')
    except Exception:
        print(f'  No old {path} to delete')

for label, builder in [('residual_stream_v2', build_rs), ('routing_v2', build_rt)]:
    try:
        ep = builder()
        res = admin.custom_api.add(endpoint=ep)
        print(f'  Registered {label}: {res}')
    except Exception as e:
        print(f'  FAILED {label}: {e}')
        sys.exit(1)

eps = admin.custom_api.api_endpoints()
print(f'  Endpoints now: {[e.path for e in eps]}')
print('[SETUP] Done.')
"

echo ""

# ---- STEP 1: Auditor 1 — capture_residual_stream (CC-on) --------------------
echo "[RUN 1/4] Auditor 1 × capture_residual_stream (n=${N_PER_CELL})"
python phase3_pysyft_driver.py \
    --endpoints capture_residual_stream \
    --datasite-url "${DATASITE_URL}" \
    --no-proxy --flatten --flatten-sides toxic,benign \
    --pairs-json "${PAIRS}" --n-requests "${N_PER_CELL}" --req-rate "${RATE}" \
    --max-new-tokens "${MAX_TOK}" \
    --auditor-email "${AUD1_EMAIL}" --auditor-password "${AUD1_PASS}" \
    --out-dir "${OUTBASE}/aud1-residual" --cell-id S5-aud1-residual --cc-state on

echo ""

# ---- STEP 2: Auditor 1 — capture_routing (CC-on) ----------------------------
echo "[RUN 2/4] Auditor 1 × capture_routing (n=${N_PER_CELL})"
python phase3_pysyft_driver.py \
    --endpoints capture_routing \
    --datasite-url "${DATASITE_URL}" \
    --no-proxy --flatten --flatten-sides toxic,benign \
    --pairs-json "${PAIRS}" --n-requests "${N_PER_CELL}" --req-rate "${RATE}" \
    --max-new-tokens "${MAX_TOK}" \
    --routing-layers "12,23,39,51,62,70" \
    --auditor-email "${AUD1_EMAIL}" --auditor-password "${AUD1_PASS}" \
    --out-dir "${OUTBASE}/aud1-routing" --cell-id S5-aud1-routing --cc-state on

echo ""

# ---- STEP 3: Auditor 2 — capture_residual_stream (CC-on) --------------------
echo "[RUN 3/4] Auditor 2 × capture_residual_stream (n=${N_PER_CELL})"
python phase3_pysyft_driver.py \
    --endpoints capture_residual_stream \
    --datasite-url "${DATASITE_URL}" \
    --no-proxy --flatten --flatten-sides toxic,benign \
    --pairs-json "${PAIRS}" --n-requests "${N_PER_CELL}" --req-rate "${RATE}" \
    --max-new-tokens "${MAX_TOK}" \
    --auditor-email "${AUD2_EMAIL}" --auditor-password "${AUD2_PASS}" \
    --out-dir "${OUTBASE}/aud2-residual" --cell-id S5-aud2-residual --cc-state on

echo ""

# ---- STEP 4: Auditor 2 — capture_routing (CC-on) ----------------------------
echo "[RUN 4/4] Auditor 2 × capture_routing (n=${N_PER_CELL})"
python phase3_pysyft_driver.py \
    --endpoints capture_routing \
    --datasite-url "${DATASITE_URL}" \
    --no-proxy --flatten --flatten-sides toxic,benign \
    --pairs-json "${PAIRS}" --n-requests "${N_PER_CELL}" --req-rate "${RATE}" \
    --max-new-tokens "${MAX_TOK}" \
    --routing-layers "12,23,39,51,62,70" \
    --auditor-email "${AUD2_EMAIL}" --auditor-password "${AUD2_PASS}" \
    --out-dir "${OUTBASE}/aud2-routing" --cell-id S5-aud2-routing --cc-state on

echo ""

# ---- STEP 5: Summary --------------------------------------------------------
echo "[SUMMARY] Aggregating results..."

python3 -c "
import pandas as pd
import json
from pathlib import Path

base = Path('${OUTBASE}')
cells = ['aud1-residual', 'aud1-routing', 'aud2-residual', 'aud2-routing']
all_dfs = []

for cell in cells:
    pq = base / cell / 'requests.parquet'
    if not pq.exists():
        print(f'  WARNING: {pq} missing')
        continue
    df = pd.read_parquet(pq)
    all_dfs.append(df)
    ok = df[df.error.isna()]
    print(f'  {cell}: n={len(ok)}/{len(df)} '
          f'auditor={ok.auditor_id.iloc[0] if len(ok) else \"?\"} '
          f'engagement={ok.engagement_id.iloc[0][:12] if len(ok) else \"?\"}... '
          f'approval_p50={ok.pysyft_approval_seconds.median()*1000:.3f}ms '
          f'ledger_p50={ok.pysyft_ledger_seconds.median()*1000:.3f}ms '
          f'wall_p50={ok.wall_seconds.median():.2f}s')

if all_dfs:
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_parquet(base / 'all_cells.parquet', index=False)
    print(f'')
    print(f'  Combined: {base / \"all_cells.parquet\"} ({len(combined)} rows)')

    # Isolation check
    ok = combined[combined.error.isna()]
    engagements = ok.groupby('auditor_id')['engagement_id'].nunique()
    print(f'')
    print(f'  Per-evaluator isolation:')
    for aud, n_eng in engagements.items():
        print(f'    {aud}: {n_eng} engagement(s)')
    cross = ok.groupby('auditor_id')['engagement_id'].first()
    same = len(cross.unique()) < len(cross)
    print(f'    Engagement IDs shared across auditors: {same}')

print('')
print('[DONE] All 4 cells complete.')
"
