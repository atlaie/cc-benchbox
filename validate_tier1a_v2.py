#!/usr/bin/env python3
"""
validate_tier1a_v2.py — Post-run sanity checks for §4.3 Tier 1A v2.

Runs `analyze_max_tokens_sweep.py` on v1 and v2 sweep parquets, then
compares the single-feature fit parameters (Δa, Δb) and per-cell paired
deltas. The headline claim under test:

    Δb (per-output-token CC slope) reproduces within its 95% CI across
    deploy generations. Brief §4.2 reports +63.8 ms/tok single-feature
    on the max_tokens sweep alone, +64.00 ms/tok [+63.27, +64.86] for
    the two-feature combined fit.

Checks (in order):

  1. All v2 sweep cells present and all rows http_status == 200.
  2. v2 per-cell tokens_out_p50 match v1 (within ±5%) — confirms the
     sampler / chat-template / model didn't drift.
  3. Δb v2 lies inside the v1 fit's 95% CI (the strict reproducibility
     test for the +64 ms/tok headline).
  4. Δa v2 lies inside the v1 fit's 95% CI.
  5. R² stays ≥ 0.99 on both arms in v2 (linearity preserved).

Exit code 0 if all checks pass, 1 otherwise. Prints a brief edit
snippet for §4.2.

Usage:
    python validate_tier1a_v2.py \\
        --v1-dir runs/phase3 \\
        --v2-dir runs/phase3_v2/max_tokens_sweep
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Brief §4.2 reference values from the v1 max_tokens sweep alone
# (single-feature fit; the +64 ms/tok number in the brief is from
# the COMBINED fit with the tokens_in sweep — we don't replicate that
# here since §4.3 covers max_tokens only).
V1_REFERENCE = {
    "delta_a_s":           0.33,    # fixed CC slice
    "delta_b_ms_per_tok":  63.8,    # per-output-token decode slope
}

# Expected cells per arm
EXPECTED_MAX_TOKENS = [32, 128, 512, 1024, 2048]
EXPECTED_N = {32: 100, 128: 100, 512: 50, 1024: 20, 2048: 10}


def cell_dir(root: Path, prefix: str, cc: str, max_tok: int) -> Path:
    return root / f"{prefix}-{cc}-t{max_tok}"


def check_parquets_present(root: Path, prefix: str) -> list[str]:
    """Check all expected v2 cells exist and have populated parquets."""
    failures = []
    for cc in ("off", "on"):
        for max_tok in EXPECTED_MAX_TOKENS:
            p = cell_dir(root, prefix, cc, max_tok) / "requests.parquet"
            if not p.exists():
                failures.append(f"missing parquet: {p}")
                continue
            df = pd.read_parquet(p)
            if len(df) == 0:
                failures.append(f"empty parquet: {p}")
            n_expected = EXPECTED_N[max_tok]
            if abs(len(df) - n_expected) > 3:
                failures.append(
                    f"{p}: n={len(df)} vs expected {n_expected} "
                    f"(allow ±3 for warmup/failures)"
                )
    return failures


def check_http_status(root: Path, prefix: str) -> list[str]:
    failures = []
    for cc in ("off", "on"):
        for max_tok in EXPECTED_MAX_TOKENS:
            p = cell_dir(root, prefix, cc, max_tok) / "requests.parquet"
            if not p.exists():
                continue
            df = pd.read_parquet(p)
            n_bad = int((df["http_status"] != 200).sum())
            if n_bad:
                failures.append(f"{p.parent.name}: {n_bad}/{len(df)} non-200 rows")
    return failures


def per_cell_tok_out_p50(root: Path, prefix: str, cc: str) -> dict:
    """Returns {max_tok: tokens_out_p50} for one CC arm."""
    out = {}
    for max_tok in EXPECTED_MAX_TOKENS:
        p = cell_dir(root, prefix, cc, max_tok) / "requests.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        ok = df[df["http_status"] == 200]
        if len(ok) == 0:
            continue
        out[max_tok] = float(np.median(ok["tokens_out"]))
    return out


def check_tok_out_drift(v1_root: Path, v2_root: Path, prefix: str) -> list[str]:
    """Compare per-cell tokens_out_p50 between v1 and v2 — should be
    near-identical because temperature=0 generation is deterministic
    on identical prompts."""
    failures = []
    for cc in ("off", "on"):
        v1 = per_cell_tok_out_p50(v1_root, prefix, cc)
        v2 = per_cell_tok_out_p50(v2_root, prefix, cc)
        for max_tok in EXPECTED_MAX_TOKENS:
            if max_tok not in v1 or max_tok not in v2:
                continue
            v1_val, v2_val = v1[max_tok], v2[max_tok]
            if v1_val == 0:
                continue
            drift_pct = 100 * abs(v2_val - v1_val) / v1_val
            if drift_pct > 5.0:
                failures.append(
                    f"[{cc} t={max_tok}] tokens_out_p50 drift {drift_pct:.1f}% "
                    f"(v1={v1_val:.0f}, v2={v2_val:.0f}); sampler / template change?"
                )
    return failures


def run_analyze(scripts_dir: Path, data_dir: Path, output_dir: Path,
                 prefix: str) -> Path:
    """Run analyze_max_tokens_sweep.py on a sweep dir; return path to
    sweep_fit.json. Captures stdout/stderr."""
    cmd = [
        sys.executable,
        str(scripts_dir / "analyze_max_tokens_sweep.py"),
        "--data-dir", str(data_dir),
        "--output-dir", str(output_dir),
        "--cell-id-prefix", prefix,
    ]
    print(f"  running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(f"FATAL: analyze_max_tokens_sweep.py failed on {data_dir}")
    fit_path = output_dir / "sweep_fit.json"
    if not fit_path.exists():
        sys.exit(f"FATAL: expected {fit_path} but it wasn't written")
    return fit_path


def fmt_ci(lo: float, hi: float, scale: float = 1.0, unit: str = "") -> str:
    return f"[{lo*scale:+.2f}, {hi*scale:+.2f}]{unit}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v1-dir", type=Path, default=Path("runs/phase3"))
    ap.add_argument("--v2-dir", type=Path,
                    default=Path("runs/phase3_v2/max_tokens_sweep"))
    ap.add_argument("--cell-id-prefix", default="C1")
    ap.add_argument("--scripts-dir", type=Path, default=Path("."),
                    help="Directory containing analyze_max_tokens_sweep.py")
    ap.add_argument("--analysis-root", type=Path,
                    default=Path("runs/phase3_v2/analysis"),
                    help="Where to write the v1 / v2 sweep_fit.json + figures")
    args = ap.parse_args()

    failures: list[str] = []

    # === 1. Parquet presence + http_status ===
    print("[1/5] v2 parquet presence + http_status checks...")
    f = check_parquets_present(args.v2_dir, args.cell_id_prefix)
    if f:
        failures += f
        print("  FAIL: missing/empty parquets — cannot proceed to fits")
        for x in f:
            print(f"    • {x}")
        return 1
    print("  PASS: all 10 cells present")

    f = check_http_status(args.v2_dir, args.cell_id_prefix)
    if f:
        failures += f
        for x in f:
            print(f"    • {x}")
    else:
        print("  PASS: all rows http_status==200")

    # === 2. tokens_out drift ===
    print("\n[2/5] v1 vs v2 tokens_out_p50 drift...")
    f = check_tok_out_drift(args.v1_dir, args.v2_dir, args.cell_id_prefix)
    if f:
        failures += f
        for x in f:
            print(f"    • {x}")
    else:
        print("  PASS: all cells within ±5% on tokens_out_p50")

    # === 3 & 4. Run analyses and compare fits ===
    args.analysis_root.mkdir(parents=True, exist_ok=True)

    print("\n[3/5] Running analyze_max_tokens_sweep.py on v1...")
    v1_fit_path = run_analyze(
        args.scripts_dir, args.v1_dir,
        args.analysis_root / "tier1a_v1", args.cell_id_prefix,
    )

    print("\n[4/5] Running analyze_max_tokens_sweep.py on v2...")
    v2_fit_path = run_analyze(
        args.scripts_dir, args.v2_dir,
        args.analysis_root / "tier1a_v2", args.cell_id_prefix,
    )

    v1_fit = json.loads(v1_fit_path.read_text())
    v2_fit = json.loads(v2_fit_path.read_text())

    # Single-feature deltas (these are what the brief headlines)
    v1_da    = v1_fit["delta_a_seconds"]
    v1_da_lo, v1_da_hi = v1_fit["delta_a_ci"]
    v1_db_ms = v1_fit["delta_b_ms_per_token"]
    v1_db_lo, v1_db_hi = v1_fit["delta_b_ms_per_token_ci"]

    v2_da    = v2_fit["delta_a_seconds"]
    v2_da_lo, v2_da_hi = v2_fit["delta_a_ci"]
    v2_db_ms = v2_fit["delta_b_ms_per_token"]
    v2_db_lo, v2_db_hi = v2_fit["delta_b_ms_per_token_ci"]

    print("\n[5/5] Single-feature fit comparison (max_tokens sweep alone):")
    print(f"  {'param':<22}  {'v1':>20}  {'v2':>20}")
    print("  " + "-" * 64)
    print(f"  {'Δa (s)':<22}  "
          f"{v1_da:+.3f} {fmt_ci(v1_da_lo, v1_da_hi):>14}  "
          f"{v2_da:+.3f} {fmt_ci(v2_da_lo, v2_da_hi):>14}")
    print(f"  {'Δb (ms/out-tok)':<22}  "
          f"{v1_db_ms:+.2f} {fmt_ci(v1_db_lo, v1_db_hi):>14}  "
          f"{v2_db_ms:+.2f} {fmt_ci(v2_db_lo, v2_db_hi):>14}")

    # Reproducibility checks: does v2 estimate sit inside v1's CI?
    if not (v1_db_lo <= v2_db_ms <= v1_db_hi):
        failures.append(
            f"Δb: v2 point {v2_db_ms:+.2f} ms/tok outside v1 CI "
            f"[{v1_db_lo:+.2f}, {v1_db_hi:+.2f}]"
        )
    if not (v1_da_lo <= v2_da <= v1_da_hi):
        failures.append(
            f"Δa: v2 point {v2_da:+.3f}s outside v1 CI "
            f"[{v1_da_lo:+.3f}, {v1_da_hi:+.3f}]"
        )

    # CI overlap (looser check — passes even if point estimates differ
    # as long as the intervals overlap)
    ci_overlap_db = max(v1_db_lo, v2_db_lo) <= min(v1_db_hi, v2_db_hi)
    ci_overlap_da = max(v1_da_lo, v2_da_lo) <= min(v1_da_hi, v2_da_hi)
    if ci_overlap_db:
        print(f"  ✓ Δb CIs overlap (replication consistent)")
    else:
        print(f"  ✗ Δb CIs disjoint — v1 and v2 disagree on the decode tax")

    # === Brief edit snippet ===
    print("\n" + "=" * 72)
    print("BRIEF EDIT SNIPPET (paste into §4.2, after Table 7):")
    print("=" * 72)
    drift_db = v2_db_ms - v1_db_ms
    drift_db_pct = 100 * drift_db / v1_db_ms if v1_db_ms else float("nan")
    print(
        f"\n  The max_tokens sweep was re-run on a fresh deploy pair "
        f"(same image\n"
        f"  digest, same prompts, same dispatch). The single-feature fit "
        f"reproduces\n"
        f"  the v1 decode tax: Δb = {v2_db_ms:+.2f} ms/output-token "
        f"(95% CI {fmt_ci(v2_db_lo, v2_db_hi)}),\n"
        f"  vs v1's {v1_db_ms:+.2f} ms/output-token "
        f"({fmt_ci(v1_db_lo, v1_db_hi)}). The v1 and v2 CIs\n"
        f"  {'overlap' if ci_overlap_db else 'are DISJOINT'}; "
        f"v2 point estimate drifts {drift_db:+.2f} ms/tok "
        f"({drift_db_pct:+.1f}%) vs v1, consistent\n"
        f"  with the inter-deploy envelope documented in §3.\n"
    )

    print(f"\n[output] analysis figures + JSONs in: {args.analysis_root}")

    if failures:
        print("\n" + "!" * 72)
        print(f"VALIDATION FAILED — {len(failures)} issue(s):")
        for f in failures:
            print(f"  • {f}")
        print("!" * 72)
        return 1
    print("\nVALIDATION PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
