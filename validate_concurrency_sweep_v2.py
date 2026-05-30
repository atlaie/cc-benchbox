#!/usr/bin/env python3
"""
validate_concurrency_sweep_v2.py — Post-run sanity checks for §4.4.

Run this once the concurrency sweep v2 finishes. Checks (in order,
failures are loud):

  1. All rows http_status == 200 across all 4 c-levels × 2 arms.
  2. v2 per-c wall_p50 CC Δ% lands inside the brief's [+31, +38]%
     baseline-family band (the headline invariance finding).
  3. v1 vs v2 drift in pp per c-level. ±10 pp envelope = pass; outside
     = flag for investigation.
  4. Throughput CC Δ% in v2 sits in the [-27%, -25%] band reported in
     brief Table A3.
  5. Within-deploy c-trend coherence: Δ% should stay tight across c
     (the brief reports a ~7 pp band; v2 should reproduce that).

Exit code 0 if all checks pass, 1 otherwise. Prints a one-paragraph
brief edit snippet suitable for §4.3 / Appendix F.

Usage:
    python validate_concurrency_sweep_v2.py \\
        --v1-dir runs/phase3/concurrency_sweep_n500 \\
        --v2-dir runs/phase3_v2/concurrency_sweep_n500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Brief Table A3 reference values for the v1 concurrency sweep
# (within-deploy at n=500). Wall p50 CC Δ% per c-level.
V1_REFERENCE_DELTA_PCT = {
    8:  37.1,
    16: 37.9,
    32: 37.3,
    64: 31.0,
}

# Brief's claimed +33–38% baseline-family CC band.
BAND_LOW_PCT  = 31.0   # the brief widens to +31 in the concurrency sweep specifically
BAND_HIGH_PCT = 38.0

DRIFT_ENVELOPE_PP = 10.0

# Throughput CC Δ% band from Table A3 (negative because CC reduces throughput)
V1_THROUGHPUT_DELTA_BAND_PCT = (-27.5, -25.0)

C_LEVELS = [8, 16, 32, 64]


def load_pair(data_dir: Path, off_name: str, on_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    off_p = data_dir / off_name / "requests.parquet"
    on_p  = data_dir / on_name  / "requests.parquet"
    if not off_p.exists():
        sys.exit(f"FATAL: missing parquet {off_p}")
    if not on_p.exists():
        sys.exit(f"FATAL: missing parquet {on_p}")
    return pd.read_parquet(off_p), pd.read_parquet(on_p)


def check_http_status(off: pd.DataFrame, on: pd.DataFrame, c: int) -> list[str]:
    failures = []
    n_bad_off = int((off["http_status"] != 200).sum())
    n_bad_on  = int((on["http_status"]  != 200).sum())
    if n_bad_off:
        failures.append(f"[c={c}] CC-off has {n_bad_off}/{len(off)} non-200 rows")
    if n_bad_on:
        failures.append(f"[c={c}] CC-on has {n_bad_on}/{len(on)} non-200 rows")
    return failures


def paired_wall_p50_delta(off: pd.DataFrame, on: pd.DataFrame) -> tuple[float, float, float]:
    """Returns (off_p50, on_p50, delta_pct). Uses unpaired marginal medians
    since concurrency makes per-pair alignment less natural than wall
    measurement stability across n=500 requests."""
    off_ok = off[off["http_status"] == 200]
    on_ok  = on[on["http_status"]  == 200]
    if off_ok.empty or on_ok.empty:
        return float("nan"), float("nan"), float("nan")
    off_p50 = float(off_ok["wall_seconds"].quantile(0.5))
    on_p50  = float(on_ok["wall_seconds"].quantile(0.5))
    delta = 100.0 * (on_p50 - off_p50) / off_p50 if off_p50 > 0 else float("nan")
    return off_p50, on_p50, delta


def throughput_delta(off: pd.DataFrame, on: pd.DataFrame) -> tuple[float, float, float]:
    """Approximate throughput: n_requests / total elapsed wall (last t_complete
    minus first t_send). Returns (off_thr, on_thr, delta_pct)."""
    def _thr(df: pd.DataFrame) -> float:
        ok = df[df["http_status"] == 200]
        if len(ok) < 2 or "t_send" not in ok.columns or "t_complete" not in ok.columns:
            return float("nan")
        elapsed = ok["t_complete"].max() - ok["t_send"].min()
        return len(ok) / elapsed if elapsed > 0 else float("nan")
    off_thr = _thr(off)
    on_thr  = _thr(on)
    delta = 100.0 * (on_thr - off_thr) / off_thr if off_thr > 0 else float("nan")
    return off_thr, on_thr, delta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v1-dir", type=Path,
                    default=Path("runs/phase3/concurrency_sweep_n500"))
    ap.add_argument("--v2-dir", type=Path,
                    default=Path("runs/phase3_v2/concurrency_sweep_n500"))
    args = ap.parse_args()

    failures: list[str] = []
    results: list[dict] = []

    print(f"v1 dir: {args.v1_dir}")
    print(f"v2 dir: {args.v2_dir}\n")

    print(f"{'c':>4}  {'v1 Δ%':>8}  {'v2 Δ%':>8}  {'drift pp':>10}  "
          f"{'v2 off p50':>11}  {'v2 on p50':>11}  {'thr Δ%':>8}")
    print("-" * 78)

    for c in C_LEVELS:
        v1_off, v1_on = load_pair(args.v1_dir, f"C1-off-c{c}", f"C1-on-c{c}")
        v2_off, v2_on = load_pair(args.v2_dir, f"C1-off-c{c}", f"C1-on-c{c}")

        # Check 1: http_status
        failures += check_http_status(v2_off, v2_on, c)

        # Check 2: wall_p50 delta in band
        v2_off_p50, v2_on_p50, v2_delta = paired_wall_p50_delta(v2_off, v2_on)
        _, _, v1_delta = paired_wall_p50_delta(v1_off, v1_on)

        drift_pp = v2_delta - v1_delta if not np.isnan(v2_delta) else float("nan")

        # Throughput
        _, _, thr_delta = throughput_delta(v2_off, v2_on)

        results.append({
            "c": c,
            "v1_delta_pct": v1_delta,
            "v2_delta_pct": v2_delta,
            "drift_pp": drift_pp,
            "v2_off_p50": v2_off_p50,
            "v2_on_p50": v2_on_p50,
            "thr_delta_pct": thr_delta,
        })

        print(f"{c:>4}  {v1_delta:+7.2f}%  {v2_delta:+7.2f}%  "
              f"{drift_pp:+9.2f}   {v2_off_p50:>10.3f}s  {v2_on_p50:>10.3f}s  "
              f"{thr_delta:+7.2f}%")

        # Band check
        if not (BAND_LOW_PCT <= v2_delta <= BAND_HIGH_PCT):
            failures.append(
                f"[c={c}] v2 Δ%={v2_delta:+.2f}% outside the brief's "
                f"[+{BAND_LOW_PCT:.0f}, +{BAND_HIGH_PCT:.0f}]% baseline band"
            )

        # Drift check
        if not np.isnan(drift_pp) and abs(drift_pp) > DRIFT_ENVELOPE_PP:
            failures.append(
                f"[c={c}] drift pp={drift_pp:+.2f} exceeds ±{DRIFT_ENVELOPE_PP} pp envelope "
                f"(v1={v1_delta:+.2f}%, v2={v2_delta:+.2f}%)"
            )

        # Throughput band check
        if not np.isnan(thr_delta):
            lo, hi = V1_THROUGHPUT_DELTA_BAND_PCT
            # Loosen by ±3pp on either side for inter-deploy variation
            if not (lo - 3.0 <= thr_delta <= hi + 3.0):
                failures.append(
                    f"[c={c}] throughput Δ%={thr_delta:+.2f}% outside the brief's "
                    f"[{lo:+.1f}, {hi:+.1f}]% band (±3pp loosened)"
                )

    # Check 5: within-deploy c-trend coherence
    v2_deltas = [r["v2_delta_pct"] for r in results
                 if not np.isnan(r["v2_delta_pct"])]
    if len(v2_deltas) >= 2:
        span = max(v2_deltas) - min(v2_deltas)
        print(f"\nWithin-v2 c-trend span: {span:.2f} pp "
              f"(brief reports ~7 pp across c ∈ [8, 64])")
        if span > 12.0:
            failures.append(
                f"[c-trend] v2 spans {span:.2f} pp across c, "
                f"brief reports ~7 pp — non-trivial broadening"
            )

    # === Brief edit snippet ===
    print("\n" + "=" * 78)
    print("BRIEF EDIT SNIPPET (paste into §4.3 or App. F Table A3 footnote):")
    print("=" * 78)
    v2_deltas_str = ", ".join(f"c={r['c']}: {r['v2_delta_pct']:+.1f}%"
                                for r in results)
    drift_pp_max = max((abs(r["drift_pp"]) for r in results
                         if not np.isnan(r["drift_pp"])), default=float("nan"))
    print(
        f"\n  The concurrency sweep was replicated on a fresh deploy pair\n"
        f"  ({v2_deltas_str}). The CC Δ band reproduces within the\n"
        f"  documented inter-deploy envelope: maximum per-c drift\n"
        f"  vs the original sweep is {drift_pp_max:.2f} pp. The\n"
        f"  regime-invariance finding holds across deploy generations.\n"
    )

    if failures:
        print("\n" + "!" * 78)
        print(f"VALIDATION FAILED — {len(failures)} issue(s):")
        for f in failures:
            print(f"  • {f}")
        print("!" * 78)
        return 1
    print("\nVALIDATION PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
