#!/usr/bin/env python3
"""
analyze_thinking.py
===================

Tier 3G analysis. Validates the Tier 1A linear-fit prediction
(`wall = 0.47 + 275.2·t/1000` for CC-on, R²=0.9999) at long
realised outputs produced by GSM8K + enable_thinking=true.

Inputs
------
- runs/phase3/C_R-off/{requests.parquet, summary.json}
- runs/phase3/C_R-on/{requests.parquet, summary.json}

Outputs
-------
- runs/phase3/tier3g_summary.json   — programmatic summary
- runs/phase3/tier3g_report.txt     — human-readable report
- stdout                            — same as the report

Statistical tests
-----------------
1. Per-cell wall_p50 and CC delta (abs and %).
2. Tier 1A linear-fit residuals on realised tok_out. For each
   request, compute observed_wall - predicted_wall(tok_out), check
   if residual mean is within bootstrap CI of zero. A large positive
   bias would indicate super-linear scaling at long outputs (the
   amortisation-falsification mechanism would have started to bend).
3. Paired bootstrap CI on Δ wall for matched (toxic, benign)
   prompt indices across CC states.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Tier 1A linear-fit coefficients (PHASE3_REFERENCE §3.7, R²=0.9999).
# y = a + b · (tok_out / 1000), where y is wall_seconds.
TIER1A_FIT = {
    "off": {"a": 0.14, "b": 211.4},
    "on":  {"a": 0.47, "b": 275.2},
}

# Tier 1A headline band, for comparison.
TIER1A_BAND = {"low_pct": 30.0, "mid_pct": 33.4, "high_pct": 38.0}


def _predict_wall(tok_out: np.ndarray, cc_state: str) -> np.ndarray:
    """Tier 1A single-feature linear fit prediction."""
    coef = TIER1A_FIT[cc_state]
    return coef["a"] + coef["b"] * (tok_out / 1000.0)


def _bootstrap_ci(
    values: np.ndarray, n_boot: int = 10_000, ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap CI of the mean. Returns (point, low, high)."""
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    boots = rng.choice(values, size=(n_boot, n), replace=True).mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    low = float(np.quantile(boots, alpha))
    high = float(np.quantile(boots, 1 - alpha))
    return (float(values.mean()), low, high)


def _load_cell(out_dir: Path, cell_id: str) -> tuple[pd.DataFrame, dict]:
    """Load requests.parquet and summary.json for one cell."""
    cell_dir = out_dir / cell_id
    parquet_path = cell_dir / "requests.parquet"
    summary_path = cell_dir / "summary.json"
    if not parquet_path.exists():
        sys.exit(f"ERROR: {parquet_path} not found. Has the cell run?")
    if not summary_path.exists():
        sys.exit(f"ERROR: {summary_path} not found.")
    df = pd.read_parquet(parquet_path)
    summary = json.loads(summary_path.read_text())
    return df, summary


def _per_cell_stats(df: pd.DataFrame, summary: dict, cell_id: str) -> dict:
    """Pull per-cell summary stats with sanity guards."""
    wall = df["wall_seconds"].to_numpy()
    tok_out = df["tokens_out"].to_numpy()
    n_success = int(df["http_status"].eq(200).sum()) if "http_status" in df else len(df)
    return {
        "cell_id": cell_id,
        "n_total": len(df),
        "n_success": n_success,
        "wall_seconds": {
            "p50": float(np.median(wall)),
            "mean": float(np.mean(wall)),
            "p10": float(np.quantile(wall, 0.10)),
            "p90": float(np.quantile(wall, 0.90)),
            "max": float(np.max(wall)),
        },
        "tokens_out": {
            "p50": float(np.median(tok_out)),
            "mean": float(np.mean(tok_out)),
            "p90": float(np.quantile(tok_out, 0.90)),
            "max": int(np.max(tok_out)),
        },
        "from_summary": {
            "wall_p50": summary.get("overall", {}).get("wall_seconds", {}).get("p50"),
        },
    }


def _linear_fit_residuals(df: pd.DataFrame, cc_state: str) -> dict:
    """Tier 1A linear-fit residuals on this cell's realised outputs."""
    wall = df["wall_seconds"].to_numpy()
    tok_out = df["tokens_out"].to_numpy().astype(float)
    predicted = _predict_wall(tok_out, cc_state)
    residual = wall - predicted

    mean_pred = float(np.mean(predicted))
    mean_obs = float(np.mean(wall))
    rel_bias_pct = 100.0 * (mean_obs - mean_pred) / mean_pred if mean_pred else float("nan")

    boot_mean, boot_lo, boot_hi = _bootstrap_ci(residual)

    return {
        "cc_state": cc_state,
        "n": len(df),
        "tok_out_p50": float(np.median(tok_out)),
        "tok_out_max": int(np.max(tok_out)),
        "mean_observed_wall_s": mean_obs,
        "mean_predicted_wall_s": mean_pred,
        "relative_bias_pct": rel_bias_pct,
        "residual_mean_s": boot_mean,
        "residual_ci95_low_s": boot_lo,
        "residual_ci95_high_s": boot_hi,
        "residual_ci_brackets_zero": (boot_lo <= 0.0 <= boot_hi),
    }


def _cc_delta(off_df: pd.DataFrame, on_df: pd.DataFrame) -> dict:
    """CC delta at p50 wall plus paired-bootstrap on matched indices."""
    off_p50 = float(np.median(off_df["wall_seconds"]))
    on_p50 = float(np.median(on_df["wall_seconds"]))
    delta_abs = on_p50 - off_p50
    delta_pct = 100.0 * delta_abs / off_p50 if off_p50 else float("nan")

    # Paired bootstrap if lengths match. The driver alternates
    # toxic/benign within each pair, so paired-by-index is the
    # closest we get without semantic prompt alignment.
    paired = None
    n = min(len(off_df), len(on_df))
    if n > 0:
        off_w = off_df["wall_seconds"].to_numpy()[:n]
        on_w = on_df["wall_seconds"].to_numpy()[:n]
        diffs = on_w - off_w
        mean_d, lo, hi = _bootstrap_ci(diffs)
        paired = {
            "n_paired": int(n),
            "mean_delta_abs_s": mean_d,
            "delta_ci95_low_s": lo,
            "delta_ci95_high_s": hi,
            "mean_delta_pct": 100.0 * mean_d / off_w.mean() if off_w.mean() else float("nan"),
        }

    return {
        "p50_off_s": off_p50,
        "p50_on_s": on_p50,
        "delta_abs_s_at_p50": delta_abs,
        "delta_pct_at_p50": delta_pct,
        "paired_bootstrap": paired,
    }


def _format_report(stats: dict) -> str:
    """Human-readable report. Folds straight into the brief."""
    lines = []
    a = lines.append
    a("=" * 70)
    a("TIER 3G — REASONING + THINKING (GSM8K, enable_thinking=true)")
    a("=" * 70)
    a("")

    a("Per-cell summary")
    a("-" * 70)
    for c in (stats["C_R-off"], stats["C_R-on"]):
        a(f"  {c['cell_id']}: n_success={c['n_success']}/{c['n_total']}  "
          f"wall_p50={c['wall_seconds']['p50']:.2f}s  "
          f"tok_out_p50={c['tokens_out']['p50']:.0f}  "
          f"tok_out_max={c['tokens_out']['max']}")
    a("")

    d = stats["cc_delta"]
    a("CC delta (p50 wall, single-cell)")
    a("-" * 70)
    a(f"  Δ abs : {d['delta_abs_s_at_p50']:+.3f} s")
    a(f"  Δ %   : {d['delta_pct_at_p50']:+.2f}%")
    if d["paired_bootstrap"]:
        pb = d["paired_bootstrap"]
        a(f"  Paired bootstrap (n={pb['n_paired']}):  "
          f"mean Δ = {pb['mean_delta_abs_s']:+.3f} s  "
          f"[{pb['delta_ci95_low_s']:+.3f}, {pb['delta_ci95_high_s']:+.3f}]  "
          f"({pb['mean_delta_pct']:+.2f}%)")
    a(f"  Tier 1A band : +{TIER1A_BAND['low_pct']:.1f}% to "
      f"+{TIER1A_BAND['high_pct']:.1f}% (mid +{TIER1A_BAND['mid_pct']:.1f}%)")
    in_band = TIER1A_BAND["low_pct"] <= d["delta_pct_at_p50"] <= TIER1A_BAND["high_pct"]
    a(f"  In band : {'YES' if in_band else 'NO'}")
    a("")

    a("Tier 1A linear-fit residuals at long outputs")
    a("-" * 70)
    for fit in (stats["fit_off"], stats["fit_on"]):
        a(f"  cc={fit['cc_state']}  n={fit['n']}  "
          f"tok_out_p50={fit['tok_out_p50']:.0f}  tok_out_max={fit['tok_out_max']}")
        a(f"    observed wall (mean):  {fit['mean_observed_wall_s']:.2f}s")
        a(f"    predicted wall (mean): {fit['mean_predicted_wall_s']:.2f}s")
        a(f"    relative bias:         {fit['relative_bias_pct']:+.2f}%")
        a(f"    residual mean:         {fit['residual_mean_s']:+.3f}s  "
          f"[{fit['residual_ci95_low_s']:+.3f}, {fit['residual_ci95_high_s']:+.3f}]")
        verdict = (
            "linear fit holds"
            if fit["residual_ci_brackets_zero"]
            else "linear fit BREAKS (residual CI excludes 0)"
        )
        a(f"    verdict:               {verdict}")
        a("")

    a("Overall verdict")
    a("-" * 70)
    fit_holds = (stats["fit_off"]["residual_ci_brackets_zero"]
                 and stats["fit_on"]["residual_ci_brackets_zero"])
    in_band = TIER1A_BAND["low_pct"] <= d["delta_pct_at_p50"] <= TIER1A_BAND["high_pct"]
    if fit_holds and in_band:
        a("  PASS: linear fit extrapolates cleanly to long reasoning")
        a("        outputs; CC delta stays in the Tier 1A band.")
    elif in_band and not fit_holds:
        a("  PARTIAL: CC delta is in-band but linear-fit residuals are")
        a("           biased — investigate whether wall scales super-")
        a("           linearly at long outputs (would indicate KV-cache")
        a("           memory-bandwidth ceiling under MKTME).")
    elif fit_holds and not in_band:
        a("  PARTIAL: linear fit holds but CC delta drifted outside the")
        a("           +30-38% band. Either good news (decode amortises")
        a("           the CC tax across reasoning trace) or a regime")
        a("           change worth flagging — check tok_out distribution.")
    else:
        a("  FAIL: linear fit AND CC band both deviate. Reasoning workload")
        a("        is a distinct regime; brief needs a separate section.")
    a("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--out-dir", type=Path, default=Path("runs/phase3"),
                    help="Directory containing C_R-off/ and C_R-on/ subdirs.")
    ap.add_argument("--no-write-files", action="store_true",
                    help="Skip writing tier3g_summary.json + tier3g_report.txt.")
    args = ap.parse_args()

    off_df, off_summary = _load_cell(args.out_dir, "C_R-off")
    on_df, on_summary = _load_cell(args.out_dir, "C_R-on")

    stats = {
        "C_R-off": _per_cell_stats(off_df, off_summary, "C_R-off"),
        "C_R-on":  _per_cell_stats(on_df,  on_summary,  "C_R-on"),
        "cc_delta": _cc_delta(off_df, on_df),
        "fit_off": _linear_fit_residuals(off_df, "off"),
        "fit_on":  _linear_fit_residuals(on_df,  "on"),
        "tier1a_fit_used": TIER1A_FIT,
    }

    report = _format_report(stats)
    print(report)

    if not args.no_write_files:
        (args.out_dir / "tier3g_summary.json").write_text(
            json.dumps(stats, indent=2, default=float), encoding="utf-8"
        )
        (args.out_dir / "tier3g_report.txt").write_text(report, encoding="utf-8")
        print(f"\nWrote {args.out_dir/'tier3g_summary.json'} and "
              f"{args.out_dir/'tier3g_report.txt'}")


if __name__ == "__main__":
    main()
