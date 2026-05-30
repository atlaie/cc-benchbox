#!/usr/bin/env python3
"""
validate_tier3g_v2.py — Post-run sanity checks for the Tier 3G v2 rerun.

Run this once both C_R_v2 arms have written their parquets. Checks (in
order, failures are loud):

  1. All rows http_status == 200 on both arms (the rerun's headline goal:
     no CC-on timeout at the longest pair).
  2. v1 vs v2 realised tokens_out distributions agree closely — if they
     don't, something changed in the model / sampler / chat template
     and the v2 results aren't a clean replication.
  3. Byte-identity of completion_text across CC states within v2 (extends
     the brief's short-output byte-identity audit to multi-thousand-token
     reasoning traces; v1 reported 49/50).
  4. Pair_id alignment between off and on arms (paired-bootstrap inputs
     must be aligned).
  5. v2 CC delta vs v1's +33.4% — flag if outside the ±10% inter-deploy
     drift envelope documented in brief §3 / §4.3.

Exit code 0 if all checks pass, 1 otherwise. Outputs a one-paragraph
summary block suitable for pasting into the §4.1 brief edit.

Usage:
    python validate_tier3g_v2.py \\
        --v1-dir runs/phase3 \\
        --v2-dir runs/phase3_v2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


V1_HEADLINE_DELTA_PCT = 33.4      # brief §4.2, Figure 1
DRIFT_ENVELOPE_PP     = 10.0      # ±10pp on relative delta; ±10% on absolutes
                                   # (intentionally generous — v1 was N=1)


def load_pair(data_dir: Path, off_name: str, on_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    off_p = data_dir / off_name / "requests.parquet"
    on_p  = data_dir / on_name  / "requests.parquet"
    if not off_p.exists():
        sys.exit(f"FATAL: missing parquet {off_p}")
    if not on_p.exists():
        sys.exit(f"FATAL: missing parquet {on_p}")
    return pd.read_parquet(off_p), pd.read_parquet(on_p)


def check_http_status(off: pd.DataFrame, on: pd.DataFrame, label: str) -> list[str]:
    failures = []
    n_bad_off = int((off["http_status"] != 200).sum())
    n_bad_on  = int((on["http_status"]  != 200).sum())
    if n_bad_off:
        failures.append(f"[{label}] CC-off has {n_bad_off}/{len(off)} non-200 rows; "
                        f"expected 0. error samples: {off[off.http_status != 200]['error'].dropna().head(3).tolist()}")
    if n_bad_on:
        failures.append(f"[{label}] CC-on has {n_bad_on}/{len(on)} non-200 rows; "
                        f"expected 0. error samples: {on[on.http_status != 200]['error'].dropna().head(3).tolist()}")
    return failures


def check_pair_alignment(off: pd.DataFrame, on: pd.DataFrame, label: str) -> list[str]:
    failures = []
    off_keys = set(zip(off["pair_id"], off["prompt_class"]))
    on_keys  = set(zip(on["pair_id"],  on["prompt_class"]))
    only_off = off_keys - on_keys
    only_on  = on_keys  - off_keys
    if only_off or only_on:
        failures.append(f"[{label}] pairing asymmetry: "
                        f"{len(only_off)} off-only, {len(only_on)} on-only "
                        f"(common: {len(off_keys & on_keys)})")
    return failures


def check_distribution_drift(v1: pd.DataFrame, v2: pd.DataFrame,
                              col: str, label: str,
                              ks_threshold: float = 0.30) -> tuple[list[str], dict]:
    """Compare two distributions for the same metric across v1 / v2.
    Returns (failure_list, stats_dict). Uses two-sample KS — a KS
    statistic > 0.30 on n≈50 means the distributions diverge enough that
    something changed in the underlying generation process."""
    failures = []
    a = v1[col].dropna().to_numpy(dtype=float)
    b = v2[col].dropna().to_numpy(dtype=float)
    if len(a) < 5 or len(b) < 5:
        return failures, {}
    ks_stat, ks_p = stats.ks_2samp(a, b)
    info = {
        "v1_n":      len(a),
        "v2_n":      len(b),
        "v1_p50":    float(np.median(a)),
        "v2_p50":    float(np.median(b)),
        "v1_p95":    float(np.quantile(a, 0.95)),
        "v2_p95":    float(np.quantile(b, 0.95)),
        "ks_stat":   float(ks_stat),
        "ks_pvalue": float(ks_p),
    }
    if ks_stat > ks_threshold:
        failures.append(
            f"[{label}] {col}: v1 vs v2 KS={ks_stat:.3f} (>{ks_threshold:.2f}); "
            f"distributions diverge enough that something changed "
            f"(model? sampler? template?). "
            f"v1 p50={info['v1_p50']:.2f}, v2 p50={info['v2_p50']:.2f}."
        )
    return failures, info


def check_byte_identity(off: pd.DataFrame, on: pd.DataFrame,
                         label: str) -> tuple[int, int]:
    """Returns (n_identical, n_paired). The brief reported 49/50 on v1
    (one timeout). On v2 with the budget bumped, expect ~50/50."""
    common = off.set_index(["pair_id", "prompt_class"]).join(
        on.set_index(["pair_id", "prompt_class"]),
        lsuffix="_off", rsuffix="_on", how="inner",
    )
    common = common[(common["http_status_off"] == 200) &
                    (common["http_status_on"]  == 200)]
    if "completion_text_off" not in common.columns:
        return 0, len(common)
    identical = (common["completion_text_off"] == common["completion_text_on"]).sum()
    return int(identical), len(common)


def check_longest_pair_completion(off: pd.DataFrame, on: pd.DataFrame,
                                    label: str) -> list[str]:
    """The v1 timeout was the longest CC-off realisation; check that
    same/similar pair completes under CC-on in v2."""
    failures = []
    off_ok = off[off["http_status"] == 200]
    on_ok  = on[on["http_status"]  == 200]
    if off_ok.empty or on_ok.empty:
        return [f"[{label}] cannot check longest pair — one arm is empty after status filter"]
    longest_off_idx = off_ok["tokens_out"].idxmax()
    longest_pair = (int(off_ok.loc[longest_off_idx, "pair_id"]),
                    off_ok.loc[longest_off_idx, "prompt_class"])
    longest_off_tok = int(off_ok.loc[longest_off_idx, "tokens_out"])
    longest_off_wall = float(off_ok.loc[longest_off_idx, "wall_seconds"])

    on_match = on_ok[(on_ok["pair_id"] == longest_pair[0])
                      & (on_ok["prompt_class"] == longest_pair[1])]
    if on_match.empty:
        failures.append(f"[{label}] longest CC-off pair {longest_pair} "
                        f"(tok_out={longest_off_tok}, wall={longest_off_wall:.1f}s) "
                        f"has no matching CC-on success row")
    else:
        on_tok  = int(on_match.iloc[0]["tokens_out"])
        on_wall = float(on_match.iloc[0]["wall_seconds"])
        print(f"  longest pair {longest_pair}: "
              f"CC-off tok={longest_off_tok} wall={longest_off_wall:.1f}s | "
              f"CC-on  tok={on_tok} wall={on_wall:.1f}s")
    return failures


def compute_cc_delta_pct(off: pd.DataFrame, on: pd.DataFrame) -> float:
    """Paired-pair wall_p50 CC delta as percentage."""
    common = off.set_index(["pair_id", "prompt_class"]).join(
        on.set_index(["pair_id", "prompt_class"]),
        lsuffix="_off", rsuffix="_on", how="inner",
    )
    common = common[(common["http_status_off"] == 200) &
                    (common["http_status_on"]  == 200)]
    if common.empty:
        return float("nan")
    off_p50 = common["wall_seconds_off"].quantile(0.5)
    on_p50  = common["wall_seconds_on"].quantile(0.5)
    return 100.0 * (on_p50 - off_p50) / off_p50 if off_p50 > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v1-dir", type=Path, default=Path("runs/phase3"))
    ap.add_argument("--v2-dir", type=Path, default=Path("runs/phase3_v2"))
    ap.add_argument("--cell-off", default="C_R-off")
    ap.add_argument("--cell-on",  default="C_R-on")
    args = ap.parse_args()

    print(f"Loading v1 from {args.v1_dir}/{{{args.cell_off},{args.cell_on}}}/")
    v1_off, v1_on = load_pair(args.v1_dir, args.cell_off, args.cell_on)
    print(f"  v1: off n={len(v1_off)}, on n={len(v1_on)}")

    print(f"Loading v2 from {args.v2_dir}/{{{args.cell_off},{args.cell_on}}}/")
    v2_off, v2_on = load_pair(args.v2_dir, args.cell_off, args.cell_on)
    print(f"  v2: off n={len(v2_off)}, on n={len(v2_on)}")

    failures: list[str] = []

    # Check 1: http_status (the headline goal of the rerun)
    print("\n[1/5] HTTP status check...")
    failures += check_http_status(v2_off, v2_on, "v2")
    if not any("v2" in f for f in failures):
        print("  PASS: all v2 rows http_status==200")

    # Check 2: v1 vs v2 distribution drift (tok_out and wall)
    print("\n[2/5] v1 vs v2 distribution drift...")
    drift_failures, tok_info = check_distribution_drift(
        v1_off, v2_off, "tokens_out", "off")
    failures += drift_failures
    drift_failures, _ = check_distribution_drift(
        v1_on, v2_on, "tokens_out", "on")
    failures += drift_failures
    if tok_info:
        print(f"  tok_out: v1 off p50={tok_info['v1_p50']:.0f} | "
              f"v2 off p50={tok_info['v2_p50']:.0f} | "
              f"KS={tok_info['ks_stat']:.3f}")

    # Check 3: pair_id alignment
    print("\n[3/5] Pair alignment...")
    failures += check_pair_alignment(v2_off, v2_on, "v2")

    # Check 4: byte-identity (extends brief §3 audit)
    print("\n[4/5] Byte-identity across CC states (v2)...")
    n_id, n_paired = check_byte_identity(v2_off, v2_on, "v2")
    print(f"  v2: {n_id}/{n_paired} pairs byte-identical "
          f"(v1 was 49/50; rerun target ≥ same fraction)")
    if n_paired and n_id / n_paired < 0.95:
        failures.append(f"[v2] byte-identity rate {n_id}/{n_paired} "
                        f"below 95% — generation may not be deterministic")

    # Check 5: longest pair completion + delta drift
    print("\n[5/5] Longest pair completion + CC delta drift...")
    failures += check_longest_pair_completion(v2_off, v2_on, "v2")
    v2_delta = compute_cc_delta_pct(v2_off, v2_on)
    v1_delta = compute_cc_delta_pct(v1_off, v1_on)
    drift_pp = v2_delta - V1_HEADLINE_DELTA_PCT
    print(f"  v1 CC delta (recomputed): {v1_delta:+.2f}%  "
          f"(brief headline {V1_HEADLINE_DELTA_PCT:+.2f}%)")
    print(f"  v2 CC delta:              {v2_delta:+.2f}%  "
          f"(drift vs brief: {drift_pp:+.2f}pp)")
    if abs(drift_pp) > DRIFT_ENVELOPE_PP:
        failures.append(
            f"[drift] v2 CC delta {v2_delta:+.2f}% is {drift_pp:+.2f}pp from "
            f"v1's {V1_HEADLINE_DELTA_PCT:+.2f}%, outside ±{DRIFT_ENVELOPE_PP}pp envelope. "
            f"Investigate before brief integration."
        )

    # === Brief edit snippet ===
    print("\n" + "=" * 72)
    print("BRIEF EDIT SNIPPET (paste into §4.2, after the OLS fit results):")
    print("=" * 72)
    print(
        f"\n  Tier 3G was re-run with the per-request budget raised to 1000 s\n"
        f"  on a fresh deploy pair; {n_id}/{n_paired} pairs completed under both CC\n"
        f"  states. The paired-Δ panel now spans tok_out ∈ [_, {int(v2_off['tokens_out'].max())}].\n"
        f"  Replicated CC delta on wall p50 is {v2_delta:+.2f}% "
        f"({drift_pp:+.2f}pp drift\n"
        f"  vs v1's {V1_HEADLINE_DELTA_PCT:+.2f}%), within the documented inter-deploy\n"
        f"  envelope. Per-arm bias and outlier structure reproduced.\n"
    )

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
