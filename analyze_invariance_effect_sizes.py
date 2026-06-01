#!/usr/bin/env python3
"""
Analysis 1+3: Cohen's d on CC delta per invariance axis + TOST equivalence test.

Computes:
  (a) Cohen's d for the CC delta (on−off) within each condition.
  (b) Cohen's d for the *difference* of CC deltas between invariance-axis
      pairs (e.g., TP=1 vs TP=8), quantifying the invariance claim.
  (c) Two one-sided t-test (TOST) for statistical equivalence on the
      invariance-axis pairs, with a configurable equivalence margin.

Input:  Per-cell requests.parquet files from the Phase 3 matrix.
Output: Markdown table + CSV for brief inclusion.

Usage:
    python analyze_invariance_effect_sizes.py \
        --data-dir runs/phase3 \
        --output-dir runs/phase3/analysis \
        --equiv-margin 3.0

Convention: follows analyze_cc_deltas.py pairing (pair_id, prompt_class).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_cell(data_dir: Path, cell_id: str) -> pd.DataFrame:
    """Load requests.parquet for a given cell_id under data_dir."""
    candidates = [
        data_dir / cell_id / "requests.parquet",
        data_dir / f"{cell_id}.parquet",
    ]
    for p in candidates:
        if p.exists():
            return pd.read_parquet(p)
    raise FileNotFoundError(
        f"No parquet found for cell '{cell_id}' under {data_dir}. "
        f"Tried: {[str(c) for c in candidates]}"
    )


def pair_deltas(df_off: pd.DataFrame, df_on: pd.DataFrame) -> np.ndarray:
    """Pair by (pair_id, prompt_class); return array of on−off wall deltas."""
    merge_keys = ["pair_id", "prompt_class"]
    # Fallback: if prompt_class not present, pair on pair_id only
    if "prompt_class" not in df_off.columns:
        merge_keys = ["pair_id"]

    merged = df_off.merge(
        df_on, on=merge_keys, suffixes=("_off", "_on"), how="inner"
    )
    if len(merged) == 0:
        raise ValueError("No matched pairs found after merge.")
    return (merged["wall_seconds_on"] - merged["wall_seconds_off"]).values


def cohens_d_paired(x: np.ndarray) -> float:
    """Cohen's d for a paired sample: mean(x) / std(x, ddof=1)."""
    return float(np.mean(x) / np.std(x, ddof=1))


def cohens_d_two_sample(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d (pooled) for two independent samples."""
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(
        ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1))
        / (nx + ny - 2)
    )
    if pooled_std == 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled_std)


def tost(
    x: np.ndarray,
    y: np.ndarray,
    margin: float,
    alpha: float = 0.05,
) -> dict:
    """
    Two one-sided t-test (TOST) for equivalence.

    Tests H0: |mean(x) - mean(y)| >= margin
    against H1: |mean(x) - mean(y)| < margin.

    Returns dict with t-stats, p-values, and equivalence conclusion.
    """
    diff = np.mean(x) - np.mean(y)
    n_x, n_y = len(x), len(y)
    se = np.sqrt(np.var(x, ddof=1) / n_x + np.var(y, ddof=1) / n_y)
    df = n_x + n_y - 2  # Welch approximation could be used; simplified here

    # Upper test: H0: diff >= +margin
    t_upper = (diff - margin) / se
    p_upper = stats.t.cdf(t_upper, df)

    # Lower test: H0: diff <= -margin
    t_lower = (diff + margin) / se
    p_lower = 1 - stats.t.cdf(t_lower, df)

    p_tost = max(p_upper, p_lower)

    return {
        "diff_mean": float(diff),
        "se": float(se),
        "margin": margin,
        "t_upper": float(t_upper),
        "p_upper": float(p_upper),
        "t_lower": float(t_lower),
        "p_lower": float(p_lower),
        "p_tost": float(p_tost),
        "equivalent": p_tost < alpha,
        "alpha": alpha,
    }


def relative_delta_pct(df_off: pd.DataFrame, df_on: pd.DataFrame) -> np.ndarray:
    """Paired relative CC delta in percent: (on−off)/off * 100."""
    merge_keys = ["pair_id", "prompt_class"]
    if "prompt_class" not in df_off.columns:
        merge_keys = ["pair_id"]
    merged = df_off.merge(
        df_on, on=merge_keys, suffixes=("_off", "_on"), how="inner"
    )
    off = merged["wall_seconds_off"].values
    on = merged["wall_seconds_on"].values
    return ((on - off) / off) * 100.0


# ---------------------------------------------------------------------------
# Cell definitions — adapt to your directory naming
# ---------------------------------------------------------------------------

# Each invariance axis is a list of (label, off_cell_id, on_cell_id) tuples.
# Modify these to match your actual directory names under --data-dir.
DEFAULT_INVARIANCE_AXES = {
    "Dispatch regime (GLM-MoE baseline)": [
        ("sequential", "C0_baseline_off", "C0_baseline_on"),
        ("streaming", "C0_baseline_streaming_off", "C0_baseline_streaming_on"),
        ("concurrent c=8", "C0_baseline_concurrent_c8_off", "C0_baseline_concurrent_c8_on"),
    ],
    "Architecture (TP=8, sequential)": [
        ("GLM-MoE", "C0_baseline_off", "C0_baseline_on"),
        ("Llama-70B-dense", "C0_llama70b_tp8_off", "C0_llama70b_tp8_on"),
    ],
    "Tensor parallelism (Llama-70B-dense, sequential)": [
        ("TP=8", "C0_llama70b_tp8_off", "C0_llama70b_tp8_on"),
        ("TP=1", "C0_llama70b_tp1_off", "C0_llama70b_tp1_on"),
    ],
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Effect sizes and TOST equivalence for CC-delta invariance axes."
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Root directory containing per-cell subdirs with requests.parquet.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for CSV/MD. Defaults to --data-dir/analysis.",
    )
    parser.add_argument(
        "--equiv-margin", type=float, default=3.0,
        help="TOST equivalence margin in percentage points (default: 3.0 pp).",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Significance level (default: 0.05).",
    )
    parser.add_argument(
        "--cell-map-json", type=Path, default=None,
        help="Optional JSON mapping overriding DEFAULT_INVARIANCE_AXES. "
             "Format: {axis_name: [[label, off_cell, on_cell], ...]}",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir or data_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load cell map
    if args.cell_map_json and args.cell_map_json.exists():
        import json
        with open(args.cell_map_json) as f:
            axes = json.load(f)
    else:
        axes = DEFAULT_INVARIANCE_AXES
        print(
            "Using DEFAULT_INVARIANCE_AXES. If cell IDs don't match your "
            "directory layout, pass --cell-map-json with the correct mapping.",
            file=sys.stderr,
        )

    # ---- Part A: Per-condition Cohen's d on the CC delta ----
    rows_per_condition = []
    axis_deltas = {}  # axis_name -> {label: relative_delta_pct array}

    for axis_name, conditions in axes.items():
        axis_deltas[axis_name] = {}
        for label, off_id, on_id in conditions:
            try:
                df_off = load_cell(data_dir, off_id)
                df_on = load_cell(data_dir, on_id)
            except FileNotFoundError as e:
                print(f"SKIP {label}: {e}", file=sys.stderr)
                continue

            abs_deltas = pair_deltas(df_off, df_on)
            rel_deltas = relative_delta_pct(df_off, df_on)
            axis_deltas[axis_name][label] = rel_deltas

            d_abs = cohens_d_paired(abs_deltas)
            d_rel = cohens_d_paired(rel_deltas)

            rows_per_condition.append({
                "axis": axis_name,
                "condition": label,
                "n_pairs": len(abs_deltas),
                "mean_abs_delta_s": float(np.mean(abs_deltas)),
                "mean_rel_delta_pct": float(np.mean(rel_deltas)),
                "sd_rel_delta_pct": float(np.std(rel_deltas, ddof=1)),
                "cohens_d_abs": d_abs,
                "cohens_d_rel": d_rel,
            })

    df_conditions = pd.DataFrame(rows_per_condition)

    # ---- Part B: Cross-condition Cohen's d and TOST ----
    rows_invariance = []

    for axis_name, conditions in axes.items():
        labels_with_data = [
            l for l in [c[0] for c in conditions]
            if l in axis_deltas.get(axis_name, {})
        ]
        if len(labels_with_data) < 2:
            continue

        # Pairwise comparisons within each axis
        for i in range(len(labels_with_data)):
            for j in range(i + 1, len(labels_with_data)):
                l_a, l_b = labels_with_data[i], labels_with_data[j]
                d_a = axis_deltas[axis_name][l_a]
                d_b = axis_deltas[axis_name][l_b]

                d_cross = cohens_d_two_sample(d_a, d_b)
                tost_result = tost(d_a, d_b, args.equiv_margin, args.alpha)

                rows_invariance.append({
                    "axis": axis_name,
                    "comparison": f"{l_a} vs {l_b}",
                    "mean_A_pct": float(np.mean(d_a)),
                    "mean_B_pct": float(np.mean(d_b)),
                    "diff_pp": tost_result["diff_mean"],
                    "cohens_d": d_cross,
                    "tost_margin_pp": args.equiv_margin,
                    "tost_p": tost_result["p_tost"],
                    "equivalent": tost_result["equivalent"],
                })

    df_invariance = pd.DataFrame(rows_invariance)

    # ---- Output ----
    csv_cond = output_dir / "effect_sizes_per_condition.csv"
    csv_inv = output_dir / "effect_sizes_invariance.csv"
    md_path = output_dir / "effect_sizes.md"

    df_conditions.to_csv(csv_cond, index=False)
    df_invariance.to_csv(csv_inv, index=False)

    with open(md_path, "w") as f:
        f.write("# Effect sizes and TOST equivalence\n\n")
        f.write("## Per-condition Cohen's d on the paired CC delta\n\n")
        f.write(df_conditions.to_markdown(index=False, floatfmt=".3f"))
        f.write("\n\n")
        f.write("## Cross-condition invariance: Cohen's d + TOST\n\n")
        f.write(f"Equivalence margin: ±{args.equiv_margin} pp, α={args.alpha}\n\n")
        if len(df_invariance) > 0:
            f.write(df_invariance.to_markdown(index=False, floatfmt=".3f"))
        else:
            f.write("*No invariance comparisons computed (cell data not found).*\n")
        f.write("\n")

    print(f"Wrote: {csv_cond}")
    print(f"Wrote: {csv_inv}")
    print(f"Wrote: {md_path}")

    # ---- Summary to stdout ----
    if len(df_conditions) > 0:
        print("\n--- Per-condition Cohen's d (relative CC delta) ---")
        for _, r in df_conditions.iterrows():
            print(f"  {r['condition']:30s}  d={r['cohens_d_rel']:+.2f}  "
                  f"Δ={r['mean_rel_delta_pct']:+.1f}% ± {r['sd_rel_delta_pct']:.1f}%")

    if len(df_invariance) > 0:
        print(f"\n--- Invariance TOST (margin=±{args.equiv_margin} pp) ---")
        for _, r in df_invariance.iterrows():
            eq_str = "EQUIVALENT" if r["equivalent"] else "NOT equivalent"
            print(f"  {r['comparison']:35s}  d={r['cohens_d']:+.3f}  "
                  f"diff={r['diff_pp']:+.2f} pp  p_TOST={r['tost_p']:.4f}  {eq_str}")


if __name__ == "__main__":
    main()
