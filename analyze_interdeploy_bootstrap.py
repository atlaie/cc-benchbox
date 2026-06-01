#!/usr/bin/env python3
"""
Analysis 4: Inter-deploy variance propagation into CC-delta CIs.

The brief's CIs are within-run only. This script implements a hierarchical
(two-level) bootstrap that resamples deploys at the outer level and prompts
within deploys at the inner level, propagating inter-deploy drift into the
CI for the relative CC delta.

Requires at least 2 deploy pairs for the same condition. The concurrency
sweep has two independent deploy pairs (deploy 1 + deploy 2); the
max_tokens sweep has a primary + replication deploy.

Input:  Per-cell requests.parquet files tagged with a deploy_id column,
        OR separate directories per deploy (deploy1/, deploy2/).
Output: Hierarchical-bootstrap CIs vs within-run CIs (CSV + Markdown).

Usage:
    python analyze_interdeploy_bootstrap.py \
        --data-dir runs/phase3 \
        --output-dir runs/phase3/analysis \
        --n-boot 10000 --alpha 0.05

Convention: BCa is not straightforward for hierarchical designs, so this
uses percentile bootstrap (which is standard for cluster/hierarchical).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def load_cell(data_dir: Path, cell_id: str) -> pd.DataFrame:
    candidates = [
        data_dir / cell_id / "requests.parquet",
        data_dir / f"{cell_id}.parquet",
    ]
    for p in candidates:
        if p.exists():
            return pd.read_parquet(p)
    raise FileNotFoundError(f"No parquet for '{cell_id}' under {data_dir}")


def pair_and_compute_relative_delta(
    df_off: pd.DataFrame, df_on: pd.DataFrame
) -> pd.DataFrame:
    """Pair by (pair_id, prompt_class), return DataFrame with delta_pct column."""
    merge_keys = ["pair_id", "prompt_class"]
    if "prompt_class" not in df_off.columns:
        merge_keys = ["pair_id"]
    m = df_off.merge(df_on, on=merge_keys, suffixes=("_off", "_on"), how="inner")
    m["delta_pct"] = ((m["wall_seconds_on"] - m["wall_seconds_off"]) / m["wall_seconds_off"]) * 100.0
    return m


def within_run_percentile_ci(
    deltas: np.ndarray, n_boot: int, alpha: float, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Standard percentile bootstrap CI on a single sample."""
    n = len(deltas)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = np.mean(deltas[idx])
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(np.mean(deltas)), float(lo), float(hi)


def hierarchical_bootstrap(
    deploy_deltas: list[np.ndarray],
    n_boot: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """
    Two-level percentile bootstrap.

    Outer: resample deploys with replacement.
    Inner: resample prompts within each selected deploy.

    This propagates both inter-deploy and within-deploy variance.
    """
    n_deploys = len(deploy_deltas)
    boot_means = np.empty(n_boot)

    for b in range(n_boot):
        # Outer: resample deploys
        deploy_idx = rng.integers(0, n_deploys, size=n_deploys)
        # Inner: for each selected deploy, resample prompts
        resampled_means = []
        for di in deploy_idx:
            d = deploy_deltas[di]
            inner_idx = rng.integers(0, len(d), size=len(d))
            resampled_means.append(np.mean(d[inner_idx]))
        boot_means[b] = np.mean(resampled_means)

    grand_mean = float(np.mean([np.mean(d) for d in deploy_deltas]))
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return grand_mean, lo, hi


# ---------------------------------------------------------------------------
# Deploy-pair definitions — adapt to your layout
# ---------------------------------------------------------------------------

# Each entry: (label, [(off_cell_deploy_i, on_cell_deploy_i), ...])
# The list has one tuple per deploy pair.
DEFAULT_DEPLOY_PAIRS = {
    "baseline (concurrency sweep, c=8)": [
        ("conc_sweep_d1_c8_off", "conc_sweep_d1_c8_on"),
        ("conc_sweep_d2_c8_off", "conc_sweep_d2_c8_on"),
    ],
    "baseline (concurrency sweep, c=16)": [
        ("conc_sweep_d1_c16_off", "conc_sweep_d1_c16_on"),
        ("conc_sweep_d2_c16_off", "conc_sweep_d2_c16_on"),
    ],
    "baseline (concurrency sweep, c=32)": [
        ("conc_sweep_d1_c32_off", "conc_sweep_d1_c32_on"),
        ("conc_sweep_d2_c32_off", "conc_sweep_d2_c32_on"),
    ],
    "baseline (concurrency sweep, c=64)": [
        ("conc_sweep_d1_c64_off", "conc_sweep_d1_c64_on"),
        ("conc_sweep_d2_c64_off", "conc_sweep_d2_c64_on"),
    ],
}


def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical bootstrap propagating inter-deploy variance."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deploy-map-json", type=Path, default=None,
        help="JSON mapping: {label: [[off1, on1], [off2, on2], ...]}",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir or data_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    if args.deploy_map_json and args.deploy_map_json.exists():
        import json
        with open(args.deploy_map_json) as f:
            deploy_map = json.load(f)
    else:
        deploy_map = DEFAULT_DEPLOY_PAIRS
        print("Using DEFAULT_DEPLOY_PAIRS; pass --deploy-map-json to override.",
              file=sys.stderr)

    rows = []

    for label, pairs in deploy_map.items():
        deploy_deltas = []
        skip = False

        for off_id, on_id in pairs:
            try:
                df_off = load_cell(data_dir, off_id)
                df_on = load_cell(data_dir, on_id)
            except FileNotFoundError as e:
                print(f"SKIP {label} (missing deploy): {e}", file=sys.stderr)
                skip = True
                break
            merged = pair_and_compute_relative_delta(df_off, df_on)
            deploy_deltas.append(merged["delta_pct"].values)

        if skip or len(deploy_deltas) < 2:
            if not skip:
                print(f"SKIP {label}: need ≥2 deploy pairs, got {len(deploy_deltas)}",
                      file=sys.stderr)
            continue

        n_deploys = len(deploy_deltas)
        total_n = sum(len(d) for d in deploy_deltas)

        # Hierarchical CI
        h_mean, h_lo, h_hi = hierarchical_bootstrap(
            deploy_deltas, args.n_boot, args.alpha, rng
        )

        # Within-run CIs (per deploy)
        within_cis = []
        for i, d in enumerate(deploy_deltas):
            w_mean, w_lo, w_hi = within_run_percentile_ci(
                d, args.n_boot, args.alpha, rng
            )
            within_cis.append((w_mean, w_lo, w_hi))

        # Pooled within-run CI (concatenate all deploys, ignore deploy structure)
        pooled_all = np.concatenate(deploy_deltas)
        p_mean, p_lo, p_hi = within_run_percentile_ci(
            pooled_all, args.n_boot, args.alpha, rng
        )

        rows.append({
            "cell": label,
            "n_deploys": n_deploys,
            "total_n": total_n,
            "hierarchical_mean_pct": h_mean,
            "hierarchical_lo_pct": h_lo,
            "hierarchical_hi_pct": h_hi,
            "hierarchical_width_pp": h_hi - h_lo,
            "pooled_within_mean_pct": p_mean,
            "pooled_within_lo_pct": p_lo,
            "pooled_within_hi_pct": p_hi,
            "pooled_within_width_pp": p_hi - p_lo,
            "width_ratio": (h_hi - h_lo) / max(p_hi - p_lo, 1e-9),
        })

        # Per-deploy detail
        for i, (w_mean, w_lo, w_hi) in enumerate(within_cis):
            print(f"  {label} deploy {i+1}: "
                  f"Δ={w_mean:.1f}% [{w_lo:.1f}, {w_hi:.1f}]")

    df = pd.DataFrame(rows)

    csv_path = output_dir / "interdeploy_bootstrap.csv"
    md_path = output_dir / "interdeploy_bootstrap.md"

    df.to_csv(csv_path, index=False)
    with open(md_path, "w") as f:
        f.write("# Inter-deploy variance propagation (hierarchical bootstrap)\n\n")
        f.write(f"Two-level percentile bootstrap: {args.n_boot} resamples, "
                f"α={args.alpha}. Outer level resamples deploys; inner level "
                f"resamples prompts within each selected deploy.\n\n")
        if len(df) > 0:
            f.write(df.to_markdown(index=False, floatfmt=".2f"))
        else:
            f.write("*No cells with ≥2 deploy pairs found.*\n")
        f.write("\n\n")
        f.write("**width_ratio** > 1 means the hierarchical CI is wider than the "
                "pooled within-run CI — the excess width is the inter-deploy "
                "variance component. A ratio near 1 means inter-deploy drift "
                "is negligible relative to within-run prompt variance.\n")

    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {md_path}")

    if len(df) > 0:
        print(f"\n--- Hierarchical vs within-run CI widths ---")
        for _, r in df.iterrows():
            print(f"  {r['cell']:40s}  "
                  f"hierarch=[{r['hierarchical_lo_pct']:.1f}, {r['hierarchical_hi_pct']:.1f}] "
                  f"({r['hierarchical_width_pp']:.1f} pp)  "
                  f"within=[{r['pooled_within_lo_pct']:.1f}, {r['pooled_within_hi_pct']:.1f}] "
                  f"({r['pooled_within_width_pp']:.1f} pp)  "
                  f"ratio={r['width_ratio']:.2f}x")


if __name__ == "__main__":
    main()
