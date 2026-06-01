#!/usr/bin/env python3
"""
Analysis 2: Post-hoc minimum detectable effect (MDE) for paired CC-delta design.

Given the observed within-pair standard deviation of the CC delta at each cell,
computes the minimum difference in relative CC delta (in percentage points)
detectable at the specified power and significance level. This turns
"sub-1 pp differences are treated as noise floor" into a calibrated statement.

Input:  Per-cell requests.parquet files.
Output: MDE table (CSV + Markdown).

Usage:
    python analyze_mde.py \
        --data-dir runs/phase3 \
        --output-dir runs/phase3/analysis \
        --power 0.80 --alpha 0.05
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def load_cell(data_dir: Path, cell_id: str) -> pd.DataFrame:
    candidates = [
        data_dir / cell_id / "requests.parquet",
        data_dir / f"{cell_id}.parquet",
    ]
    for p in candidates:
        if p.exists():
            return pd.read_parquet(p)
    raise FileNotFoundError(f"No parquet for '{cell_id}' under {data_dir}")


def pair_relative_delta(df_off: pd.DataFrame, df_on: pd.DataFrame) -> np.ndarray:
    merge_keys = ["pair_id", "prompt_class"]
    if "prompt_class" not in df_off.columns:
        merge_keys = ["pair_id"]
    m = df_off.merge(df_on, on=merge_keys, suffixes=("_off", "_on"), how="inner")
    off = m["wall_seconds_off"].values
    on = m["wall_seconds_on"].values
    return ((on - off) / off) * 100.0


def mde_paired(sd: float, n: int, alpha: float, power: float) -> float:
    """
    Minimum detectable effect for a paired t-test.

    MDE = (t_{α/2, n-1} + t_{1-β, n-1}) * sd / sqrt(n)

    Returns MDE in the same units as sd.
    """
    df = n - 1
    t_alpha = stats.t.ppf(1 - alpha / 2, df)
    t_power = stats.t.ppf(power, df)
    return (t_alpha + t_power) * sd / np.sqrt(n)


# Default cells — adapt to your naming
DEFAULT_CELLS = [
    ("baseline (GLM-MoE, seq)", "C0_baseline_off", "C0_baseline_on"),
    ("repe_bundle", "C1_repe_bundle_off", "C1_repe_bundle_on"),
    ("routing", "C2_routing_off", "C2_routing_on"),
    ("steering", "C4_steering_off", "C4_steering_on"),
    ("gradient", "C3_gradient_off", "C3_gradient_on"),
    ("dense TP=8", "C0_llama70b_tp8_off", "C0_llama70b_tp8_on"),
    ("dense TP=1", "C0_llama70b_tp1_off", "C0_llama70b_tp1_on"),
]


def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc MDE for paired CC-delta design."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--power", type=float, default=0.80)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--cell-list-json", type=Path, default=None,
        help="JSON list of [label, off_cell, on_cell] triples.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir or data_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.cell_list_json and args.cell_list_json.exists():
        import json
        with open(args.cell_list_json) as f:
            cells = json.load(f)
    else:
        cells = DEFAULT_CELLS
        print("Using DEFAULT_CELLS; pass --cell-list-json to override.",
              file=sys.stderr)

    rows = []
    for label, off_id, on_id in cells:
        try:
            df_off = load_cell(data_dir, off_id)
            df_on = load_cell(data_dir, on_id)
        except FileNotFoundError as e:
            print(f"SKIP {label}: {e}", file=sys.stderr)
            continue

        rel_deltas = pair_relative_delta(df_off, df_on)
        n = len(rel_deltas)
        sd = float(np.std(rel_deltas, ddof=1))
        mean_delta = float(np.mean(rel_deltas))
        mde = mde_paired(sd, n, args.alpha, args.power)

        rows.append({
            "cell": label,
            "n_pairs": n,
            "mean_delta_pct": mean_delta,
            "sd_delta_pct": sd,
            "mde_pp": mde,
            "power": args.power,
            "alpha": args.alpha,
        })

    df = pd.DataFrame(rows)

    csv_path = output_dir / "mde_table.csv"
    md_path = output_dir / "mde_table.md"

    df.to_csv(csv_path, index=False)
    with open(md_path, "w") as f:
        f.write(f"# Minimum detectable effect (MDE) at {args.power:.0%} power, "
                f"α={args.alpha}\n\n")
        f.write("MDE is the smallest difference in relative CC delta (pp) "
                "detectable by a paired t-test at the given power and α.\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".2f"))
        f.write("\n\n")
        f.write("Interpretation: differences smaller than MDE cannot be "
                "distinguished from noise at this sample size. The brief's "
                "\"sub-1 pp treated as noise floor\" is calibrated if MDE ≤ ~1 pp.\n")

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")

    print(f"\n--- MDE at {args.power:.0%} power, α={args.alpha} ---")
    for _, r in df.iterrows():
        print(f"  {r['cell']:30s}  n={r['n_pairs']:4d}  "
              f"sd={r['sd_delta_pct']:.2f} pp  MDE={r['mde_pp']:.2f} pp")


if __name__ == "__main__":
    main()
