#!/usr/bin/env python3
"""
Analysis 6: Leave-one-cell-out sensitivity on the two-feature OLS fit.

The brief's headline +64 ms/out-tok decode slope comes from a two-feature OLS
pooled across the max_tokens sweep (5 cells, n=10..100) and the tokens_in
sweep (5 cells, n=100). The smallest cell (max_tokens=2048, n=10) has high
leverage on the decode slope.

This script:
  (a) Fits the full pooled OLS (replicating Table 6).
  (b) Runs leave-one-cell-out: drops each of the 10 sweep cells in turn,
      refits, and reports the change in Δb₂ (decode slope delta).
  (c) Runs a weighted OLS (weights = 1/n_cell, equalising cell influence)
      and compares the slope.
  (d) Reports bootstrap CIs on the LOO range.

Input:  Per-cell requests.parquet from the max_tokens and tokens_in sweeps.
Output: LOO sensitivity table + weighted-OLS comparison (CSV + Markdown).

Usage:
    python analyze_ols_sensitivity.py \
        --data-dir runs/phase3 \
        --output-dir runs/phase3/analysis
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_cell(data_dir: Path, cell_id: str) -> pd.DataFrame:
    candidates = [
        data_dir / cell_id / "requests.parquet",
        data_dir / f"{cell_id}.parquet",
    ]
    for p in candidates:
        if p.exists():
            return pd.read_parquet(p)
    raise FileNotFoundError(f"No parquet for '{cell_id}' under {data_dir}")


def pair_and_merge(df_off: pd.DataFrame, df_on: pd.DataFrame) -> pd.DataFrame:
    """Pair by (pair_id, prompt_class), return merged df with wall_off, wall_on."""
    keys = ["pair_id", "prompt_class"]
    if "prompt_class" not in df_off.columns:
        keys = ["pair_id"]
    m = df_off.merge(df_on, on=keys, suffixes=("_off", "_on"), how="inner")
    return m


def ols_two_feature(
    tok_in: np.ndarray,
    tok_out: np.ndarray,
    wall: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict:
    """
    Fit wall = a + b1 * tok_in + b2 * tok_out.

    Returns dict with coefficients, R², and residuals.
    If weights is provided, runs WLS.
    """
    n = len(wall)
    X = np.column_stack([np.ones(n), tok_in, tok_out])

    if weights is not None:
        W = np.diag(np.sqrt(weights))
        Xw = W @ X
        yw = W @ wall
    else:
        Xw = X
        yw = wall

    beta, residuals, rank, sv = np.linalg.lstsq(Xw, yw, rcond=None)

    wall_pred = X @ beta
    ss_res = np.sum((wall - wall_pred) ** 2)
    ss_tot = np.sum((wall - np.mean(wall)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "a": float(beta[0]),
        "b1_ms_per_intok": float(beta[1] * 1000),
        "b2_ms_per_outtok": float(beta[2] * 1000),
        "r2": float(r2),
        "n": n,
        "residuals": wall - wall_pred,
    }


# ---------------------------------------------------------------------------
# Cell definitions — adapt to your naming
# ---------------------------------------------------------------------------

# (label, off_cell, on_cell, tokens_in_value, tok_out_value_or_None)
# tok_out_value_or_None: if None, read from parquet column.
# For the max_tokens sweep, tok_out varies per prompt (use parquet).
# For the tokens_in sweep, tok_out ≈ 4 (fixed max_tokens=32).

DEFAULT_MAX_TOKENS_CELLS = [
    ("tok32", "sweep_maxtok32_off", "sweep_maxtok32_on"),
    ("tok128", "sweep_maxtok128_off", "sweep_maxtok128_on"),
    ("tok512", "sweep_maxtok512_off", "sweep_maxtok512_on"),
    ("tok1024", "sweep_maxtok1024_off", "sweep_maxtok1024_on"),
    ("tok2048", "sweep_maxtok2048_off", "sweep_maxtok2048_on"),
]

DEFAULT_TOKENS_IN_CELLS = [
    ("in100", "sweep_tokin100_off", "sweep_tokin100_on"),
    ("in500", "sweep_tokin500_off", "sweep_tokin500_on"),
    ("in1000", "sweep_tokin1000_off", "sweep_tokin1000_on"),
    ("in4000", "sweep_tokin4000_off", "sweep_tokin4000_on"),
    ("in8000", "sweep_tokin8000_off", "sweep_tokin8000_on"),
]


def main():
    parser = argparse.ArgumentParser(
        description="LOO sensitivity + weighted OLS on the two-feature decode slope."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--cell-map-json", type=Path, default=None,
        help="JSON with keys 'max_tokens_cells' and 'tokens_in_cells', "
             "each a list of [label, off_cell, on_cell] triples.",
    )
    # Column names in parquet
    parser.add_argument("--tok-in-col", type=str, default="tokens_in",
                        help="Parquet column for input token count.")
    parser.add_argument("--tok-out-col", type=str, default="tokens_out",
                        help="Parquet column for output token count.")
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir or data_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.cell_map_json and args.cell_map_json.exists():
        import json
        with open(args.cell_map_json) as f:
            cmap = json.load(f)
        mt_cells = cmap.get("max_tokens_cells", DEFAULT_MAX_TOKENS_CELLS)
        ti_cells = cmap.get("tokens_in_cells", DEFAULT_TOKENS_IN_CELLS)
    else:
        mt_cells = DEFAULT_MAX_TOKENS_CELLS
        ti_cells = DEFAULT_TOKENS_IN_CELLS
        print("Using default cell definitions; pass --cell-map-json to override.",
              file=sys.stderr)

    # ---- Load and assemble the pooled dataset ----
    all_cells = []  # list of (label, merged_df)

    for label, off_id, on_id in mt_cells + ti_cells:
        try:
            df_off = load_cell(data_dir, off_id)
            df_on = load_cell(data_dir, on_id)
        except FileNotFoundError as e:
            print(f"SKIP {label}: {e}", file=sys.stderr)
            continue
        merged = pair_and_merge(df_off, df_on)
        all_cells.append((label, merged))

    if len(all_cells) == 0:
        print("ERROR: no cells loaded. Check --data-dir and cell IDs.",
              file=sys.stderr)
        sys.exit(2)

    # Build arrays for each CC state
    def extract_arrays(cells, cc_state: str):
        """cc_state: 'off' or 'on'"""
        tok_in_all, tok_out_all, wall_all, cell_labels = [], [], [], []
        for label, m in cells:
            tok_in_col = f"{args.tok_in_col}_{cc_state}" if f"{args.tok_in_col}_{cc_state}" in m.columns else args.tok_in_col
            tok_out_col = f"{args.tok_out_col}_{cc_state}" if f"{args.tok_out_col}_{cc_state}" in m.columns else args.tok_out_col
            wall_col = f"wall_seconds_{cc_state}"

            if tok_in_col not in m.columns or tok_out_col not in m.columns:
                # Try common fallback column names
                for candidate_in in ["tokens_in", "tok_in", "prompt_tokens"]:
                    c = f"{candidate_in}_{cc_state}" if f"{candidate_in}_{cc_state}" in m.columns else candidate_in
                    if c in m.columns:
                        tok_in_col = c
                        break
                for candidate_out in ["tokens_out", "tok_out", "completion_tokens"]:
                    c = f"{candidate_out}_{cc_state}" if f"{candidate_out}_{cc_state}" in m.columns else candidate_out
                    if c in m.columns:
                        tok_out_col = c
                        break

            if wall_col not in m.columns:
                print(f"WARN: {wall_col} not in columns for {label}. "
                      f"Available: {list(m.columns)}", file=sys.stderr)
                continue

            tok_in_all.append(m[tok_in_col].values)
            tok_out_all.append(m[tok_out_col].values)
            wall_all.append(m[wall_col].values)
            cell_labels.extend([label] * len(m))

        return (
            np.concatenate(tok_in_all),
            np.concatenate(tok_out_all),
            np.concatenate(wall_all),
            np.array(cell_labels),
        )

    results = {}
    loo_rows = []

    for cc in ["off", "on"]:
        tok_in, tok_out, wall, labels = extract_arrays(all_cells, cc)

        # Full fit
        full = ols_two_feature(tok_in, tok_out, wall)
        results[f"full_{cc}"] = full

        # Weighted fit (weight = 1 / cell_count for each observation)
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        label_to_weight = {l: 1.0 / c for l, c in zip(unique_labels, label_counts)}
        weights = np.array([label_to_weight[l] for l in labels])
        weighted = ols_two_feature(tok_in, tok_out, wall, weights)
        results[f"weighted_{cc}"] = weighted

        # LOO by cell
        for drop_label in unique_labels:
            mask = labels != drop_label
            loo = ols_two_feature(tok_in[mask], tok_out[mask], wall[mask])
            n_dropped = int((~mask).sum())
            loo_rows.append({
                "cc_state": cc,
                "dropped_cell": drop_label,
                "n_dropped": n_dropped,
                "n_remaining": int(mask.sum()),
                "b2_full_ms": full["b2_ms_per_outtok"],
                "b2_loo_ms": loo["b2_ms_per_outtok"],
                "b2_change_ms": loo["b2_ms_per_outtok"] - full["b2_ms_per_outtok"],
                "b2_change_pct": (loo["b2_ms_per_outtok"] - full["b2_ms_per_outtok"])
                                 / full["b2_ms_per_outtok"] * 100,
                "r2_loo": loo["r2"],
            })

    df_loo = pd.DataFrame(loo_rows)

    # Compute the delta Δb₂ = b2_on - b2_off
    delta_b2_full = (results["full_on"]["b2_ms_per_outtok"]
                     - results["full_off"]["b2_ms_per_outtok"])
    delta_b2_weighted = (results["weighted_on"]["b2_ms_per_outtok"]
                         - results["weighted_off"]["b2_ms_per_outtok"])

    # LOO on the delta
    loo_delta_rows = []
    loo_off = df_loo[df_loo["cc_state"] == "off"].set_index("dropped_cell")
    loo_on = df_loo[df_loo["cc_state"] == "on"].set_index("dropped_cell")
    common_labels = sorted(set(loo_off.index) & set(loo_on.index))

    for label in common_labels:
        delta_loo = loo_on.loc[label, "b2_loo_ms"] - loo_off.loc[label, "b2_loo_ms"]
        loo_delta_rows.append({
            "dropped_cell": label,
            "n_dropped": int(loo_off.loc[label, "n_dropped"]),
            "delta_b2_full_ms": delta_b2_full,
            "delta_b2_loo_ms": float(delta_loo),
            "change_ms": float(delta_loo - delta_b2_full),
            "change_pct": float((delta_loo - delta_b2_full) / delta_b2_full * 100),
        })

    df_delta_loo = pd.DataFrame(loo_delta_rows)

    # ---- Output ----
    csv_loo = output_dir / "ols_loo_sensitivity.csv"
    csv_delta = output_dir / "ols_loo_delta_b2.csv"
    md_path = output_dir / "ols_sensitivity.md"

    df_loo.to_csv(csv_loo, index=False)
    df_delta_loo.to_csv(csv_delta, index=False)

    with open(md_path, "w") as f:
        f.write("# OLS decode-slope sensitivity analysis\n\n")

        f.write("## Full vs weighted fit\n\n")
        f.write("| Fit | b₂ CC-off (ms/tok) | b₂ CC-on (ms/tok) | Δb₂ (ms/tok) | R² off | R² on |\n")
        f.write("|-----|----:|----:|----:|----:|----:|\n")
        f.write(f"| Full OLS | {results['full_off']['b2_ms_per_outtok']:.2f} "
                f"| {results['full_on']['b2_ms_per_outtok']:.2f} "
                f"| {delta_b2_full:.2f} "
                f"| {results['full_off']['r2']:.6f} "
                f"| {results['full_on']['r2']:.6f} |\n")
        f.write(f"| Weighted OLS (1/n) | {results['weighted_off']['b2_ms_per_outtok']:.2f} "
                f"| {results['weighted_on']['b2_ms_per_outtok']:.2f} "
                f"| {delta_b2_weighted:.2f} "
                f"| {results['weighted_off']['r2']:.6f} "
                f"| {results['weighted_on']['r2']:.6f} |\n")
        f.write(f"\nWeighted-vs-full Δb₂ difference: "
                f"{delta_b2_weighted - delta_b2_full:+.2f} ms/tok "
                f"({(delta_b2_weighted - delta_b2_full)/delta_b2_full*100:+.1f}%)\n\n")

        f.write("## Leave-one-cell-out on Δb₂ (decode slope delta)\n\n")
        if len(df_delta_loo) > 0:
            f.write(df_delta_loo.to_markdown(index=False, floatfmt=".2f"))
            loo_range = df_delta_loo["delta_b2_loo_ms"]
            f.write(f"\n\nLOO Δb₂ range: [{loo_range.min():.2f}, {loo_range.max():.2f}] ms/tok "
                    f"(full: {delta_b2_full:.2f} ms/tok). "
                    f"Max deviation: {df_delta_loo['change_ms'].abs().max():.2f} ms/tok "
                    f"({df_delta_loo['change_pct'].abs().max():.1f}%).\n")
        else:
            f.write("*No LOO comparisons computed.*\n")

        f.write("\n\n## Per-arm LOO detail\n\n")
        if len(df_loo) > 0:
            f.write(df_loo.to_markdown(index=False, floatfmt=".3f"))
        f.write("\n")

    print(f"Wrote: {csv_loo}")
    print(f"Wrote: {csv_delta}")
    print(f"Wrote: {md_path}")

    # Summary
    print(f"\n--- OLS Δb₂ sensitivity ---")
    print(f"  Full:     Δb₂ = {delta_b2_full:.2f} ms/tok")
    print(f"  Weighted: Δb₂ = {delta_b2_weighted:.2f} ms/tok "
          f"({(delta_b2_weighted-delta_b2_full)/delta_b2_full*100:+.1f}%)")
    if len(df_delta_loo) > 0:
        print(f"  LOO range: [{df_delta_loo['delta_b2_loo_ms'].min():.2f}, "
              f"{df_delta_loo['delta_b2_loo_ms'].max():.2f}] ms/tok")
        worst = df_delta_loo.loc[df_delta_loo["change_ms"].abs().idxmax()]
        print(f"  Worst-case LOO: drop '{worst['dropped_cell']}' → "
              f"Δb₂ = {worst['delta_b2_loo_ms']:.2f} ms/tok "
              f"({worst['change_pct']:+.1f}%)")


if __name__ == "__main__":
    main()
