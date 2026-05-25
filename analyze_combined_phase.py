#!/usr/bin/env python3
"""
analyze_combined_phase.py — Combined two-feature OLS fit on the original
max_tokens sweep (varying tok_out at fixed prompt distribution, tok_in
∈ [12, 148]) and the new tok_in sweep (varying tok_in ∈ [100, 8000] at
fixed max_new_tokens=32 on RULER single-needle NIAH prompts).

The original sweep alone identifies b2 (decode slope) tightly via output
length variation but b1 (prefill slope) is weakly identified because
tok_in varies only over a ~12× range. The tok_in sweep extends tok_in to
80× and pins b1. Combined, both slopes — and the fixed slice a — are
well-identified, suitable for a Table 7 refresh in the brief.

Usage:
    python analyze_combined_phase.py \\
        --max-tokens-dir runs/phase3 \\
        --max-tokens-prefix C1 \\
        --tok-in-dir runs/phase3/sweep_tok_in \\
        --tok-in-prefix CI \\
        --output-dir runs/phase3/combined_phase \\
        --n-resamples 10000

Outputs in --output-dir:
    combined_phase_fit.json                          — OLS coefs + BCa CIs
    combined_phase_table.md                          — markdown for brief
    figures/combined_phase_decomposition.{png,pdf}   — updated figure
    diagnostic_per_cell.csv                          — per-cell sanity
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Make sibling analysis scripts importable when invoked from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from analyze_phase_decomposition import (
        fit_phase_decomposition,
        default_regimes,
        write_tables,
        figure_phase_decomposition,
    )
    from analyze_max_tokens_sweep import (
        discover_cells as discover_max_tokens_cells,
        load_sweep as load_max_tokens_sweep,
        REQUIRED_COLS,
    )
except ImportError as e:
    sys.exit(
        f"FATAL: cannot import sibling analysis scripts: {e}\n"
        f"Place this file alongside analyze_phase_decomposition.py and "
        f"analyze_max_tokens_sweep.py, or add their dir to PYTHONPATH."
    )


TOK_IN_CELL_RE = re.compile(
    r"^(?P<prefix>[A-Za-z0-9]+)-(?P<cc>off|on)-i(?P<tok_in_target>\d+)$"
)


@dataclass
class TokInCell:
    cell_id: str
    cc: str
    tok_in_target: int
    path: Path


def discover_tok_in_cells(data_dir: Path, prefix: str) -> list[TokInCell]:
    if not data_dir.exists():
        sys.exit(f"FATAL: tok_in dir does not exist: {data_dir}")
    cells: list[TokInCell] = []
    for sub in sorted(data_dir.iterdir()):
        if not sub.is_dir():
            continue
        m = TOK_IN_CELL_RE.match(sub.name)
        if not m or m.group("prefix") != prefix:
            continue
        parquet = sub / "requests.parquet"
        if not parquet.exists():
            warnings.warn(f"{sub.name}: no requests.parquet, skipping")
            continue
        cells.append(
            TokInCell(
                cell_id=sub.name,
                cc=m.group("cc"),
                tok_in_target=int(m.group("tok_in_target")),
                path=sub,
            )
        )
    cells.sort(key=lambda c: (c.cc, c.tok_in_target))
    return cells


def load_tok_in_sweep(
    cells: list[TokInCell], max_new_tokens_fixed: int = 32
) -> pd.DataFrame:
    if not cells:
        sys.exit("FATAL: no tok_in cells discovered")
    frames: list[pd.DataFrame] = []
    for c in cells:
        df = pd.read_parquet(c.path / "requests.parquet")
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            sys.exit(
                f"FATAL: {c.cell_id}: missing required columns "
                f"{sorted(missing)} in requests.parquet"
            )
        df = df.copy()
        df["cell_id"] = c.cell_id
        df["cc"] = c.cc
        df["tok_in_target"] = c.tok_in_target
        df["max_new_tokens"] = max_new_tokens_fixed
        df["sweep"] = "tok_in"
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def write_diagnostic(df: pd.DataFrame, out_path: Path) -> None:
    """Per-cell sanity table — written to CSV and stdout."""
    g = df.groupby(["sweep", "cc", "cell_id"], as_index=False).agg(
        n=("wall_seconds", "size"),
        wall_p50=("wall_seconds", lambda s: float(np.percentile(s, 50))),
        wall_mean=("wall_seconds", "mean"),
        tok_in_p50=("tokens_in", lambda s: float(np.percentile(s, 50))),
        tok_out_p50=("tokens_out", lambda s: float(np.percentile(s, 50))),
        tok_out_max=("tokens_out", "max"),
    )
    g = g.sort_values(["sweep", "cc", "cell_id"]).reset_index(drop=True)
    g.to_csv(out_path, index=False)
    print(f"  wrote {out_path}")
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 160,
        "display.float_format", lambda x: f"{x:.4f}",
    ):
        print(g.to_string(index=False))


def _print_fit(fit) -> None:
    """Defensive print: deltas + CIs are guaranteed; per-arm coefs best-effort."""
    print()
    print(
        f"  n_paired_obs={fit.n_paired_obs}  "
        f"R²_off={fit.r2_off:.4f}  R²_on={fit.r2_on:.4f}"
    )
    for arm in ("off", "on"):
        try:
            a = getattr(fit, f"a_{arm}")
            b1 = getattr(fit, f"b1_{arm}")
            b2 = getattr(fit, f"b2_{arm}")
        except AttributeError:
            continue
        print(
            f"  {arm.upper():3s}: a={a:+.4f} s   "
            f"b1={b1 * 1000:+.4f} ms/in-tok   "
            f"b2={b2 * 1000:+.4f} ms/out-tok"
        )
    print(
        f"  Δa   = {fit.delta_a:+.4f} s        "
        f"CI [{fit.delta_a_ci[0]:+.4f}, {fit.delta_a_ci[1]:+.4f}]"
    )
    print(
        f"  Δb1  = {fit.delta_b1 * 1000:+.4f} ms/in-tok   "
        f"CI [{fit.delta_b1_ci[0] * 1000:+.4f}, "
        f"{fit.delta_b1_ci[1] * 1000:+.4f}]"
    )
    print(
        f"  Δb2  = {fit.delta_b2 * 1000:+.4f} ms/out-tok  "
        f"CI [{fit.delta_b2_ci[0] * 1000:+.4f}, "
        f"{fit.delta_b2_ci[1] * 1000:+.4f}]"
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--max-tokens-dir",
        type=Path,
        required=True,
        help="Directory containing <prefix>-<cc>-t<num>/ cells (existing sweep)",
    )
    ap.add_argument("--max-tokens-prefix", default="C1")
    ap.add_argument(
        "--tok-in-dir",
        type=Path,
        required=True,
        help="Directory containing <prefix>-<cc>-i<num>/ cells (new sweep)",
    )
    ap.add_argument("--tok-in-prefix", default="CI")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--n-resamples", type=int, default=10000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument(
        "--max-new-tokens-tok-in",
        type=int,
        default=32,
        help="max_new_tokens value to record for tok_in cells "
        "(must match the YAML; default 32)",
    )
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. load max_tokens sweep ----
    print(
        f"[1/5] discovering max_tokens cells under "
        f"{args.max_tokens_dir} prefix={args.max_tokens_prefix!r}"
    )
    mt_cells = discover_max_tokens_cells(
        args.max_tokens_dir, args.max_tokens_prefix
    )
    if not mt_cells:
        sys.exit(
            f"FATAL: no max_tokens cells matching "
            f"{args.max_tokens_prefix}-(off|on)-t* under {args.max_tokens_dir}"
        )
    df_mt = load_max_tokens_sweep(mt_cells)
    df_mt["sweep"] = "max_tokens"
    print(
        f"  loaded {len(df_mt)} rows across {len(mt_cells)} cells "
        f"({df_mt['cc'].value_counts().to_dict()})"
    )

    # ---- 2. load tok_in sweep ----
    print(
        f"[2/5] discovering tok_in cells under "
        f"{args.tok_in_dir} prefix={args.tok_in_prefix!r}"
    )
    ti_cells = discover_tok_in_cells(args.tok_in_dir, args.tok_in_prefix)
    if not ti_cells:
        sys.exit(
            f"FATAL: no tok_in cells matching "
            f"{args.tok_in_prefix}-(off|on)-i* under {args.tok_in_dir}"
        )
    df_ti = load_tok_in_sweep(
        ti_cells, max_new_tokens_fixed=args.max_new_tokens_tok_in
    )
    print(
        f"  loaded {len(df_ti)} rows across {len(ti_cells)} cells "
        f"({df_ti['cc'].value_counts().to_dict()})"
    )

    # ---- 3. sanity checks ----
    print("[3/5] sanity checks")

    # 3a — pair_id namespaces must not collide on the bootstrap key
    key_cols = ["pair_id", "prompt_class", "max_new_tokens"]
    mt_keys = set(map(tuple, df_mt[key_cols].values.tolist()))
    ti_keys = set(map(tuple, df_ti[key_cols].values.tolist()))
    collisions = mt_keys & ti_keys
    if collisions:
        sample = sorted(collisions)[:5]
        sys.exit(
            f"FATAL: {len(collisions)} pair_id-key collisions between "
            f"sweeps (sample: {sample}). Re-namespace pair_ids in "
            f"build_ruler_pairs.py (currently 11000+ namespace)."
        )
    print(
        f"  ✓ no key collisions on {key_cols} "
        f"(mt:{len(mt_keys)} keys, ti:{len(ti_keys)} keys)"
    )

    # 3b — prompt_class labels (paired bootstrap intersects on this key)
    pc_mt = df_mt["prompt_class"].value_counts().to_dict()
    pc_ti = df_ti["prompt_class"].value_counts().to_dict()
    print(f"  prompt_class in max_tokens sweep: {pc_mt}")
    print(f"  prompt_class in tok_in sweep:     {pc_ti}")
    if not (set(pc_mt) & set(pc_ti)):
        print(
            "  ✓ prompt_class labels disjoint across sweeps "
            "(combined fit treats them as independent strata)"
        )
    else:
        warnings.warn(
            f"prompt_class labels overlap between sweeps: "
            f"{sorted(set(pc_mt) & set(pc_ti))}. The paired bootstrap will "
            f"still pair correctly because the join key includes pair_id and "
            f"max_new_tokens, but verify this is intentional."
        )

    # 3c — required columns
    needed = list(REQUIRED_COLS) + [
        "cell_id", "cc", "sweep", "max_new_tokens"
    ]
    for col in needed:
        if col not in df_mt.columns:
            sys.exit(f"FATAL: max_tokens sweep missing column {col!r}")
        if col not in df_ti.columns:
            sys.exit(f"FATAL: tok_in sweep missing column {col!r}")
    print(f"  ✓ all required columns present in both sweeps")

    # ---- 4. concat + write per-cell diagnostic ----
    df = pd.concat([df_mt, df_ti], ignore_index=True)
    print(
        f"[4/5] combined: {len(df)} rows, "
        f"tok_in ∈ [{df['tokens_in'].min():.0f}, "
        f"{df['tokens_in'].max():.0f}], "
        f"tok_out ∈ [{df['tokens_out'].min():.0f}, "
        f"{df['tokens_out'].max():.0f}], "
        f"max_new_tokens ∈ "
        f"{sorted(df['max_new_tokens'].unique().tolist())}"
    )
    write_diagnostic(df, args.output_dir / "diagnostic_per_cell.csv")

    # ---- 5. fit + emit ----
    print(
        f"\n[5/5] fitting two-feature OLS with paired bootstrap "
        f"(n_resamples={args.n_resamples}, α={args.alpha})"
    )
    fit = fit_phase_decomposition(
        df, n_resamples=args.n_resamples, alpha=args.alpha
    )
    _print_fit(fit)

    regimes = default_regimes(fit)
    print(f"\n  writing tables and figure to {args.output_dir}")
    write_tables(fit, regimes, args.output_dir)
    figure_phase_decomposition(fit, regimes, args.output_dir)
    print("  done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
