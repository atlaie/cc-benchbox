#!/usr/bin/env python3
"""
analyze_pysyft_overhead.py - per-stage decomposition of PySyft overhead.

Inputs:
  --pysyft-parquet  runs/pysyft/<cell>/requests.parquet   (this driver's output)
  --egress-parquet  runs/phase3/<cell>/requests.parquet   (phase3_egress_driver_v2 output)
                    OR a combined egress parquet carrying a `condition` column.

The two parquets MUST be from runs at the same cc-state, same model, same
prompt set, with the pair_id / prompt_class columns aligned. The egress
run plays the role of "PySyft baseline" — same encoder, same CC condition,
no PySyft layer. Per-pair_id deltas isolate the PySyft layer's overhead.

PER-ENDPOINT COMPARISON (v2):
    The PySyft driver hits all four endpoints per pair. Pooling them into one
    median before differencing against a single egress condition conflates
    endpoints with very different payload weights (routing's 75-layer payload
    vs a steering call). Instead we compare each PySyft endpoint against the
    egress `condition` that exercises the SAME encoder workload:

        capture_residual_stream  -> repe_bundle
        capture_attention_stats  -> repe_bundle
        capture_routing          -> routing
        apply_steering           -> steer

    The egress parquet may be a single-condition file (no `condition` column;
    we then trust --egress-condition or fall back to comparing every PySyft
    endpoint against that one file) or a combined file with a `condition`
    column (we slice per endpoint). Unmatched endpoints are reported, not
    silently dropped.

Output:
  - <out-dir>/pysyft_overhead_table.csv
  - <out-dir>/pysyft_overhead_table.md     (drop-in for the brief's §4.3)
  - <out-dir>/figures/stage_decomposition.{png,pdf}

The within-PySyft table matches the brief's Table 9 row schema:

  Stage                              p50 (ms)   95% CI       % of total
  workflow dispatch + auditor lookup ...
  engagement cap + session start     ...
  encoder call (loopback)            ...
  engagement-ledger insert           ...
  response assembly                  ...
  TOTAL PySyft layer                 ...
  --- (separate, laptop-measured) ---
  serialise + transport              ...

Bootstrap CIs use the same paired-BCa machinery as analyze_cc_deltas — same
n_resamples default (10k), same alpha (0.05). Imports the helper from
analyze_cc_deltas to keep the methodology aligned without duplicating code.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reuse the brief's BCa paired bootstrap. analyze_cc_deltas.py is at the
# cc-benchbox root so this works when both scripts are in the same dir.
from analyze_cc_deltas import bootstrap_paired_delta

# Endpoint -> egress condition. Mirrors phase3_pysyft_driver.ENDPOINT_TO_EGRESS_CONDITION;
# duplicated here so the analysis script has no import dependency on the driver
# (which imports syft). Keep the two in sync.
ENDPOINT_TO_EGRESS_CONDITION = {
    "capture_residual_stream":  "repe_bundle",
    "capture_attention_stats":  "repe_bundle",
    "capture_routing":          "routing",
    "apply_steering":           "steer",
}


# Stage column → display name. Order = top-down decomposition. These are the
# server-measured PySyft governance stages; their sum is pysyft_total.
STAGES = [
    ("pysyft_workflow_seconds",          "workflow dispatch + auditor lookup"),
    ("pysyft_approval_seconds",          "engagement cap + session start"),
    ("pysyft_encoder_seconds",           "encoder call (loopback to /v1/egress_eval)"),
    ("pysyft_ledger_seconds",            "engagement-ledger insert"),
    ("pysyft_response_assembly_seconds", "response assembly (build return dict)"),
]

# Backward-compat: a parquet produced by the pre-patch driver carries
# `pysyft_bundle_return_seconds` instead of `pysyft_response_assembly_seconds`.
_LEGACY_STAGE_RENAME = {
    "pysyft_bundle_return_seconds": "pysyft_response_assembly_seconds",
}


def _pct(xs: np.ndarray, q: float) -> float:
    return float(np.percentile(xs, q))


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename legacy columns so downstream code sees the v2 schema."""
    present = {old: new for old, new in _LEGACY_STAGE_RENAME.items()
               if old in df.columns and new not in df.columns}
    if present:
        df = df.rename(columns=present)
    return df


def stage_table(
    pysyft_df: pd.DataFrame,
    egress_df: pd.DataFrame | None,
    n_resamples: int,
    alpha: float,
    egress_condition: str | None = None,
) -> pd.DataFrame:
    """Build the per-stage decomposition table plus per-endpoint PySyft-vs-egress
    deltas.

    Within-PySyft: distribution of each stage's wall, pooled across endpoints
    (the governance stages are payload-independent, so pooling is fair here).
    Per-endpoint total: a separate block of paired-BCa deltas, one row per
    PySyft endpoint vs its matched egress condition.
    """
    pysyft_df = _normalize_columns(pysyft_df)
    ok = pysyft_df[pysyft_df["error"].isna() | (pysyft_df["error"] == "")]
    if ok.empty:
        raise ValueError("no successful pysyft rows")

    pysyft_total = ok["pysyft_total_seconds"].to_numpy()
    rows: list[dict] = []
    for col, label in STAGES:
        if col not in ok.columns:
            continue
        v = ok[col].to_numpy()
        rows.append({
            "block": "pysyft_stage",
            "stage": label,
            "p50_ms": _pct(v, 50) * 1000,
            "p95_ms": _pct(v, 95) * 1000,
            "mean_ms": float(v.mean()) * 1000,
            "pct_of_total_p50": 100 * _pct(v, 50) / max(_pct(pysyft_total, 50), 1e-9),
        })
    rows.append({
        "block": "pysyft_stage",
        "stage": "TOTAL PySyft layer (server-measured)",
        "p50_ms": _pct(pysyft_total, 50) * 1000,
        "p95_ms": _pct(pysyft_total, 95) * 1000,
        "mean_ms": float(pysyft_total.mean()) * 1000,
        "pct_of_total_p50": 100.0,
    })

    # Separate, laptop-measured serialise+transport line (NOT a PySyft stage).
    if "transport_serialize_seconds" in ok.columns:
        tx = ok["transport_serialize_seconds"].to_numpy()
        rows.append({
            "block": "transport",
            "stage": "serialise + transport (laptop-measured, post-return)",
            "p50_ms": _pct(tx, 50) * 1000,
            "p95_ms": _pct(tx, 95) * 1000,
            "mean_ms": float(tx.mean()) * 1000,
            "pct_of_total_p50": float("nan"),
        })

    # ----- Per-endpoint PySyft-vs-egress paired delta -----
    if egress_df is not None:
        egress_df = _normalize_columns(egress_df)
        e_ok = egress_df[egress_df["error"].isna() | (egress_df["error"] == "")]
        has_cond_col = "condition" in e_ok.columns

        for ep in sorted(ok["endpoint"].unique()):
            target_cond = ENDPOINT_TO_EGRESS_CONDITION.get(ep)
            # Resolve the egress slice for this endpoint.
            if has_cond_col and target_cond is not None:
                e_slice = e_ok[e_ok["condition"] == target_cond]
            elif egress_condition is not None and target_cond is not None \
                    and egress_condition != target_cond:
                # Single-condition egress file that doesn't match this endpoint:
                # skip with a recorded note rather than a misleading comparison.
                rows.append({
                    "block": "delta",
                    "stage": f"Δ wall_p50  {ep} − egress[{target_cond}]",
                    "p50_ms": float("nan"), "p95_ms": float("nan"),
                    "mean_ms": float("nan"), "pct_of_total_p50": float("nan"),
                    "note": f"egress file is condition={egress_condition!r}, "
                            f"endpoint needs {target_cond!r}; skipped",
                })
                continue
            else:
                # Single-condition file we trust matches (or no mapping known).
                e_slice = e_ok

            if e_slice.empty:
                rows.append({
                    "block": "delta",
                    "stage": f"Δ wall_p50  {ep} − egress[{target_cond}]",
                    "p50_ms": float("nan"), "p95_ms": float("nan"),
                    "mean_ms": float("nan"), "pct_of_total_p50": float("nan"),
                    "note": f"no egress rows for condition {target_cond!r}",
                })
                continue

            pys_grouped = (
                ok[ok["endpoint"] == ep]
                .groupby(["pair_id", "prompt_class"])["wall_seconds"]
                .median().reset_index().rename(columns={"wall_seconds": "pys_wall"})
            )
            eg_grouped = (
                e_slice.groupby(["pair_id", "prompt_class"])["wall_seconds"]
                .median().reset_index().rename(columns={"wall_seconds": "eg_wall"})
            )
            merged = pys_grouped.merge(eg_grouped, on=["pair_id", "prompt_class"],
                                       how="inner")
            n_unmatched = max(len(pys_grouped), len(eg_grouped)) - len(merged)
            if len(merged) >= 2:
                pys_arr = merged["pys_wall"].to_numpy()
                eg_arr = merged["eg_wall"].to_numpy()
                point, lo, hi = bootstrap_paired_delta(
                    eg_arr, pys_arr,
                    statistic=lambda x: float(np.median(x)),
                    n_resamples=n_resamples, alpha=alpha,
                )
                note = ""
                if n_unmatched:
                    note = f"{n_unmatched} unmatched pair(s) dropped"
                rows.append({
                    "block": "delta",
                    "stage": f"Δ wall_p50  {ep} − egress[{target_cond}]",
                    "p50_ms": point * 1000,
                    "p95_ms": float("nan"),
                    "mean_ms": float("nan"),
                    "pct_of_total_p50": float("nan"),
                    "ci_low_ms": lo * 1000,
                    "ci_high_ms": hi * 1000,
                    "n_paired": int(len(merged)),
                    "note": note,
                })
            else:
                rows.append({
                    "block": "delta",
                    "stage": f"Δ wall_p50  {ep} − egress[{target_cond}]",
                    "p50_ms": float("nan"), "p95_ms": float("nan"),
                    "mean_ms": float("nan"), "pct_of_total_p50": float("nan"),
                    "note": f"only {len(merged)} matched pair(s); need >=2",
                })
    return pd.DataFrame(rows)


def write_markdown(table: pd.DataFrame, path: Path) -> None:
    lines = ["# PySyft overhead — stage decomposition (Table 9)\n"]
    disp = table.copy()
    for col in ("p50_ms", "p95_ms", "mean_ms", "ci_low_ms", "ci_high_ms"):
        if col in disp.columns:
            disp[col] = disp[col].round(2)
    if "pct_of_total_p50" in disp.columns:
        disp["pct_of_total_p50"] = disp["pct_of_total_p50"].round(1)
    # Drop the internal `block` column from the rendered table.
    if "block" in disp.columns:
        disp = disp.drop(columns=["block"])
    lines.append(disp.to_markdown(index=False))
    path.write_text("\n".join(lines))


def figure_stage_decomposition(
    pysyft_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    pysyft_df = _normalize_columns(pysyft_df)
    ok = pysyft_df[pysyft_df["error"].isna() | (pysyft_df["error"] == "")]
    if ok.empty:
        return
    stage_cols = [(c, l) for c, l in STAGES if c in ok.columns]
    medians_ms = [float(np.median(ok[col].to_numpy())) * 1000 for col, _ in stage_cols]
    labels = [lbl for _, lbl in stage_cols]

    fig, ax = plt.subplots(figsize=(8, 4.0))
    y = np.arange(len(labels))[::-1]
    bars = ax.barh(y, medians_ms, color="#47A5AD", edgecolor="white", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("p50 wall (ms)")
    ax.set_title("PySyft overhead — per-stage decomposition (median across requests)")
    ax.grid(True, axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Inline value labels.
    for b, v in zip(bars, medians_ms):
        ax.text(b.get_width() + max(medians_ms) * 0.01,
                b.get_y() + b.get_height() / 2,
                f"{v:.1f}", va="center", fontsize=9)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(fig_dir / "stage_decomposition.png", dpi=150, bbox_inches="tight")
    fig.savefig(fig_dir / "stage_decomposition.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--pysyft-parquet", type=Path, required=True)
    ap.add_argument("--egress-parquet", type=Path, default=None,
                    help="Optional. Paired baseline for PySyft - egress delta. "
                         "May be single-condition or a combined parquet with a "
                         "`condition` column.")
    ap.add_argument("--egress-condition", type=str, default=None,
                    help="Condition name of a single-condition egress parquet "
                         "(repe_bundle | routing | steer | baseline). Used to "
                         "match endpoints when the parquet has no `condition` "
                         "column.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-resamples", type=int, default=10_000)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    pysyft_df = pd.read_parquet(args.pysyft_parquet)
    egress_df = pd.read_parquet(args.egress_parquet) if args.egress_parquet else None

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table = stage_table(pysyft_df, egress_df,
                        n_resamples=args.n_resamples, alpha=args.alpha,
                        egress_condition=args.egress_condition)
    table.to_csv(args.out_dir / "pysyft_overhead_table.csv", index=False)
    write_markdown(table, args.out_dir / "pysyft_overhead_table.md")
    figure_stage_decomposition(pysyft_df, args.out_dir)
    print(table.to_string(index=False))
    print(f"\n[out] {args.out_dir / 'pysyft_overhead_table.csv'}")
    print(f"[out] {args.out_dir / 'pysyft_overhead_table.md'}")
    print(f"[out] {args.out_dir / 'figures/stage_decomposition.{{png,pdf}}'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())