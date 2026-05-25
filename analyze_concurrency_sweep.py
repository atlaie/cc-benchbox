#!/usr/bin/env python3
"""
analyze_concurrency_sweep.py — summarise the Task D concurrency sweep.

Reads per-cell summary.json + requests.parquet from
    runs/phase3/concurrency_sweep/C1-{off,on}-c{8,16,32,64}/
and emits:
    runs/phase3/concurrency_sweep/sweep_table.md   — markdown table
    runs/phase3/concurrency_sweep/sweep_table.csv  — same, machine-readable
    runs/phase3/concurrency_sweep/sweep_figure.{png,pdf}  — Δ% vs c

For paired BCa CIs, use the project's existing analyze_cc_deltas.py
on the same directory; this script reports point estimates per cell and
per-c paired deltas, formatted to match the brief's Table 3b /
Table A3 style.

Usage:
    python analyze_concurrency_sweep.py \\
        --data-dir runs/phase3/concurrency_sweep \\
        --c-values 8,16,32,64
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


def _load_cell(cell_dir: Path) -> dict[str, Any]:
    """Return {summary: dict, walls: ndarray, ttfts: ndarray} or raise."""
    summary_path = cell_dir / "summary.json"
    parquet_path = cell_dir / "requests.parquet"
    if not summary_path.exists():
        raise FileNotFoundError(f"{summary_path} not found")
    summary = json.loads(summary_path.read_text())
    if not parquet_path.exists():
        # Fall back to summary stats only — bootstrap won't be possible.
        return {"summary": summary, "walls": None, "ttfts": None}
    df = pd.read_parquet(parquet_path)
    ok = df[df["error"].isna() | (df["error"] == "")]
    walls = ok["wall_seconds"].to_numpy(dtype=np.float64)
    ttfts = ok["ttft_seconds"].dropna().to_numpy(dtype=np.float64) \
        if "ttft_seconds" in ok.columns else np.array([])
    return {"summary": summary, "walls": walls, "ttfts": ttfts}


def _pct(arr: np.ndarray, p: float) -> Optional[float]:
    if arr.size == 0:
        return None
    return float(np.percentile(arr, p))


def _paired_delta_p50(off: np.ndarray, on: np.ndarray) -> dict[str, Optional[float]]:
    """Unpaired p50 delta. Concurrent dispatch breaks request-order pairing
    (the c=N driver issues requests in send-time-staggered order, and
    completion order is determined by the server's continuous-batching
    scheduler, which differs between CC-off and CC-on arms). Falls back to
    distribution-level p50 delta — comparable to the brief's Table 3b
    treatment of the existing c=8 cell."""
    if off.size == 0 or on.size == 0:
        return {"off_p50": None, "on_p50": None, "delta_abs": None, "delta_pct": None}
    off_p50 = float(np.percentile(off, 50))
    on_p50 = float(np.percentile(on, 50))
    return {
        "off_p50": off_p50,
        "on_p50": on_p50,
        "delta_abs": on_p50 - off_p50,
        "delta_pct": 100.0 * (on_p50 - off_p50) / off_p50,
    }


def _cell_throughput(s: dict[str, Any]) -> Optional[float]:
    return s.get("cell_throughput_req_per_s")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--data-dir", type=Path, required=True,
                    help="Directory containing C1-{off,on}-c<N>/ cells")
    ap.add_argument("--c-values", default="8,16,32,64",
                    help="Comma-separated concurrency levels (default: 8,16,32,64)")
    ap.add_argument("--cell-prefix", default="C1",
                    help="Cell ID prefix; cells named <prefix>-<cc>-c<N> (default: C1)")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Where to write outputs (default: --data-dir)")
    args = ap.parse_args()

    c_values = [int(x) for x in args.c_values.split(",")]
    out_dir = args.output_dir or args.data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    print(f"[1/3] loading cells from {args.data_dir}")
    for c in c_values:
        record: dict[str, Any] = {"concurrency": c}
        for cc in ("off", "on"):
            cell_id = f"{args.cell_prefix}-{cc}-c{c}"
            cell_dir = args.data_dir / cell_id
            try:
                data = _load_cell(cell_dir)
            except FileNotFoundError as e:
                print(f"  WARN: {cell_id}: {e}", file=sys.stderr)
                record[f"{cc}_n_success"] = None
                continue
            s = data["summary"]
            walls = data["walls"]
            ttfts = data["ttfts"]
            n_success = s.get("n_success")
            wall_p50 = _pct(walls, 50) if walls is not None else \
                s.get("overall", {}).get("wall_seconds", {}).get("p50")
            wall_p95 = _pct(walls, 95) if walls is not None else \
                s.get("overall", {}).get("wall_seconds", {}).get("p95")
            ttft_p50 = _pct(ttfts, 50) if ttfts is not None and ttfts.size > 0 else \
                s.get("overall", {}).get("ttft_seconds", {}).get("p50")
            ttft_p95 = _pct(ttfts, 95) if ttfts is not None and ttfts.size > 0 else \
                s.get("overall", {}).get("ttft_seconds", {}).get("p95")
            thr = _cell_throughput(s)
            record[f"{cc}_n_success"] = n_success
            record[f"{cc}_wall_p50"] = wall_p50
            record[f"{cc}_wall_p95"] = wall_p95
            record[f"{cc}_ttft_p50"] = ttft_p50
            record[f"{cc}_ttft_p95"] = ttft_p95
            record[f"{cc}_throughput"] = thr
            record[f"{cc}_walls"] = walls
            record[f"{cc}_ttfts"] = ttfts
            print(f"  {cell_id}: n={n_success}  "
                  f"wall_p50={wall_p50:.3f}s  "
                  f"ttft_p50={ttft_p50:.3f}s  "
                  if (wall_p50 is not None and ttft_p50 is not None)
                  else f"  {cell_id}: incomplete data")
        rows.append(record)

    # ---- per-c paired deltas ----
    print(f"[2/3] computing paired deltas per c")
    for r in rows:
        off_w = r.get("off_walls")
        on_w = r.get("on_walls")
        off_t = r.get("off_ttfts")
        on_t = r.get("on_ttfts")
        if off_w is not None and on_w is not None:
            d = _paired_delta_p50(off_w, on_w)
            r["wall_delta_abs"] = d["delta_abs"]
            r["wall_delta_pct"] = d["delta_pct"]
        if off_t is not None and on_t is not None \
                and off_t.size > 0 and on_t.size > 0:
            d = _paired_delta_p50(off_t, on_t)
            r["ttft_delta_abs"] = d["delta_abs"]
            r["ttft_delta_pct"] = d["delta_pct"]
        if r.get("off_throughput") and r.get("on_throughput"):
            r["throughput_delta_pct"] = 100.0 * (
                r["on_throughput"] - r["off_throughput"]
            ) / r["off_throughput"]

    # Drop the raw arrays before serialising
    table_cols = [
        "concurrency",
        "off_n_success", "on_n_success",
        "off_wall_p50", "on_wall_p50", "wall_delta_abs", "wall_delta_pct",
        "off_ttft_p50", "on_ttft_p50", "ttft_delta_abs", "ttft_delta_pct",
        "off_throughput", "on_throughput", "throughput_delta_pct",
    ]
    table = pd.DataFrame([{k: r.get(k) for k in table_cols} for r in rows])

    # ---- write outputs ----
    print(f"[3/3] writing outputs to {out_dir}")
    csv_path = out_dir / "sweep_table.csv"
    table.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"  wrote {csv_path}")

    md_path = out_dir / "sweep_table.md"
    with md_path.open("w") as f:
        f.write("# Task D — concurrency sweep results\n\n")
        f.write("Per-c, GLM-5.1-FP8 baseline, sequential paired off/on within "
                "a single deploy session. Δ p50 is distribution-level "
                "(unpaired — see note in script). For paired BCa CIs run "
                "`analyze_cc_deltas.py --data-dir runs/phase3/concurrency_sweep`.\n\n")
        f.write("## Table — wall and TTFT p50 by concurrency\n\n")
        f.write("| c | n off / on | wall p50 off (s) | wall p50 on (s) | "
                "Δ wall abs (s) | Δ wall % | TTFT p50 off (s) | "
                "TTFT p50 on (s) | Δ TTFT % |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            def _fmt(v, prec=3, suffix=""):
                if v is None:
                    return "—"
                return f"{v:.{prec}f}{suffix}"
            f.write(
                f"| {r['concurrency']} "
                f"| {r.get('off_n_success', '—')} / {r.get('on_n_success', '—')} "
                f"| {_fmt(r.get('off_wall_p50'))} "
                f"| {_fmt(r.get('on_wall_p50'))} "
                f"| {_fmt(r.get('wall_delta_abs'), prec=3, suffix=' s' if r.get('wall_delta_abs') is not None else '')} "
                f"| {_fmt(r.get('wall_delta_pct'), prec=1, suffix='%' if r.get('wall_delta_pct') is not None else '')} "
                f"| {_fmt(r.get('off_ttft_p50'))} "
                f"| {_fmt(r.get('on_ttft_p50'))} "
                f"| {_fmt(r.get('ttft_delta_pct'), prec=1, suffix='%' if r.get('ttft_delta_pct') is not None else '')} |\n"
            )
        f.write("\n## Table — throughput by concurrency\n\n")
        f.write("| c | thr off (r/s) | thr on (r/s) | Δ thr % |\n")
        f.write("|---|---|---|---|\n")
        for r in rows:
            def _fmt(v, prec=3, suffix=""):
                if v is None:
                    return "—"
                return f"{v:.{prec}f}{suffix}"
            f.write(
                f"| {r['concurrency']} "
                f"| {_fmt(r.get('off_throughput'))} "
                f"| {_fmt(r.get('on_throughput'))} "
                f"| {_fmt(r.get('throughput_delta_pct'), prec=1, suffix='%' if r.get('throughput_delta_pct') is not None else '')} |\n"
            )
    print(f"  wrote {md_path}")

    # ---- figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cs = [r["concurrency"] for r in rows
              if r.get("wall_delta_pct") is not None]
        wall_pcts = [r["wall_delta_pct"] for r in rows
                     if r.get("wall_delta_pct") is not None]
        ttft_pcts = [r.get("ttft_delta_pct") for r in rows
                     if r.get("wall_delta_pct") is not None]
        thr_pcts = [r.get("throughput_delta_pct") for r in rows
                    if r.get("wall_delta_pct") is not None]

        fig, ax = plt.subplots(1, 1, figsize=(7, 4.5), dpi=140)
        ax.plot(cs, wall_pcts, "o-", color="#1f6f8b",
                label="Δ wall p50 (%)", linewidth=2, markersize=8)
        ax.plot(cs, [t for t in ttft_pcts if t is not None],
                "s--", color="#c44536",
                label="Δ TTFT p50 (%)", linewidth=1.5, markersize=7)
        ax.plot(cs, [t for t in thr_pcts if t is not None],
                "^:", color="#7e8c4d",
                label="Δ throughput (%)", linewidth=1.5, markersize=7)
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.axhspan(33, 38, alpha=0.12, color="#1f6f8b",
                   label="+33–38% baseline band (brief)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(cs)
        ax.set_xticklabels([str(c) for c in cs])
        ax.set_xlabel("concurrency (in-flight requests)")
        ax.set_ylabel("CC delta (%)")
        ax.set_title("GLM-5.1-FP8 baseline — CC overhead vs concurrency")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fpath = out_dir / f"sweep_figure.{ext}"
            fig.savefig(fpath)
            print(f"  wrote {fpath}")
        plt.close(fig)
    except ImportError:
        print("  matplotlib not installed; skipping figure")

    # ---- terminal recap ----
    print()
    print("=" * 64)
    print("  CONCURRENCY SWEEP — wall p50 CC delta")
    print("=" * 64)
    for r in rows:
        c = r["concurrency"]
        off = r.get("off_wall_p50")
        on = r.get("on_wall_p50")
        dpct = r.get("wall_delta_pct")
        if off is None or on is None or dpct is None:
            print(f"  c={c:>3}:  incomplete")
            continue
        print(f"  c={c:>3}:  off={off:.3f}s  on={on:.3f}s  Δ={dpct:+.1f}%")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
