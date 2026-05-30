#!/usr/bin/env python3
"""
analyze_concurrency_sweep.py — Task D concurrency sweep figure.

Drop-in update for the brief's Figure 5. Produces a three-panel figure:

  (a) Top-left  — CC overhead % on wall p50 + TTFT p50 vs c, with the
                  second deploy replicate overlaid. Paired-BCa 95% CIs;
                  +33-38% baseline-family band shaded behind. Carries the
                  within-deploy invariance AND the inter-deploy drift
                  story in one panel.

  (b) Top-right — Absolute throughput (r/s) on CC-off and CC-on vs c.
                  Promotes throughput to its own axis with a parallel-
                  efficiency annotation. The "uplift on both arms, no
                  amortisation of the CC tax" story reads visually
                  instead of as a -25% line jammed next to a +35% line.

  (c) Bottom    — Per-pair Δwall distributions per c level. n=500 paired
                  observations per cell, rendered as violins with median
                  annotation; secondary deploy overlaid as open violins.
                  Demonstrates the invariance is structural, not a median
                  artifact, and that inter-deploy drift is a level shift
                  not a distribution-shape change.

Pairing convention matches analyze_cc_deltas: by (pair_id, prompt_class)
across CC-off / CC-on arms within the same deploy. Under continuous
batching the wall-clock arrival order differs between arms, but request
identity is stable, so identity-pairing is valid (and tighter than
distribution-level pairing v1 used).

Usage:
    python analyze_concurrency_sweep.py \\
        --data-dir runs/phase3/concurrency_sweep_n500 \\
        [--secondary-data-dir runs/phase3/concurrency_sweep_n500_v2] \\
        --c-values 8,16,32,64 \\
        --output-dir runs/phase3/concurrency_sweep_n500/analysis
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reuse the project-wide conventions.
from analyze_cc_deltas import (
    PALETTE,
    CC_BAND_PCT,
    LABEL_WALL_P50_S,
    setup_style,
    bootstrap_paired_delta,
    _save,
    _quantile,
)

setup_style()


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def _load_cell(cell_dir: Path) -> dict[str, Any]:
    """Return {summary, df, walls, ttfts}. `df` is the success-filtered
    requests frame (kept for paired alignment); `walls`/`ttfts` are the
    unpaired arrays for distribution-level fallback statistics."""
    summary_path = cell_dir / "summary.json"
    parquet_path = cell_dir / "requests.parquet"
    if not summary_path.exists():
        raise FileNotFoundError(f"{summary_path} not found")
    summary = json.loads(summary_path.read_text())
    if not parquet_path.exists():
        return {"summary": summary, "df": None, "walls": None, "ttfts": None}
    df = pd.read_parquet(parquet_path)
    if "error" in df.columns:
        ok = df[df["error"].isna() | (df["error"] == "")]
    else:
        ok = df
    walls = ok["wall_seconds"].to_numpy(dtype=np.float64)
    ttfts = (ok["ttft_seconds"].dropna().to_numpy(dtype=np.float64)
             if "ttft_seconds" in ok.columns else np.array([]))
    return {"summary": summary, "df": ok, "walls": walls, "ttfts": ttfts}


def _paired_arrays(off_df: pd.DataFrame | None, on_df: pd.DataFrame | None,
                   col: str) -> tuple[np.ndarray, np.ndarray]:
    """Align off/on by (pair_id, prompt_class). Continuous batching changes
    wall-clock arrival order across arms, but the request identity is
    stable, so identity-pairing is valid and tighter than distribution-
    level pairing. Returns (off_arr, on_arr) of equal length."""
    if off_df is None or on_df is None:
        return np.array([]), np.array([])
    if col not in off_df.columns or col not in on_df.columns:
        return np.array([]), np.array([])
    off_p = off_df.set_index(["pair_id", "prompt_class"])
    on_p = on_df.set_index(["pair_id", "prompt_class"])
    common = sorted(set(off_p.index) & set(on_p.index))
    if len(common) < 2:
        return np.array([]), np.array([])
    off_a = off_p.loc[common, col]
    on_a = on_p.loc[common, col]
    # Drop pairs where either side is NaN.
    paired = pd.concat([off_a.rename("off"), on_a.rename("on")], axis=1).dropna()
    return paired["off"].to_numpy(dtype=np.float64), paired["on"].to_numpy(dtype=np.float64)


def _load_deploy(data_dir: Path, c_values: list[int],
                 cell_prefix: str) -> dict[int, dict]:
    """Load all cells for one deploy into {c: {off: cell, on: cell}}."""
    deploy: dict[int, dict] = {}
    for c in c_values:
        rec: dict[str, Any] = {"off": None, "on": None}
        for cc in ("off", "on"):
            cell_id = f"{cell_prefix}-{cc}-c{c}"
            try:
                rec[cc] = _load_cell(data_dir / cell_id)
            except FileNotFoundError as e:
                print(f"  WARN: {cell_id}: {e}", file=sys.stderr)
        deploy[c] = rec
    return deploy


# ----------------------------------------------------------------------------
# Per-cell stats
# ----------------------------------------------------------------------------

def _compute_row(c: int, cell_off: dict | None, cell_on: dict | None,
                 n_resamples: int, alpha: float) -> dict:
    """Per-c row: paired-BCa wall/TTFT deltas + throughput with bootstrap CIs."""
    row: dict[str, Any] = {"concurrency": c}
    if cell_off is None or cell_on is None:
        return row

    s_off, s_on = cell_off["summary"], cell_on["summary"]
    row["off_throughput"] = s_off.get("cell_throughput_req_per_s")
    row["on_throughput"] = s_on.get("cell_throughput_req_per_s")
    if row["off_throughput"] and row["on_throughput"]:
        row["throughput_delta_pct"] = (
            100 * (row["on_throughput"] - row["off_throughput"])
            / row["off_throughput"]
        )

    df_off, df_on = cell_off["df"], cell_on["df"]

    # Paired wall delta with BCa CI
    off_w, on_w = _paired_arrays(df_off, df_on, "wall_seconds")
    row["n_paired"] = len(off_w)
    if len(off_w) >= 2:
        point, lo, hi = bootstrap_paired_delta(
            off_w, on_w, _quantile(0.5),
            n_resamples=n_resamples, alpha=alpha,
        )
        off_p50 = float(np.median(off_w))
        row["wall_off_p50"] = off_p50
        row["wall_on_p50"] = float(np.median(on_w))
        row["wall_delta_abs"] = point
        row["wall_delta_pct"] = 100 * point / off_p50 if off_p50 > 0 else np.nan
        row["wall_delta_pct_lo"] = 100 * lo / off_p50 if off_p50 > 0 else np.nan
        row["wall_delta_pct_hi"] = 100 * hi / off_p50 if off_p50 > 0 else np.nan
        row["wall_pair_deltas"] = on_w - off_w

    # Paired TTFT delta with BCa CI
    off_t, on_t = _paired_arrays(df_off, df_on, "ttft_seconds")
    if len(off_t) >= 2:
        point, lo, hi = bootstrap_paired_delta(
            off_t, on_t, _quantile(0.5),
            n_resamples=n_resamples, alpha=alpha,
        )
        off_p50 = float(np.median(off_t))
        row["ttft_off_p50"] = off_p50
        row["ttft_on_p50"] = float(np.median(on_t))
        row["ttft_delta_pct"] = 100 * point / off_p50 if off_p50 > 0 else np.nan
        row["ttft_delta_pct_lo"] = 100 * lo / off_p50 if off_p50 > 0 else np.nan
        row["ttft_delta_pct_hi"] = 100 * hi / off_p50 if off_p50 > 0 else np.nan

    # Throughput point estimate + bootstrap CIs, both via Little's law
    # (throughput ≈ c / mean(wall_seconds)). Using a consistent estimator
    # avoids the asymmetric / negative yerr that arose from pairing
    # summary.json's real-timestamp point with a Little's-law CI.
    off_walls = cell_off.get("walls")
    on_walls = cell_on.get("walls")
    if (off_walls is not None and on_walls is not None
            and len(off_walls) >= 10 and len(on_walls) >= 10):
        off_thr = c / float(np.mean(off_walls))
        on_thr = c / float(np.mean(on_walls))
        row["off_throughput"] = off_thr
        row["on_throughput"] = on_thr
        row["throughput_delta_pct"] = (
            100 * (on_thr - off_thr) / off_thr if off_thr > 0 else np.nan
        )

        rng_thr = np.random.default_rng(0xC0DECCC0 + int(c))
        idx_off = rng_thr.integers(0, len(off_walls),
                                   size=(n_resamples, len(off_walls)))
        idx_on = rng_thr.integers(0, len(on_walls),
                                  size=(n_resamples, len(on_walls)))
        thr_off_s = c / off_walls[idx_off].mean(axis=1)
        thr_on_s = c / on_walls[idx_on].mean(axis=1)
        q_lo, q_hi = alpha / 2, 1 - alpha / 2
        row["off_throughput_lo"] = float(np.quantile(thr_off_s, q_lo))
        row["off_throughput_hi"] = float(np.quantile(thr_off_s, q_hi))
        row["on_throughput_lo"] = float(np.quantile(thr_on_s, q_lo))
        row["on_throughput_hi"] = float(np.quantile(thr_on_s, q_hi))

    return row

def _rows_for_deploy(deploy: dict[int, dict], c_values: list[int],
                     n_resamples: int, alpha: float) -> list[dict]:
    return [_compute_row(c, deploy[c]["off"], deploy[c]["on"],
                         n_resamples, alpha)
            for c in c_values]


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------

TTFT_COLOR = "#c44536"   # warm red, matches v1


def make_figure(rows_primary: list[dict], rows_secondary: list[dict] | None,
                out_dir: Path) -> None:
    """Three-panel concurrency figure. See module docstring for layout."""
    fig = plt.figure(figsize=(13.6, 8.8))
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1.0, 1.10],
        hspace=0.42, wspace=0.22,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    _panel_a_overhead(ax_a, rows_primary, rows_secondary)
    _panel_b_throughput(ax_b, rows_primary, rows_secondary)   # <-- now passes secondary
    _panel_c_distribution(ax_c, rows_primary, rows_secondary)

    fig.suptitle("GLM-5.1-fp8 baseline — CC overhead vs concurrency",
                 fontsize=13.5, y=0.995, fontweight="bold")
    _save(fig, out_dir / "sweep_figure")
    print(f"  wrote {out_dir / 'sweep_figure.{png,pdf}'}")

def _panel_a_overhead(ax: plt.Axes,
                      rows_p: list[dict],
                      rows_s: list[dict] | None) -> None:
    """(a) Legend → upper-left ncol=2 short labels; drift box → lower-right."""
    from matplotlib.patches import Patch

    valid_p = [r for r in rows_p if "wall_delta_pct" in r]
    if not valid_p:
        ax.text(0.5, 0.5, "(no paired data)", ha="center", va="center",
                transform=ax.transAxes, color=PALETTE["neutral"])
        return

    def _arrs(rows, key):
        return np.array([r.get(key, np.nan) for r in rows
                         if "wall_delta_pct" in r])

    cs = _arrs(valid_p, "concurrency").astype(int)
    wp, wp_lo, wp_hi = (_arrs(valid_p, k) for k in
                        ("wall_delta_pct", "wall_delta_pct_lo", "wall_delta_pct_hi"))
    tp, tp_lo, tp_hi = (_arrs(valid_p, k) for k in
                        ("ttft_delta_pct", "ttft_delta_pct_lo", "ttft_delta_pct_hi"))

    ax.axhspan(*CC_BAND_PCT, color=PALETTE["muted"], alpha=0.30, zorder=0)
    ax.axhline(0, color=PALETTE["neutral"], linewidth=0.8, linestyle=":",
               alpha=0.6, zorder=1)

    h_w1 = ax.errorbar(cs, wp, yerr=[wp - wp_lo, wp_hi - wp],
                       fmt="o-", color=PALETTE["on"], ecolor=PALETTE["on"],
                       markersize=8.5, linewidth=2.4,
                       capsize=5, capthick=2.0, elinewidth=2.0,
                       markeredgecolor="white", markeredgewidth=1.2,
                       label="wall  D1", zorder=5)
    valid_t = ~np.isnan(tp)
    h_t1 = None
    if valid_t.any():
        h_t1 = ax.errorbar(cs[valid_t], tp[valid_t],
                           yerr=[tp[valid_t] - tp_lo[valid_t],
                                 tp_hi[valid_t] - tp[valid_t]],
                           fmt="s-", color=TTFT_COLOR, ecolor=TTFT_COLOR,
                           markersize=7.5, linewidth=2.0,
                           capsize=5, capthick=2.0, elinewidth=1.8,
                           markeredgecolor="white", markeredgewidth=1.2,
                           label="TTFT  D1", zorder=5)

    h_w2 = h_t2 = None
    if rows_s:
        valid2 = [r for r in rows_s if "wall_delta_pct" in r]
        if valid2:
            cs2 = _arrs(valid2, "concurrency").astype(int)
            wp2, wp2_lo, wp2_hi = (_arrs(valid2, k) for k in
                                   ("wall_delta_pct", "wall_delta_pct_lo",
                                    "wall_delta_pct_hi"))
            tp2, tp2_lo, tp2_hi = (_arrs(valid2, k) for k in
                                   ("ttft_delta_pct", "ttft_delta_pct_lo",
                                    "ttft_delta_pct_hi"))

            ax.plot(cs2, wp2, "--", color=PALETTE["on"], linewidth=1.4,
                    alpha=0.85, zorder=3)
            h_w2 = ax.errorbar(cs2, wp2, yerr=[wp2 - wp2_lo, wp2_hi - wp2],
                               fmt="o", color=PALETTE["on"], ecolor=PALETTE["on"],
                               markersize=8.5, markerfacecolor="white",
                               markeredgewidth=2.2, linestyle="None",
                               capsize=5, capthick=1.6, elinewidth=1.5,
                               label="wall  D2", zorder=4)
            valid_t2 = ~np.isnan(tp2)
            if valid_t2.any():
                ax.plot(cs2[valid_t2], tp2[valid_t2], "--", color=TTFT_COLOR,
                        linewidth=1.2, alpha=0.85, zorder=3)
                h_t2 = ax.errorbar(cs2[valid_t2], tp2[valid_t2],
                                   yerr=[tp2[valid_t2] - tp2_lo[valid_t2],
                                         tp2_hi[valid_t2] - tp2[valid_t2]],
                                   fmt="s", color=TTFT_COLOR, ecolor=TTFT_COLOR,
                                   markersize=7.5, markerfacecolor="white",
                                   markeredgewidth=2.2, linestyle="None",
                                   capsize=5, capthick=1.6, elinewidth=1.5,
                                   label="TTFT  D2", zorder=4)

            wall_drift = float(np.median(wp) - np.median(wp2))
            drift_lines = ["inter-deploy drift (median across c):",
                           f"  wall    {wall_drift:+5.1f} pp"]
            if valid_t.any() and valid_t2.any():
                ttft_drift = float(np.median(tp[valid_t])
                                   - np.median(tp2[valid_t2]))
                drift_lines.append(f"  TTFT   {ttft_drift:+5.1f} pp")
            ax.text(
                0.985, 0.04, "\n".join(drift_lines),
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8.8, color=PALETTE["annotation"],
                family="DejaVu Sans Mono",
                bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                          edgecolor="#CCCCCC", linewidth=0.7, alpha=0.95),
            )

    ax.set_xscale("log", base=2)
    ax.set_xticks(cs)
    ax.set_xticklabels([str(c) for c in cs])
    ax.set_xlabel("concurrency (in-flight requests)")
    ax.set_ylabel(r"CC overhead on $p_{50}$  (%)")

    band_patch = Patch(facecolor=PALETTE["muted"], alpha=0.30,
                       label=f"{CC_BAND_PCT[0]:.0f}-{CC_BAND_PCT[1]:.0f}% band")
    handles = [h for h in (h_w1, h_t1, h_w2, h_t2) if h is not None] + [band_patch]
    ax.legend(handles=handles, loc="upper left", fontsize=8.5,
              ncol=3, framealpha=0.95, columnspacing=0.9,
              handlelength=1.6, handletextpad=0.5)
    # Headroom so the upper-left legend clears the curves.
    all_y = np.concatenate([wp, wp_hi]
                           + ([tp_hi[valid_t]] if valid_t.any() else []))
    ax.set_ylim(-2, float(all_y.max()) + 12)
    ax.set_title("(a)  CC overhead vs concurrency — two deploys",
                 loc="left", fontsize=11.5, pad=8)


def _panel_b_throughput(ax: plt.Axes, rows_p: list[dict],
                        rows_s: list[dict] | None = None) -> None:
    """(b) Tighter callout (3 lines) at lower-right; no data overlap."""
    cs = np.array([r["concurrency"] for r in rows_p], dtype=float)
    off_t = np.array([r.get("off_throughput", np.nan) for r in rows_p])
    on_t = np.array([r.get("on_throughput", np.nan) for r in rows_p])
    off_lo = np.array([r.get("off_throughput_lo", np.nan) for r in rows_p])
    off_hi = np.array([r.get("off_throughput_hi", np.nan) for r in rows_p])
    on_lo = np.array([r.get("on_throughput_lo", np.nan) for r in rows_p])
    on_hi = np.array([r.get("on_throughput_hi", np.nan) for r in rows_p])

    yerr_off = ([off_t - off_lo, off_hi - off_t]
                if not np.isnan(off_lo).all() else None)
    yerr_on = ([on_t - on_lo, on_hi - on_t]
               if not np.isnan(on_lo).all() else None)

    ax.errorbar(cs, off_t, yerr=yerr_off,
                fmt="o-", color=PALETTE["off"],
                markersize=9, linewidth=2.4,
                markeredgecolor=PALETTE["annotation"], markeredgewidth=1.0,
                markerfacecolor="white", ecolor=PALETTE["annotation"],
                capsize=4, capthick=1.4, elinewidth=1.2,
                label="CC-off  D1", zorder=4)
    ax.errorbar(cs, on_t, yerr=yerr_on,
                fmt="o-", color=PALETTE["on"],
                markersize=9, linewidth=2.4,
                markeredgecolor="white", markeredgewidth=1.0,
                ecolor=PALETTE["on"],
                capsize=4, capthick=1.4, elinewidth=1.2,
                label="CC-on  D1", zorder=4)

    if rows_s:
        cs2 = np.array([r["concurrency"] for r in rows_s], dtype=float)
        off2 = np.array([r.get("off_throughput", np.nan) for r in rows_s])
        on2 = np.array([r.get("on_throughput", np.nan) for r in rows_s])
        ax.plot(cs2, off2, "o--", color=PALETTE["off"],
                markersize=7.5, linewidth=1.4,
                markerfacecolor="white", markeredgewidth=1.6,
                markeredgecolor=PALETTE["annotation"],
                alpha=0.85, label="CC-off  D2", zorder=3)
        ax.plot(cs2, on2, "o--", color=PALETTE["on"],
                markersize=7.5, linewidth=1.4,
                markerfacecolor="white", markeredgewidth=1.6,
                markeredgecolor=PALETTE["on"],
                alpha=0.85, label="CC-on  D2", zorder=3)

    # Tight 3-line callout — no internal blank line, smaller font
    finite = ~(np.isnan(off_t) | np.isnan(on_t))
    if finite.sum() >= 2:
        cs_f, off_f, on_f = cs[finite], off_t[finite], on_t[finite]
        c_lo, c_hi = cs_f[0], cs_f[-1]
        scale_c = c_hi / c_lo
        scale_off = off_f[-1] / off_f[0]
        scale_on = on_f[-1] / on_f[0]
        eff_off = 100 * scale_off / scale_c
        eff_on = 100 * scale_on / scale_c
        thr_d_lo = 100 * (on_f[0] - off_f[0]) / off_f[0]
        thr_d_hi = 100 * (on_f[-1] - off_f[-1]) / off_f[-1]
        ax.text(
            0.985, 0.05,
            f"c={c_lo:.0f}→{c_hi:.0f} (×{scale_c:.0f}):\n"
            f"  off ×{scale_off:.2f} ({eff_off:.0f}%)   "
            f"on ×{scale_on:.2f} ({eff_on:.0f}%)\n"
            f"thr Δ:  {thr_d_lo:+.0f}% (c={c_lo:.0f})  →  "
            f"{thr_d_hi:+.0f}% (c={c_hi:.0f})",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, color=PALETTE["annotation"],
            family="DejaVu Sans Mono",
            bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                      edgecolor="#CCCCCC", linewidth=0.7, alpha=0.95),
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(cs)
    ax.set_xticklabels([f"{int(c)}" for c in cs])
    ax.set_xlabel("concurrency (in-flight requests)")
    ax.set_ylabel("throughput (req/s)")
    ax.legend(loc="upper left", fontsize=8.8, ncol=2, framealpha=0.95,
              columnspacing=0.9, handlelength=1.6, handletextpad=0.5)
    ax.set_title("(b)  Throughput scales on both arms",
                 loc="left", fontsize=11.5, pad=8)


def _panel_c_distribution(ax: plt.Axes,
                          rows_p: list[dict],
                          rows_s: list[dict] | None) -> None:
    """(c) Boxplots replace violins. Scatter overlay (deploy 1 only) shows
    n=500 density + outliers. No per-cell median labels — they cascade
    rightward and collide with the next cell. One corner summary instead."""
    from matplotlib.patches import Patch

    valid_p = [r for r in rows_p if r.get("wall_pair_deltas") is not None
               and len(r["wall_pair_deltas"]) > 0]
    if not valid_p:
        ax.text(0.5, 0.5, "(no paired Δwall data)", ha="center", va="center",
                transform=ax.transAxes, color=PALETTE["neutral"])
        return

    positions = np.arange(len(valid_p), dtype=float)
    has_sec = bool(rows_s)
    pos_p = positions - 0.20 if has_sec else positions
    pos_s_base = positions + 0.20 if has_sec else None
    box_w = 0.30 if has_sec else 0.45

    data_p = [r["wall_pair_deltas"] for r in valid_p]
    cs_p = [r["concurrency"] for r in valid_p]

    # Collect data first (so scatter has it)
    data_s, pos_s = [], []
    if has_sec:
        for i, c in enumerate(cs_p):
            rec = next((r for r in rows_s if r["concurrency"] == c
                        and r.get("wall_pair_deltas") is not None
                        and len(r["wall_pair_deltas"]) > 0), None)
            if rec is not None:
                data_s.append(rec["wall_pair_deltas"])
                pos_s.append(pos_s_base[i])

    # Primary scatter — dense, visible (kept from prior patch)
    rng = np.random.default_rng(20260527)
    for i, d in enumerate(data_p):
        jitter = rng.uniform(-0.09, 0.09, len(d))
        ax.scatter(pos_p[i] + jitter, d,
                   s=10, color="#1a5258", alpha=0.32,
                   edgecolor="none", zorder=2)
    # Secondary scatter — was missing in the prior patch
    if has_sec:
        for i, d in enumerate(data_s if data_s else []):
            jitter = rng.uniform(-0.09, 0.09, len(d))
            ax.scatter(pos_s[i] + jitter, d,
                       s=10, color=PALETTE["on"], alpha=0.30,
                       edgecolor="none", zorder=2)
    # Primary boxplots — filled
    ax.boxplot(data_p, positions=pos_p, widths=box_w,
               patch_artist=True, showfliers=False,
               boxprops=dict(facecolor=PALETTE["on"], alpha=0.50,
                             edgecolor=PALETTE["annotation"], linewidth=1.2),
               medianprops=dict(color="#0c3a3f", linewidth=2.8),
               whiskerprops=dict(color=PALETTE["annotation"], linewidth=1.2),
               capprops=dict(color=PALETTE["annotation"], linewidth=1.2),
               zorder=4)

    # Secondary boxplots — lighter fill
    if data_s:
        ax.boxplot(data_s, positions=pos_s, widths=box_w,
                    patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor=PALETTE["on"], alpha=0.20,
                                    edgecolor=PALETTE["on"], linewidth=1.4),
                    medianprops=dict(color=PALETTE["on"], linewidth=2.4),
                    whiskerprops=dict(color=PALETTE["on"], linewidth=1.0),
                    capprops=dict(color=PALETTE["on"], linewidth=1.0),
                    zorder=4)

    # Compute Δmedian summary for the title
    delta_str = ""
    if data_s:
        deltas_ms = [(float(np.median(data_p[i])) - float(np.median(data_s[i]))) * 1000
                     for i in range(min(len(data_p), len(data_s)))]
        delta_str = (f"  ·  Δmedian (D1−D2): "
                     f"{min(deltas_ms):+.0f} to {max(deltas_ms):+.0f} ms across c")

    # Headroom for upper-left legend above c=64 outliers
    all_data = np.concatenate(data_p + data_s)
    ax.set_ylim(float(all_data.min()) - 0.5, float(all_data.max()) + 2.5)

    ax.axhline(0, color=PALETTE["neutral"], linewidth=0.8,
               linestyle=":", alpha=0.6, zorder=1)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"c={c}" for c in cs_p])
    ax.set_xlabel("concurrency")
    ax.set_ylabel(r"$\Delta$ wall (s)")

    n_per_cell = len(data_p[0])
    handles = [Patch(facecolor=PALETTE["on"], edgecolor=PALETTE["annotation"],
                     alpha=0.50, label=f"deploy 1")]
    if data_s:
        handles.append(Patch(facecolor=PALETTE["on"], edgecolor=PALETTE["on"],
                             alpha=0.20, linewidth=1.4,
                             label=f"deploy 2"))
    ax.legend(handles=handles, loc="upper left", fontsize=9.5,
              ncol=1, framealpha=0.95)
    ax.set_title(
        "(c)  Paired Δwall per prompt — boxplots per cell  "
        f"(CC-on $-$ CC-off, n={n_per_cell};  box = IQR,"
        f"whiskers = 1.5·IQR)",
        loc="left", fontsize=10.5, pad=8,
    )


# ----------------------------------------------------------------------------
# Tables (preserved from v1, with paired-BCa numbers replacing the
# v1 unpaired distribution-level deltas)
# ----------------------------------------------------------------------------

def write_tables(rows_p: list[dict], rows_s: list[dict] | None,
                 out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sweep_table.csv"
    md_path = out_dir / "sweep_table.md"

    def _row_view(r: dict, deploy: str) -> dict:
        return {
            "deploy": deploy,
            "concurrency": r["concurrency"],
            "n_paired": r.get("n_paired"),
            "wall_off_p50_s": r.get("wall_off_p50"),
            "wall_on_p50_s": r.get("wall_on_p50"),
            "wall_delta_abs_s": r.get("wall_delta_abs"),
            "wall_delta_pct": r.get("wall_delta_pct"),
            "wall_delta_pct_lo": r.get("wall_delta_pct_lo"),
            "wall_delta_pct_hi": r.get("wall_delta_pct_hi"),
            "ttft_off_p50_s": r.get("ttft_off_p50"),
            "ttft_on_p50_s": r.get("ttft_on_p50"),
            "ttft_delta_pct": r.get("ttft_delta_pct"),
            "ttft_delta_pct_lo": r.get("ttft_delta_pct_lo"),
            "ttft_delta_pct_hi": r.get("ttft_delta_pct_hi"),
            "off_throughput": r.get("off_throughput"),
            "on_throughput": r.get("on_throughput"),
            "throughput_delta_pct": r.get("throughput_delta_pct"),
        }

    rows_flat = [_row_view(r, "primary") for r in rows_p]
    if rows_s:
        rows_flat += [_row_view(r, "secondary") for r in rows_s]
    df = pd.DataFrame(rows_flat)
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"  wrote {csv_path}")

    def _fmt(v, prec=2, suffix=""):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return f"{v:.{prec}f}{suffix}"

    lines = ["# Task D — concurrency sweep (n=500 per cell)\n",
             "Paired BCa CIs computed by aligning off/on requests on "
             "(pair_id, prompt_class). Wall p50 Δ is the paired-median "
             "shift; CIs are bootstrap 95%, n_resamples=10000.\n"]
    for deploy_name, rows in (("Primary", rows_p),
                              (("Secondary", rows_s) if rows_s else (None, None))):
        if not deploy_name:
            continue
        lines.append(f"## {deploy_name} deploy — wall + TTFT\n")
        lines.append("| c | n paired | wall off (s) | wall on (s) | "
                     "Δ wall % [95% CI] | TTFT off (s) | TTFT on (s) | "
                     "Δ TTFT % [95% CI] |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in rows:
            wpct = r.get("wall_delta_pct")
            wlo, whi = r.get("wall_delta_pct_lo"), r.get("wall_delta_pct_hi")
            tpct = r.get("ttft_delta_pct")
            tlo, thi = r.get("ttft_delta_pct_lo"), r.get("ttft_delta_pct_hi")
            wci = (f"{_fmt(wpct, 1, '%')} [{_fmt(wlo, 1)}, {_fmt(whi, 1)}]"
                   if wpct is not None and not np.isnan(wpct) else "—")
            tci = (f"{_fmt(tpct, 1, '%')} [{_fmt(tlo, 1)}, {_fmt(thi, 1)}]"
                   if tpct is not None and not np.isnan(tpct) else "—")
            lines.append(
                f"| {r['concurrency']} | {r.get('n_paired', '—')} "
                f"| {_fmt(r.get('wall_off_p50'), 3)} "
                f"| {_fmt(r.get('wall_on_p50'), 3)} | {wci} "
                f"| {_fmt(r.get('ttft_off_p50'), 3)} "
                f"| {_fmt(r.get('ttft_on_p50'), 3)} | {tci} |"
            )
        lines.append("")
        lines.append(f"### {deploy_name} deploy — throughput\n")
        lines.append("| c | thr off (r/s) | thr on (r/s) | Δ thr % |")
        lines.append("|---|---|---|---|")
        for r in rows:
            lines.append(
                f"| {r['concurrency']} "
                f"| {_fmt(r.get('off_throughput'), 3)} "
                f"| {_fmt(r.get('on_throughput'), 3)} "
                f"| {_fmt(r.get('throughput_delta_pct'), 1, '%')} |"
            )
        lines.append("")
    md_path.write_text("\n".join(lines))
    print(f"  wrote {md_path}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--data-dir", type=Path, required=True,
                    help="Primary deploy: directory with C1-{off,on}-c<N>/ cells")
    ap.add_argument("--secondary-data-dir", type=Path, default=None,
                    help="Optional secondary (replicate) deploy with the same "
                         "cell naming convention; overlaid on panel (a) and (c).")
    ap.add_argument("--c-values", default="8,16,32,64",
                    help="Comma-separated concurrency levels (default: 8,16,32,64)")
    ap.add_argument("--cell-prefix", default="C1",
                    help="Cell ID prefix; cells named <prefix>-<cc>-c<N>")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Where to write outputs (default: --data-dir)")
    ap.add_argument("--n-resamples", type=int, default=10_000)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    c_values = [int(x) for x in args.c_values.split(",")]
    out_dir = args.output_dir or args.data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] loading primary deploy from {args.data_dir}")
    deploy_p = _load_deploy(args.data_dir, c_values, args.cell_prefix)
    rows_p = _rows_for_deploy(deploy_p, c_values, args.n_resamples, args.alpha)

    rows_s: list[dict] | None = None
    if args.secondary_data_dir is not None:
        print(f"[2/4] loading secondary deploy from {args.secondary_data_dir}")
        deploy_s = _load_deploy(args.secondary_data_dir, c_values, args.cell_prefix)
        rows_s = _rows_for_deploy(deploy_s, c_values, args.n_resamples, args.alpha)
    else:
        print(f"[2/4] no secondary deploy (skipping)")

    print(f"[3/4] writing tables to {out_dir}")
    write_tables(rows_p, rows_s, out_dir)

    print(f"[4/4] writing figure to {out_dir}")
    make_figure(rows_p, rows_s, out_dir)

    # Terminal recap
    print()
    print("=" * 68)
    print("  CONCURRENCY SWEEP — wall p50 CC delta (paired BCa)")
    print("=" * 68)
    for deploy_name, rows in (("primary", rows_p),
                              (("secondary", rows_s) if rows_s else (None, None))):
        if not deploy_name:
            continue
        print(f"  [{deploy_name}]")
        for r in rows:
            c = r["concurrency"]
            wpct = r.get("wall_delta_pct")
            wlo = r.get("wall_delta_pct_lo")
            whi = r.get("wall_delta_pct_hi")
            if wpct is None or np.isnan(wpct):
                print(f"    c={c:>3}:  incomplete")
                continue
            print(f"    c={c:>3}:  Δ={wpct:+.1f}%  [{wlo:+.1f}, {whi:+.1f}]")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())