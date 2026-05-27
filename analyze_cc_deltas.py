#!/usr/bin/env python3
"""
analyze_cc_deltas.py

Bootstrap confidence intervals + visualizations for the Phase 3
CC overhead pre-pilot data.

Usage:
    python analyze_cc_deltas.py \\
        --data-dir runs/phase3 \\
        --output-dir runs/phase3/analysis \\
        [--n-resamples 10000] [--alpha 0.05] [--trim-transient]

Inputs (one of):
    --data-dir DIR            Directory with one subdir per cell, each
                              containing `requests.parquet`. Cell IDs
                              must match the CELLS table below.
    --combined-parquet PATH   A single concatenated parquet file with a
                              `cell` column.

Outputs (under --output-dir):
    ci_table.csv              All deltas with point estimates and 95% CIs
    ci_table.md               Same, markdown-formatted for the report
    figures/forest.{png,pdf}  Headline CC delta forest plot
    figures/ecdf_grid.{png,pdf}      Wall-time ECDFs (6 panels, v2.5 layout)
    figures/delta_heatmap.{png,pdf}  CC delta % across percentiles
    figures/delta_heatmap_exec.{png,pdf}  Compact heatmap variant for the
                                          executive summary; drops p99,
                                          bolds p50, thick row-group divider
    figures/invariance_panel.{png,pdf}    Merged H1/H2/H3a invariance plot (v2.5)
    figures/phase_decomposition.{png,pdf} Stacked prefill+decode bars (v2.5)
    figures/tail_anomalies.{png,pdf}      Per-cell max/p95 ratio diagnostic
    figures/debug/paired_grid.{png,pdf}   Paired CC-off vs CC-on (--debug-plots)
    figures/debug_timelines/*.{png,pdf}   Per-cell wall timelines (--debug-plots)

Conventions:
    - Paired bootstrap by pair_id within a (condition, regime) for off/on
      deltas. Unpaired across regimes / conditions.
    - BCa for percentile deltas (p50, p95). Plain percentile for means.
    - Payload delta is computed as a null-hypothesis control: its CI
      should straddle zero exactly.

v2.1 update: added the EII-4 steering condition (C5-off / C5-on, sequential
dispatch). Steer is an active-intervention condition; per the report's §3.8,
its CC delta sits in the regime-invariant band established for passive
workloads.

v2.4 update: added the H3 dense-model replication cells (C0-off / C0-on on
Llama-3.1-70B-FP8 TP=8) and the H3a single-GPU ablation cells (C0-off-tp1 /
C0-on-tp1 on Llama-3.1-70B-FP8 TP=1). Both pairs are sequential-dispatch
baseline-workload cells on a different model from the C1-C5 GLM-5.1-FP8
matrix and exist to test whether the +33% CC overhead is MoE-specific (H2)
or multi-GPU-specific (H3a). Per the report's §3.9/§3.10, both hypotheses
are falsified: C0 sits at +38.4%, C0-tp1 at +38.1%, both within the same
+33-38% band as C1. Compound condition names (`baseline_70b`,
`baseline_70b_tp1`) carry the deployment-config axis without restructuring
the (condition, regime) grouping the rest of the script relies on.

v2.5 update: information-density pass on the main-text figures.
  - figure_invariance_panel REPLACES figure_regime_invariance and
    figure_architecture_invariance (which were structurally the same plot
    on different axes). Three row-groups (regime / architecture / TP),
    one shared x-axis, one figure slot in the report.
  - figure_forest: x-axis cropped to the data range, per-row absolute Δ
    in seconds and n annotated inline, empirical +33-38% "platform CC band"
    shaded behind the data.
  - figure_ecdf collapses the 3 GLM-5.1 baseline regimes into ONE overlay
    panel and the 2 Llama-70B configs into ONE overlay panel, dropping
    the grid from 12 panel slots (10 used) to 6 panel slots (all used).
  - figure_delta_heatmap: row-group separator between baseline-family and
    instrumented rows, a `med (p50-p99)` summary column, luminance-aware
    text colour.
  - figure_phase_decomposition: stacked-bar variant — total wall_p50 split
    into prefill+decode, off vs on side-by-side per regime. Carries both
    the absolute structure AND the differential.
v2.7 update: added figure_delta_heatmap_exec — a stripped-down heatmap for
  the executive summary. Drops p99 (single-request tail-outlier artifacts in
  the off cells of routing and baseline/c8 dominated the eye at exec-summary
  scale without adding to the structural finding). Bolds the p50 column and
  the baseline-vs-instrumented divider. No external colorbar. Compact
  footprint suitable for embedding alongside the "At a glance" table.

  Also hoisted HEATMAP_BASELINE_BLOCK / HEATMAP_INSTRUMENTED_BLOCK and the
  _heatmap_data() helper out of figure_delta_heatmap, so both heatmap
  functions stay structurally aligned automatically.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import DegenerateDataWarning

# scipy floods stderr with DegenerateDataWarning when BCa can't compute the
# acceleration (e.g. constant statistic, very small N). We catch the failure
# inside bootstrap_*_delta and fall back to plain percentile; the warnings
# themselves are noise.
warnings.filterwarnings("ignore", category=DegenerateDataWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")


# ----------------------------------------------------------------------------
# Plotting style
# ----------------------------------------------------------------------------

# Custom palette: CC-off recedes to neutral grey, CC-on is the highlighted state
# (teal) since it's the thing under test. Condition colours are Paul Tol bright.
# `steer` is the EII-4 active-intervention condition added in v2.1.
# v2.4: `baseline_70b` and `baseline_70b_tp1` are the H3/H3a cross-model and
# cross-TP baseline conditions; coloured in the blue family to signal that
# they are all baseline-workload variants differing on deployment axes.
PALETTE = {
    "off":               "#C3C3C3",   # neutral grey
    "on":                "#47A5AD",   # teal
    "baseline":          "#4477AA",   # Paul Tol bright blue — GLM-5.1 / TP=8
    "baseline_70b":      "#66CCEE",   # bright cyan       — Llama-70B / TP=8 (H3 cells)
    "baseline_70b_tp1":  "#332288",   # dark blue         — Llama-70B / TP=1 (H3a cells)
    "baseline_harmbench":"#88CCEE",
    "repe_bundle":       "#228833",   # green
    "gradient":          "#EE6677",   # red-pink
    "routing":           "#AA3377",   # purple
    "steer":             "#CCBB44",   # Paul Tol bright yellow; flags the active-intervention condition
    "neutral":           "#555555",   # dark grey for reference lines/text
    "muted":             "#BBBBBB",   # gridlines, faint shading
    "annotation":        "#222222",
}

# Human-readable display labels for plot rows / panel titles. Falls back to
# the condition key if unmapped. Used by _disp_label() to keep figure copy
# consistent across forest, heatmap, ECDF, and the new architecture-invariance
# plot — without restructuring the underlying (condition, regime) grouping.
CONDITION_DISPLAY = {
    "baseline":          "baseline (GLM-5.1, TP=8)",
    "baseline_70b":      "baseline (Llama-70B, TP=8)",
    "baseline_70b_tp1":  "baseline (Llama-70B, TP=1)",
    "baseline_harmbench":"baseline (GLM-5.1, HarmBench)",
    "repe_bundle":       "repe_bundle",
    "gradient":          "gradient",
    "routing":           "routing",
    "steer":             "steer",
}

# Set of all conditions that share the "baseline workload" semantics — used to
# group baseline-vs-instrumented in the forest plot. Anchored explicitly rather
# than via startswith() so a future condition called e.g. `baseline_xtra` won't
# silently get grouped in.
BASELINE_CONDITIONS = {"baseline", "baseline_70b", "baseline_70b_tp1","baseline_harmbench"}

# Empirical "platform CC band" — the +33-38% range that every baseline-workload
# cell in the report falls into (C1 sequential 33.4%, C0 sequential 38.3%,
# C0-tp1 sequential 38.1%). Used as a shaded reference in the forest plot,
# the invariance panel, and (implicitly) the heatmap. Anchored to the report's
# §3.9 / §3.10 finding rather than recomputed from data so the band is stable
# across reruns with slightly noisier samples.
CC_BAND_PCT: tuple[float, float] = (33.0, 38.0)


def _disp_label(cond: str, regime: str | None = None) -> str:
    """Human-readable label for a (condition, regime) pair."""
    cond_label = CONDITION_DISPLAY.get(cond, cond)
    if regime is None:
        return cond_label
    return f"{cond_label} / {regime}"


# Math-rendered axis labels (v2.6). Using inline LaTeX gives p_{50}(t_{wall})
# style notation that reads as proper math rather than snake_case identifiers.
LABEL_WALL_P50_S    = r"$p_{50}(t_{\mathrm{wall}})$  (s)"
LABEL_WALL_P50_PCT  = r"CC overhead on $p_{50}(t_{\mathrm{wall}})$  (%)"
LABEL_DELTA_P50_PCT = r"$\Delta p_{50}$"


def setup_style() -> None:
    """Apply consistent plotting style. Called once at module load.

    v2.6: global font sizes bumped ~20% across the board for publication
    legibility. Earlier values fit too many figures into the visual budget;
    main-text figures need to read at conventional academic figure widths
    (single-column ~3.5", double-column ~7"). Per-function explicit
    fontsize overrides (where set) are also bumped in sync below."""
    plt.rcParams.update({
        # Typography — single sans-serif, hierarchical sizing
        "font.family":      "DejaVu Sans",
        "font.size":         11.5,
        "axes.titlesize":    13.0,
        "axes.titleweight":  "normal",
        "axes.labelsize":    12.0,
        "xtick.labelsize":   11.0,
        "ytick.labelsize":   11.0,
        "legend.fontsize":   10.5,
        "figure.titlesize":  14.0,
        "figure.titleweight": "normal",

        # Chart-junk reduction
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    1.0,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "xtick.major.size":  3.5,
        "ytick.major.size":  3.5,

        # Gridlines — subtle, not dominant
        "axes.grid":         True,
        "grid.color":        "#CCCCCC",
        "grid.linewidth":    0.6,
        "grid.alpha":        0.6,
        "axes.axisbelow":    True,

        # Lines and markers — bumped per user pref
        "lines.linewidth":   2.4,
        "lines.markersize":  6.0,
        "scatter.marker":    "o",

        # Legend — flat box, no shadow
        "legend.frameon":    True,
        "legend.facecolor":  "white",
        "legend.edgecolor":  "#CCCCCC",
        "legend.framealpha": 0.95,
        "legend.borderpad":  0.4,

        # High-quality output
        "savefig.dpi":       200,
        "savefig.bbox":      "tight",
        "figure.dpi":        110,
    })


setup_style()

# ----------------------------------------------------------------------------
# Cell registry
# ----------------------------------------------------------------------------

# Authoritative per-cell metadata. Used to overwrite per-row values from
# the parquet (in case any run inherited stale values from a prior config).
# v2.4: added C0 (Llama-70B TP=8) and C0-tp1 (Llama-70B TP=1) baseline pairs.
CELLS: list[dict] = [
    {"cell": "C1-off",        "condition": "baseline",          "cc": "off", "regime": "sequential"},
    {"cell": "C1-on",         "condition": "baseline",          "cc": "on",  "regime": "sequential"},
    {"cell": "C2-off",        "condition": "repe_bundle",       "cc": "off", "regime": "sequential"},
    {"cell": "C2-on",         "condition": "repe_bundle",       "cc": "on",  "regime": "sequential"},
    {"cell": "C3-off",        "condition": "gradient",          "cc": "off", "regime": "sequential"},
    {"cell": "C3-on",         "condition": "gradient",          "cc": "on",  "regime": "sequential"},
    {"cell": "C4-off",        "condition": "routing",           "cc": "off", "regime": "sequential"},
    {"cell": "C4-on",         "condition": "routing",           "cc": "on",  "regime": "sequential"},
    {"cell": "C1-off-stream", "condition": "baseline",          "cc": "off", "regime": "streaming"},
    {"cell": "C1-on-stream",  "condition": "baseline",          "cc": "on",  "regime": "streaming"},
    {"cell": "C1-off-c8",     "condition": "baseline",          "cc": "off", "regime": "concurrent_c8"},
    {"cell": "C1-on-c8",      "condition": "baseline",          "cc": "on",  "regime": "concurrent_c8"},
    # v2.1: EII-4 active-intervention condition (sequential dispatch).
    {"cell": "C5-off",        "condition": "steer",             "cc": "off", "regime": "sequential"},
    {"cell": "C5-on",         "condition": "steer",             "cc": "on",  "regime": "sequential"},
    # v2.4: H3 dense-model replication on Llama-3.1-70B-FP8 (TP=8, sequential).
    # Same vLLM image / same prompts / different model. Tests H2 (MoE-specific
    # compute pattern as source of CC overhead) — falsified per report §3.9.
    {"cell": "C0-off",        "condition": "baseline_70b",      "cc": "off", "regime": "sequential"},
    {"cell": "C0-on",         "condition": "baseline_70b",      "cc": "on",  "regime": "sequential"},
    # v2.4: H3a single-GPU ablation on Llama-3.1-70B-FP8 (TP=1, sequential).
    # Same model as C0 but tensor-parallel-size=1. Tests H3a (multi-GPU
    # coordination as source of CC overhead) — falsified per report §3.10.
    {"cell": "C0-off-tp1",    "condition": "baseline_70b_tp1",  "cc": "off", "regime": "sequential"},
    {"cell": "C0-on-tp1",     "condition": "baseline_70b_tp1",  "cc": "on",  "regime": "sequential"},
    {"cell": "C1-off-attest", "condition": "baseline_attest", "cc": "off", "regime": "sequential"},
    {"cell": "C1-on-attest",  "condition": "baseline_attest", "cc": "on",  "regime": "sequential"},
    # v2.X: HarmBench alternative-corpus baseline replication on GLM-5.1 TP=8.
    # Same model / same hardware / different prompt distribution from ToxicChat
    # (the C1-C5 default). Tests whether the +33-38% CC band is corpus-specific.
    {"cell": "C6-off",       "condition": "baseline_harmbench", "cc": "off", "regime": "sequential"},
    {"cell": "C6-on",        "condition": "baseline_harmbench", "cc": "on",  "regime": "sequential"},
]
CELL_META = {c["cell"]: c for c in CELLS}

# Per-cell request cadence (req/s); used as throughput denominator for
# rate-limited cells where wall-time throughput would understate capacity.
# For sequential cells we use cell_wall_seconds from summary.json if available;
# otherwise fall back to N / max(wall_cumulative).
# v2.4: C0 cells use req_rate=1.0 per report §2.4 — sub-2-second walls fit
# under the no-queue-buildup ceiling, unlike the GLM cells at 6-15s walls.
REQ_RATE_BY_CELL = {
    "C1-off": 0.15, "C1-on": 0.15,
    "C2-off": 0.15, "C2-on": 0.15,
    "C3-off": 0.05, "C3-on": 0.05,   # rate-limited
    "C4-off": 0.15, "C4-on": 0.15,
    "C1-off-stream": 1.0, "C1-on-stream": 1.0,
    "C1-off-c8": 8.0, "C1-on-c8": 8.0,
    "C5-off": 0.15, "C5-on": 0.15,
    "C0-off": 1.0, "C0-on": 1.0,             # Llama-70B TP=8; sub-2s walls
    "C0-off-tp1": 1.0, "C0-on-tp1": 1.0,     # Llama-70B TP=1; ~1.3s walls
    "C6-off":    0.15, "C6-on":    0.15,    # NEW
}

# Required and optional columns. Script fails loudly if required is missing.
REQUIRED_COLS = {"pair_id", "prompt_class", "wall_seconds", "payload_bytes"}
OPTIONAL_COLS = {"ttft_seconds", "itl_p50_seconds", "tokens_in", "tokens_out",
                 "n_chunks", "request_id", "http_status", "error"}

# Streaming / concurrent regimes for which TTFT is expected.
TTFT_REGIMES = {"streaming", "concurrent_c8"}

CONCURRENT_TRANSIENT_N = 12  # rows to optionally trim from c=8 cells (per §3.7)

# Canonical 3x4 layout for multi-panel grids.
# Row 1 = same condition (baseline, GLM-5.1 TP=8) across measurement regimes
#         — tests regime sensitivity.
# Row 2 = different instrumentation conditions at the same regime (sequential,
#         GLM-5.1 TP=8) — tests workload dependence.
# Row 3 = baseline workload across deployment configurations at sequential
#         dispatch — tests architecture (H2) and multi-GPU (H3a) sensitivity.
# (condition, regime) tuples; None means leave the panel blank.
PANEL_GRID: list[list[tuple[str, str] | None]] = [
    [("baseline",         "sequential"), ("baseline",         "streaming"),    ("baseline",         "concurrent_c8"), None],
    [("repe_bundle",      "sequential"), ("gradient",         "sequential"),   ("routing",          "sequential"),   ("steer", "sequential")],
    [("baseline_70b",     "sequential"), ("baseline_70b_tp1", "sequential"),    None,                                None],
]
PANEL_GRID_ROW_LABELS = [
    "C1 baseline across measurement regimes",
    "non-baseline conditions (sequential dispatch)",
    "H3/H3a cross-model and cross-TP baselines (sequential dispatch)",
]

# Canonical heatmap row order, hoisted out of figure_delta_heatmap so the
# exec-summary variant stays structurally aligned with the full version
# without duplicating the row list.
HEATMAP_BASELINE_BLOCK: list[tuple[str, str]] = [
    ("baseline",          "sequential"),
    ("baseline",          "streaming"),
    ("baseline",          "concurrent_c8"),
    ("baseline_70b",      "sequential"),    # H3 cross-model
    ("baseline_70b_tp1",  "sequential"),    # H3a single-GPU
    ("baseline_harmbench", "sequential"),
]
HEATMAP_INSTRUMENTED_BLOCK: list[tuple[str, str]] = [
    # v2.8: reordered to descending p50 so the negative-cell row
    # (repe_bundle's p90 = -8.5 in the exec heatmap, plus routing/gradient
    # p99 anomalies in the full heatmap) lands at the bottom of the block
    # rather than adjacent to the baseline/instrumented divider.
    ("gradient",          "sequential"),
    ("routing",           "sequential"),
    ("steer",             "sequential"),
    ("repe_bundle",       "sequential"),
]


def _heatmap_data(df: pd.DataFrame, percentiles: list[int]) -> tuple[pd.DataFrame, int]:
    """Shared data prep for both heatmap figures.

    Walks `HEATMAP_BASELINE_BLOCK + HEATMAP_INSTRUMENTED_BLOCK`, computes
    per-percentile CC delta (%) for each present (condition, regime), and
    appends a `med` summary column = median across the requested percentiles.

    Returns (heatmap_df_indexed_by_label, n_baseline_rows_present). The
    second value anchors the row-group separator in the rendered figure.
    """
    rows: list[dict] = []
    baseline_present = 0
    for cond, regime in HEATMAP_BASELINE_BLOCK + HEATMAP_INSTRUMENTED_BLOCK:
        sub = df[(df["condition"] == cond) & (df["regime"] == regime)]
        off = sub[sub["cc"] == "off"]["wall_seconds"].dropna().to_numpy()
        on  = sub[sub["cc"] == "on"]["wall_seconds"].dropna().to_numpy()
        if len(off) == 0 or len(on) == 0:
            continue
        row: dict = {"label": _disp_label(cond, regime)}
        per_pctl: list[float] = []
        for p in percentiles:
            off_v = float(np.quantile(off, p / 100))
            on_v  = float(np.quantile(on,  p / 100))
            v = 100 * (on_v - off_v) / off_v if off_v > 0 else float("nan")
            row[f"p{p}"] = v
            per_pctl.append(v)
        row["med"] = float(np.nanmedian(per_pctl))
        rows.append(row)
        if (cond, regime) in HEATMAP_BASELINE_BLOCK:
            baseline_present += 1
    if not rows:
        return pd.DataFrame(), 0
    return pd.DataFrame(rows).set_index("label"), baseline_present

# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------

def load_cells(data_dir: Path | None,
               combined_parquet: Path | None) -> pd.DataFrame:
    """
    Load all cells into a single long DataFrame. Either reads one parquet per
    cell from data_dir/<cell_id>/requests.parquet, or one combined parquet
    with a `cell` column.

    Returns a DataFrame with these columns at minimum:
        cell, condition, cc, regime, pair_id, prompt_class,
        wall_seconds, payload_bytes, request_idx (0..N-1 within cell)
    plus ttft_seconds where present.
    """
    if combined_parquet is not None:
        df = pd.read_parquet(combined_parquet)
        if "cell" not in df.columns:
            sys.exit(f"FATAL: combined parquet {combined_parquet} missing 'cell' column")
        frames = [d for _, d in df.groupby("cell")]
    elif data_dir is not None:
        frames = []
        for cell in CELLS:
            cell_id = cell["cell"]
            p = data_dir / cell_id / "requests.parquet"
            if not p.exists():
                warnings.warn(f"missing parquet for cell {cell_id} at {p} -- skipping")
                continue
            cdf = pd.read_parquet(p)
            cdf["cell"] = cell_id
            frames.append(cdf)
        if not frames:
            sys.exit(f"FATAL: no parquets found under {data_dir}")
    else:
        sys.exit("FATAL: must specify either --data-dir or --combined-parquet")

    out = []
    for cdf in frames:
        cell_id = cdf["cell"].iloc[0]
        if cell_id not in CELL_META:
            warnings.warn(f"unknown cell {cell_id} in data; skipping")
            continue
        # Schema validation
        missing = REQUIRED_COLS - set(cdf.columns)
        if missing:
            sys.exit(f"FATAL: cell {cell_id} missing required columns: {sorted(missing)}\n"
                     f"        found: {sorted(cdf.columns)}")
        # Overwrite metadata from registry (defensive)
        meta = CELL_META[cell_id]
        cdf = cdf.copy()
        cdf["condition"] = meta["condition"]
        cdf["cc"] = meta["cc"]
        cdf["regime"] = meta["regime"]
        # Order within cell — useful for the timeline plot and transient trimming
        if "request_id" in cdf.columns:
            cdf = cdf.sort_values("request_id").reset_index(drop=True)
        cdf["request_idx"] = np.arange(len(cdf))
        # Derive decode time = wall − TTFT (the "after first token" phase).
        # Available only on streaming and concurrent cells.
        if "ttft_seconds" in cdf.columns:
            cdf["decode_seconds"] = cdf["wall_seconds"] - cdf["ttft_seconds"]
        out.append(cdf)

    df = pd.concat(out, ignore_index=True)

    # Sanity: every cell should have CC-off and CC-on counterpart present, with
    # the same pair_id set. We don't enforce this hard, but we report it.
    for condition in df["condition"].unique():
        for regime in df[df["condition"] == condition]["regime"].unique():
            sub = df[(df["condition"] == condition) & (df["regime"] == regime)]
            off = sub[sub["cc"] == "off"]["pair_id"].dropna().astype(int).unique()
            on  = sub[sub["cc"] == "on"]["pair_id"].dropna().astype(int).unique()
            common = set(off) & set(on)
            only_off = set(off) - set(on)
            only_on  = set(on) - set(off)
            if only_off or only_on:
                print(f"  [warn] {condition}/{regime}: "
                      f"{len(common)} paired, {len(only_off)} off-only, "
                      f"{len(only_on)} on-only pair_ids")

    return df


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------

@dataclass
class CIResult:
    """A point estimate with its bootstrap CI and metadata."""
    metric: str
    cell_or_pair: str
    n: int
    point: float
    ci_low: float
    ci_high: float
    method: str       # "BCa-paired", "BCa-unpaired", "percentile", ...
    notes: str = ""


def _median(x: np.ndarray) -> float:
    return float(np.median(x))


def _quantile(q: float) -> Callable[[np.ndarray], float]:
    return lambda x: float(np.quantile(x, q))


def _mean(x: np.ndarray) -> float:
    return float(np.mean(x))


def bootstrap_paired_delta(off: np.ndarray,
                           on: np.ndarray,
                           statistic: Callable[[np.ndarray], float],
                           n_resamples: int = 10_000,
                           alpha: float = 0.05,
                           seed: int = 0xC0DECC) -> tuple[float, float, float]:
    """
    Paired bootstrap: resample pair_id indices with replacement, recompute
    statistic on the resampled off-array and on-array, return the delta.
    BCa-corrected CI.

    Requires off.shape == on.shape and aligned by pair_id at the caller.
    """
    assert off.shape == on.shape, f"paired arrays misaligned: {off.shape} vs {on.shape}"
    n = len(off)
    if n < 2:
        return float("nan"), float("nan"), float("nan")

    point = statistic(on) - statistic(off)

    # scipy.stats.bootstrap handles BCa with paired data via vectorized=False
    def delta(off_s, on_s):
        return statistic(on_s) - statistic(off_s)

    rng = np.random.default_rng(seed)
    try:
        res = stats.bootstrap(
            (off, on),
            statistic=delta,
            paired=True,
            n_resamples=n_resamples,
            confidence_level=1 - alpha,
            method="BCa",
            random_state=rng,
            vectorized=False,
        )
        return point, float(res.confidence_interval.low), float(res.confidence_interval.high)
    except Exception as e:
        # BCa can degenerate (e.g. acceleration division-by-zero on constant data).
        # Fall back to plain percentile bootstrap.
        idx = rng.integers(0, n, size=(n_resamples, n))
        deltas = np.array([statistic(on[ix]) - statistic(off[ix]) for ix in idx])
        return point, float(np.quantile(deltas, alpha / 2)), float(np.quantile(deltas, 1 - alpha / 2))


def bootstrap_unpaired_delta(off: np.ndarray,
                             on: np.ndarray,
                             statistic: Callable[[np.ndarray], float],
                             n_resamples: int = 10_000,
                             alpha: float = 0.05,
                             seed: int = 0xC0DECC) -> tuple[float, float, float]:
    """Unpaired bootstrap with BCa."""
    if len(off) < 2 or len(on) < 2:
        return float("nan"), float("nan"), float("nan")

    point = statistic(on) - statistic(off)

    def delta(off_s, on_s):
        return statistic(on_s) - statistic(off_s)

    rng = np.random.default_rng(seed)
    try:
        res = stats.bootstrap(
            (off, on),
            statistic=delta,
            paired=False,
            n_resamples=n_resamples,
            confidence_level=1 - alpha,
            method="BCa",
            random_state=rng,
            vectorized=False,
        )
        return point, float(res.confidence_interval.low), float(res.confidence_interval.high)
    except Exception:
        rng2 = np.random.default_rng(seed)
        off_idx = rng2.integers(0, len(off), size=(n_resamples, len(off)))
        on_idx  = rng2.integers(0, len(on),  size=(n_resamples, len(on)))
        deltas = np.array([statistic(on[oi]) - statistic(off[fi])
                           for oi, fi in zip(on_idx, off_idx)])
        return point, float(np.quantile(deltas, alpha / 2)), float(np.quantile(deltas, 1 - alpha / 2))


def cell_throughput_req_per_s(cell_df: pd.DataFrame) -> float:
    """Approximate cell throughput. Uses request_idx span / total wall budget."""
    # Crude but adequate: N / total elapsed wall time, treating requests as serial.
    if "wall_seconds" not in cell_df.columns:
        return float("nan")
    return len(cell_df) / float(cell_df["wall_seconds"].sum())


# ----------------------------------------------------------------------------
# Delta-table assembly
# ----------------------------------------------------------------------------

@dataclass
class DeltaRow:
    condition: str
    regime: str
    metric: str
    off_point: float
    on_point: float
    abs_delta: float
    abs_ci_low: float
    abs_ci_high: float
    rel_delta_pct: float
    rel_ci_low_pct: float
    rel_ci_high_pct: float
    n_off: int
    n_on: int
    n_paired: int
    method: str
    notes: str = ""


METRICS = [
    ("wall_p50", "wall_seconds", _quantile(0.5)),
    ("wall_p90", "wall_seconds", _quantile(0.9)),
    ("wall_p95", "wall_seconds", _quantile(0.95)),
    ("wall_mean", "wall_seconds", _mean),
    ("ttft_p50", "ttft_seconds", _quantile(0.5)),
    ("ttft_p95", "ttft_seconds", _quantile(0.95)),
    ("ttft_mean", "ttft_seconds", _mean),
    ("decode_p50", "decode_seconds", _quantile(0.5)),
    ("decode_p95", "decode_seconds", _quantile(0.95)),
    ("payload_p50", "payload_bytes", _quantile(0.5)),
]


def compute_deltas(df: pd.DataFrame,
                   n_resamples: int = 10_000,
                   alpha: float = 0.05,
                   trim_transient_c8: bool = False) -> pd.DataFrame:
    """
    For every (condition, regime), compute CC-on minus CC-off deltas with
    BCa CIs for the metrics in METRICS. Paired by pair_id where both arms
    have the same pair_ids.
    """
    rows: list[DeltaRow] = []

    groups = df.groupby(["condition", "regime"])
    for (condition, regime), sub in groups:
        if trim_transient_c8:
            if regime != "concurrent_c8":
                # Skip non-c8 cells when in trim mode; their values are identical
                # to the full-data pass and we don't want duplicate rows.
                continue
            sub = sub[sub["request_idx"] >= CONCURRENT_TRANSIENT_N]
        off = sub[sub["cc"] == "off"]
        on  = sub[sub["cc"] == "on"]
        if len(off) == 0 or len(on) == 0:
            print(f"  [skip] {condition}/{regime}: missing arm")
            continue

        # Align by (pair_id, prompt_class) for paired bootstrap. pair_id alone
        # is non-unique: toxic + benign share the same pair_id by construction.
        off_p = off.set_index(["pair_id", "prompt_class"])
        on_p  = on.set_index(["pair_id", "prompt_class"])
        common = sorted(set(off_p.index) & set(on_p.index))

        for metric_name, col, statfn in METRICS:
            if col not in df.columns:
                continue
            # Skip TTFT / decode metrics for sequential cells (TTFT not recorded)
            if col in ("ttft_seconds", "decode_seconds") and regime not in TTFT_REGIMES:
                continue

            off_vals_all = off[col].dropna().to_numpy(dtype=float)
            on_vals_all  = on[col].dropna().to_numpy(dtype=float)
            if len(off_vals_all) == 0 or len(on_vals_all) == 0:
                continue

            off_point = statfn(off_vals_all)
            on_point  = statfn(on_vals_all)

            # Paired arrays (intersection of pairing keys, NaN-dropped)
            if common:
                off_paired = off_p.loc[common, col].dropna()
                on_paired  = on_p.loc[common, col].dropna()
                # Intersection of remaining keys
                common_clean = sorted(set(off_paired.index) & set(on_paired.index))
                off_arr = off_paired.loc[common_clean].to_numpy(dtype=float)
                on_arr  = on_paired.loc[common_clean].to_numpy(dtype=float)
            else:
                off_arr = on_arr = np.array([], dtype=float)

            # Constant-data null control (e.g. payload_bytes always 0):
            # BCa undefined; report Δ=0 with CI=[0,0].
            if len(off_arr) >= 2 and np.all(off_arr == on_arr) and np.std(off_arr) == 0:
                point, lo, hi = 0.0, 0.0, 0.0
                method = "constant-data"
            elif len(off_arr) >= 2 and len(off_arr) == len(on_arr):
                point, lo, hi = bootstrap_paired_delta(
                    off_arr, on_arr, statfn,
                    n_resamples=n_resamples, alpha=alpha,
                )
                method = "BCa-paired"
            else:
                point, lo, hi = bootstrap_unpaired_delta(
                    off_vals_all, on_vals_all, statfn,
                    n_resamples=n_resamples, alpha=alpha,
                )
                method = "BCa-unpaired"

            rel_denom = abs(off_point) if abs(off_point) > 1e-12 else float("nan")
            if method == "constant-data":
                # Δ=0 with zero baseline: relative delta is by convention 0%
                rel_delta, rel_lo, rel_hi = 0.0, 0.0, 0.0
            elif rel_denom == rel_denom:  # not nan
                rel_delta = 100 * point / rel_denom
                rel_lo    = 100 * lo / rel_denom
                rel_hi    = 100 * hi / rel_denom
            else:
                rel_delta = rel_lo = rel_hi = float("nan")

            notes = ""
            if trim_transient_c8 and regime == "concurrent_c8":
                notes = f"trimmed first {CONCURRENT_TRANSIENT_N} rows (matrix-fill transient)"

            rows.append(DeltaRow(
                condition=condition, regime=regime, metric=metric_name,
                off_point=off_point, on_point=on_point,
                abs_delta=point, abs_ci_low=lo, abs_ci_high=hi,
                rel_delta_pct=rel_delta, rel_ci_low_pct=rel_lo, rel_ci_high_pct=rel_hi,
                n_off=len(off_vals_all), n_on=len(on_vals_all),
                n_paired=len(off_arr),
                method=method, notes=notes,
            ))

    return pd.DataFrame([r.__dict__ for r in rows])


# ----------------------------------------------------------------------------
# Output: tables
# ----------------------------------------------------------------------------

def write_tables(deltas: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    deltas.to_csv(out_dir / "ci_table.csv", index=False)

    # Markdown: one table per condition × regime
    lines: list[str] = ["# CC overhead — bootstrap CIs (95%)\n"]
    for (cond, regime), sub in deltas.groupby(["condition", "regime"]):
        lines.append(f"## {_disp_label(cond, regime)}\n")
        sub = sub.copy()
        sub_disp = pd.DataFrame({
            "metric":       sub["metric"],
            "off":          sub["off_point"].round(4),
            "on":           sub["on_point"].round(4),
            "Δ abs":        sub["abs_delta"].round(4),
            "Δ abs 95% CI": [f"[{lo:.4f}, {hi:.4f}]"
                              for lo, hi in zip(sub["abs_ci_low"], sub["abs_ci_high"])],
            "Δ %":          sub["rel_delta_pct"].round(2),
            "Δ % 95% CI":   [f"[{lo:.2f}, {hi:.2f}]"
                              for lo, hi in zip(sub["rel_ci_low_pct"], sub["rel_ci_high_pct"])],
            "n_off":        sub["n_off"],
            "n_on":         sub["n_on"],
            "n_paired":     sub["n_paired"],
            "method":       sub["method"],
        })
        lines.append(sub_disp.to_markdown(index=False))
        lines.append("")
        if sub["notes"].astype(bool).any():
            for n in sub["notes"].unique():
                if n:
                    lines.append(f"*Notes:* {n}\n")
    (out_dir / "ci_table.md").write_text("\n".join(lines))

    print(f"  wrote {out_dir / 'ci_table.csv'}")
    print(f"  wrote {out_dir / 'ci_table.md'}")


# ----------------------------------------------------------------------------
# Output: figures
# ----------------------------------------------------------------------------

def _save(fig: plt.Figure, path_no_ext: Path) -> None:
    path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{path_no_ext}.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{path_no_ext}.pdf", bbox_inches="tight")
    plt.close(fig)


def _row_color(condition: str) -> str:
    """Map a condition to its palette colour. Falls back to PALETTE['annotation']
    for any condition without a dedicated colour."""
    return PALETTE.get(condition, PALETTE["annotation"])


def figure_forest(deltas: pd.DataFrame, out_dir: Path) -> None:
    """Headline figure: per-cell CC delta in relative and absolute terms.

    LEFT panel  — relative CC overhead (%) on wall_p50 with paired-BCa 95% CIs.
                  Empirical +33-38% platform band shaded behind the data.
    RIGHT panel — absolute wall_p50 (s) as a dumbbell from CC-off to CC-on.
                  Hollow circle = CC-off, filled circle = CC-on; both in the
                  row's condition colour. Dumbbell length encodes the absolute
                  delta; no separate annotation is needed.

    Both panels share row order (baseline family first in canonical block
    order, then instrumented conditions sorted by relative CC delta) and a
    single colour axis: condition. CC state is encoded as marker fill, never
    as colour. A thick (1.6pt) separator divides the two row blocks, matching
    the convention used in figure_delta_heatmap_exec.

    Streaming cells have n=50; all others n=100 — noted in the caption rather
    than inline, to keep the figure free of per-row annotation noise.
    """
    fig_dir = out_dir / "figures"
    sub = deltas[deltas["metric"] == "wall_p50"].copy()
    if sub.empty:
        return
    sub["label"] = [_disp_label(c, r) for c, r in zip(sub["condition"], sub["regime"])]

    # Row ordering ---------------------------------------------------------
    # Row ordering: sort by CC-off baseline wall_p50, ascending — fastest
    # at top, slowest at bottom. The GLM/Llama two-tone colour scheme
    # carries the architectural grouping that the prior family/instrumented
    # split anchored visually.
    order = sub.sort_values("off_point")["label"].tolist()

    m = sub.set_index("label").reindex(order)
    y = np.arange(len(order))
    valid = m["rel_delta_pct"].notna().to_numpy()
    label_to_cond = dict(zip(sub["label"], sub["condition"]))
    LLAMA_COLOR     = "#EE7733"          # Paul Tol orange  — Llama (cross-architecture check)
    GLM_BASE_COLOR  = PALETTE["on"]      # teal             — GLM baseline (anchors the +33-38% band)
    GLM_INSTR_COLOR = "#4477AA"          # Paul Tol blue    — GLM instrumented (tests against the band)
    def _row_color3(cond: str) -> str:
        if cond.startswith("baseline_70b"):
            return LLAMA_COLOR
        if cond in BASELINE_CONDITIONS:
            return GLM_BASE_COLOR
        return GLM_INSTR_COLOR
    colors = [_row_color3(label_to_cond.get(lbl, "")) for lbl in order]
    off_p = m["off_point"].to_numpy()
    on_p  = m["on_point"].to_numpy()

    # Figure layout --------------------------------------------------------
    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(13.0, 0.60 * len(order) + 2.2),
        gridspec_kw={"width_ratios": [1.0, 0.80], "wspace": 0.04},
        sharey=True,
    )

    # === LEFT PANEL: relative CC overhead (%) ===========================
    ax_l.axvspan(CC_BAND_PCT[0], CC_BAND_PCT[1],
                 color=PALETTE["muted"], alpha=0.28, zorder=0)
    ax_l.axvline(0, color=PALETTE["neutral"], linewidth=1.0, linestyle="--",
                 alpha=0.55, zorder=1)

    x_vals  = m["rel_delta_pct"].to_numpy()
    xerr_lo = (m["rel_delta_pct"] - m["rel_ci_low_pct"]).to_numpy()
    xerr_hi = (m["rel_ci_high_pct"] - m["rel_delta_pct"]).to_numpy()
    for i, c in enumerate(colors):
        if not valid[i]:
            continue
        ax_l.errorbar([x_vals[i]], [y[i]],
                      xerr=[[xerr_lo[i]], [xerr_hi[i]]],
                      fmt="o", capsize=4, color=c, ecolor=c,
                      markersize=7.5, linewidth=2.4, capthick=2.0, zorder=3)
        lo, hi = m["rel_ci_low_pct"].iat[i], m["rel_ci_high_pct"].iat[i]
        ax_l.text(hi + 0.9, y[i],
                  f"{x_vals[i]:+.1f}%  [{lo:+.1f}, {hi:+.1f}]",
                  va="center", ha="left", fontsize=10.5, color=c)
    # Colour legend explaining the three-tone row scheme.
    from matplotlib.patches import Patch
    color_legend = [
        Patch(facecolor=GLM_BASE_COLOR,  edgecolor="white", label="GLM-5.1 baseline"),
        Patch(facecolor=GLM_INSTR_COLOR, edgecolor="white", label="GLM-5.1 instrumented"),
        Patch(facecolor=LLAMA_COLOR,     edgecolor="white", label="Llama-3.1-70B"),
    ]
    ax_l.legend(handles=color_legend, loc="lower right",
                framealpha=0.95, fontsize=10)
    lo_arr = m["rel_ci_low_pct"].to_numpy()[valid]
    hi_arr = m["rel_ci_high_pct"].to_numpy()[valid]
    ax_l.set_xlim(min(-3.0, float(lo_arr.min()) - 2.0),
                  float(hi_arr.max()) + 30.0)
    ax_l.set_xlabel(LABEL_WALL_P50_PCT)
    ax_l.set_yticks(y)
    ax_l.set_yticklabels([lbl.replace(" / sequential", "") for lbl in order], fontsize = 14)
    ax_l.set_xticklabels(labels = plt.gca().get_xticklabels(), fontsize = 14)

    for i in range(len(order)):
        if not valid[i] or np.isnan(off_p[i]) or np.isnan(on_p[i]):
            continue
        # Connecting line in the row's condition colour.
        ax_r.hlines(y[i], off_p[i], on_p[i],
                    color=colors[i], linewidth=2.8, alpha=0.65, zorder=2)
        # Hollow = CC-off, filled = CC-on. Both in condition colour.
        ax_r.scatter([off_p[i]], [y[i]], facecolor="white",
                     edgecolor=colors[i], linewidth=2.2, s=82, zorder=4)
        ax_r.scatter([on_p[i]], [y[i]], facecolor=colors[i],
                     edgecolor=colors[i], linewidth=1.6, s=82, zorder=4)

    # Marker-shape legend: encodes CC state without introducing a colour.
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor="white",
               markeredgecolor=PALETTE["annotation"],
               markeredgewidth=2.0, markersize=9, label="CC-off"),
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor=PALETTE["annotation"],
               markeredgecolor=PALETTE["annotation"],
               markeredgewidth=1.6, markersize=9, label="CC-on"),
    ]
    ax_r.legend(handles=legend_handles, loc="lower right",
                framealpha=0.95, fontsize=14)

    all_vals = np.concatenate([off_p[valid], on_p[valid]])
    pad = all_vals.max() * 0.05
    ax_r.set_xlim(0.0, all_vals.max() + pad)
    ax_r.set_xlabel(LABEL_WALL_P50_S)
    ax_r.set_xticklabels(labels = plt.gca().get_xticklabels(), fontsize = 14)
    ax_r.tick_params(axis="y", left=False, labelleft=False)

    # Two-line title matching figure_delta_heatmap_exec / figure_phase_decomposition.
    fig.suptitle(
        "Relative and absolute CC overhead per cell",weight = 'bold',
        fontsize=11.5, y=1.005,
    )
    plt.tight_layout()
    _save(fig, fig_dir / "forest")
    print(f"  wrote {fig_dir / 'forest.{png,pdf}'}")


def figure_ecdf(df: pd.DataFrame, out_dir: Path) -> None:
    """Wall-time ECDFs, CC-off vs CC-on (v2.5 layout: 2x3 = 6 panels).

    The v2.4 3x4 grid had 12 panel slots for 10 cells, with 2 empty slots
    and a fragmented visual story (10 small ECDF pairs to scan). v2.5 collapses:

      Panel 1: GLM-5.1 baseline across 3 regimes, overlaid — 3 off-curves
               (grey, line-style differentiated) + 3 on-curves (teal,
               same line styles). The visual question H1 asks ("does
               regime shift the curve?") is answered by overlay.
      Panel 2: Llama-70B baseline at TP=8 and TP=1, overlaid — same logic
               for H3a.
      Panels 3-6: the four instrumentation conditions (repe_bundle,
                  gradient, routing, steer), each its own panel since
                  wall scales differ by ~3x across them and overlay
                  would compress the bulk.

    Per-panel Δp50 and Δp95 annotations carry both the central shift
    and the tail shift.
    """
    fig_dir = out_dir / "figures"

    REGIME_STYLES = [
        ("sequential",   "-",  3.0),
        ("streaming",    "--", 2.4),
        ("concurrent_c8", ":", 2.4),
    ]
    TP_STYLES = [
        ("baseline_70b",     "TP=8", "-",  3.0),
        ("baseline_70b_tp1", "TP=1", "--", 2.4),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2))

    def _annot_dp50_arrow(ax: plt.Axes, off: np.ndarray, on: np.ndarray) -> None:
        """v2.6: replace the Δp50/Δp95 text box with a bidirectional arrow at
        y=0.5 spanning p50_off → p50_on. The arrow endpoints sit exactly
        where each ECDF crosses 0.5 (i.e. on the median); the geometry is
        the proof of the median shift."""
        if len(off) == 0 or len(on) == 0:
            return
        p50_off, p50_on = float(np.median(off)), float(np.median(on))
        dp50 = 100 * (p50_on - p50_off) / p50_off if p50_off > 0 else float("nan")
        ax.annotate("", xy=(p50_on, 0.5), xytext=(p50_off, 0.5),
                    xycoords="data", textcoords="data",
                    arrowprops=dict(arrowstyle="<->",
                                     color=PALETTE["annotation"],
                                     lw=1.8, shrinkA=0, shrinkB=0),
                    zorder=5)
        mid_x = (p50_off + p50_on) / 2
        ax.text(mid_x, 0.555,
                f"{LABEL_DELTA_P50_PCT} = {dp50:+.1f}%",
                ha="center", va="bottom", fontsize=11,
                color=PALETTE["annotation"],
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="#CCCCCC", linewidth=0.5, alpha=0.92),
                zorder=6)

    def _plot_ecdf(ax: plt.Axes, x: np.ndarray, color: str, ls: str, lw: float,
                   label: str | None = None) -> None:
        if len(x) == 0:
            return
        xs = np.sort(x)
        ax.plot(xs, np.linspace(0, 1, len(xs), endpoint=False),
                color=color, linewidth=lw, linestyle=ls, label=label)

    # --- Panel (0, 0): baseline GLM-5.1 across regimes -----------------------
    ax = axes[0, 0]
    all_w = []
    for (regime, ls, lw) in REGIME_STYLES:
        sub = df[(df["condition"] == "baseline") & (df["regime"] == regime)]
        off = sub[sub["cc"] == "off"]["wall_seconds"].dropna().to_numpy()
        on  = sub[sub["cc"] == "on" ]["wall_seconds"].dropna().to_numpy()
        if len(off) and len(on):
            _plot_ecdf(ax, off, PALETTE["off"], ls, lw, label=f"off, {regime}")
            _plot_ecdf(ax, on,  PALETTE["on"],  ls, lw, label=f"on, {regime}")
            all_w.extend([off, on])
    if all_w:
        joint = np.concatenate(all_w)
        # v2.6: tighter clip — p98*1.05 instead of p99*1.05 — to drop long
        # singular-outlier tails (these are the tail-anomalies figure's job).
        ax.set_xlim(left=max(0.0, joint.min() - 0.3),
                    right=float(np.quantile(joint, 0.98)) * 1.05)
    ax.set_title("H1 — GLM-5.1 baseline / regime sweep")
    ax.set_ylabel("ECDF")
    ax.legend(loc="lower right", fontsize=9.5, ncol=1)

    # Δp50 arrow for the regime overlay: anchor on sequential.
    seq_off = df[(df["condition"] == "baseline") & (df["regime"] == "sequential")
                  & (df["cc"] == "off")]["wall_seconds"].dropna().to_numpy()
    seq_on  = df[(df["condition"] == "baseline") & (df["regime"] == "sequential")
                  & (df["cc"] == "on")]["wall_seconds"].dropna().to_numpy()
    _annot_dp50_arrow(ax, seq_off, seq_on)

    # --- Panel (0, 1): Llama-70B TP=8 vs TP=1 -------------------------------
    ax = axes[0, 1]
    all_w = []
    for (cond, lbl, ls, lw) in TP_STYLES:
        sub = df[(df["condition"] == cond) & (df["regime"] == "sequential")]
        off = sub[sub["cc"] == "off"]["wall_seconds"].dropna().to_numpy()
        on  = sub[sub["cc"] == "on" ]["wall_seconds"].dropna().to_numpy()
        if len(off) and len(on):
            _plot_ecdf(ax, off, PALETTE["off"], ls, lw, label=f"off, {lbl}")
            _plot_ecdf(ax, on,  PALETTE["on"],  ls, lw, label=f"on, {lbl}")
            all_w.extend([off, on])
    if all_w:
        joint = np.concatenate(all_w)
        ax.set_xlim(left=max(0.0, joint.min() - 0.1),
                    right=float(np.quantile(joint, 0.98)) * 1.05)
    ax.set_title("H3a — Llama-70B baseline / TP sweep")
    ax.legend(loc="lower right", fontsize=9.5)
    # Δp50 arrow for the TP=8 anchor:
    sub_anc = df[(df["condition"] == "baseline_70b") & (df["regime"] == "sequential")]
    _annot_dp50_arrow(ax,
                       sub_anc[sub_anc["cc"] == "off"]["wall_seconds"].dropna().to_numpy(),
                       sub_anc[sub_anc["cc"] == "on"]["wall_seconds"].dropna().to_numpy())

    # --- Panels (0, 2) and (1, 0..2): the four instrumentation conditions ---
    INSTR = [
        ("repe_bundle", "sequential", axes[0, 2]),
        ("gradient",    "sequential", axes[1, 0]),
        ("routing",     "sequential", axes[1, 1]),
        ("steer",       "sequential", axes[1, 2]),
    ]
    for (cond, regime, ax) in INSTR:
        sub = df[(df["condition"] == cond) & (df["regime"] == regime)]
        off = sub[sub["cc"] == "off"]["wall_seconds"].dropna().to_numpy()
        on  = sub[sub["cc"] == "on" ]["wall_seconds"].dropna().to_numpy()
        if len(off) == 0 or len(on) == 0:
            ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                    transform=ax.transAxes, color=PALETTE["neutral"], fontsize=11)
            ax.set_title(_disp_label(cond, regime))
            continue
        _plot_ecdf(ax, off, PALETTE["off"], "-", 2.6, label=f"CC-off (n={len(off)})")
        _plot_ecdf(ax, on,  PALETTE["on"],  "-", 2.6, label=f"CC-on  (n={len(on)})")
        # p50 reference lines — light, since the arrow already marks the gap
        p50_off, p50_on = float(np.median(off)), float(np.median(on))
        ax.axvline(p50_off, color=PALETTE["off"], linewidth=1.0,
                   linestyle=":", alpha=0.5)
        ax.axvline(p50_on, color=PALETTE["on"], linewidth=1.0,
                   linestyle=":", alpha=0.5)
        # v2.6: tighter clip at p98*1.05
        joint = np.concatenate([off, on])
        ax.set_xlim(left=max(0.0, joint.min() - 0.3),
                    right=float(np.quantile(joint, 0.98)) * 1.05)
        ax.set_title(_disp_label(cond, regime))
        ax.legend(loc="lower right", fontsize=9.5)
        _annot_dp50_arrow(ax, off, on)

    # Axis labels on the perimeter only
    for ax in axes[1, :]:
        ax.set_xlabel(LABEL_WALL_P50_S.replace("(s)", "(s)"))  # keep prose label; math is in p_{50}
    axes[1, 0].set_ylabel("ECDF")

    fig.suptitle("Wall-time ECDFs: CC-off vs CC-on  "
                 "(overlay panels for invariance axes; small multiples for workloads)",
                 y=1.005, fontsize=12.5)
    plt.tight_layout()
    _save(fig, fig_dir / "ecdf_grid")
    print(f"  wrote {fig_dir / 'ecdf_grid.{png,pdf}'}")


def figure_tail_anomalies(df: pd.DataFrame, out_dir: Path) -> None:
    """Per-cell tail anomaly diagnostic.

    Top panel (log y): max / p95 ratio per cell. Ratio of 1 = no rogue tail;
    higher = a few requests dominate the max. Anomaly threshold at 2×.
    v2.1: C5-off and C5-on both clear the 2× threshold due to the first-call
    vllm-lens steering-hook init artifact (documented in report §3.8).

    Bottom panel: count of requests above 3× p50, rendered as stems
    (integer-valued, so bars look heavy)."""
    fig_dir = out_dir / "figures"

    rows = []
    cell_order = [c["cell"] for c in CELLS]
    for cell in cell_order:
        sub = df[df["cell"] == cell]
        if sub.empty:
            continue
        w = sub["wall_seconds"].dropna().to_numpy()
        if len(w) < 5:
            continue
        p50 = float(np.median(w))
        p95 = float(np.quantile(w, 0.95))
        mx  = float(w.max())
        meta = CELL_META.get(cell, {})
        rows.append({
            "cell": cell,
            "condition": meta.get("condition", "?"),
            "max_over_p95": mx / p95 if p95 > 0 else np.nan,
            "n_above_3x_p50": int((w > 3 * p50).sum()),
            "n": len(w),
        })
    if not rows:
        return
    tail = pd.DataFrame(rows)
    colors = [PALETTE.get(c, PALETTE["neutral"]) for c in tail["condition"]]

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 5.4), sharex=True,
                              gridspec_kw={"height_ratios": [2.5, 1]})
    x = np.arange(len(tail))

    # --- top: max/p95 on log scale ---
    bars = axes[0].bar(x, tail["max_over_p95"], color=colors,
                        edgecolor="white", linewidth=0.5, width=0.7)
    axes[0].axhline(2.0, color=PALETTE["neutral"], linestyle="--",
                    linewidth=1.2, alpha=0.8)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("max / p95 wall (log)")
    axes[0].set_title("Tail anomalies per cell", pad=8)
    axes[0].grid(True, axis="y", alpha=0.4, which="both")

    # Threshold label
    axes[0].text(len(tail) - 0.5, 2.0, " anomaly threshold (2×)",
                 va="center", ha="left", fontsize=8.5,
                 color=PALETTE["neutral"])

    # Annotate the worst offender — placed inline to the left so it doesn't push
    # past the data area on log scale and disrupt title placement.
    flagged = tail[tail["max_over_p95"] >= 2.0].sort_values("max_over_p95", ascending=False)
    if not flagged.empty:
        worst = flagged.iloc[0]
        worst_idx = list(tail["cell"]).index(worst["cell"])
        axes[0].annotate(f"{worst['cell']}\n{worst['max_over_p95']:.1f}×",
                          xy=(worst_idx, worst["max_over_p95"]),
                          xytext=(worst_idx - 2.5, worst["max_over_p95"] * 0.85),
                          fontsize=8.5, color=PALETTE["annotation"],
                          ha="right", va="center",
                          arrowprops=dict(arrowstyle="->", color=PALETTE["annotation"],
                                          linewidth=0.8, alpha=0.7,
                                          connectionstyle="arc3,rad=0.0"))

    # --- bottom: integer counts as stems ---
    counts = tail["n_above_3x_p50"].to_numpy()
    # Stems only (lines + zero baseline), markers via scatter for per-cell colour.
    axes[1].vlines(x, 0, counts, color=PALETTE["neutral"], linewidth=1.0, alpha=0.6)
    axes[1].scatter(x, counts, c=colors, s=55, edgecolor="white", linewidth=0.6,
                    zorder=3)
    axes[1].axhline(0, color=PALETTE["muted"], linewidth=0.5)
    axes[1].set_ylabel("count(wall > 3·p50)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tail["cell"], rotation=45, ha="right")
    axes[1].set_ylim(bottom=-0.3, top=max(counts.max() + 1, 2))
    axes[1].grid(True, axis="y", alpha=0.4)

    # Per-condition legend on the top panel — v2.1: include steer
    # v2.4: include baseline_70b family. Legend order matches the canonical
    # heatmap row order (baseline family first, then non-baseline by delta).
    from matplotlib.patches import Patch
    legend_conditions = ["baseline", "baseline_70b", "baseline_70b_tp1",
                          "repe_bundle", "gradient", "routing", "steer"]
    legend_handles = [Patch(facecolor=PALETTE[c], edgecolor="white", linewidth=0.5,
                             label=CONDITION_DISPLAY.get(c, c))
                      for c in legend_conditions if c in PALETTE]
    axes[0].legend(handles=legend_handles, loc="upper right", ncol=2, frameon=True,
                    fontsize=8)

    plt.tight_layout()
    _save(fig, fig_dir / "tail_anomalies")
    print(f"  wrote {fig_dir / 'tail_anomalies.{png,pdf}'}")


def figure_timelines_debug(df: pd.DataFrame, out_dir: Path) -> None:
    """Per-cell wall timeline. Useful for interactive diagnosis but bulky for
    the paper. Emitted only under --debug-plots."""
    fig_dir = out_dir / "figures" / "debug_timelines"
    count = 0
    for cell, sub in df.groupby("cell"):
        sub = sub.sort_values("request_idx")
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(sub["request_idx"], sub["wall_seconds"],
                marker="o", markersize=3, linewidth=0.7, alpha=0.8)
        ax.set_xlabel("request order within cell")
        ax.set_ylabel("wall_seconds")
        ax.set_title(f"{cell}  ({sub['condition'].iloc[0]} / {sub['regime'].iloc[0]} / CC-{sub['cc'].iloc[0]}, n={len(sub)})")
        ax.grid(True, alpha=0.3)
        _save(fig, fig_dir / f"timeline_{cell}")
        count += 1
    print(f"  wrote {count} debug timeline figures to {fig_dir}")

def figure_delta_heatmap(df: pd.DataFrame, out_dir: Path,
                          n_resamples: int = 5_000, alpha: float = 0.05) -> None:
    """
    Heatmap of CC delta (%) across percentiles {50, 75, 90, 95, 99} ×
    (condition, regime). Diverging palette centred at 0 (overhead is red,
    rare negative deltas — host-noise artifacts — are blue).

    v2.5 enhancements:
      - Trailing summary column showing median across p50-p99 for the row.
        Lets the reader see "what's the row's headline CC delta" without
        eye-averaging five cells.
      - Row-group separator (thin black line) between the baseline-family
        rows and the instrumentation-condition rows, anchoring the H2/H3a
        narrative visually.
      - Luminance-aware text colour: cells are pre-classified by their
        normalized-cmap luminance, not by a fixed |v| threshold. Fixes the
        readability dip at +25-35% (light salmon background).

    v2.7: data prep extracted to `_heatmap_data` so the exec-summary
    variant (figure_delta_heatmap_exec) stays structurally aligned with
    this figure without duplicating the row list.
    """
    from matplotlib.colors import TwoSlopeNorm

    fig_dir = out_dir / "figures"
    pctls = [50, 75, 90, 95, 99]
    hm, baseline_present_count = _heatmap_data(df, pctls)
    if hm.empty:
        return

    main_cols = [f"p{p}" for p in pctls]
    summary_col = "med"

    # Colour normalization. Clip negative tail at -20; upper at 70.
    vmin, vcenter, vmax = -20.0, 0.0, 70.0
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r")

    # Figure width grows by one column for the summary
    fig, (ax_main, ax_sum) = plt.subplots(
        1, 2, figsize=(8.6, 0.55 * len(hm) + 1.7),
        gridspec_kw={"width_ratios": [len(main_cols), 1.05], "wspace": 0.04},
    )

    main_data = hm[main_cols].to_numpy()
    sum_data  = hm[[summary_col]].to_numpy()

    im_main = ax_main.imshow(main_data, aspect="auto", cmap=cmap, norm=norm)
    im_sum  = ax_sum.imshow(sum_data,  aspect="auto", cmap=cmap, norm=norm)

    def _annotate(ax: plt.Axes, data: np.ndarray, cols: list[str]) -> None:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                if np.isnan(v):
                    continue
                # Luminance-aware text colour using the cmap RGB.
                rgba = cmap(norm(v))
                # Relative luminance (sRGB approximation)
                lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                text_colour = "white" if lum < 0.50 else "#1A1A1A"
                ax.text(j, i, f"{v:+.1f}",
                        ha="center", va="center", color=text_colour, fontsize=9)

    _annotate(ax_main, main_data, main_cols)
    _annotate(ax_sum,  sum_data,  [summary_col])

    # Tick / label config
    ax_main.set_xticks(np.arange(len(main_cols)))
    ax_main.set_xticklabels(main_cols)
    ax_main.set_yticks(np.arange(len(hm)))
    ax_main.set_yticklabels(hm.index)
    ax_main.set_xlabel("wall-time percentile")
    ax_main.grid(False); ax_main.tick_params(length=0)

    ax_sum.set_xticks([0])
    ax_sum.set_xticklabels(["med\n(p50-p99)"], fontsize=8.5)
    ax_sum.set_yticks([])
    ax_sum.grid(False); ax_sum.tick_params(length=0)
    # Visual divide between the main matrix and the summary column
    ax_sum.spines["left"].set_visible(True)
    ax_sum.spines["left"].set_color("#1A1A1A")
    ax_sum.spines["left"].set_linewidth(1.2)

    # Row-group separator (between baseline block and instrumented block)
    if 0 < baseline_present_count < len(hm):
        sep_y = baseline_present_count - 0.5
        for ax in (ax_main, ax_sum):
            ax.axhline(sep_y, color="#1A1A1A", linewidth=1.4, alpha=0.9, zorder=10)

    cbar = fig.colorbar(im_main, ax=[ax_main, ax_sum], fraction=0.025, pad=0.02)
    cbar.set_label("CC overhead %", rotation=90, labelpad=8)
    cbar.outline.set_linewidth(0.5)

    ax_main.set_title("CC overhead across the wall-time distribution"
                      "  (line = baseline-family / instrumented divide)",
                      pad=10, loc="left")
    _save(fig, fig_dir / "delta_heatmap")
    print(f"  wrote {fig_dir / 'delta_heatmap.{png,pdf}'}")


def figure_delta_heatmap_exec(df: pd.DataFrame, out_dir: Path) -> None:
    """Compact, exec-summary-grade heatmap of CC delta (%).

    Differences from `figure_delta_heatmap`:
      - Drops the p99 column. The two negative p99 cells visible in the full
        heatmap (routing -69%, baseline/concurrent_c8 -8.5%) are explicitly
        attributed to single-request tail outliers in the off cells (report
        §3.3) and would dominate the eye in a small exec-summary render
        without adding to the structural finding. p95 retains the tail
        information without the artifact.
      - Hierarchical y-axis labels: short leaf names per row (e.g. just
        "streaming", "TP=8", "gradient"), with bold group headers and
        vertical brackets to the left grouping by deployment config.
      - Headline p50 column rendered with bold text and faint vertical
        separators, so the eye lands on the headline number first.
      - Thick (1.6pt) divider between the baseline-family block and the
        instrumented block.
      - Two-line title leading with the structural finding.
      - Compact footprint with a narrow vertical colorbar.
    """
    from matplotlib.colors import TwoSlopeNorm

    fig_dir = out_dir / "figures"
    pctls = [50, 75, 90, 95]
    hm, baseline_present_count = _heatmap_data(df, pctls)
    if hm.empty:
        return

    main_cols = [f"p{p}" for p in pctls]
    summary_col = "med"
    p50_col_idx = main_cols.index("p50")

    vmin, vcenter, vmax = -15.0, 0.0, 60.0
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r")

    fig, (ax_main, ax_sum) = plt.subplots(
        1, 2, figsize=(8.6, 0.44 * len(hm) + 1.7),
        gridspec_kw={"width_ratios": [len(main_cols), 1.0], "wspace": 0.05},
    )

    main_data = hm[main_cols].to_numpy()
    sum_data  = hm[[summary_col]].to_numpy()

    im_main = ax_main.imshow(main_data, aspect="auto", cmap=cmap, norm=norm)
    ax_sum.imshow(sum_data, aspect="auto", cmap=cmap, norm=norm)

    def _annotate(ax: plt.Axes, data: np.ndarray, bold_col_idx: int | None = None) -> None:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                if np.isnan(v):
                    continue
                rgba = cmap(norm(v))
                lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                text_colour = "white" if lum < 0.50 else "#1A1A1A"
                weight = "bold" if (bold_col_idx is not None and j == bold_col_idx) else "normal"
                ax.text(j, i, f"{v:+.1f}",
                        ha="center", va="center", color=text_colour,
                        fontsize=9.5, fontweight=weight)

    _annotate(ax_main, main_data, bold_col_idx=p50_col_idx)
    _annotate(ax_sum,  sum_data,  bold_col_idx=0)

    # === Hierarchical y-axis labels =====================================
    # Replace verbose "baseline (GLM-5.1, TP=8) / streaming"-style labels
    # with short leaves + bracketed group headers on the left.
    def _short_leaf(label: str) -> str:
        cond_part, _, regime = label.partition(" / ")
        regime_pretty = {"concurrent_c8": "concurrent c=8"}.get(regime, regime)
        if "Llama-70B" in cond_part:
            if "TP=8" in cond_part: return "TP=8"
            if "TP=1" in cond_part: return "TP=1"
        if "GLM-5.1" in cond_part:
            return regime_pretty
        return cond_part   # instrumented row — use the condition name itself

    def _group_key(label: str) -> str:
        cond_part, _, _ = label.partition(" / ")
        if "Llama-70B" in cond_part: return "llama70b"
        if "GLM-5.1"  in cond_part: return "glm"
        return "instrumented"

    GROUP_LABELS = {
        "glm":          "GLM-5.1 TP=8\nbaseline",
        "llama70b":     "Llama-70B\nbaseline",
        "instrumented": "GLM-5.1 TP=8\ninstrumented",
    }

    short_labels = [_short_leaf(lbl) for lbl in hm.index]
    group_keys   = [_group_key(lbl)  for lbl in hm.index]

    # Compute contiguous group spans
    group_spans: list[tuple[str, int, int]] = []
    cur_key, cur_start = None, 0
    for i, k in enumerate(group_keys):
        if k != cur_key:
            if cur_key is not None:
                group_spans.append((cur_key, cur_start, i - 1))
            cur_key, cur_start = k, i
    if cur_key is not None:
        group_spans.append((cur_key, cur_start, len(group_keys) - 1))

    ax_main.set_yticks(np.arange(len(hm)))
    ax_main.set_yticklabels(short_labels, fontsize=10)
    ax_main.tick_params(axis="y", length=0, pad=3)

    # Bracketed group labels on the left. Drawn in data coords with
    # clip_on=False so they extend outside the axes; bbox_inches="tight"
    # in _save() expands the saved bbox to include them.
    bracket_x   = -1.85    # vertical bracket line
    label_x     = -2.15    # group label text
    cap_dx      = 0.13     # horizontal cap length at bracket top/bottom
    pad_y       = 0.32     # vertical padding for the bracket
    for key, i_start, i_end in group_spans:
        y_center = (i_start + i_end) / 2
        ax_main.text(label_x, y_center, GROUP_LABELS.get(key, key),
                     ha="right", va="center", fontsize=10.0,
                     fontweight="bold", color=PALETTE["annotation"],
                     clip_on=False)
        if i_end > i_start:
            ax_main.plot([bracket_x, bracket_x],
                         [i_start - pad_y, i_end + pad_y],
                         color=PALETTE["annotation"], linewidth=1.2,
                         clip_on=False)
            for y_cap in (i_start - pad_y, i_end + pad_y):
                ax_main.plot([bracket_x, bracket_x + cap_dx], [y_cap, y_cap],
                             color=PALETTE["annotation"], linewidth=1.2,
                             clip_on=False)

    # X tick labels: bold the p50 header.
    ax_main.set_xticks(np.arange(len(main_cols)))
    ax_main.set_xticklabels(main_cols)
    for j, tick in enumerate(ax_main.get_xticklabels()):
        if j == p50_col_idx:
            tick.set_fontweight("bold")
            tick.set_fontsize(11)
    ax_main.set_xlabel("wall-time percentile", fontsize=10)
    ax_main.grid(False); ax_main.tick_params(axis="x", length=0)

    ax_sum.set_xticks([0])
    ax_sum.set_xticklabels([f"med\n(p50-p{pctls[-1]})"], fontsize=8.5)
    ax_sum.set_yticks([])
    ax_sum.grid(False); ax_sum.tick_params(length=0)
    ax_sum.spines["left"].set_visible(True)
    ax_sum.spines["left"].set_color("#1A1A1A")
    ax_sum.spines["left"].set_linewidth(1.2)

    # Thick row-group separator between baseline-family and instrumented blocks.
    if 0 < baseline_present_count < len(hm):
        sep_y = baseline_present_count - 0.5
        for ax in (ax_main, ax_sum):
            ax.axhline(sep_y, color="#1A1A1A", linewidth=1.6, alpha=1.0, zorder=10)

    # Faint vertical separators around the p50 column.
    ax_main.axvline(p50_col_idx + 0.5, color="#1A1A1A", linewidth=0.9,
                    alpha=0.55, zorder=8)
    if p50_col_idx > 0:
        ax_main.axvline(p50_col_idx - 0.5, color="#1A1A1A", linewidth=0.9,
                        alpha=0.55, zorder=8)

    cbar = fig.colorbar(im_main, ax=[ax_main, ax_sum], fraction=0.022, pad=0.02)
    cbar.set_label("CC overhead %", rotation=90, labelpad=6, fontsize=9)
    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(labelsize=8.5)

    # Two-line title: structural takeaway above, methodology below.
    fig.suptitle(
        "Per-cell CC overhead across the wall-time distribution\n"
        "Baseline workloads cluster at +30-38%; instrumented span +9 to +60% "
        r"depending on per-token compute weight",
        fontsize=11.5, y=1.015,
    )
    _save(fig, fig_dir / "delta_heatmap_exec")
    print(f"  wrote {fig_dir / 'delta_heatmap_exec.{png,pdf}'}")

def figure_invariance_panel(deltas: pd.DataFrame, out_dir: Path) -> None:
    """
    Merged H1 + H2 + H3a invariance figure (v2.5 — replaces the prior
    figure_regime_invariance + figure_architecture_invariance).

    The two prior figures were structurally the same plot (rel CC delta with
    paired-BCa CIs against a "band" reference) on different axes. Merging
    saves a figure slot and presents the full invariance story coherently:
    the +33-38% platform CC band survives every axis of variation tested.

    Layout: a single x-axis (CC overhead on wall_p50, %), three vertically
    stacked row-groups:

        Group A (H1): regime — sequential, streaming, concurrent_c8
                    (GLM-5.1 / TP=8, baseline workload)
        Group B (H2): architecture — GLM-5.1 vs Llama-70B (both TP=8,
                    sequential dispatch)
        Group C (H3a): tensor-parallel size — TP=8 vs TP=1
                    (both Llama-70B, sequential dispatch)

    The GLM-5.1-sequential and Llama-70B-TP=8 points anchor multiple groups
    and are repeated inline for visual clarity, with the anchor flag in the
    row label.
    """
    fig_dir = out_dir / "figures"
    sub = deltas[deltas["metric"] == "wall_p50"].copy()
    if sub.empty:
        return
    sub_idx = sub.set_index(["condition", "regime"])

    def _get(cond: str, regime: str) -> dict | None:
        try:
            r = sub_idx.loc[(cond, regime)]
        except KeyError:
            return None
        return {
            "delta": float(r["rel_delta_pct"]),
            "lo":    float(r["rel_ci_low_pct"]),
            "hi":    float(r["rel_ci_high_pct"]),
            "cond":  cond,
        }

    # Build the three groups. Each row: (label, point-dict, anchor_note).
    groups: list[tuple[str, list[tuple[str, dict, str]]]] = []

    # Group A — regime
    rows_a = []
    for regime in ["sequential", "streaming", "concurrent_c8"]:
        p = _get("baseline", regime)
        if p is None: continue
        label = regime if regime != "concurrent_c8" else "concurrent c=8"
        rows_a.append((label, p, ""))
    if rows_a:
        groups.append(("H1 — regime (GLM-5.1 / TP=8)", rows_a))

    # Group B — architecture (both TP=8, sequential)
    rows_b = []
    for cond, lbl in [("baseline", "GLM-5.1 (MoE)"),
                      ("baseline_70b", "Llama-70B (dense)")]:
        p = _get(cond, "sequential")
        if p is None: continue
        rows_b.append((lbl, p, ""))
    if len(rows_b) >= 2:
        groups.append(("H2 — architecture (TP=8, sequential)", rows_b))

    # Group C — TP size (both Llama-70B, sequential)
    rows_c = []
    for cond, lbl in [("baseline_70b", "TP=8 (multi-GPU)"),
                      ("baseline_70b_tp1", "TP=1 (single-GPU)")]:
        p = _get(cond, "sequential")
        if p is None: continue
        rows_c.append((lbl, p, ""))
    if len(rows_c) >= 2:
        groups.append(("H3a — tensor-parallel size (Llama-70B, sequential)", rows_c))

    if not groups:
        print("  [skip] invariance_panel: insufficient data")
        return

    # Flatten with group separators baked in as None rows.
    flat: list = []   # entries are (label, point | None, group_header | None)
    for g_idx, (header, rows) in enumerate(groups):
        flat.append(("__HEADER__", None, header))
        for (lbl, p, note) in rows:
            flat.append((lbl, p, None))
        # spacer except after the last group
        if g_idx < len(groups) - 1:
            flat.append(("__SPACER__", None, None))

    # Render top-to-bottom in conventional reading order: flip the y indices.
    n_rows = len(flat)
    y_pos = np.arange(n_rows)[::-1]

    fig, ax = plt.subplots(figsize=(10.8, 0.50 * n_rows + 1.8))

    # CC band
    ax.axvspan(CC_BAND_PCT[0], CC_BAND_PCT[1],
               color=PALETTE["muted"], alpha=0.30, zorder=0,
               label=f"platform CC band ({CC_BAND_PCT[0]:.0f}-{CC_BAND_PCT[1]:.0f}%)")
    ax.axvline(0, color=PALETTE["neutral"], linewidth=1.0, linestyle="--",
               alpha=0.6, zorder=1)

    # Track data-min / data-max for the crop
    all_his = []
    all_los = []

    # Build y-tick labels (one per row). Headers are bolded after the fact.
    yticklabels: list[str] = []
    header_rows: list[int] = []   # which y-positions are headers (for bolding)
    spacer_rows: list[int] = []

    for i, (lbl, p, hdr) in enumerate(flat):
        y = int(y_pos[i])
        if lbl == "__HEADER__":
            yticklabels.append(hdr)
            header_rows.append(y)
            continue
        if lbl == "__SPACER__":
            yticklabels.append("")
            spacer_rows.append(y)
            continue
        # data row
        d, lo, hi, cond = p["delta"], p["lo"], p["hi"], p["cond"]
        color = _row_color(cond)
        ax.errorbar([d], [y], xerr=[[d - lo], [hi - d]],
                    fmt="o", capsize=5, color=color, ecolor=color,
                    markersize=8.0, linewidth=2.6, capthick=2.2, zorder=3)
        ax.text(hi + 1.0, y,
                f"{d:.1f}%  [{lo:.1f}, {hi:.1f}]",
                va="center", ha="left", fontsize=11, color=color)
        yticklabels.append(lbl)
        all_his.append(hi); all_los.append(lo)

    if not all_his:
        print("  [skip] invariance_panel: no plottable rows")
        plt.close(fig); return

    # Now register the y-ticks. After matplotlib creates the Text objects we can
    # style the header rows (bold, slightly larger).
    ax.set_yticks(y_pos)
    # We need to feed the labels in the SAME ORDER as y_pos (which is reversed
    # via [::-1] above), so re-zip by sorting on the y values.
    label_by_y = {int(y_pos[i]): yticklabels[i] for i in range(n_rows)}
    ordered_labels = [label_by_y[int(y)] for y in y_pos]
    ax.set_yticklabels(ordered_labels)

    # Bold and left-align the header rows; reduce alpha of spacer-row whitespace.
    header_set = set(header_rows)
    for tick_y, tick_label in zip(y_pos, ax.get_yticklabels()):
        if int(tick_y) in header_set:
            tick_label.set_fontweight("bold")
            tick_label.set_fontsize(11.5)
            tick_label.set_color(PALETTE["annotation"])

    # Crop x-axis. Lower: a bit below the data minimum or 0, whichever is
    # more negative. Upper: room for inline labels.
    x_min = min(0.0, min(all_los) - 2.0)
    x_max = max(all_his) + 18.0
    ax.set_xlim(x_min, x_max)

    ax.set_xlabel(LABEL_WALL_P50_PCT)
    ax.set_title("CC overhead does not depend on regime, model architecture, or tensor-parallel size\n"
                 r"(wall $p_{50}$, paired BCa 95% CIs; shaded = +33-38% empirical platform band)",
                 pad=10, fontsize=12.5)
    # No explicit legend — the title names the shaded band and the colours
    # match the forest/heatmap conventions documented elsewhere.

    ax.tick_params(axis="y", length=0)
    plt.tight_layout()
    _save(fig, fig_dir / "invariance_panel")
    print(f"  wrote {fig_dir / 'invariance_panel.{png,pdf}'}")


def figure_phase_decomposition(deltas: pd.DataFrame, out_dir: Path,
                                df: pd.DataFrame | None = None) -> None:
    """Decoding-dominated CC tax (v2.5: stacked-bar variant).

    Previous version (v2.4) showed only the 4 relative-delta bars (% on a
    bare axis). Information density was low. v2.5 stacks the absolute
    wall_p50 into its two phases (prefill + decode) and renders off vs on
    side-by-side per regime. The visual carries:
      - the absolute structure (prefill is a tiny fraction of the total)
      - the differential under CC (decode segment grows; prefill barely
        moves)
      - inline Δ-ms annotations per phase, in absolute milliseconds

    Falls back to the v2.4 % bars if df is None (i.e. only deltas available).
    """
    fig_dir = out_dir / "figures"
    sub_d = deltas[deltas["metric"].isin(["ttft_p50", "decode_p50", "wall_p50"])
                    & (deltas["condition"] == "baseline")].copy()
    if sub_d.empty:
        print("  no phase-decomposition data (no TTFT cells)")
        return
    regimes = [r for r in ["streaming", "concurrent_c8"] if r in sub_d["regime"].unique()]
    if not regimes:
        return

    # Extract per-regime absolute p50s for prefill (TTFT) and decode (wall − TTFT)
    # Two paths: prefer df (richer; we can compute the on/off p50 directly).
    # Otherwise fall back to deltas off_point / on_point fields.
    rows = []
    for regime in regimes:
        if df is not None:
            for cc in ("off", "on"):
                sub_r = df[(df["condition"] == "baseline") & (df["regime"] == regime)
                           & (df["cc"] == cc)]
                if sub_r.empty or "ttft_seconds" not in sub_r.columns:
                    continue
                ttft = sub_r["ttft_seconds"].dropna().to_numpy()
                wall = sub_r["wall_seconds"].dropna().to_numpy()
                if len(ttft) == 0 or len(wall) == 0:
                    continue
                rows.append({
                    "regime": regime, "cc": cc,
                    "prefill_s":  float(np.median(ttft)),
                    "decode_s":   float(np.median(wall) - np.median(ttft)),
                    "wall_s":     float(np.median(wall)),
                })
        else:
            ttft_row    = sub_d[(sub_d["regime"] == regime) & (sub_d["metric"] == "ttft_p50")]
            dec_row     = sub_d[(sub_d["regime"] == regime) & (sub_d["metric"] == "decode_p50")]
            if ttft_row.empty or dec_row.empty:
                continue
            for cc, col in [("off", "off_point"), ("on", "on_point")]:
                rows.append({
                    "regime": regime, "cc": cc,
                    "prefill_s": float(ttft_row[col].iloc[0]),
                    "decode_s":  float(dec_row[col].iloc[0]),
                    "wall_s":    float(ttft_row[col].iloc[0]) + float(dec_row[col].iloc[0]),
                })

    if not rows:
        print("  [skip] phase_decomposition: no per-cell TTFT data available")
        return

    pdf = pd.DataFrame(rows)
    bar_w = 0.35

    fig, ax = plt.subplots(figsize=(8.6, 4.8))

    x_centers = np.arange(len(regimes))
    color_prefill = PALETTE["off"]   # neutral grey (small contribution)
    color_decode  = PALETTE["on"]    # teal (where the tax lives)
    # Slightly different shades for off vs on so the eye can pair them
    # (CC-off uses lighter / outlined, CC-on uses filled)
    pair_offsets = {"off": -bar_w / 2, "on": +bar_w / 2}

    for i, regime in enumerate(regimes):
        for cc in ("off", "on"):
            row = pdf[(pdf["regime"] == regime) & (pdf["cc"] == cc)]
            if row.empty: continue
            r = row.iloc[0]
            x = x_centers[i] + pair_offsets[cc]
            alpha_face = 0.55 if cc == "off" else 0.95
            edge = PALETTE["annotation"]
            # Prefill at the bottom
            ax.bar(x, r["prefill_s"], bar_w, color=color_prefill,
                   edgecolor=edge, linewidth=0.7, alpha=alpha_face,
                   label="prefill (TTFT)" if (i == 0 and cc == "off") else None,
                   zorder=2)
            # Decode on top
            ax.bar(x, r["decode_s"], bar_w, bottom=r["prefill_s"],
                   color=color_decode, edgecolor=edge, linewidth=0.7,
                   alpha=alpha_face,
                   label="decoding (wall − TTFT)" if (i == 0 and cc == "off") else None,
                   zorder=2)
            # CC state label just under each bar (small, italic)
            ax.text(x, -0.35, f"CC-{cc}", ha="center", va="top",
                    fontsize=8.5, color=PALETTE["annotation"], fontstyle="italic")
            # Total at top
            ax.text(x, r["wall_s"] + 0.06, f"{r['wall_s']:.2f} s",
                    ha="center", va="bottom", fontsize=8.5,
                    color=PALETTE["annotation"], fontweight="bold")

        # Δ-ms annotations between off and on bars
        off_row = pdf[(pdf["regime"] == regime) & (pdf["cc"] == "off")]
        on_row  = pdf[(pdf["regime"] == regime) & (pdf["cc"] == "on")]
        if not off_row.empty and not on_row.empty:
            off_r, on_r = off_row.iloc[0], on_row.iloc[0]
            d_pref_ms = (on_r["prefill_s"] - off_r["prefill_s"]) * 1000
            d_dec_ms  = (on_r["decode_s"]  - off_r["decode_s"])  * 1000
            d_pref_pct = 100 * (on_r["prefill_s"] - off_r["prefill_s"]) / off_r["prefill_s"] \
                         if off_r["prefill_s"] > 0 else np.nan
            d_dec_pct  = 100 * (on_r["decode_s"]  - off_r["decode_s"])  / off_r["decode_s"]  \
                         if off_r["decode_s"] > 0 else np.nan
            # Position labels at mid-height of each phase segment, centered between bars
            mid_x = x_centers[i]
            # prefill labels (low — both off and on prefill are at the bottom)
            mid_y_pref = (off_r["prefill_s"] + on_r["prefill_s"]) / 4
            ax.annotate(f"ΔTTFT = {d_pref_ms:+.0f} ms\n({d_pref_pct:+.1f}%)",
                        xy=(mid_x, mid_y_pref), ha="center", va="center",
                        fontsize=8, color=PALETTE["annotation"],
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                  edgecolor="#CCCCCC", linewidth=0.5, alpha=0.92))
            # decode labels (high — at the centre of the decode segment of on_r)
            mid_y_dec = off_r["prefill_s"] + off_r["decode_s"] / 2
            ax.annotate(f"Δdecode = {d_dec_ms:+.0f} ms\n({d_dec_pct:+.1f}%)",
                        xy=(mid_x, mid_y_dec), ha="center", va="center",
                        fontsize=8, color=PALETTE["annotation"],
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                  edgecolor="#CCCCCC", linewidth=0.5, alpha=0.92))

    ax.set_xticks(x_centers)
    ax.set_xticklabels([r if r != "concurrent_c8" else "concurrent c=8" for r in regimes],
                        fontsize=10, fontweight="bold")
    # Move the regime tick labels down a bit so they sit below the CC-off/on annotations
    ax.tick_params(axis="x", pad=22, length=0)
    ax.set_ylabel("wall_p50 (s) — prefill + decoding")
    ax.set_xlim(-0.5, len(regimes) - 0.5)
    # Generous lower margin for the CC-off/on captions
    cur_lo, cur_hi = ax.get_ylim()
    ax.set_ylim(-0.95, cur_hi + 0.5)
    # No bottom spine clutter under the bars
    ax.spines["bottom"].set_position(("data", 0))

    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("CC tax lands on decoding, not prefill\n"
                 "(wall_p50 stacked into prefill + decoding for CC-off vs CC-on)",
                 pad=10)
    plt.tight_layout()
    _save(fig, fig_dir / "phase_decomposition")
    print(f"  wrote {fig_dir / 'phase_decomposition.{png,pdf}'}")


def figure_paired_debug(df: pd.DataFrame, out_dir: Path) -> None:
    """Paired CC-off vs CC-on scatter grid. Moved to --debug-plots since the
    headline insight is already in the forest plot and heatmap; this is a
    diagnostic of the paired-bootstrap assumption."""
    fig_dir = out_dir / "figures" / "debug"
    nrows, ncols = len(PANEL_GRID), len(PANEL_GRID[0])
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 4.0 * nrows))
    if nrows == 1:
        axes = np.array([axes])

    for r, row in enumerate(PANEL_GRID):
        for c, key in enumerate(row):
            ax = axes[r, c]
            if key is None:
                ax.axis("off")
                continue
            cond, regime = key
            sub = df[(df["condition"] == cond) & (df["regime"] == regime)]
            off = sub[sub["cc"] == "off"].set_index(["pair_id", "prompt_class"])["wall_seconds"]
            on  = sub[sub["cc"] == "on"].set_index(["pair_id", "prompt_class"])["wall_seconds"]
            common = sorted(set(off.index) & set(on.index))
            if len(common) < 2:
                ax.text(0.5, 0.5, "(no paired data)", ha="center", va="center",
                        transform=ax.transAxes, color=PALETTE["neutral"])
                ax.set_title(_disp_label(cond, regime), fontsize=9.5)
                continue
            x_tox = np.array([off.loc[k] for k in common if k[1] == "toxic"])
            y_tox = np.array([on.loc[k]  for k in common if k[1] == "toxic"])
            x_ben = np.array([off.loc[k] for k in common if k[1] == "benign"])
            y_ben = np.array([on.loc[k]  for k in common if k[1] == "benign"])
            all_x = np.concatenate([x_tox, x_ben])
            all_y = np.concatenate([y_tox, y_ben])
            p99_joint = max(np.quantile(all_x, 0.99), np.quantile(all_y, 0.99))
            lim = p99_joint * 1.10

            if len(x_tox):
                ax.scatter(np.clip(x_tox, 0, lim), np.clip(y_tox, 0, lim),
                           alpha=0.6, s=22, color=PALETTE["on"],
                           edgecolor="white", linewidth=0.4,
                           label=f"toxic (n={len(x_tox)})")
            if len(x_ben):
                ax.scatter(np.clip(x_ben, 0, lim), np.clip(y_ben, 0, lim),
                           alpha=0.6, s=22, color=PALETTE["off"],
                           edgecolor="white", linewidth=0.4,
                           label=f"benign (n={len(x_ben)})")

            p50_off = float(np.median(all_x))
            p50_on  = float(np.median(all_y))
            shift_ratio = p50_on / p50_off if p50_off > 0 else 1.0
            xs = np.array([0, lim])
            ax.plot(xs, xs, color=PALETTE["neutral"], linewidth=1.2,
                    linestyle="--", alpha=0.7)
            ax.plot(xs, xs * shift_ratio, color=PALETTE["annotation"],
                    linewidth=1.4, linestyle="--", alpha=0.8,
                    label=f"y = {shift_ratio:.2f}·x")

            n_off_x = int((all_x > lim).sum())
            n_off_y = int((all_y > lim).sum())
            if n_off_x or n_off_y:
                ax.annotate(f"{max(n_off_x, n_off_y)} outlier(s) clipped",
                            xy=(0.97, 0.93), xycoords="axes fraction",
                            ha="right", va="top", fontsize=8,
                            color=PALETTE["annotation"],
                            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                                      edgecolor="#DDDDDD", linewidth=0.5, alpha=0.92))

            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_aspect("equal")
            ax.set_title(f"{_disp_label(cond, regime)}  (n={len(common)})", fontsize=9.5)
            ax.legend(loc="upper left", fontsize=8)
            if c == 0:
                ax.set_ylabel("CC-on wall (s)")
            if r == nrows - 1:
                ax.set_xlabel("CC-off wall (s)")

    fig.suptitle("Paired CC-off vs CC-on wall times (DEBUG)", y=1.005)
    plt.tight_layout()
    _save(fig, fig_dir / "paired_grid")
    print(f"  wrote {fig_dir / 'paired_grid.{png,pdf}'}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                  description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="Directory with one subdir per cell (each with requests.parquet)")
    ap.add_argument("--combined-parquet", type=Path, default=None,
                    help="Single concatenated parquet with a `cell` column")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--n-resamples", type=int, default=10_000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--trim-transient", action="store_true",
                    help="Trim first 12 rows of concurrent_c8 cells (matrix-fill transient)")
    ap.add_argument("--debug-plots", action="store_true",
                    help="Additionally emit per-cell timeline figures (18 files) for "
                         "interactive diagnosis. Not needed for the paper.")
    args = ap.parse_args()

    print(f"Loading data...")
    df = load_cells(args.data_dir, args.combined_parquet)
    print(f"  loaded {len(df)} rows across {df['cell'].nunique()} cells")

    print(f"\nComputing bootstrap CIs (n_resamples={args.n_resamples}, alpha={args.alpha})...")
    deltas_full = compute_deltas(df, n_resamples=args.n_resamples, alpha=args.alpha,
                                  trim_transient_c8=False)
    if args.trim_transient:
        print(f"  also computing trimmed (first {CONCURRENT_TRANSIENT_N} rows of c=8 dropped)...")
        deltas_trim = compute_deltas(df, n_resamples=args.n_resamples, alpha=args.alpha,
                                      trim_transient_c8=True)
        deltas_trim["notes"] = deltas_trim["notes"].fillna("") + " (trimmed)"
        deltas = pd.concat([deltas_full, deltas_trim], ignore_index=True)
    else:
        deltas = deltas_full
    print(f"  computed {len(deltas)} delta rows")

    print(f"\nWriting tables to {args.output_dir}...")
    write_tables(deltas, args.output_dir)

    print(f"\nGenerating figures...")
    figure_forest(deltas_full, args.output_dir)
    figure_invariance_panel(deltas_full, args.output_dir)        # v2.5: merged H1+H2+H3a
    figure_ecdf(df, args.output_dir)
    figure_tail_anomalies(df, args.output_dir)
    figure_delta_heatmap(df, args.output_dir, n_resamples=args.n_resamples, alpha=args.alpha)
    figure_delta_heatmap_exec(df, args.output_dir)
    figure_phase_decomposition(deltas_full, args.output_dir, df=df)  # v2.5: pass df for stacked-bar
    if args.debug_plots:
        figure_paired_debug(df, args.output_dir)
        figure_timelines_debug(df, args.output_dir)
    print(f"  all figures written to {args.output_dir / 'figures'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
