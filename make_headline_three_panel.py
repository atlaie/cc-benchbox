#!/usr/bin/env python3
"""
make_headline_three_panel.py — three-finding headline figure for the brief.

Extends the two-panel headline to cover all three empirical findings:

  LEFT   (spans full height)
         Platform overhead converges to +30%.
         Wall p50 vs realised tokens_out for CC-off and CC-on, with
         per-CC-state single-feature OLS fits overlaid. Annotated with
         the +30% decode asymptote and the two-feature Δb₂ slope.
         Invariance inset in the lower-right corner.

  RIGHT-TOP
         Tier-1 governed-egress is a net wall saving.
         Compressed cost-benefit decomposition (encoder cost vs network
         savings vs net) plus pooled Student-t 95% CI — single compact
         panel (no per-replicate dots).

  RIGHT-BOTTOM
         Governance accounting is invisible against the gated computation.
         Log-scale horizontal lollipop chart: four stages from sub-ms
         governance to ~14 s wall. Visual punch: 4 orders of magnitude.

Usage:
    python make_headline_three_panel.py \\
        --max-tokens-dir runs/phase3 \\
        --max-tokens-prefix C1 \\
        --tok-in-dir runs/phase3/sweep_tok_in \\
        --tok-in-prefix CI \\
        --output-dir figures

Outputs:
    <output-dir>/headline_three_panel.{png,pdf}
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

# Make sibling analysis scripts importable when invoked from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_cc_deltas import PALETTE, setup_style, _save  # type: ignore
from analyze_max_tokens_sweep import (  # type: ignore
    discover_cells as discover_max_tokens_cells,
    load_sweep as load_max_tokens_sweep,
    compute_per_cell,
    fit_linear_paired,
)
from analyze_combined_phase import (  # type: ignore
    discover_tok_in_cells,
    load_tok_in_sweep,
)
from analyze_phase_decomposition import fit_phase_decomposition  # type: ignore

setup_style()


# ─────────────────────────────────────────────────────────────────────────────
# Hardcoded supporting data (stable values from the published matrix)
# ─────────────────────────────────────────────────────────────────────────────

# Invariance forest: five axes from §3.3 of the brief.
INVARIANCE_FOREST: list[tuple[str, float, float, float]] = [
    ("GLM-MoE  /  sequential",            33.4, 33.0, 34.0),
    ("GLM-MoE  /  streaming",             29.6, 28.7, 30.9),
    ("GLM-MoE  /  concurrent c=8",        31.1, 20.3, 36.2),
    ("Llama-70B dense  /  TP=8",          38.3, 36.7, 39.9),
    ("Llama-70B dense  /  TP=1",          38.1, 36.6, 39.7),
]

# Overnight egress replicate summary (5 paired reps × N=100).
EGRESS_REPLICATES_DEFAULT: list[dict] = [
    {"replicate_id": "rep1", "raw_passthrough_s": 10.9111, "full_tier1_s": 10.0342},
    {"replicate_id": "rep2", "raw_passthrough_s": 10.8663, "full_tier1_s": 10.0055},
    {"replicate_id": "rep3", "raw_passthrough_s": 10.8841, "full_tier1_s": 10.0053},
    {"replicate_id": "rep4", "raw_passthrough_s": 11.3436, "full_tier1_s":  9.9693},
    {"replicate_id": "rep5", "raw_passthrough_s": 10.9722, "full_tier1_s": 10.0324},
]

# Supporting facts from §4 of the brief.
EGRESS_ENCODER_COST_MS = 238
EGRESS_BUNDLE_KB       = 60

# Governance data from Table 10 / §5 of the brief.
# (label, p50_ms, lo_ms, hi_ms, is_governance_stage)
# Cross-cell ranges from the 4-cell comprehensive run (2 auditors × 2 endpoints):
#   approval p50: [0.80, 0.82] ms; ledger p50: [0.25, 0.28] ms
#   encoder p50: [9030, 9470] ms; wall p50: [13320, 14100] ms
# Governance sum = approval + ledger; range propagated by adding extremes.
GOVERNANCE_STAGES: list[tuple[str, float, float, float, bool]] = [
    ("ledger insert",                0.28,   0.25,   0.28,   True),
    ("approval gate",                0.80,   0.80,   0.82,   True),
    ("governance sum",               1.08,   1.05,   1.10,   True),
    ("encoder + model\nforward pass", 9470.0, 9030.0, 9470.0, False),
    ("wall (end-to-end)",           14050.0, 13320.0, 14100.0, False),
]


def load_egress_replicates(csv_path: Path | None) -> list[dict]:
    if csv_path is None:
        return EGRESS_REPLICATES_DEFAULT
    df = pd.read_csv(csv_path)
    required = {"replicate_id", "raw_passthrough_s", "full_tier1_s"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"FATAL: {csv_path}: missing columns {sorted(missing)}")
    return df.to_dict("records")


# ─────────────────────────────────────────────────────────────────────────────
# Left panel: output-length sweep + decode asymptote + invariance inset
# ─────────────────────────────────────────────────────────────────────────────

def _draw_left_panel(ax, stats_list, fit_1f, fit_2f):
    """Wall vs realised tokens_out, with per-CC-state single-feature OLS fits
    overlaid, annotated with the +30% decode asymptote and the two-feature
    decode slope."""
    sl = sorted(stats_list, key=lambda s: s.tokens_out_p50)
    x = np.array([s.tokens_out_p50 for s in sl])
    off_p50 = np.array([s.wall_p50_off for s in sl])
    on_p50  = np.array([s.wall_p50_on  for s in sl])

    ax.plot(x, off_p50, "-o", color=PALETTE["off"], markersize=8,
            markeredgecolor="white", markeredgewidth=1.2,
            linewidth=2.4, label="CC-off  (paired p50)", zorder=4)
    ax.plot(x, on_p50,  "-o", color=PALETTE["on"],  markersize=8,
            markeredgecolor="white", markeredgewidth=1.2,
            linewidth=2.4, label="CC-on  (paired p50)", zorder=4)

    # Single-feature OLS fit lines.
    x_max_data = float(x.max())
    x_fit = np.linspace(0, x_max_data * 1.05, 200)
    ax.plot(x_fit, fit_1f.a_off + fit_1f.b_off * x_fit,
            "--", color=PALETTE["off"], linewidth=1.3, alpha=0.85, zorder=3)
    ax.plot(x_fit, fit_1f.a_on  + fit_1f.b_on  * x_fit,
            "--", color=PALETTE["on"],  linewidth=1.3, alpha=0.85, zorder=3)

    # +30% decode asymptote annotation.
    anchor_x = x_max_data
    anchor_y_off = fit_1f.a_off + fit_1f.b_off * anchor_x
    anchor_y_on  = fit_1f.a_on  + fit_1f.b_on  * anchor_x
    rel_pct = 100.0 * (anchor_y_on - anchor_y_off) / max(anchor_y_off, 1e-6)

    arrow = FancyArrowPatch(
        (anchor_x * 1.02, anchor_y_off),
        (anchor_x * 1.02, anchor_y_on),
        arrowstyle="<|-|>", mutation_scale=14,
        color=PALETTE["annotation"], linewidth=1.8, zorder=5,
    )
    ax.add_patch(arrow)

    delta_b2_ms = fit_2f.delta_b2 * 1000.0
    ax.annotate(
        f"+{rel_pct:.0f}% decode asymptote\n"
        rf"$\Delta b_2 = +{delta_b2_ms:.1f}$ ms / out-tok",
        xy=(anchor_x * 1.02, (anchor_y_off + anchor_y_on) / 2),
        xytext=(anchor_x * 0.40, (anchor_y_off + anchor_y_on) / 2),
        fontsize=11.0, color=PALETTE["annotation"],
        ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.35",
                  facecolor="white", edgecolor="#CCCCCC",
                  linewidth=0.7, alpha=0.98),
        arrowprops=dict(arrowstyle="->",
                        color=PALETTE["annotation"],
                        lw=1.0, alpha=0.85,
                        connectionstyle="arc3,rad=-0.2"),
        zorder=6,
    )

    ax.set_xlabel(r"realised  $p_{50}(\mathrm{tokens\_out})$")
    ax.set_ylabel(r"wall  $p_{50}(t_{\mathrm{wall}})$  (s)")
    ax.set_xlim(left=0, right=x_max_data * 1.20)
    y_max = max(on_p50.max(), anchor_y_on) * 1.18
    ax.set_ylim(bottom=0, top=y_max)
    ax.legend(loc="upper left", fontsize=10.5, framealpha=0.96)
    ax.set_title("Platform overhead converges to a +30% decode asymptote",
                 loc="left", fontsize=12.5, pad=12, fontweight="bold")

    # Invariance inset.
    ax_inset = ax.inset_axes([0.60, 0.06, 0.38, 0.34], zorder=10)
    ax_inset.set_facecolor("white")
    ax_inset.patch.set_alpha(1.0)
    for spine in ax_inset.spines.values():
        spine.set_edgecolor("#BBBBBB")
        spine.set_linewidth(0.8)
    _draw_invariance_inset(ax_inset, INVARIANCE_FOREST)


def _draw_invariance_inset(ax, forest_rows):
    """Compressed forest of invariance axes."""
    ax.axvspan(33.0, 38.0, color=PALETTE["muted"], alpha=0.35, zorder=1)
    ax.axvline(33.0, color=PALETTE["muted"], linewidth=0.5, alpha=0.7, zorder=2)
    ax.axvline(38.0, color=PALETTE["muted"], linewidth=0.5, alpha=0.7, zorder=2)

    n = len(forest_rows)
    ys = np.arange(n)[::-1]

    for y, (label, pt, lo, hi) in zip(ys, forest_rows):
        ax.errorbar([pt], [y], xerr=[[pt - lo], [hi - pt]],
                    fmt="o", markersize=5.5, color=PALETTE["on"],
                    markerfacecolor=PALETTE["on"],
                    markeredgecolor="white", markeredgewidth=0.9,
                    ecolor=PALETTE["on"], elinewidth=1.6, capsize=3,
                    capthick=1.3, zorder=4)
        ax.text(hi + 1.2, y, f"+{pt:.1f}%", fontsize=7.5,
                color=PALETTE["annotation"], va="center", ha="left",
                family="DejaVu Sans Mono")

    ax.set_yticks(ys)
    ax.set_yticklabels([row[0] for row in forest_rows], fontsize=7.8)
    ax.set_xlim(15, 50)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel(r"$\Delta$ wall p50 (%)", fontsize=8.0, labelpad=2)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.tick_params(axis="y", labelsize=7.8, pad=2)
    ax.set_title("Invariant across regime, architecture, TP size",
                 fontsize=8.5, pad=3, loc="left")


# ─────────────────────────────────────────────────────────────────────────────
# Right-top: compressed egress cost-benefit + pooled CI
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EgressPooled:
    n: int
    deltas_ms: np.ndarray
    median_ms: float
    mean_ms: float
    ci_lo_ms: float
    ci_hi_ms: float


def _pool_egress(replicates: list[dict], alpha: float = 0.05) -> EgressPooled:
    deltas_s = np.array(
        [r["raw_passthrough_s"] - r["full_tier1_s"] for r in replicates],
        dtype=float,
    )
    deltas_ms = deltas_s * 1000.0
    n = len(deltas_ms)
    if n < 2:
        raise RuntimeError("need ≥2 replicates for a CI")
    mean = float(deltas_ms.mean())
    median = float(np.median(deltas_ms))
    sd = float(deltas_ms.std(ddof=1))
    se = sd / np.sqrt(n)
    t = stats.t.ppf(1 - alpha / 2, df=n - 1)
    return EgressPooled(
        n=n, deltas_ms=deltas_ms,
        median_ms=median, mean_ms=mean,
        ci_lo_ms=float(mean - t * se), ci_hi_ms=float(mean + t * se),
    )


def _draw_right_top(ax, pooled: EgressPooled):
    """Compressed egress panel: cost-benefit bars + pooled CI line."""
    encoder_cost   = EGRESS_ENCODER_COST_MS                    # +238 ms
    net_saving     = float(pooled.median_ms)                   # +879 ms
    network_saving = encoder_cost + net_saving                 # = 1117 ms
    ratio          = network_saving / encoder_cost

    rows = [
        ("encoder cost",      -encoder_cost,    PALETTE["off"], 0.60, "normal"),
        ("network savings",   +network_saving,  PALETTE["on"],  0.40, "normal"),
        ("net wall saving",   +net_saving,      PALETTE["on"],  1.00, "bold"),
    ]

    for i, (label, val, color, alpha_val, weight) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh([y], [val], color=color, alpha=alpha_val, height=0.55,
                edgecolor="white", linewidth=0.8, zorder=3)
        sign = "+" if val >= 0 else "−"
        ha = "left" if val >= 0 else "right"
        offset = 30 if val >= 0 else -30
        combined = f"{sign}{abs(val):.0f} ms"
        ax.text(val + offset, y, combined,
                fontsize=9.5, va="center", ha=ha,
                color=PALETTE["annotation"], fontweight=weight,
                family="DejaVu Sans Mono")
        # Category label on the left
        ax.text(-encoder_cost * 5.0, y, label,
                fontsize=9.0, va="center", ha="left",
                color=PALETTE["annotation"], fontweight=weight)

    ax.axvline(0, color="#888888", linewidth=0.8, linestyle="--",
               alpha=0.6, zorder=1)

    # Standalone CI bracket below the bars: median marker + 95% CI.
    y_ci = -0.65
    ax.errorbar(
        [pooled.median_ms], [y_ci],
        xerr=[[pooled.median_ms - pooled.ci_lo_ms],
              [pooled.ci_hi_ms - pooled.median_ms]],
        fmt="s", markersize=6, color=PALETTE["on"],
        markerfacecolor=PALETTE["on"], markeredgecolor="white",
        markeredgewidth=1.0,
        ecolor=PALETTE["on"], elinewidth=2.2,
        capsize=5, capthick=1.8, zorder=5,
    )
    ax.text(
        pooled.ci_hi_ms + 25, y_ci,
        f"median +{pooled.median_ms:.0f} ms,  "
        f"95% CI [{pooled.ci_lo_ms:+.0f}, {pooled.ci_hi_ms:+.0f}]  "
        f"(R={pooled.n} replicates)",
        fontsize=8.5, va="center", ha="left",
        color=PALETTE["annotation"],
        family="DejaVu Sans Mono",
    )

    ax.set_yticks([])
    xmin = -encoder_cost * 5.2
    xmax = max(network_saving, pooled.ci_hi_ms) * 1.60
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-1.1, len(rows) - 0.5)

    ax.set_title(
        "Tier-1 governed-egress is a net wall saving",
        loc="left", fontsize=11.5, pad=8, fontweight="bold",
    )
    # Ratio annotation top-right.
    ax.text(
        0.99, 0.95,
        f"network savings exceed encoder cost by {ratio:.1f}×",
        transform=ax.transAxes,
        fontsize=8.5, ha="right", va="top", style="italic",
        color=PALETTE["annotation"], alpha=0.85,
    )
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", labelsize=8.0)
    ax.set_xlabel(r"wall delta (ms)", fontsize=8.5, labelpad=2)


# ─────────────────────────────────────────────────────────────────────────────
# Right-bottom: governance magnitude (log-scale lollipop)
# ─────────────────────────────────────────────────────────────────────────────

def _draw_right_bottom(ax):
    """Log-scale horizontal lollipop: governance stages vs gated computation.

    The visual argument: governance accounting (≈1 ms) is 4 orders of
    magnitude below the gated computation it wraps (≈9.5 s), and invisible
    against end-to-end wall (≈14 s). Error bars show cross-cell range from
    the 4-cell comprehensive run (Table 10).
    """
    stages = GOVERNANCE_STAGES
    n = len(stages)
    ys = np.arange(n)[::-1]
    vals = [s[1] for s in stages]
    los  = [s[2] for s in stages]
    his  = [s[3] for s in stages]
    is_gov = [s[4] for s in stages]
    labels = [s[0] for s in stages]

    # Shaded band for the governance region (sub-ms to ~1 ms).
    gov_max = max(v for v, g in zip(vals, is_gov) if g) * 2.0
    ax.axvspan(0.01, gov_max, color=PALETTE["muted"], alpha=0.22, zorder=1)

    for y, val, lo, hi, gov, label in zip(ys, vals, los, his, is_gov, labels):
        color = PALETTE["on"] if gov else PALETTE["off"]
        marker_alpha = 1.0 if gov else 0.75
        marker_size = 11 if gov else 9

        # Lollipop stem.
        ax.plot([0.1, val], [y, y], color=color, linewidth=1.8,
                alpha=0.35, zorder=2)
        # Marker with cross-cell error bar.
        ax.errorbar(
            [val], [y],
            xerr=[[val - lo], [hi - val]],
            fmt="o", color=color, markersize=marker_size,
            markeredgecolor="white", markeredgewidth=1.3,
            alpha=marker_alpha,
            ecolor=color, elinewidth=1.8, capsize=4, capthick=1.3,
            zorder=4,
        )

        # Value label to the right of the dot / error bar.
        label_x = hi * 1.35 if hi > val else val * 1.35
        if val < 1.0:
            val_str = f"{val:.2f} ms"
        elif val < 10.0:
            val_str = f"{val:.1f} ms"
        elif val < 1000.0:
            val_str = f"{val:.0f} ms"
        else:
            val_str = f"{val / 1000:.1f} s"

        ax.text(label_x, y, val_str, fontsize=9.5,
                color=PALETTE["annotation"], va="center", ha="left",
                family="DejaVu Sans Mono",
                fontweight="bold" if gov else "normal")

    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=9.0)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 50_000)
    ax.set_ylim(-0.6, n - 0.4)

    # Custom tick labels: 0.1ms, 1ms, 10ms, 100ms, 1s, 10s.
    ax.set_xticks([0.1, 1, 10, 100, 1000, 10000])
    ax.set_xticklabels(["0.1 ms", "1 ms", "10 ms", "100 ms", "1 s", "10 s"],
                       fontsize=8.0)
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.tick_params(axis="y", labelsize=9.0, pad=4)

    # Magnitude annotation: governance sum → encoder bracket.
    gov_sum_y = ys[is_gov.index(True) + 2]   # governance sum row
    # Place the annotation between governance sum and encoder+model.
    ax.annotate(
        "4 orders of magnitude",
        xy=(1.08, gov_sum_y),
        xytext=(50, (gov_sum_y + ys[3]) / 2),
        fontsize=9.5, color=PALETTE["annotation"],
        ha="left", va="center", style="italic",
        arrowprops=dict(arrowstyle="-",
                        color=PALETTE["annotation"],
                        lw=0.8, alpha=0.5),
    )

    ax.set_title(
        "Governance accounting is invisible against gated computation",
        loc="left", fontsize=11.5, pad=8, fontweight="bold",
    )
    # Sub-annotation with the percentage.
    ax.text(
        0.99, 0.95,
        "governance sum ≈ 1.1 ms = 0.008% of wall\n"
        "sub-ms and payload-invariant across 32× payload range",
        transform=ax.transAxes,
        fontsize=8.5, ha="right", va="top", style="italic",
        color=PALETTE["annotation"], alpha=0.85,
    )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--max-tokens-dir", type=Path, required=True)
    ap.add_argument("--max-tokens-prefix", default="C1")
    ap.add_argument("--tok-in-dir", type=Path, required=True)
    ap.add_argument("--tok-in-prefix", default="CI")
    ap.add_argument("--max-new-tokens-tok-in", type=int, default=32)
    ap.add_argument("--egress-replicates-csv", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--n-resamples", type=int, default=10_000)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ─── Left panel data ────────────────────────────────────────────────
    print(f"[1/4] discovering max_tokens cells under {args.max_tokens_dir} "
          f"(prefix={args.max_tokens_prefix})")
    mt_cells = discover_max_tokens_cells(args.max_tokens_dir,
                                         args.max_tokens_prefix)
    if not mt_cells:
        sys.exit(f"FATAL: no cells matching {args.max_tokens_prefix}-(off|on)-t* "
                 f"under {args.max_tokens_dir}")
    print(f"  found {len(mt_cells)} cells")

    df_mt = load_max_tokens_sweep(mt_cells)
    df_mt["sweep"] = "max_tokens"
    print(f"  loaded {len(df_mt)} rows")

    print(f"[2/4] discovering tok_in cells under {args.tok_in_dir} "
          f"(prefix={args.tok_in_prefix})")
    ti_cells = discover_tok_in_cells(args.tok_in_dir, args.tok_in_prefix)
    if not ti_cells:
        sys.exit(f"FATAL: no cells matching {args.tok_in_prefix}-(off|on)-i* "
                 f"under {args.tok_in_dir}")
    print(f"  found {len(ti_cells)} cells")

    df_ti = load_tok_in_sweep(ti_cells,
                              max_new_tokens_fixed=args.max_new_tokens_tok_in)
    df_ti["sweep"] = "tok_in"
    print(f"  loaded {len(df_ti)} rows")

    df_combined = pd.concat([df_mt, df_ti], ignore_index=True)

    print(f"[3/4] fitting models")
    stats_list = compute_per_cell(df_mt, n_resamples=args.n_resamples,
                                 alpha=args.alpha)
    if len(stats_list) < 2:
        sys.exit(f"FATAL: only {len(stats_list)} cells with paired data")

    fit_1f = fit_linear_paired(df_mt, n_resamples=args.n_resamples,
                               alpha=args.alpha)
    fit_2f = fit_phase_decomposition(df_combined,
                                     n_resamples=args.n_resamples,
                                     alpha=args.alpha)
    print(f"  1-feature  Δb  = {fit_1f.delta_b * 1000:+.3f} ms/out-tok")
    print(f"  2-feature  Δb₂ = {fit_2f.delta_b2 * 1000:+.3f} ms/out-tok")

    # ─── Right panel data ───────────────────────────────────────────────
    replicates = load_egress_replicates(args.egress_replicates_csv)
    pooled = _pool_egress(replicates, alpha=args.alpha)
    print(f"  egress: median={pooled.median_ms:+.0f} ms  "
          f"95% CI [{pooled.ci_lo_ms:+.0f}, {pooled.ci_hi_ms:+.0f}] ms")

    # ─── Build figure ───────────────────────────────────────────────────
    print(f"[4/4] rendering figure")
    fig = plt.figure(figsize=(15.5, 8.0))

    # 2-column layout: left spans full height, right splits into two rows.
    gs_outer = fig.add_gridspec(
        nrows=1, ncols=2,
        width_ratios=[1.25, 1.0],
        wspace=0.22,
    )

    # LEFT: single axes spanning full height.
    ax_left = fig.add_subplot(gs_outer[0, 0])
    _draw_left_panel(ax_left, stats_list, fit_1f, fit_2f)

    # RIGHT: two rows.
    gs_right = gs_outer[0, 1].subgridspec(
        nrows=2, ncols=1,
        height_ratios=[0.75, 1.0],
        hspace=0.50,
    )
    ax_rt = fig.add_subplot(gs_right[0])
    ax_rb = fig.add_subplot(gs_right[1])

    _draw_right_top(ax_rt, pooled)
    _draw_right_bottom(ax_rb)

    out_base = args.output_dir / "headline_three_panel"
    _save(fig, out_base)
    print(f"  wrote {out_base}.png / .pdf")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
