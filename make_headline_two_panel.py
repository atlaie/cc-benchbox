#!/usr/bin/env python3
"""
make_headline_two_panel.py — new headline figure for the brief.

Builds a two-panel figure that argues both empirical findings of the brief
in a single visual:

  LEFT  Platform overhead converges to +30%.
        Wall p50 vs realised tokens_out for CC-off and CC-on (max_tokens
        sweep), with per-CC-state single-feature OLS fits overlaid.
        Annotated with the +30% decode asymptote and the two-feature
        Delta-b2 slope. A small inset forest in an empty corner condenses
        the invariance results (regime / architecture / TP size) to point
        estimates with CIs.

  RIGHT  In-TEE governed-egress pipeline is a net wall saving.
         Stacked layout: top sub-panel shows the cost-benefit
         decomposition (encoder cost vs network savings vs net) on a
         shared wall-delta axis; bottom sub-panel shows the per-replicate
         paired deltas (raw passthrough − full Tier-1, positive = saving)
         plus the pooled Student-t 95% CI, with supporting facts
         (mechanism, determinism) annotated alongside.

Reuses the same palette, save helper, and bootstrap conventions as the
sibling analysis scripts. Inputs are the same as analyze_combined_phase.py
(two sweep directories), so the figure stays in lock-step with the
combined two-feature fit reported in Table 7 of the brief.

Usage:
    python make_headline_two_panel.py \\
        --max-tokens-dir runs/phase3 \\
        --max-tokens-prefix C1 \\
        --tok-in-dir runs/phase3/sweep_tok_in \\
        --tok-in-prefix CI \\
        --output-dir figures

Optional:
    --egress-replicates-csv PATH
        CSV with columns: replicate_id, raw_passthrough_s, full_tier1_s.
        If omitted, uses the values from the overnight summary baked into
        this script (rep1-rep5).

Outputs:
    <output-dir>/headline_two_panel.{png,pdf}
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
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

# Invariance forest: five axes from §3.4 / §3.5 of the brief.
# Each row: (label, point_pct, ci_lo_pct, ci_hi_pct).
# Sourced from Figure 1 / Figure 3 of the brief; paired BCa 95% CIs.
INVARIANCE_FOREST: list[tuple[str, float, float, float]] = [
    ("GLM-MoE  /  sequential",            33.4, 33.0, 34.0),
    ("GLM-MoE  /  streaming",             29.6, 28.7, 30.9),
    ("GLM-MoE  /  concurrent c=8",        31.1, 20.3, 36.2),
    ("Llama-70B dense  /  TP=8",          38.3, 36.7, 39.9),
    ("Llama-70B dense  /  TP=1",          38.1, 36.6, 39.7),
]

# Overnight egress replicate summary (5 paired reps × N=100 each, same TEE
# deployment v0.0.32-egress). Source: overnight.sh run summary.
EGRESS_REPLICATES_DEFAULT: list[dict] = [
    {"replicate_id": "rep1", "raw_passthrough_s": 10.9111, "full_tier1_s": 10.0342},
    {"replicate_id": "rep2", "raw_passthrough_s": 10.8663, "full_tier1_s": 10.0055},
    {"replicate_id": "rep3", "raw_passthrough_s": 10.8841, "full_tier1_s": 10.0053},
    {"replicate_id": "rep4", "raw_passthrough_s": 11.3436, "full_tier1_s":  9.9693},
    {"replicate_id": "rep5", "raw_passthrough_s": 10.9722, "full_tier1_s": 10.0324},
]

# Supporting facts from §4 of the brief.
EGRESS_ENCODER_COST_MS = 238       # server-side compute, Table 10
EGRESS_BUNDLE_KB       = 60        # bounded across 2.7-25 MB raw payloads
EGRESS_RAW_PAYLOAD_MB  = 4.0       # repe_bundle p50 raw payload
EGRESS_COV_FULL_PCT    = 0.25      # full Tier-1 wall coefficient of variation
EGRESS_COV_RAW_PCT     = 1.6       # raw passthrough wall coefficient of variation


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
    decode slope.

    Changes vs v1:
      • Dropped the per-cell marginal-CI wedge fill_between (it was a
        visual artefact of small-N at large max_tokens, not a meaningful
        uncertainty band on the CC delta — pairing tightens the delta
        far below the marginal wedge).
      • Inset positioned in the lower-right corner below both fit lines,
        with an opaque white background so the data is unambiguous.
      • Inset shrunk to ~36% of axes width to leave the upper data clear.
    """
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

    # Single-feature OLS fit lines (1D in tokens_out; dashed).
    x_max_data = float(x.max())
    x_fit = np.linspace(0, x_max_data * 1.05, 200)
    ax.plot(x_fit, fit_1f.a_off + fit_1f.b_off * x_fit,
              "--", color=PALETTE["off"], linewidth=1.3, alpha=0.85, zorder=3)
    ax.plot(x_fit, fit_1f.a_on  + fit_1f.b_on  * x_fit,
              "--", color=PALETTE["on"],  linewidth=1.3, alpha=0.85, zorder=3)

    # +30% decode asymptote annotation at the rightmost measured point.
    anchor_x = x_max_data
    anchor_y_off = fit_1f.a_off + fit_1f.b_off * anchor_x
    anchor_y_on  = fit_1f.a_on  + fit_1f.b_on  * anchor_x
    rel_pct_at_anchor = 100.0 * (anchor_y_on - anchor_y_off) / max(anchor_y_off, 1e-6)

    arrow = FancyArrowPatch(
        (anchor_x * 1.02, anchor_y_off),
        (anchor_x * 1.02, anchor_y_on),
        arrowstyle="<|-|>", mutation_scale=14,
        color=PALETTE["annotation"], linewidth=1.8, zorder=5,
    )
    ax.add_patch(arrow)

    delta_b2_ms = fit_2f.delta_b2 * 1000.0
    # Anchor the annotation BELOW the arrow midpoint so its box doesn't
    # crowd the panel title. Use va="center" so the box is centered on
    # xytext, not extending upward from it.
    ax.annotate(
        f"+{rel_pct_at_anchor:.0f}% decode asymptote\n"
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
    # Extra headroom above the highest data point so the title and any
    # near-top legend / annotation do not crowd.
    y_max = max(on_p50.max(), anchor_y_on) * 1.18
    ax.set_ylim(bottom=0, top=y_max)
    ax.legend(loc="upper left", fontsize=10.5, framealpha=0.96)
    ax.set_title("Platform overhead converges to a +30% decode asymptote",
                   loc="left", fontsize=12.5, pad=12, fontweight="bold")

    # Inset forest in the lower-right corner, sized to sit below both
    # fit lines and the data markers. Opaque white background so any
    # incidental clipping is unambiguous.
    ax_inset = ax.inset_axes(
        [0.60, 0.06, 0.38, 0.34],
        zorder=10,
    )
    ax_inset.set_facecolor("white")
    ax_inset.patch.set_alpha(1.0)
    for spine in ax_inset.spines.values():
        spine.set_edgecolor("#BBBBBB")
        spine.set_linewidth(0.8)
    _draw_invariance_inset(ax_inset, INVARIANCE_FOREST)


def _draw_invariance_inset(ax, forest_rows):
    """Compressed forest of invariance axes. CIs shown as horizontal bars;
    +33-38% band shaded behind. Each row gets a one-line label."""
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
# Right panel: cost-benefit (top) + replicates with mechanism (bottom)
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
    """Pooled Student-t CI on the mean of per-replicate paired deltas
    (raw_passthrough − full_tier1). N=5 is small; Student-t is the
    appropriate inference for the pooled mean."""
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


def _draw_right_top_costbenefit(ax, pooled: EgressPooled):
    """Cost-benefit decomposition as three horizontal bars on a shared
    wall-delta axis. The visual is "encoder cost is dwarfed by the
    network savings; net is a clear positive saving"."""
    # Use the measured pooled median for the net; derive network savings
    # as encoder + net (matches the brief's "exceeded by ~4.6×" claim).
    encoder_cost   = EGRESS_ENCODER_COST_MS                    # +238 ms
    net_saving     = float(pooled.median_ms)                   # +879 ms (measured)
    network_saving = encoder_cost + net_saving                 # = 1117 ms (derived)
    ratio          = network_saving / encoder_cost             # ≈ 4.7×

    # x-axis convention: positive = saving, negative = cost.
    rows = [
        # (label, value_signed, color, alpha, fontweight)
        ("encoder cost",      -encoder_cost,    PALETTE["off"], 0.60, "normal"),
        ("network savings",   +network_saving,  PALETTE["on"],  0.40, "normal"),
        ("net wall saving",   +net_saving,      PALETTE["on"],  1.00, "bold"),
    ]

    for i, (label, val, color, alpha, weight) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh([y], [val], color=color, alpha=alpha, height=0.62,
                  edgecolor="white", linewidth=0.8, zorder=3)
        # Single combined label per bar: "<value> ms  <category>". Anchored
        # outside the bar with ha matching the bar direction, so cost labels
        # extend left and saving labels extend right. This eliminates the
        # dual-text overlap of the previous version (where value and
        # category were two separate text() calls that crowded each other).
        sign = "+" if val >= 0 else "−"
        ha = "left" if val >= 0 else "right"
        offset = 30 if val >= 0 else -30
        combined = f"{sign}{abs(val):.0f} ms  {label}"
        ax.text(val + offset, y, combined,
                  fontsize=10.0, va="center", ha=ha,
                  color=PALETTE["annotation"], fontweight=weight,
                  family="DejaVu Sans Mono")

    ax.axvline(0, color="#888888", linewidth=0.8, linestyle="--",
                 alpha=0.6, zorder=1)
    ax.set_yticks([])
    # Wider symmetric padding to fit the combined "<value> ms <category>"
    # labels without clipping. xmin needs to accommodate "−238 ms encoder
    # cost" extending left of the encoder bar; xmax needs to fit
    # "+1117 ms network savings" extending right of the longest bar.
    xmin = -encoder_cost * 4.5
    xmax = network_saving * 1.45
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.6, len(rows) - 0.4)

    ax.set_title(
        "Tier-1 governed-egress is a net wall saving",
        loc="left", fontsize=12.5, pad=12, fontweight="bold",
    )
    # Sub-info as inline italic line at top-right of the panel.
    ax.text(
        0.99, 0.96,
        f"network savings exceed encoder cost by {ratio:.1f}×",
        transform=ax.transAxes,
        fontsize=9.5, ha="right", va="top", style="italic",
        color=PALETTE["annotation"], alpha=0.85,
    )
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.set_xlabel(r"wall delta (ms):  $\leftarrow$ cost  |  saving  $\rightarrow$",
                    fontsize=8.5, labelpad=2)


def _draw_right_main_replicates(ax, replicates: list[dict], pooled: EgressPooled):
    """Per-replicate dots, pooled mean ± CI, supporting-facts annotation.

    Layout (y, top→bottom):
      n+1 .. 2   five individual replicates
      1          pooled mean ± Student-t 95% CI (square marker, thick CI bar)
      0..-3      empty area beneath; the mechanism + determinism annotation
                 box anchors to the right side
    """
    deltas_ms = pooled.deltas_ms
    n = pooled.n
    y_reps = np.arange(2, n + 2)[::-1]   # rep1 at top
    y_pooled = 1

    ax.axvline(0, color="#888888", linewidth=0.8, linestyle="--",
                 alpha=0.6, zorder=1)

    # Faint trace lines from rep label to data point for easier reading.
    for y, delta in zip(y_reps, deltas_ms):
        ax.plot([0, delta], [y, y], color=PALETTE["on"],
                  alpha=0.18, linewidth=1.0, zorder=2)

    # Individual reps
    for y, delta, rep in zip(y_reps, deltas_ms,
                                [r["replicate_id"] for r in replicates]):
        is_outlier = delta > pooled.median_ms * 1.4
        ax.plot(delta, y, "o", color=PALETTE["on"], markersize=10,
                  markeredgecolor="white", markeredgewidth=1.3,
                  alpha=0.95 if is_outlier else 0.85, zorder=4)
        # Replicate label on the left side at x just past zero
        ax.text(-30, y, rep, fontsize=9.5,
                  color=PALETTE["annotation"], va="center", ha="right",
                  family="DejaVu Sans Mono")
        # Value label to the right of the dot
        suffix = "  (outlier)" if is_outlier else ""
        ax.text(delta + 25, y, f"+{delta:.0f} ms{suffix}",
                  fontsize=9.0, color=PALETTE["annotation"],
                  va="center", ha="left", alpha=0.85,
                  family="DejaVu Sans Mono",
                  fontstyle="italic" if is_outlier else "normal")

    # Pooled mean ± CI
    ax.errorbar([pooled.mean_ms], [y_pooled],
                  xerr=[[pooled.mean_ms - pooled.ci_lo_ms],
                          [pooled.ci_hi_ms - pooled.mean_ms]],
                  fmt="s", markersize=14, color=PALETTE["on"],
                  markerfacecolor=PALETTE["on"],
                  markeredgecolor="white", markeredgewidth=1.6,
                  ecolor=PALETTE["on"], elinewidth=3.4, capsize=8,
                  capthick=2.4, zorder=5)
    ax.text(-30, y_pooled, "pooled", fontsize=10.5, fontweight="bold",
              color=PALETTE["annotation"], va="center", ha="right",
              family="DejaVu Sans Mono")
    # Pooled-stats label BELOW the bracket (va="top"), so it doesn't crowd
    # the rep5 marker above. The previous version anchored above with
    # va="bottom" and overlapped the rep5 row.
    ax.text(
        pooled.mean_ms, y_pooled - 0.55,
        f"median +{pooled.median_ms:.0f},  mean +{pooled.mean_ms:.0f},  "
        f"95% CI [+{pooled.ci_lo_ms:.0f}, +{pooled.ci_hi_ms:.0f}] ms",
        fontsize=9.0, color=PALETTE["annotation"],
        ha="center", va="top",
        family="DejaVu Sans Mono",
        bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                    edgecolor="#CCCCCC", linewidth=0.6, alpha=0.96),
        zorder=6,
    )

    # Supporting facts: two-block annotation below the pooled bracket
    # and its stats label. Vertically separated to avoid overlap.
    mech_text = (
        f"Mechanism:  raw ~{EGRESS_RAW_PAYLOAD_MB:.0f} MB payload  →  "
        f"{EGRESS_BUNDLE_KB} KB signed bundle\n"
        f"   bundle size bounded across realised raw payloads 2.7–25 MB\n"
        f"   (byte-identical at tok128; $\\mathcal{{O}}(1)$ in sequence length)"
    )
    det_text = (
        f"Determinism:  CoV {EGRESS_COV_FULL_PCT}% (full Tier-1)  vs  "
        f"{EGRESS_COV_RAW_PCT}% (raw passthrough)\n"
        f"   → in-TEE pipeline is ~{EGRESS_COV_RAW_PCT / EGRESS_COV_FULL_PCT:.0f}× "
        f"more wall-time-deterministic"
    )
    ax.text(
        -30, -1.7, mech_text,
        fontsize=8.8, color=PALETTE["annotation"],
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=PALETTE["muted"],
                    edgecolor="#CCCCCC", linewidth=0.6, alpha=0.40),
    )
    ax.text(
        -30, -4.0, det_text,
        fontsize=8.8, color=PALETTE["annotation"],
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=PALETTE["muted"],
                    edgecolor="#CCCCCC", linewidth=0.6, alpha=0.40),
    )

    # Y-axis
    ax.set_yticks([])
    ax.set_ylim(-5.6, n + 2.2)

    # X-axis: shared scale with the cost-benefit top panel for visual
    # alignment. Must use the same xmin / xmax expressions as
    # _draw_right_top_costbenefit so the "+net wall saving" bar above
    # lines up with the pooled bracket below at the same data x.
    encoder_cost = EGRESS_ENCODER_COST_MS
    network_saving = encoder_cost + float(pooled.median_ms)
    xmin = -encoder_cost * 4.5
    xmax = network_saving * 1.45
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(
        r"wall delta (ms):  raw passthrough  $-$  full Tier-1  "
        r"(positive = full Tier-1 finishes sooner)",
        fontsize=9.5, labelpad=4,
    )

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", labelsize=8.5)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--max-tokens-dir", type=Path, required=True,
                      help="Dir with <prefix>-<cc>-t<num>/ cells from "
                            "phase3_sweep_max_tokens.py")
    ap.add_argument("--max-tokens-prefix", default="C1")
    ap.add_argument("--tok-in-dir", type=Path, required=True,
                      help="Dir with <prefix>-<cc>-i<num>/ cells from the "
                            "tok_in sweep")
    ap.add_argument("--tok-in-prefix", default="CI")
    ap.add_argument("--max-new-tokens-tok-in", type=int, default=32,
                      help="max_new_tokens value to record for tok_in cells "
                            "(must match the YAML; default 32)")
    ap.add_argument("--egress-replicates-csv", type=Path, default=None,
                      help="Optional CSV with columns: replicate_id, "
                            "raw_passthrough_s, full_tier1_s. Defaults to "
                            "the overnight summary baked into this script.")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--n-resamples", type=int, default=10_000)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ─── Left panel data: max_tokens sweep + combined two-feature fit ───
    print(f"[1/4] discovering max_tokens cells under {args.max_tokens_dir} "
          f"(prefix={args.max_tokens_prefix})")
    mt_cells = discover_max_tokens_cells(args.max_tokens_dir, args.max_tokens_prefix)
    if not mt_cells:
        sys.exit(f"FATAL: no cells matching {args.max_tokens_prefix}-(off|on)-t* "
                  f"under {args.max_tokens_dir}")
    print(f"  found {len(mt_cells)} cells")

    df_mt = load_max_tokens_sweep(mt_cells)
    df_mt["sweep"] = "max_tokens"
    print(f"  loaded {len(df_mt)} rows "
          f"({df_mt['cc'].value_counts().to_dict()})")

    print(f"[2/4] discovering tok_in cells under {args.tok_in_dir} "
          f"(prefix={args.tok_in_prefix})")
    ti_cells = discover_tok_in_cells(args.tok_in_dir, args.tok_in_prefix)
    if not ti_cells:
        sys.exit(f"FATAL: no cells matching {args.tok_in_prefix}-(off|on)-i* "
                  f"under {args.tok_in_dir}")
    print(f"  found {len(ti_cells)} cells")

    df_ti = load_tok_in_sweep(ti_cells, max_new_tokens_fixed=args.max_new_tokens_tok_in)
    df_ti["sweep"] = "tok_in"
    print(f"  loaded {len(df_ti)} rows "
          f"({df_ti['cc'].value_counts().to_dict()})")

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
    print(f"  single-feature  Δa  = {fit_1f.delta_a:+.3f} s "
          f"[{fit_1f.delta_a_ci[0]:+.3f}, {fit_1f.delta_a_ci[1]:+.3f}]")
    print(f"                  Δb  = {fit_1f.delta_b * 1000:+.3f} ms/out-tok "
          f"[{fit_1f.delta_b_ci[0] * 1000:+.3f}, {fit_1f.delta_b_ci[1] * 1000:+.3f}]")
    print(f"  two-feature     Δb₂ = {fit_2f.delta_b2 * 1000:+.3f} ms/out-tok "
          f"[{fit_2f.delta_b2_ci[0] * 1000:+.3f}, {fit_2f.delta_b2_ci[1] * 1000:+.3f}]")

    # ─── Right panel data: egress replicates ────────────────────────────
    replicates = load_egress_replicates(args.egress_replicates_csv)
    pooled = _pool_egress(replicates, alpha=args.alpha)
    print(f"  egress replicates  N={pooled.n}  median={pooled.median_ms:+.0f} ms  "
          f"mean={pooled.mean_ms:+.0f} ms  95% CI "
          f"[{pooled.ci_lo_ms:+.0f}, {pooled.ci_hi_ms:+.0f}] ms")

    # ─── Build figure ───────────────────────────────────────────────────
    print(f"[4/4] rendering figure")
    fig = plt.figure(figsize=(14.4, 6.6))
    gs_outer = fig.add_gridspec(
        nrows=1, ncols=2, width_ratios=[1.32, 1.0], wspace=0.20,
    )

    # LEFT: single axes with embedded inset
    ax_l = fig.add_subplot(gs_outer[0, 0])
    _draw_left_panel(ax_l, stats_list, fit_1f, fit_2f)

    # RIGHT: nested 2-row gridspec (cost-benefit on top, replicates below)
    gs_right = gs_outer[0, 1].subgridspec(
        nrows=2, ncols=1, height_ratios=[0.50, 1.0], hspace=0.55,
    )
    ax_r_top = fig.add_subplot(gs_right[0])
    ax_r_main = fig.add_subplot(gs_right[1])

    _draw_right_top_costbenefit(ax_r_top, pooled)
    _draw_right_main_replicates(ax_r_main, replicates, pooled)

    # No suptitle: the two column-heading set_titles above the LEFT panel
    # and ax_r_top already carry the "two findings" framing. Adding a
    # suptitle on top of that crowds the layout and collides with the
    # right-column heading.

    out_base = args.output_dir / "headline_two_panel"
    _save(fig, out_base)
    print(f"  wrote {out_base}.png")
    print(f"  wrote {out_base}.pdf")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())