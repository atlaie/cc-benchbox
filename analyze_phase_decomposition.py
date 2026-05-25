#!/usr/bin/env python3
"""
analyze_phase_decomposition.py — Tier 1A follow-on.

Re-fits the max_tokens sweep data with a two-feature linear model

    wall = a + b1·tok_in + b2·tok_out

per CC state. Separates the per-input-token (prefill) CC tax (Δb1) from the
per-output-token (decode) CC tax (Δb2), which the single-feature fit in
analyze_max_tokens_sweep.py collapses together.

Why this matters
----------------
The single-feature fit (`wall = a + b·tok_out`) absorbs any per-input-token
cost into the slope b and intercept a, distorted by the correlation between
tok_in and tok_out across the dataset. At the dataset median this is
roughly correct, but it CANNOT be used to extrapolate to long-context
evaluation regimes where tok_in is an order of magnitude larger than what
was measured.

The two-feature fit is unbiased under the assumption that wall is linear
in both tok_in and tok_out, which is the standard prefill+decode model:
  - Prefill processes the full input in one forward pass; per-input-token
    cost is small (parallel, batched).
  - Decode runs sequentially per output token.

The CC delta (paired bootstrap across CC states):
  - Δa   = per-request CC overhead (handshake, page-table setup)
  - Δb1  = per-input-token CC overhead (prefill phase, KV-cache write
            under MKTME during prompt processing)
  - Δb2  = per-output-token CC overhead (decode phase, KV-cache read+write
            under MKTME for each generated token)

Three regimes to interpret the fit:
  - ToxicChat (data we measured): tok_in_p50 ≈ 40, tok_out_p50 ≈ 200
  - AISI long-context eval:       tok_in ≈ 800, tok_out ≈ 500
  - AISI reasoning trace:         tok_in ≈ 800, tok_out ≈ 5000

Inputs
------
    --data-dir DIR           Same as analyze_max_tokens_sweep.
    --cell-id-prefix STR     Default C1.
    --output-dir DIR
    --n-resamples N          Bootstrap iterations (default 10000)
    --alpha A                CI level (default 0.05 → 95% CI)

Outputs (under --output-dir):
    sweep_phase_fit.json                     Two-feature fit parameters with CIs
    sweep_phase_decomposition.md             Markdown summary table
    figures/sweep_phase_decomposition.{png,pdf}  Headline decomposition figure
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reuse companion-module conventions.
from analyze_cc_deltas import (
    PALETTE,
    setup_style,
    _save,
)
from analyze_max_tokens_sweep import (
    discover_cells,
    load_sweep,
)

warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")

setup_style()


# Local labels for new units
LABEL_TOKENS_IN  = r"$\mathrm{tokens\_in}$"
LABEL_TOKENS_OUT = r"$\mathrm{tokens\_out}$"
LABEL_DELTA_WALL_S = r"CC tax on  $t_{\mathrm{wall}}$  (s)"


# ----------------------------------------------------------------------------
# Two-feature paired bootstrap
# ----------------------------------------------------------------------------

@dataclass
class PhaseFit:
    """wall = a + b1·tok_in + b2·tok_out  per CC state.

    All slopes are stored in seconds-per-token; the markdown / JSON output
    converts to ms-per-token for readability.
    """
    # CC-off
    a_off: float;  a_off_ci:  tuple[float, float]
    b1_off: float; b1_off_ci: tuple[float, float]   # per-input-token (prefill)
    b2_off: float; b2_off_ci: tuple[float, float]   # per-output-token (decode)
    # CC-on
    a_on: float;   a_on_ci:   tuple[float, float]
    b1_on: float;  b1_on_ci:  tuple[float, float]
    b2_on: float;  b2_on_ci:  tuple[float, float]
    # Paired differences (computed per-resample for tight CIs)
    delta_a:  float;  delta_a_ci:  tuple[float, float]
    delta_b1: float;  delta_b1_ci: tuple[float, float]
    delta_b2: float;  delta_b2_ci: tuple[float, float]
    # Diagnostics
    r2_off: float
    r2_on:  float
    n_paired_obs: int
    tok_in_range:  tuple[float, float]
    tok_out_range: tuple[float, float]
    method: str

    def to_dict(self) -> dict:
        return {
            "a_off_seconds": self.a_off, "a_off_ci": list(self.a_off_ci),
            "b1_off_ms_per_input_token":  self.b1_off * 1000,
            "b1_off_ci_ms": [self.b1_off_ci[0] * 1000, self.b1_off_ci[1] * 1000],
            "b2_off_ms_per_output_token": self.b2_off * 1000,
            "b2_off_ci_ms": [self.b2_off_ci[0] * 1000, self.b2_off_ci[1] * 1000],
            "a_on_seconds": self.a_on, "a_on_ci": list(self.a_on_ci),
            "b1_on_ms_per_input_token":   self.b1_on * 1000,
            "b1_on_ci_ms":  [self.b1_on_ci[0] * 1000, self.b1_on_ci[1] * 1000],
            "b2_on_ms_per_output_token":  self.b2_on * 1000,
            "b2_on_ci_ms":  [self.b2_on_ci[0] * 1000, self.b2_on_ci[1] * 1000],
            "delta_a_seconds":           self.delta_a,
            "delta_a_ci":                list(self.delta_a_ci),
            "delta_b1_ms_per_input_token":  self.delta_b1 * 1000,
            "delta_b1_ci_ms":               [self.delta_b1_ci[0] * 1000,
                                              self.delta_b1_ci[1] * 1000],
            "delta_b2_ms_per_output_token": self.delta_b2 * 1000,
            "delta_b2_ci_ms":               [self.delta_b2_ci[0] * 1000,
                                              self.delta_b2_ci[1] * 1000],
            "r2_off": self.r2_off,
            "r2_on":  self.r2_on,
            "n_paired_obs": self.n_paired_obs,
            "tok_in_range":  list(self.tok_in_range),
            "tok_out_range": list(self.tok_out_range),
            "method": self.method,
        }


def _lstsq_2feat(x1: np.ndarray, x2: np.ndarray,
                  y: np.ndarray) -> tuple[float, float, float, float]:
    """Closed-form OLS for y = a + b1·x1 + b2·x2. Returns (a, b1, b2, R²)."""
    X = np.column_stack([np.ones_like(x1), x1, x2])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b1, b2 = float(beta[0]), float(beta[1]), float(beta[2])
    y_hat = a + b1 * x1 + b2 * x2
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return a, b1, b2, r2


def fit_phase_decomposition(df: pd.DataFrame, n_resamples: int = 10_000,
                              alpha: float = 0.05) -> PhaseFit:
    """Paired bootstrap of wall = a + b1·tok_in + b2·tok_out per CC state.

    Resampling unit is the (pair_id, prompt_class, max_new_tokens) triple
    paired across CC arms. The Δ CIs reflect within-resample correlation
    between off and on (they share the same prompt), which is much tighter
    than combining marginal CIs.
    """
    off_p = df[df["cc"] == "off"].set_index(
        ["pair_id", "prompt_class", "max_new_tokens"])
    on_p  = df[df["cc"] == "on" ].set_index(
        ["pair_id", "prompt_class", "max_new_tokens"])
    common = sorted(set(off_p.index) & set(on_p.index))
    if len(common) < 10:
        raise RuntimeError(f"too few paired observations: {len(common)}")

    off_w   = off_p.loc[common, "wall_seconds"].to_numpy(dtype=float)
    on_w    = on_p.loc[common, "wall_seconds"].to_numpy(dtype=float)
    tok_in  = off_p.loc[common, "tokens_in"].to_numpy(dtype=float)   # same on both
    tok_out_off = off_p.loc[common, "tokens_out"].to_numpy(dtype=float)
    tok_out_on  = on_p.loc[common, "tokens_out"].to_numpy(dtype=float)

    # Sanity: tok_in should be identical across CC states for the same prompt.
    tok_in_on = on_p.loc[common, "tokens_in"].to_numpy(dtype=float)
    n_mismatch = int(np.sum(tok_in != tok_in_on))
    if n_mismatch:
        warnings.warn(f"tok_in differs between CC states for {n_mismatch} "
                       f"of {len(common)} paired obs; using off-side values")

    a_off_pt, b1_off_pt, b2_off_pt, r2_off = _lstsq_2feat(
        tok_in, tok_out_off, off_w
    )
    a_on_pt,  b1_on_pt,  b2_on_pt,  r2_on  = _lstsq_2feat(
        tok_in_on, tok_out_on, on_w
    )

    rng = np.random.default_rng(0xC0FFEE)
    n = len(common)
    idx = rng.integers(0, n, size=(n_resamples, n))

    a_off_s  = np.empty(n_resamples)
    b1_off_s = np.empty(n_resamples)
    b2_off_s = np.empty(n_resamples)
    a_on_s   = np.empty(n_resamples)
    b1_on_s  = np.empty(n_resamples)
    b2_on_s  = np.empty(n_resamples)

    for i in range(n_resamples):
        ix = idx[i]
        a_off_s[i], b1_off_s[i], b2_off_s[i], _ = _lstsq_2feat(
            tok_in[ix], tok_out_off[ix], off_w[ix]
        )
        a_on_s[i], b1_on_s[i], b2_on_s[i], _ = _lstsq_2feat(
            tok_in_on[ix], tok_out_on[ix], on_w[ix]
        )

    da_s  = a_on_s  - a_off_s
    db1_s = b1_on_s - b1_off_s
    db2_s = b2_on_s - b2_off_s
    q = (alpha / 2, 1 - alpha / 2)

    def _ci(arr: np.ndarray) -> tuple[float, float]:
        return float(np.quantile(arr, q[0])), float(np.quantile(arr, q[1]))

    return PhaseFit(
        a_off=a_off_pt, a_off_ci=_ci(a_off_s),
        b1_off=b1_off_pt, b1_off_ci=_ci(b1_off_s),
        b2_off=b2_off_pt, b2_off_ci=_ci(b2_off_s),
        a_on=a_on_pt, a_on_ci=_ci(a_on_s),
        b1_on=b1_on_pt, b1_on_ci=_ci(b1_on_s),
        b2_on=b2_on_pt, b2_on_ci=_ci(b2_on_s),
        delta_a=float(a_on_pt - a_off_pt), delta_a_ci=_ci(da_s),
        delta_b1=float(b1_on_pt - b1_off_pt), delta_b1_ci=_ci(db1_s),
        delta_b2=float(b2_on_pt - b2_off_pt), delta_b2_ci=_ci(db2_s),
        r2_off=r2_off, r2_on=r2_on,
        n_paired_obs=n,
        tok_in_range=(float(tok_in.min()), float(tok_in.max())),
        tok_out_range=(
            float(min(tok_out_off.min(), tok_out_on.min())),
            float(max(tok_out_off.max(), tok_out_on.max())),
        ),
        method="paired-bootstrap-OLS-2feat",
    )


# ----------------------------------------------------------------------------
# Regime predictions
# ----------------------------------------------------------------------------

@dataclass
class Regime:
    name: str
    tok_in: float
    tok_out: float
    extrapolation: bool   # True if outside the data range


def default_regimes(fit: PhaseFit) -> list[Regime]:
    """Three regimes spanning what was measured and what AISI's deep evals
    look like. Marks each as extrapolation-or-not based on whether its
    (tok_in, tok_out) lies inside the fit's data range."""
    tin_lo, tin_hi = fit.tok_in_range
    tout_lo, tout_hi = fit.tok_out_range

    # Empirical median is anchored; pick the middle of the input range
    # and a middle output length. (Median is fine for narrative.)
    tin_median  = (tin_lo + tin_hi) / 2
    tout_median = (tout_lo + tout_hi) / 2

    return [
        Regime(name="pre-pilot data (median)",
                tok_in=tin_median, tok_out=tout_median,
                extrapolation=False),
        Regime(name="long-context eval\n(tok_in ≈ 800)",
                tok_in=800, tok_out=500,
                extrapolation=(800 > tin_hi or 500 > tout_hi)),
        Regime(name="reasoning trace\n(tok_out ≈ 5000)",
                tok_in=800, tok_out=5000,
                extrapolation=(800 > tin_hi or 5000 > tout_hi)),
    ]


def predict_delta_wall(fit: PhaseFit, tok_in: float,
                        tok_out: float) -> dict:
    """Returns the predicted CC tax at (tok_in, tok_out) decomposed into
    fixed / prefill / decode contributions in seconds. Uses point estimates;
    propagated-CI prediction is overkill at this stage."""
    fixed   = fit.delta_a
    prefill = fit.delta_b1 * tok_in
    decode  = fit.delta_b2 * tok_out
    total   = fixed + prefill + decode
    return {
        "fixed":   fixed,
        "prefill": prefill,
        "decode":  decode,
        "total":   total,
    }


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------

def write_tables(fit: PhaseFit, regimes: list[Regime],
                  out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "sweep_phase_fit.json").write_text(
        json.dumps(fit.to_dict(), indent=2)
    )

    lines: list[str] = [
        "# Phase-decomposed CC overhead — two-feature fit\n",
        "## Fit parameters per CC state\n",
        "| Param | CC-off | CC-on | Δ (on − off) | 95% CI of Δ |",
        "|---|---|---|---|---|",
    ]
    lines.append(
        f"| intercept a (s)                | {fit.a_off:.3f} | "
        f"{fit.a_on:.3f} | {fit.delta_a:+.3f} | "
        f"[{fit.delta_a_ci[0]:+.3f}, {fit.delta_a_ci[1]:+.3f}] |"
    )
    lines.append(
        f"| b1 prefill slope (ms / in-tok) | {fit.b1_off * 1000:.2f} | "
        f"{fit.b1_on * 1000:.2f} | {fit.delta_b1 * 1000:+.2f} | "
        f"[{fit.delta_b1_ci[0] * 1000:+.2f}, "
        f"{fit.delta_b1_ci[1] * 1000:+.2f}] |"
    )
    lines.append(
        f"| b2 decode slope (ms / out-tok) | {fit.b2_off * 1000:.2f} | "
        f"{fit.b2_on * 1000:.2f} | {fit.delta_b2 * 1000:+.2f} | "
        f"[{fit.delta_b2_ci[0] * 1000:+.2f}, "
        f"{fit.delta_b2_ci[1] * 1000:+.2f}] |"
    )
    lines.append("")
    lines.append(
        f"R² off = {fit.r2_off:.4f};  R² on = {fit.r2_on:.4f};  "
        f"n = {fit.n_paired_obs} paired observations;  "
        f"tok_in range [{fit.tok_in_range[0]:.0f}, "
        f"{fit.tok_in_range[1]:.0f}];  "
        f"tok_out range [{fit.tok_out_range[0]:.0f}, "
        f"{fit.tok_out_range[1]:.0f}]."
    )
    lines.append("")
    lines.append("## Predicted CC tax per regime (Δ wall in seconds)\n")
    lines.append("| Regime | tok_in | tok_out | Δa (fixed) | "
                  "Δb1·tok_in (prefill) | Δb2·tok_out (decode) | total |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in regimes:
        p = predict_delta_wall(fit, r.tok_in, r.tok_out)
        tag = " (extrap.)" if r.extrapolation else ""
        name = r.name.replace("\n", " ")
        lines.append(
            f"| {name}{tag} | {r.tok_in:.0f} | {r.tok_out:.0f} | "
            f"{p['fixed']:+.2f} s | {p['prefill']:+.2f} s | "
            f"{p['decode']:+.2f} s | **{p['total']:+.2f} s** |"
        )

    (out_dir / "sweep_phase_decomposition.md").write_text("\n".join(lines))

    print(f"  wrote {out_dir / 'sweep_phase_fit.json'}")
    print(f"  wrote {out_dir / 'sweep_phase_decomposition.md'}")


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------

def figure_phase_decomposition(fit: PhaseFit, regimes: list[Regime],
                                out_dir: Path) -> None:
    """Two-panel headline:

    LEFT  — forest plot of the three Δ parameters with paired-bootstrap 95%
            CIs. Each parameter on its own row with its own units, since
            mixing s/req, ms/in-tok, ms/out-tok on one axis would mislead.
    RIGHT — stacked-bar decomposition at three regimes. Each bar's total
            height is the predicted CC tax in seconds; segments show
            fixed / prefill / decode contributions. Annotated regime labels
            below; extrapolation flagged in italics.
    """
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(13.6, 5.2))
    gs = fig.add_gridspec(
        nrows=3, ncols=2, width_ratios=[0.9, 1.1],
        hspace=0.55, wspace=0.20,
    )
    ax_a  = fig.add_subplot(gs[0, 0])
    ax_b1 = fig.add_subplot(gs[1, 0])
    ax_b2 = fig.add_subplot(gs[2, 0])
    ax_r  = fig.add_subplot(gs[:, 1])

    # ---- LEFT: three small forest panels ----
    def _draw_forest(ax: plt.Axes, point: float,
                       lo: float, hi: float,
                       units: str, label: str,
                       color: str = PALETTE["on"]) -> None:
        ax.axvline(0, color=PALETTE["neutral"], linewidth=1.0,
                    linestyle="--", alpha=0.55, zorder=1)
        ax.errorbar([point], [0],
                     xerr=[[point - lo], [hi - point]],
                     fmt="o", capsize=6, color=color, ecolor=color,
                     markersize=9, linewidth=2.6, capthick=2.2,
                     markeredgecolor="white", markeredgewidth=1.2, zorder=3)
        # Inline numeric label to the right of the upper CI bound.
        ax.text(
            hi, 0.18,
            f"{point:+.2f}  [{lo:+.2f}, {hi:+.2f}]  {units}",
            ha="right", va="bottom", fontsize=10.5,
            color=PALETTE["annotation"],
        )
        span = max(abs(lo), abs(hi))
        ax.set_xlim(-span * 1.15, span * 1.55)
        ax.set_ylim(-0.6, 0.6)
        ax.set_yticks([])
        ax.set_title(label, fontsize=11, loc="left", pad=3)
        ax.tick_params(axis="x", labelsize=9.5)

    _draw_forest(
        ax_a,
        fit.delta_a, fit.delta_a_ci[0], fit.delta_a_ci[1],
        "s",
        r"$\Delta a$  —  fixed CC overhead (s / request)",
    )
    _draw_forest(
        ax_b1,
        fit.delta_b1 * 1000,
        fit.delta_b1_ci[0] * 1000, fit.delta_b1_ci[1] * 1000,
        "ms",
        r"$\Delta b_1$  —  prefill CC tax (ms / input-token)",
    )
    _draw_forest(
        ax_b2,
        fit.delta_b2 * 1000,
        fit.delta_b2_ci[0] * 1000, fit.delta_b2_ci[1] * 1000,
        "ms",
        r"$\Delta b_2$  —  decode CC tax (ms / output-token)",
    )

    # ---- RIGHT: stacked-bar regime decomposition ----
    n_reg = len(regimes)
    x = np.arange(n_reg)
    bar_w = 0.55

    colors = {
        "fixed":   PALETTE["off"],
        "prefill": PALETTE["routing"],     # purple, distinct from decode
        "decode":  PALETTE["on"],
    }

    bottoms = np.zeros(n_reg)
    contributions: dict[str, np.ndarray] = {}
    for key in ("fixed", "prefill", "decode"):
        contributions[key] = np.array([
            predict_delta_wall(fit, r.tok_in, r.tok_out)[key]
            for r in regimes
        ])

    for key, label in [("fixed", "Δa  (fixed)"),
                        ("prefill", "Δb₁·tok_in  (prefill)"),
                        ("decode", "Δb₂·tok_out  (decode)")]:
        vals = contributions[key]
        ax_r.bar(x, vals, bar_w, bottom=bottoms,
                   color=colors[key], edgecolor="white", linewidth=0.7,
                   alpha=0.92, label=label, zorder=2)
        # Inline segment label only if segment is large enough to read
        for xi, vi, b in zip(x, vals, bottoms):
            if vi >= 0.5:    # at least 0.5s to bother labelling
                ax_r.text(xi, b + vi / 2,
                           f"{vi:.1f}s",
                           ha="center", va="center", fontsize=9.5,
                           color=PALETTE["annotation"], fontweight="bold")
        bottoms = bottoms + vals

    # Total at top of each bar
    for xi, total in zip(x, bottoms):
        ax_r.text(xi, total + bottoms.max() * 0.03,
                   f"total {total:+.1f} s",
                   ha="center", va="bottom", fontsize=10.5,
                   color=PALETTE["annotation"], fontweight="bold")

    # X-axis labels: regime names + extrapolation flag
    labels: list[str] = []
    for r in regimes:
        nm = r.name
        if r.extrapolation:
            nm = nm + "\n(extrapolation)"
        labels.append(nm)
    ax_r.set_xticks(x)
    ax_r.set_xticklabels(labels, fontsize=10)
    ax_r.set_ylabel(LABEL_DELTA_WALL_S)
    ax_r.set_xlim(-0.6, n_reg - 0.4)
    ax_r.set_ylim(0, bottoms.max() * 1.18)
    ax_r.legend(loc="upper left", fontsize=10, framealpha=0.95)
    ax_r.set_title("Predicted CC tax decomposition by deployment regime",
                    pad=10, fontsize=12)
    ax_r.tick_params(axis="x", pad=2)

    fig.suptitle(
        "Phase decomposition of the CC tax  —  prefill vs decode\n"
        r"two-feature fit: $t_{\mathrm{wall}}  =  a + b_1 \cdot "
        r"\mathrm{tok}_\mathrm{in} + b_2 \cdot \mathrm{tok}_\mathrm{out}$,"
        "  paired bootstrap on (pair_id, prompt_class, max_tokens)",
        fontsize=11.5, y=1.04,
    )
    _save(fig, fig_dir / "sweep_phase_decomposition")
    print(f"  wrote {fig_dir / 'sweep_phase_decomposition.{png,pdf}'}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--cell-id-prefix", default="C1")
    ap.add_argument("--n-resamples", type=int, default=10_000)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    print(f"Discovering sweep cells under {args.data_dir} "
          f"(prefix={args.cell_id_prefix})...")
    cells = discover_cells(args.data_dir, args.cell_id_prefix)
    if not cells:
        sys.exit(f"FATAL: no cells matching {args.cell_id_prefix}-(off|on)-t* "
                 f"under {args.data_dir}")
    print(f"  found {len(cells)} cells")

    print("\nLoading data...")
    df = load_sweep(cells)
    print(f"  loaded {len(df)} rows  "
          f"({df['cc'].value_counts().to_dict()})")

    # Sanity: do we have meaningful variance in tok_in? The fit needs it.
    tok_in_iqr = float(np.quantile(df["tokens_in"], 0.75)
                        - np.quantile(df["tokens_in"], 0.25))
    tok_in_range = float(df["tokens_in"].max() - df["tokens_in"].min())
    print(f"  tok_in IQR={tok_in_iqr:.0f}, range={tok_in_range:.0f}")
    if tok_in_range < 20:
        print("  [warn] tok_in range is tight; b1 identifiability is weak. "
              "Consider a long-prompt sweep to get a sharper Δb1 estimate.")

    print(f"\nFitting two-feature linear model "
          f"(paired bootstrap, n_resamples={args.n_resamples})...")
    fit = fit_phase_decomposition(df, n_resamples=args.n_resamples,
                                    alpha=args.alpha)
    print(f"  Δa  (fixed)        = {fit.delta_a:+.3f} s "
          f"[{fit.delta_a_ci[0]:+.3f}, {fit.delta_a_ci[1]:+.3f}]")
    print(f"  Δb1 (prefill tax)  = {fit.delta_b1 * 1000:+.3f} ms/in-tok "
          f"[{fit.delta_b1_ci[0] * 1000:+.3f}, "
          f"{fit.delta_b1_ci[1] * 1000:+.3f}]")
    print(f"  Δb2 (decode tax)   = {fit.delta_b2 * 1000:+.3f} ms/out-tok "
          f"[{fit.delta_b2_ci[0] * 1000:+.3f}, "
          f"{fit.delta_b2_ci[1] * 1000:+.3f}]")
    print(f"  R²: off={fit.r2_off:.4f}, on={fit.r2_on:.4f}  "
          f"(n_paired_obs={fit.n_paired_obs})")

    regimes = default_regimes(fit)
    print(f"\nPredicted CC tax per regime:")
    for r in regimes:
        p = predict_delta_wall(fit, r.tok_in, r.tok_out)
        tag = "  [EXTRAPOLATION]" if r.extrapolation else ""
        nm = r.name.replace("\n", " ")
        print(f"  {nm:38}  fixed={p['fixed']:+.2f}s  "
              f"prefill={p['prefill']:+.2f}s  "
              f"decode={p['decode']:+.2f}s  total={p['total']:+.2f}s{tag}")

    print(f"\nWriting outputs to {args.output_dir}...")
    write_tables(fit, regimes, args.output_dir)
    figure_phase_decomposition(fit, regimes, args.output_dir)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
