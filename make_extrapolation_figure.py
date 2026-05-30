#!/usr/bin/env python3
"""
make_extrapolation_figure.py — Extrapolation-check figure for the brief's §4.2.

Produces a 3-panel PDF:
  Top:    wall vs realised tok_out on GSM8K + thinking-mode prompts, with
          the linear fit from the max_tokens sweep extending through the
          GSM8K range. Five max_tokens-sweep per-cell median points are
          overlaid as hollow squares to anchor the "extrapolation from
          here to there" framing.
  Middle: per-arm residuals (observed − fit) vs tok_out. Horizontal
          dashed lines mark the per-CC-state mean residuals; mean and
          bias % annotated inline at the right edge of the panel,
          vertically offset (CC-on above its line, CC-off below) to
          prevent collision. |z| > 2.5 outliers annotated.
  Bottom: per-pair Δwall = wall_on − wall_off vs tok_out. Dashed line is
          the predicted delta Δwall = Δa + Δb · tok_out implied by the
          per-arm fits (Δa = +0.33 s, Δb = +63.8 ms/tok). Directly
          visualises the CC-tax extrapolation claim — that the +30%
          decode-asymptote tax holds across the realised-length range.

Filters http_status == 200 (drops timeout failures).

Pairing: tries common pair-key columns (prompt_id, request_id, pair_idx,
prompt_hash) and falls back to positional pairing on the RAW dataframes
if none found. Pairing happens before per-arm status filtering, so a
per-arm timeout doesn't desync positional alignment.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd


# Single-feature OLS coefficients from the max_tokens sweep
# (PHASE3_REFERENCE §3.7, R²=0.9999).
# wall = a + b · (tok_out / 1000), wall in seconds, b in ms/tok.
FIT = {"off": (0.14, 211.4), "on": (0.47, 275.2)}
MAX_TOKENS_SWEEP_CEILING = 958   # p50(tok_out) at max_tokens=2048 in the sweep

# Per-cell median points from the max_tokens sweep (Table 5 in the brief).
SWEEP_POINTS = pd.DataFrame([
    {"tok_out":  32, "wall_off":   6.81, "wall_on":   9.01},
    {"tok_out": 128, "wall_off":  26.33, "wall_on":  34.57},
    {"tok_out": 512, "wall_off": 107.60, "wall_on": 141.00},
    {"tok_out": 607, "wall_off": 127.50, "wall_on": 167.60},
    {"tok_out": 958, "wall_off": 201.40, "wall_on": 262.60},
])

PALETTE = {
    "off":         "#C3C3C3",
    "on":          "#47A5AD",
    "delta":       "#2C2C2C",  # neutral charcoal for the per-pair Δ panel
    "neutral":     "#555555",
    "muted":       "#BBBBBB",
    "annotation":  "#222222",
}

OUTLIER_Z = 2.5


def pair_data(off_df: pd.DataFrame, on_df: pd.DataFrame) -> pd.DataFrame:
    """Return per-pair dataframe with both arms' columns suffixed _off / _on.

    Tries common pair-key columns first; falls back to positional row
    alignment on the RAW data, which is robust to per-arm failures so
    long as both arms' rows were written in matching order.
    """
    candidate_keys = ["prompt_id", "request_id", "pair_idx", "prompt_hash"]
    common = [k for k in candidate_keys
              if k in off_df.columns and k in on_df.columns]

    if common:
        key = common[0]
        merged = off_df.merge(
            on_df, on=key, suffixes=("_off", "_on"), how="inner",
        )
        print(f"  paired on '{key}': {len(merged)} pairs")
        return merged

    # Positional fallback.
    n = min(len(off_df), len(on_df))
    if len(off_df) != len(on_df):
        print(f"  warning: arm lengths differ "
              f"(off={len(off_df)}, on={len(on_df)}); pairing first {n} by row order")
    o = off_df.iloc[:n].reset_index(drop=True)
    x = on_df.iloc[:n].reset_index(drop=True)
    cols_of_interest = ["wall_seconds", "tokens_out", "tokens_in", "http_status"]
    common_cols = [c for c in cols_of_interest
                   if c in o.columns and c in x.columns]
    merged = pd.DataFrame({
        **{f"{c}_off": o[c] for c in common_cols},
        **{f"{c}_on":  x[c] for c in common_cols},
    })
    print(f"  positionally paired (no pair-key column found): {len(merged)} pairs")
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument(
        "--data-dir", type=Path, default=Path("runs/phase3"),
        help="Directory containing the GSM8K-thinking cells "
             "(legacy names: C_R-off/, C_R-on/).",
    )
    ap.add_argument(
        "--out", type=Path, default=Path("figures/tier3g_extrapolation.pdf"),
        help="Output PDF path.",
    )
    args = ap.parse_args()

    off_raw = pd.read_parquet(args.data_dir / "C_R-off" / "requests.parquet")
    on_raw  = pd.read_parquet(args.data_dir / "C_R-on"  / "requests.parquet")
    off_ok = off_raw[off_raw.http_status == 200].copy()
    on_ok  = on_raw[on_raw.http_status  == 200].copy()
    off_ok = off_ok[off_ok.tokens_out > 0]
    on_ok  = on_ok[on_ok.tokens_out  > 0]

    # Pair on the raw data, then filter to pairs where both arms succeeded.
    paired_raw = pair_data(off_raw, on_raw)
    paired = paired_raw[
    (paired_raw.http_status_off == 200) & (paired_raw.http_status_on == 200) &
    (paired_raw.tokens_out_off  > 0) & (paired_raw.tokens_out_on  > 0)].copy()
    paired["delta_wall"] = paired.wall_seconds_on - paired.wall_seconds_off
    # Per the brief, 49/50 pairs have byte-identical realised tok_out across
    # CC states; using the off arm's tok_out as the x-coord is unambiguous.
    paired["tokens_out_pair"] = paired.tokens_out_off
    print(f"  paired-both-OK: {len(paired)}")

    # Per-arm stats for the residual panel.
    stats = {}
    for label, df in [("off", off_ok), ("on", on_ok)]:
        a, b = FIT[label]
        df["predicted"] = a + b * df.tokens_out / 1000.0
        df["residual"]  = df.wall_seconds - df.predicted
        sigma = df.residual.std(ddof=0)
        z = (df.residual - df.residual.mean()) / sigma if sigma > 0 else 0
        df["z_residual"] = z
        df["is_outlier"] = z.abs() > OUTLIER_Z
        bias_pct = 100.0 * df.residual.mean() / df.predicted.mean()
        stats[label] = {
            "n": len(df),
            "mean_residual": df.residual.mean(),
            "bias_pct": bias_pct,
        }

    x_max = float(max(off_ok.tokens_out.max(), on_ok.tokens_out.max())) * 1.05
    x_line = np.linspace(0, x_max, 200)

    fig, (ax_top, ax_mid, ax_bot) = plt.subplots(
        3, 1, figsize=(7.8, 9.2), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0, 1.2], "hspace": 0.10},
    )

    # ====================== TOP: wall vs tok_out ======================
    # Fit lines first (zorder=2), GSM8K dots next, sweep medians on top.
    for label in ("off", "on"):
        a, b = FIT[label]
        c = PALETTE[label]
        ax_top.plot(
            x_line, a + b * x_line / 1000.0,
            "--", color=c, alpha=0.9, linewidth=1.6, zorder=2,
        )
    for label, df in [("off", off_ok), ("on", on_ok)]:
        c = PALETTE[label]
        ax_top.scatter(
            df.tokens_out, df.wall_seconds,
            s=30, alpha=0.78, color=c, edgecolor="white", linewidth=0.5,
            label=f"GSM8K + thinking, CC-{label}  (n = {len(df)})",
            zorder=3,
        )
    for label in ("off", "on"):
        c = PALETTE[label]
        col = "wall_off" if label == "off" else "wall_on"
        ax_top.scatter(
            SWEEP_POINTS["tok_out"], SWEEP_POINTS[col],
            s=80, facecolor="white", edgecolor=c, linewidth=1.8,
            marker="s", zorder=4,
            label=f"max_tokens sweep cell medians, CC-{label}",
        )

    ax_top.axvline(
        MAX_TOKENS_SWEEP_CEILING, color=PALETTE["annotation"],
        linestyle=":", alpha=0.55, linewidth=1.0, zorder=1,
    )

    # Fit-parameter inset (top-left). Plain underscores — no LaTeX escape.
    a_off, b_off = FIT["off"]
    a_on,  b_on  = FIT["on"]
    da, db = a_on - a_off, b_on - b_off
    fit_text = (
        "max_tokens sweep linear fit:  "
        r"wall = $a + b \cdot$ tok_out"
        f"\n   CC-off:  a = {a_off:.2f} s,  b = {b_off:.1f} ms/tok"
        f"\n   CC-on:   a = {a_on:.2f} s,  b = {b_on:.1f} ms/tok"
        f"\n   $\\Delta a$ = {da:+.2f} s,  $\\Delta b$ = {db:+.1f} ms/tok"
    )
    ax_top.text(
        0.025, 0.97, fit_text,
        transform=ax_top.transAxes,
        ha="left", va="top", fontsize=9.0,
        family="DejaVu Sans Mono", color=PALETTE["annotation"],
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                  edgecolor="#CCCCCC", linewidth=0.6, alpha=0.95),
    )

    # Fit-range label — positioned in (data-x, axes-fraction-y) so it sits
    # at a deterministic vertical position regardless of the y-axis limits.
    # y = 0.55 (axes fraction) is below the fit-equation inset (which spans
    # ~0.72–0.97) and above the upper edge of the data cluster.
    fitrange_trans = blended_transform_factory(
        ax_top.transData, ax_top.transAxes,
    )
    ax_top.text(
        MAX_TOKENS_SWEEP_CEILING + 25, 0.55,
        f"fit range\n(tok_out $\\leq$ {MAX_TOKENS_SWEEP_CEILING})",
        transform=fitrange_trans,
        rotation=0, fontsize=8.5, va="center", ha="left",
        color=PALETTE["neutral"],
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor="#DDDDDD", linewidth=0.5, alpha=0.92),
    )

    ax_top.set_ylabel("Wall time (s)")
    ax_top.set_title(
        "Linear fit extrapolates to GSM8K $+$ thinking-mode realised lengths",
        fontsize=11.5, pad=8,
    )
    ax_top.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
    ax_top.grid(True, alpha=0.25)

    # ====================== MIDDLE: residuals ======================
    all_res = np.concatenate([
        off_ok.residual.to_numpy(), on_ok.residual.to_numpy(),
    ])
    res_top_y = max(float(all_res.max()), 5.0) * 1.20
    res_bot_y = min(float(all_res.min()), -5.0) * 1.15
    ax_mid.set_ylim(res_bot_y, res_top_y)

    ax_mid.axhline(
        0, color=PALETTE["annotation"], linestyle="-",
        linewidth=1.3, alpha=0.85, zorder=1,
    )

    # Inline-label transform: x in axes fraction, y in data coords.
    inline_trans = blended_transform_factory(
        ax_mid.transAxes, ax_mid.transData,
    )

    for label, df in [("off", off_ok), ("on", on_ok)]:
        c = PALETTE[label]
        s = stats[label]
        ax_mid.scatter(
            df.tokens_out, df.residual,
            s=24, alpha=0.7, color=c, edgecolor="white", linewidth=0.4,
            zorder=3, label=f"CC-{label}  (n = {s['n']})",
        )
        ax_mid.axhline(
            s["mean_residual"], color=c, linestyle="--",
            linewidth=1.2, alpha=0.80, zorder=2,
        )
        # Inline bias annotation — INSIDE the panel near the right edge,
        # vertically offset away from its line so the two arms' labels
        # never overlap.
        va = "bottom" if s["mean_residual"] >= 0 else "top"
        y_pad = (res_top_y - res_bot_y) * 0.02
        y_text = (s["mean_residual"] + y_pad if va == "bottom"
                  else s["mean_residual"] - y_pad)
        ax_mid.text(
            0.985, y_text,
            f"CC-{label}:  $\\bar r$ = {s['mean_residual']:+.2f} s,  "
            f"bias = {s['bias_pct']:+.2f}%",
            transform=inline_trans, ha="right", va=va,
            fontsize=8.5, color=c, family="DejaVu Sans Mono",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="none", alpha=0.88),
            zorder=4,
        )
        # Outliers
        for _, row in df[df["is_outlier"]].iterrows():
            dx = -120 if row.tokens_out > x_max * 0.6 else +90
            dy_sign = -1 if row.residual > 0 else +1
            dy = dy_sign * (res_top_y - res_bot_y) * 0.15
            ha = "right" if dx < 0 else "left"
            ax_mid.annotate(
                f"|z| = {abs(row.z_residual):.1f}\n({row.residual:+.1f} s)",
                xy=(row.tokens_out, row.residual),
                xytext=(row.tokens_out + dx, row.residual + dy),
                fontsize=8.0, color=PALETTE["annotation"], ha=ha, va="center",
                arrowprops=dict(arrowstyle="->", color=PALETTE["annotation"],
                                linewidth=0.7, alpha=0.75,
                                connectionstyle="arc3,rad=0.0"),
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="#DDDDDD", linewidth=0.5, alpha=0.94),
                zorder=5,
            )

    ax_mid.axvline(
        MAX_TOKENS_SWEEP_CEILING, color=PALETTE["annotation"],
        linestyle=":", alpha=0.45, linewidth=1.0, zorder=1,
    )
    ax_mid.set_ylabel("Residual (s)")
    # Legend in lower-left — empty region (no near-zero residuals at low x
    # extend that far down; outliers are upper-left; inline labels are right).
    ax_mid.legend(loc="lower left", fontsize=9.0, framealpha=0.95)
    ax_mid.grid(True, alpha=0.25)

    # ====================== BOTTOM: per-pair Δwall ======================
    delta_line = da + db * x_line / 1000.0

    ax_bot.plot(
        x_line, delta_line,
        "--", color=PALETTE["delta"], alpha=0.85, linewidth=1.6, zorder=2,
        label=(f"predicted from per-arm fits:  "
               f"$\\Delta$wall = {da:+.2f} s + "
               f"{db:.1f} ms/tok $\\cdot$ tok_out"),
    )
    ax_bot.scatter(
        paired.tokens_out_pair, paired.delta_wall,
        s=30, alpha=0.80, color=PALETTE["delta"], edgecolor="white",
        linewidth=0.5, zorder=3,
        label=f"observed per-pair $\\Delta$wall  (n = {len(paired)})",
    )

    # CC-tax slope annotation right on the predicted line — makes the
    # brief's headline +64 ms/tok number visible without chasing the legend.
    x_anno = x_max * 0.55
    y_anno = da + db * x_anno / 1000.0
    ax_bot.annotate(
        f"slope = $\\Delta b$ = {db:.1f} ms/tok",
        xy=(x_anno, y_anno),
        xytext=(x_anno + 200, y_anno - (db * x_max / 1000.0) * 0.25),
        fontsize=9.0, color=PALETTE["delta"], ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color=PALETTE["delta"],
                        linewidth=0.7, alpha=0.6,
                        connectionstyle="arc3,rad=0.0"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#CCCCCC", linewidth=0.5, alpha=0.95),
    )

    ax_bot.axvline(
        MAX_TOKENS_SWEEP_CEILING, color=PALETTE["annotation"],
        linestyle=":", alpha=0.45, linewidth=1.0, zorder=1,
    )
    ax_bot.set_xlim(0, x_max)
    ax_bot.set_xlabel("Realised $\\mathrm{tokens\\_out}$")
    ax_bot.set_ylabel(r"$\Delta$wall = on $-$ off (s)")
    ax_bot.legend(loc="upper left", fontsize=8.5, framealpha=0.95)
    ax_bot.grid(True, alpha=0.25)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Wrote {args.out}")
    for label in ("off", "on"):
        s = stats[label]
        df = off_ok if label == "off" else on_ok
        print(f"  CC-{label}: n={s['n']}, mean residual={s['mean_residual']:+.3f}s, "
              f"σ={df.residual.std(ddof=0):.3f}s, bias={s['bias_pct']:+.2f}%, "
              f"outliers={int(df.is_outlier.sum())}")
    pred_mean = da + db * paired.tokens_out_pair.mean() / 1000.0
    obs_mean = paired.delta_wall.mean()
    print(f"  Δ panel: n={len(paired)} pairs; observed mean Δwall = {obs_mean:+.2f}s, "
          f"predicted (at mean tok_out) = {pred_mean:+.2f}s")


if __name__ == "__main__":
    main()