#!/usr/bin/env python3
"""
make_tier3g_figure.py — Tier 3G extrapolation-check figure.

Produces a 2-panel PDF:
  Top:    wall vs realised tok_out, both CC states, with the Tier 1A
          linear fit extending through the Tier 3G range. Vertical line
          at tok_out=958 marks the Tier 1A measurement boundary.
  Bottom: residuals (observed − fit) vs tok_out. Centered on zero =
          linear fit extrapolates cleanly.

Filters http_status == 200 (drops timeout failures).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Tier 1A two-feature OLS coefficients (PHASE3_REFERENCE §3.7, R²=0.9999).
# y = a + b · (tok_out / 1000), with y in seconds.
TIER1A_FIT = {"off": (0.14, 211.4), "on": (0.47, 275.2)}
TIER1A_MAX_MEASURED_TOK_OUT = 958  # p50 at max_tokens=2048 in the sweep

# Teal / grey palette to roughly match the Pour Demain brief style.
PALETTE = {"off": "#7E8B95", "on": "#1F8A92"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--tier3g-dir", type=Path, default=Path("runs/phase3"),
                    help="Directory containing C_R-off/ and C_R-on/ subdirs.")
    ap.add_argument("--out", type=Path,
                    default=Path("figures/tier3g_extrapolation.pdf"),
                    help="Output PDF path (created if needed).")
    args = ap.parse_args()

    off = pd.read_parquet(args.tier3g_dir / "C_R-off" / "requests.parquet")
    on  = pd.read_parquet(args.tier3g_dir / "C_R-on"  / "requests.parquet")
    off_ok = off[off.http_status == 200].copy()
    on_ok  = on[on.http_status  == 200].copy()

    for label, df in [("off", off_ok), ("on", on_ok)]:
        a, b = TIER1A_FIT[label]
        df["predicted"] = a + b * df.tokens_out / 1000.0
        df["residual"]  = df.wall_seconds - df.predicted

    x_max = float(max(off_ok.tokens_out.max(), on_ok.tokens_out.max())) * 1.05
    x_line = np.linspace(0, x_max, 200)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7.2, 6.0), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.08},
    )

    # ---------- Top: wall vs tok_out ----------
    for label, df in [("off", off_ok), ("on", on_ok)]:
        c = PALETTE[label]
        ax_top.scatter(df.tokens_out, df.wall_seconds,
                       s=28, alpha=0.75, color=c, edgecolor="white",
                       linewidth=0.5, label=f"Tier 3G CC-{label} (n={len(df)})")
        a, b = TIER1A_FIT[label]
        ax_top.plot(x_line, a + b * x_line / 1000.0,
                    "--", color=c, alpha=0.7, linewidth=1.4,
                    label=f"Tier 1A fit CC-{label}: $a={a}+b={b}\\cdot t/1000$")

    ax_top.axvline(TIER1A_MAX_MEASURED_TOK_OUT, color="black",
                   linestyle=":", alpha=0.35, linewidth=1.0)
    ax_top.text(TIER1A_MAX_MEASURED_TOK_OUT + 30, ax_top.get_ylim()[1] * 0.04,
                "Tier 1A max measured", rotation=90, fontsize=8,
                va="bottom", ha="left", color="black", alpha=0.6)

    ax_top.set_ylabel("Wall (seconds)")
    ax_top.set_title("Tier 1A linear fit extrapolates to GSM8K + thinking-mode realised lengths",
                     fontsize=11)
    ax_top.legend(loc="upper left", fontsize=8, frameon=False)
    ax_top.grid(True, alpha=0.25)

    # ---------- Bottom: residuals ----------
    for label, df in [("off", off_ok), ("on", on_ok)]:
        c = PALETTE[label]
        mean_res = df.residual.mean()
        ax_bot.scatter(df.tokens_out, df.residual,
                       s=22, alpha=0.7, color=c, edgecolor="white",
                       linewidth=0.4,
                       label=f"CC-{label}: $\\bar r=${mean_res:+.2f}s")
    ax_bot.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
    ax_bot.axvline(TIER1A_MAX_MEASURED_TOK_OUT, color="black",
                   linestyle=":", alpha=0.35, linewidth=1.0)
    ax_bot.set_xlabel("Realised $\\mathrm{tokens\\_out}$")
    ax_bot.set_ylabel("Residual (s)")
    ax_bot.legend(loc="upper left", fontsize=8, frameon=False)
    ax_bot.grid(True, alpha=0.25)

    ax_bot.set_xlim(0, x_max)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Wrote {args.out}")
    print(f"  CC-off: n={len(off_ok)}, mean residual={off_ok.residual.mean():+.3f}s, "
          f"σ={off_ok.residual.std():.3f}s")
    print(f"  CC-on:  n={len(on_ok)},  mean residual={on_ok.residual.mean():+.3f}s, "
          f"σ={on_ok.residual.std():.3f}s")


if __name__ == "__main__":
    main()