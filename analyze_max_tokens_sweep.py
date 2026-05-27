#!/usr/bin/env python3
"""
analyze_max_tokens_sweep.py — Tier 1A analysis.

Consumes per-iteration parquets from `phase3_sweep_max_tokens.py` and
produces the unified amortisation + phase-decomposition figure plus
headline fit numbers.

Two fits are computed on the same paired data:

  - Single-feature: `wall = a + b * tokens_out` per CC state. Gives the
    fixed CC slice (Δa, seconds) and per-token CC slope (Δb, ms/token).
    These are the dashed lines on the left panel of the unified figure
    and the asymptote on the top-right panel.

  - Two-feature:    `wall = a + b1*tokens_in + b2*tokens_out` per CC state.
    Decomposes per-request wall into a fixed slice, a per-input-token
    (prefill) cost, and a per-output-token (decode) cost. The CC deltas
    on the three parameters are the phase-decomposition forest shown
    on the bottom-right panel.

The companion analysis pipeline `analyze_cc_deltas.py` handles the 18-cell
H1/H2/H3a invariance matrix and produces the figures behind §3 of the
technical report. This script handles ONLY the max_tokens sweep — a single
condition × single regime × N max_tokens values — and produces figures
appropriate to a curve, not a categorical matrix. Stylistic conventions
(palette, bootstrap, save helper, math-rendered labels) are imported from
analyze_cc_deltas so the two pipelines render consistently in the report.

Inputs:
    --data-dir DIR
        Directory containing per-iteration subdirs written by
        phase3_sweep_max_tokens.py. Expected layout:
            <data-dir>/
                C1-off-t32/requests.parquet
                C1-off-t32/summary.json
                C1-off-t128/...
                ...
                C1-on-t32/...
                ...
                sweep-c1-off-<ts>.json     (sweep manifest, optional)
                sweep-c1-on-<ts>.json

    --cell-id-prefix C1
        Match the prefix you passed to phase3_sweep_max_tokens.py.

Outputs (under --output-dir):
    sweep_table.csv / sweep_table.md         Per-max_tokens stats with CIs
    sweep_fit.json                           Single-feature fit parameters
    sweep_fit_two_feature.json               Two-feature fit parameters
    figures/sweep_amortization.{png,pdf}     Unified 3-panel headline figure
    figures/sweep_overhead_pct.{png,pdf}     Δ% curve (standalone)
    figures/sweep_eos_distribution.{png,pdf} Diagnostic: tokens_out per cap

Mechanism interpretation (the experimental question):
    If amortisation holds (single-feature reading):
      - a_on  - a_off  > 0       (fixed CC slice per request, in seconds)
      - b_on  - b_off  ≈ 0       (per-token decode is CC-invariant)
      - Δ% drops monotonically from +33% (32 tokens) toward small % at 2048

    If per-token tax holds:
      - a_on  - a_off  ≈ 0
      - b_on  - b_off  > 0       (CC tax accrues per generated token)
      - Δ% stays flat at +33% across the sweep

    Two-feature decomposition separates the per-token tax further:
      - Δb1 ≈ 0 and Δb2 > 0      → decode-phase tax only (memory encryption
                                    of KV-cache traffic on autoregressive
                                    forward passes)
      - Δb1 > 0 and Δb2 > 0      → both prefill and decode pay
      - Δb1 > 0 and Δb2 ≈ 0      → prefill-only tax (unexpected on a TEE
                                    with memory encryption; would suggest
                                    a different mechanism)

The plots make the distinction visible without needing to read the fit.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import DegenerateDataWarning

# Reuse companion module's conventions. Keeps the two pipelines visually
# coherent and avoids re-implementing the bootstrap.
from analyze_cc_deltas import (
    PALETTE,
    setup_style,
    bootstrap_paired_delta,
    _save,
    _median,
    _quantile,
    LABEL_WALL_P50_S,
    LABEL_WALL_P50_PCT,
)

warnings.filterwarnings("ignore", category=DegenerateDataWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")

setup_style()


CELL_PATTERN = re.compile(r"^(?P<prefix>[A-Za-z0-9]+)-(?P<cc>off|on)-t(?P<max_tokens>\d+)$")

# Required columns. Matches what phase3_vllm_driver writes.
# `tokens_in` is required for the two-feature (prefill vs decode) fit.
REQUIRED_COLS = {"pair_id", "prompt_class", "wall_seconds",
                 "tokens_in", "tokens_out", "payload_bytes"}


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------

@dataclass
class SweepCell:
    cell_id: str
    cc: str
    max_new_tokens: int
    path: Path

def load_ttft_cells(data_dir: Path) -> pd.DataFrame | None:
    """Load baseline streaming + concurrent_c8 cells for the empirical
    phase-split panel of figure_amortization. Returns None if the
    directory has no usable cells."""
    if not data_dir.exists():
        return None
    frames = []
    specs = [
        ("baseline", "streaming",     ["C1-off-stream", "C1-on-stream"]),
        ("baseline", "concurrent_c8", ["C1-off-c8",     "C1-on-c8"]),
    ]
    for condition, regime, cells in specs:
        for cid in cells:
            p = data_dir / cid / "requests.parquet"
            if not p.exists():
                continue
            df = pd.read_parquet(p)
            df["condition"] = condition
            df["regime"]    = regime
            df["cc"]        = "off" if "-off" in cid else "on"
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None

def discover_cells(data_dir: Path, prefix: str) -> list[SweepCell]:
    """Find all per-iteration subdirs under data_dir whose names match the
    sweep cell pattern. Returns sorted-by-(cc, max_tokens) for stable output."""
    cells: list[SweepCell] = []
    for sub in sorted(data_dir.iterdir()):
        if not sub.is_dir():
            continue
        m = CELL_PATTERN.match(sub.name)
        if not m or m.group("prefix") != prefix:
            continue
        parquet = sub / "requests.parquet"
        if not parquet.exists():
            warnings.warn(f"{sub.name}: no requests.parquet, skipping")
            continue
        cells.append(SweepCell(
            cell_id=sub.name,
            cc=m.group("cc"),
            max_new_tokens=int(m.group("max_tokens")),
            path=sub,
        ))
    cells.sort(key=lambda c: (c.cc, c.max_new_tokens))
    return cells


def load_sweep(cells: list[SweepCell]) -> pd.DataFrame:
    """Concatenate all sweep cells into one long DataFrame with cc and
    max_new_tokens columns."""
    if not cells:
        sys.exit("FATAL: no sweep cells discovered")

    frames: list[pd.DataFrame] = []
    for c in cells:
        df = pd.read_parquet(c.path / "requests.parquet")
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            sys.exit(f"FATAL: {c.cell_id}: missing required columns {sorted(missing)}")
        df = df.copy()
        df["cell_id"] = c.cell_id
        df["cc"] = c.cc
        df["max_new_tokens"] = c.max_new_tokens
        # Verify the summary.json max_new_tokens matches the cell_id (catches
        # operator mistakes like running phase3_run_cell.py on a sweep cell with
        # the default max_new_tokens).
        smry_path = c.path / "summary.json"
        if smry_path.exists():
            try:
                smry = json.loads(smry_path.read_text())
                if "max_new_tokens" in smry and int(smry["max_new_tokens"]) != c.max_new_tokens:
                    warnings.warn(
                        f"{c.cell_id}: summary.max_new_tokens={smry['max_new_tokens']} "
                        f"disagrees with cell_id={c.max_new_tokens}; trusting cell_id"
                    )
            except Exception:
                pass
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    # Sanity reports
    for max_tok in sorted(df["max_new_tokens"].unique()):
        sub = df[df["max_new_tokens"] == max_tok]
        n_off = int((sub["cc"] == "off").sum())
        n_on  = int((sub["cc"] == "on").sum())
        if n_off == 0 or n_on == 0:
            warnings.warn(f"max_tokens={max_tok}: missing arm "
                          f"(n_off={n_off}, n_on={n_on}); will be excluded "
                          f"from paired analyses")
    return df


# ----------------------------------------------------------------------------
# Per-cell statistics
# ----------------------------------------------------------------------------

@dataclass
class CellStats:
    max_new_tokens: int
    n_off: int
    n_on: int
    n_paired: int
    wall_p50_off: float
    wall_p50_on: float
    abs_delta_s: float
    abs_ci_low: float
    abs_ci_high: float
    rel_delta_pct: float
    rel_ci_low_pct: float
    rel_ci_high_pct: float
    tokens_out_p50_off: float
    tokens_out_p50_on: float
    tokens_out_p50: float                    # median of (off, on)
    per_token_overhead_ms: float             # paired median of per-pair (Δwall)/tokens_out
    per_token_ci_low_ms: float
    per_token_ci_high_ms: float
    n_eos_early_off: int                     # # prompts that EOSed before cap (off)
    n_eos_early_on: int


def _per_token_overhead_ms(off_w: np.ndarray, on_w: np.ndarray,
                            tokens_out: np.ndarray) -> float:
    """Paired-median of (on - off) wall in ms divided by per-pair tokens_out.
    Returns nan if no positive-token pairs."""
    mask = tokens_out > 0
    if not mask.any():
        return float("nan")
    return float(np.median((on_w[mask] - off_w[mask]) * 1000.0 / tokens_out[mask]))


def compute_per_cell(df: pd.DataFrame, n_resamples: int = 10_000,
                      alpha: float = 0.05) -> list[CellStats]:
    """For each max_tokens value present with both CC arms, compute the
    full stats set with paired BCa CIs."""
    out: list[CellStats] = []
    for max_tok in sorted(df["max_new_tokens"].unique()):
        sub = df[df["max_new_tokens"] == max_tok]
        off = sub[sub["cc"] == "off"]
        on  = sub[sub["cc"] == "on"]
        if len(off) == 0 or len(on) == 0:
            continue

        off_p = off.set_index(["pair_id", "prompt_class"])
        on_p  = on.set_index(["pair_id", "prompt_class"])
        common = sorted(set(off_p.index) & set(on_p.index))
        if len(common) < 2:
            warnings.warn(f"max_tokens={max_tok}: fewer than 2 paired rows; "
                          f"skipping bootstrap")
            continue

        off_w = off_p.loc[common, "wall_seconds"].to_numpy(dtype=float)
        on_w  = on_p.loc[common, "wall_seconds"].to_numpy(dtype=float)
        off_tok = off_p.loc[common, "tokens_out"].to_numpy(dtype=float)
        on_tok  = on_p.loc[common, "tokens_out"].to_numpy(dtype=float)
        # At temperature=0 these should match exactly across CC states. Use the
        # off-side value for the per-token denominator (any divergence is a
        # sampler / scheduler artefact and should be tiny).
        tokens_for_denom = off_tok

        # Marginal medians (unpaired): used for table display + plot anchors.
        wall_p50_off = float(np.median(off["wall_seconds"].dropna()))
        wall_p50_on  = float(np.median(on["wall_seconds"].dropna()))
        tok_p50_off  = float(np.median(off["tokens_out"].dropna()))
        tok_p50_on   = float(np.median(on["tokens_out"].dropna()))
        tok_p50_joint = float(np.median(np.concatenate([off_tok, on_tok])))

        # Paired BCa on wall_p50 delta (matches analyze_cc_deltas convention).
        point, lo, hi = bootstrap_paired_delta(
            off_w, on_w, _quantile(0.5),
            n_resamples=n_resamples, alpha=alpha,
        )

        rel = 100 * point / wall_p50_off if wall_p50_off > 0 else float("nan")
        rel_lo = 100 * lo / wall_p50_off if wall_p50_off > 0 else float("nan")
        rel_hi = 100 * hi / wall_p50_off if wall_p50_off > 0 else float("nan")

        # Per-pair per-token overhead: bootstrap the paired median directly.
        rng = np.random.default_rng(0xC0DECC)
        per_pair = (on_w - off_w) * 1000.0 / np.maximum(tokens_for_denom, 1.0)
        pt_point = float(np.median(per_pair))
        # Plain percentile bootstrap on the median (BCa overkill here).
        idx = rng.integers(0, len(per_pair), size=(n_resamples, len(per_pair)))
        pt_samples = np.median(per_pair[idx], axis=1)
        pt_lo = float(np.quantile(pt_samples, alpha / 2))
        pt_hi = float(np.quantile(pt_samples, 1 - alpha / 2))

        n_eos_off = int((off["tokens_out"] < max_tok).sum())
        n_eos_on  = int((on["tokens_out"]  < max_tok).sum())

        out.append(CellStats(
            max_new_tokens=int(max_tok),
            n_off=len(off), n_on=len(on), n_paired=len(common),
            wall_p50_off=wall_p50_off, wall_p50_on=wall_p50_on,
            abs_delta_s=point, abs_ci_low=lo, abs_ci_high=hi,
            rel_delta_pct=rel, rel_ci_low_pct=rel_lo, rel_ci_high_pct=rel_hi,
            tokens_out_p50_off=tok_p50_off, tokens_out_p50_on=tok_p50_on,
            tokens_out_p50=tok_p50_joint,
            per_token_overhead_ms=pt_point,
            per_token_ci_low_ms=pt_lo, per_token_ci_high_ms=pt_hi,
            n_eos_early_off=n_eos_off, n_eos_early_on=n_eos_on,
        ))
    return out


# ----------------------------------------------------------------------------
# Linear fit: wall = a + b * tokens_out, per CC state
# ----------------------------------------------------------------------------

@dataclass
class FitResult:
    a_off: float
    b_off: float
    a_off_ci: tuple[float, float]
    b_off_ci: tuple[float, float]
    a_on: float
    b_on: float
    a_on_ci: tuple[float, float]
    b_on_ci: tuple[float, float]
    delta_a: float                # fixed CC slice (seconds per request)
    delta_a_ci: tuple[float, float]
    delta_b: float                # per-token CC slope (seconds per token)
    delta_b_ci: tuple[float, float]
    n_paired_pairs: int
    method: str

    def to_dict(self) -> dict:
        return {
            "a_off": self.a_off, "b_off": self.b_off,
            "a_off_ci": list(self.a_off_ci), "b_off_ci": list(self.b_off_ci),
            "a_on": self.a_on, "b_on": self.b_on,
            "a_on_ci": list(self.a_on_ci), "b_on_ci": list(self.b_on_ci),
            "delta_a_seconds":  self.delta_a,
            "delta_a_ci":       list(self.delta_a_ci),
            "delta_b_seconds_per_token": self.delta_b,
            "delta_b_ci":       list(self.delta_b_ci),
            "delta_b_ms_per_token":      self.delta_b * 1000.0,
            "delta_b_ms_per_token_ci":   [self.delta_b_ci[0] * 1000.0,
                                          self.delta_b_ci[1] * 1000.0],
            "n_paired_pairs":   self.n_paired_pairs,
            "method":           self.method,
        }


def fit_linear_paired(df: pd.DataFrame, n_resamples: int = 10_000,
                       alpha: float = 0.05) -> FitResult:
    """Bootstrap-fit `wall = a + b * tokens_out` per CC state on all paired
    data across the sweep. Resampling unit is the (pair_id, prompt_class,
    max_new_tokens) triple — i.e., a single request observation, paired
    across CC states.

    Returns intercept/slope point estimates and 95% CIs for each CC state,
    plus the paired differences (delta_a = a_on - a_off,
    delta_b = b_on - b_off) computed per-resample so their CIs reflect
    the within-resample correlation between CC states."""

    # Build paired arrays across the full sweep.
    off_p = df[df["cc"] == "off"].set_index(
        ["pair_id", "prompt_class", "max_new_tokens"])
    on_p  = df[df["cc"] == "on"].set_index(
        ["pair_id", "prompt_class", "max_new_tokens"])
    common = sorted(set(off_p.index) & set(on_p.index))
    if len(common) < 5:
        raise RuntimeError(f"too few paired observations for linear fit "
                           f"(got {len(common)})")

    off_w   = off_p.loc[common, "wall_seconds"].to_numpy(dtype=float)
    on_w    = on_p.loc[common, "wall_seconds"].to_numpy(dtype=float)
    tok_off = off_p.loc[common, "tokens_out"].to_numpy(dtype=float)
    tok_on  = on_p.loc[common, "tokens_out"].to_numpy(dtype=float)
    # Use realised tokens_out per side (these are nearly identical at temp=0
    # but not guaranteed — sampler/scheduler can occasionally drop one token).

    def _lsq(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        # closed-form OLS; faster than scipy.linregress in a bootstrap loop
        n = len(x)
        sx, sy = x.sum(), y.sum()
        sxx = (x * x).sum()
        sxy = (x * y).sum()
        denom = n * sxx - sx * sx
        if denom == 0:
            return float("nan"), float("nan")
        b = (n * sxy - sx * sy) / denom
        a = (sy - b * sx) / n
        return float(a), float(b)

    a_off_pt, b_off_pt = _lsq(tok_off, off_w)
    a_on_pt,  b_on_pt  = _lsq(tok_on,  on_w)

    # Bootstrap. Resample paired indices with replacement.
    rng = np.random.default_rng(0xF17F17)
    n = len(common)
    idx = rng.integers(0, n, size=(n_resamples, n))
    a_off_s = np.empty(n_resamples)
    b_off_s = np.empty(n_resamples)
    a_on_s  = np.empty(n_resamples)
    b_on_s  = np.empty(n_resamples)
    for i in range(n_resamples):
        ix = idx[i]
        a_off_s[i], b_off_s[i] = _lsq(tok_off[ix], off_w[ix])
        a_on_s[i],  b_on_s[i]  = _lsq(tok_on[ix],  on_w[ix])

    da_s = a_on_s - a_off_s
    db_s = b_on_s - b_off_s
    q = (alpha / 2, 1 - alpha / 2)

    return FitResult(
        a_off=a_off_pt, b_off=b_off_pt,
        a_off_ci=(float(np.quantile(a_off_s, q[0])), float(np.quantile(a_off_s, q[1]))),
        b_off_ci=(float(np.quantile(b_off_s, q[0])), float(np.quantile(b_off_s, q[1]))),
        a_on=a_on_pt,   b_on=b_on_pt,
        a_on_ci=(float(np.quantile(a_on_s, q[0])), float(np.quantile(a_on_s, q[1]))),
        b_on_ci=(float(np.quantile(b_on_s, q[0])), float(np.quantile(b_on_s, q[1]))),
        delta_a=float(a_on_pt - a_off_pt),
        delta_a_ci=(float(np.quantile(da_s, q[0])), float(np.quantile(da_s, q[1]))),
        delta_b=float(b_on_pt - b_off_pt),
        delta_b_ci=(float(np.quantile(db_s, q[0])), float(np.quantile(db_s, q[1]))),
        n_paired_pairs=n, method="paired-bootstrap-OLS",
    )


# ----------------------------------------------------------------------------
# Two-feature fit: wall = a + b1 * tokens_in + b2 * tokens_out, per CC state
# ----------------------------------------------------------------------------
#
# The single-feature fit above collapses any per-input-token (prefill) CC
# cost into the intercept by way of the correlation between tokens_in and
# tokens_out across the sweep. ToxicChat prompt lengths vary across an
# order of magnitude (tokens_in roughly 12..148), enough to identify a
# prefill slope separately. The two-feature fit decomposes per-request
# wall into a fixed slice (`a`), a per-input-token cost (`b1`, prefill),
# and a per-output-token cost (`b2`, decode). The CC delta on each
# parameter is the headline phase-decomposition result.

@dataclass
class TwoFeatureFitResult:
    a_off: float
    a_off_ci: tuple[float, float]
    b1_off: float                              # prefill slope (s / input token)
    b1_off_ci: tuple[float, float]
    b2_off: float                              # decode slope  (s / output token)
    b2_off_ci: tuple[float, float]
    a_on: float
    a_on_ci: tuple[float, float]
    b1_on: float
    b1_on_ci: tuple[float, float]
    b2_on: float
    b2_on_ci: tuple[float, float]
    delta_a: float                             # fixed CC slice (s)
    delta_a_ci: tuple[float, float]
    delta_b1: float                            # prefill CC slope (s / in-tok)
    delta_b1_ci: tuple[float, float]
    delta_b2: float                            # decode CC slope  (s / out-tok)
    delta_b2_ci: tuple[float, float]
    r2_off: float
    r2_on: float
    n_paired_pairs: int
    method: str

    def to_dict(self) -> dict:
        return {
            "a_off": self.a_off, "a_off_ci": list(self.a_off_ci),
            "b1_off_s_per_in_tok":  self.b1_off,
            "b1_off_ms_per_in_tok": self.b1_off * 1000.0,
            "b1_off_ci": list(self.b1_off_ci),
            "b2_off_s_per_out_tok":  self.b2_off,
            "b2_off_ms_per_out_tok": self.b2_off * 1000.0,
            "b2_off_ci": list(self.b2_off_ci),
            "a_on": self.a_on, "a_on_ci": list(self.a_on_ci),
            "b1_on_s_per_in_tok":  self.b1_on,
            "b1_on_ms_per_in_tok": self.b1_on * 1000.0,
            "b1_on_ci": list(self.b1_on_ci),
            "b2_on_s_per_out_tok":  self.b2_on,
            "b2_on_ms_per_out_tok": self.b2_on * 1000.0,
            "b2_on_ci": list(self.b2_on_ci),
            "delta_a_seconds":  self.delta_a,
            "delta_a_ci":       list(self.delta_a_ci),
            "delta_b1_seconds_per_in_tok": self.delta_b1,
            "delta_b1_ms_per_in_tok":      self.delta_b1 * 1000.0,
            "delta_b1_ci":                 list(self.delta_b1_ci),
            "delta_b1_ms_per_in_tok_ci":   [self.delta_b1_ci[0] * 1000.0,
                                            self.delta_b1_ci[1] * 1000.0],
            "delta_b2_seconds_per_out_tok": self.delta_b2,
            "delta_b2_ms_per_out_tok":      self.delta_b2 * 1000.0,
            "delta_b2_ci":                  list(self.delta_b2_ci),
            "delta_b2_ms_per_out_tok_ci":   [self.delta_b2_ci[0] * 1000.0,
                                              self.delta_b2_ci[1] * 1000.0],
            "r2_off": self.r2_off, "r2_on": self.r2_on,
            "n_paired_pairs": self.n_paired_pairs,
            "method": self.method,
        }


def fit_two_feature_paired(df: pd.DataFrame, n_resamples: int = 10_000,
                            alpha: float = 0.05) -> TwoFeatureFitResult:
    """Bootstrap-fit `wall = a + b1 * tokens_in + b2 * tokens_out` per CC
    state on all paired data across the sweep. Resampling unit is the
    (pair_id, prompt_class, max_new_tokens) triple, paired across CC
    states (same as the single-feature fit).

    The paired differences (delta_a, delta_b1, delta_b2) are computed
    per-resample so their CIs reflect within-resample correlation between
    CC states. Also reports per-CC R² on the full data (no resampling).
    """

    off_p = df[df["cc"] == "off"].set_index(
        ["pair_id", "prompt_class", "max_new_tokens"])
    on_p  = df[df["cc"] == "on"].set_index(
        ["pair_id", "prompt_class", "max_new_tokens"])
    common = sorted(set(off_p.index) & set(on_p.index))
    if len(common) < 10:
        raise RuntimeError(f"too few paired observations for two-feature fit "
                           f"(got {len(common)})")

    off_w        = off_p.loc[common, "wall_seconds"].to_numpy(dtype=float)
    on_w         = on_p.loc[common,  "wall_seconds"].to_numpy(dtype=float)
    tok_in_off   = off_p.loc[common, "tokens_in"].to_numpy(dtype=float)
    tok_in_on    = on_p.loc[common,  "tokens_in"].to_numpy(dtype=float)
    tok_out_off  = off_p.loc[common, "tokens_out"].to_numpy(dtype=float)
    tok_out_on   = on_p.loc[common,  "tokens_out"].to_numpy(dtype=float)

    # Check identifiability: input-token variation must span enough range to
    # separate the prefill slope from the intercept.
    in_range = (tok_in_off.max() - tok_in_off.min())
    if in_range < 10:
        warnings.warn(f"two-feature fit: tokens_in range is only {in_range:.0f}; "
                      f"prefill slope (b1) will be poorly identified")

    def _lsq2(x1: np.ndarray, x2: np.ndarray,
              y: np.ndarray) -> tuple[float, float, float, float]:
        """Closed-form OLS for `y = a + b1*x1 + b2*x2`. Returns (a, b1, b2, R²)."""
        X = np.column_stack([np.ones_like(x1), x1, x2])
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            a, b1, b2 = float(beta[0]), float(beta[1]), float(beta[2])
            y_hat = X @ beta
            ss_res = float(((y - y_hat) ** 2).sum())
            ss_tot = float(((y - y.mean()) ** 2).sum())
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            return a, b1, b2, r2
        except np.linalg.LinAlgError:
            return float("nan"), float("nan"), float("nan"), float("nan")

    a_off_pt, b1_off_pt, b2_off_pt, r2_off_pt = _lsq2(tok_in_off, tok_out_off, off_w)
    a_on_pt,  b1_on_pt,  b2_on_pt,  r2_on_pt  = _lsq2(tok_in_on,  tok_out_on,  on_w)

    # Bootstrap. Resample paired indices with replacement.
    rng = np.random.default_rng(0xF2F2F2)
    n = len(common)
    idx = rng.integers(0, n, size=(n_resamples, n))
    a_off_s  = np.empty(n_resamples); b1_off_s = np.empty(n_resamples)
    b2_off_s = np.empty(n_resamples)
    a_on_s   = np.empty(n_resamples); b1_on_s  = np.empty(n_resamples)
    b2_on_s  = np.empty(n_resamples)
    for i in range(n_resamples):
        ix = idx[i]
        a_off_s[i], b1_off_s[i], b2_off_s[i], _ = _lsq2(
            tok_in_off[ix], tok_out_off[ix], off_w[ix])
        a_on_s[i],  b1_on_s[i],  b2_on_s[i],  _ = _lsq2(
            tok_in_on[ix],  tok_out_on[ix],  on_w[ix])

    da_s  = a_on_s  - a_off_s
    db1_s = b1_on_s - b1_off_s
    db2_s = b2_on_s - b2_off_s
    q = (alpha / 2, 1 - alpha / 2)
    def _ci(arr): return (float(np.quantile(arr, q[0])),
                          float(np.quantile(arr, q[1])))

    return TwoFeatureFitResult(
        a_off=a_off_pt,   a_off_ci=_ci(a_off_s),
        b1_off=b1_off_pt, b1_off_ci=_ci(b1_off_s),
        b2_off=b2_off_pt, b2_off_ci=_ci(b2_off_s),
        a_on=a_on_pt,     a_on_ci=_ci(a_on_s),
        b1_on=b1_on_pt,   b1_on_ci=_ci(b1_on_s),
        b2_on=b2_on_pt,   b2_on_ci=_ci(b2_on_s),
        delta_a=float(a_on_pt - a_off_pt),       delta_a_ci=_ci(da_s),
        delta_b1=float(b1_on_pt - b1_off_pt),    delta_b1_ci=_ci(db1_s),
        delta_b2=float(b2_on_pt - b2_off_pt),    delta_b2_ci=_ci(db2_s),
        r2_off=r2_off_pt, r2_on=r2_on_pt,
        n_paired_pairs=n, method="paired-bootstrap-OLS-two-feature",
    )


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------

def write_tables(stats_list: list[CellStats], fit: FitResult,
                  fit2: TwoFeatureFitResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [s.__dict__ for s in stats_list]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "sweep_table.csv", index=False)

    lines = ["# Max-tokens sweep — bootstrap CIs (95%)\n",
             "## Per-max_tokens cell statistics\n"]
    disp = pd.DataFrame({
        "max_tok":   df["max_new_tokens"],
        "n_paired":  df["n_paired"],
        "tokens_out_p50": df["tokens_out_p50"].round(0).astype(int),
        "wall_off (s)": df["wall_p50_off"].round(2),
        "wall_on (s)":  df["wall_p50_on"].round(2),
        "Δ abs (s)":    df["abs_delta_s"].round(2),
        "Δ abs 95% CI": [f"[{lo:.2f}, {hi:.2f}]"
                          for lo, hi in zip(df["abs_ci_low"], df["abs_ci_high"])],
        "Δ %":          df["rel_delta_pct"].round(1),
        "Δ % 95% CI":   [f"[{lo:.1f}, {hi:.1f}]"
                          for lo, hi in zip(df["rel_ci_low_pct"], df["rel_ci_high_pct"])],
        "Δ ms/tok":     df["per_token_overhead_ms"].round(2),
        "Δ ms/tok 95% CI": [f"[{lo:.2f}, {hi:.2f}]"
                              for lo, hi in zip(df["per_token_ci_low_ms"],
                                                df["per_token_ci_high_ms"])],
        "n_eos<cap (off/on)": [f"{a}/{b}" for a, b
                                in zip(df["n_eos_early_off"], df["n_eos_early_on"])],
    })
    lines.append(disp.to_markdown(index=False))
    lines.append("")
    lines.append("## Single-feature fit  (wall = a + b·tokens_out)\n")
    lines.append("| param | CC-off | CC-on | Δ (on − off) | 95% CI of Δ |")
    lines.append("|---|---|---|---|---|")
    lines.append(
        f"| intercept a (s)        | {fit.a_off:.3f} | {fit.a_on:.3f} "
        f"| {fit.delta_a:+.3f} | [{fit.delta_a_ci[0]:+.3f}, {fit.delta_a_ci[1]:+.3f}] |"
    )
    lines.append(
        f"| slope b (ms / token)   | {fit.b_off * 1000:.2f} | {fit.b_on * 1000:.2f} "
        f"| {fit.delta_b * 1000:+.2f} | "
        f"[{fit.delta_b_ci[0] * 1000:+.2f}, {fit.delta_b_ci[1] * 1000:+.2f}] |"
    )
    lines.append("")
    lines.append(f"_n_paired_pairs = {fit.n_paired_pairs}; method = {fit.method}_\n")

    lines.append("## Two-feature fit  (wall = a + b₁·tokens_in + b₂·tokens_out)\n")
    lines.append("| param | CC-off | CC-on | Δ (on − off) | 95% CI of Δ |")
    lines.append("|---|---|---|---|---|")
    lines.append(
        f"| intercept a (s)              | {fit2.a_off:.3f} | {fit2.a_on:.3f} "
        f"| {fit2.delta_a:+.3f} | "
        f"[{fit2.delta_a_ci[0]:+.3f}, {fit2.delta_a_ci[1]:+.3f}] |"
    )
    lines.append(
        f"| b₁ prefill (ms / in-tok)     | {fit2.b1_off * 1000:.2f} | "
        f"{fit2.b1_on * 1000:.2f} | {fit2.delta_b1 * 1000:+.2f} | "
        f"[{fit2.delta_b1_ci[0] * 1000:+.2f}, {fit2.delta_b1_ci[1] * 1000:+.2f}] |"
    )
    lines.append(
        f"| b₂ decode  (ms / out-tok)    | {fit2.b2_off * 1000:.2f} | "
        f"{fit2.b2_on * 1000:.2f} | {fit2.delta_b2 * 1000:+.2f} | "
        f"[{fit2.delta_b2_ci[0] * 1000:+.2f}, {fit2.delta_b2_ci[1] * 1000:+.2f}] |"
    )
    lines.append("")
    lines.append(f"_R² CC-off = {fit2.r2_off:.4f}; R² CC-on = {fit2.r2_on:.4f}; "
                 f"n_paired_pairs = {fit2.n_paired_pairs}; method = {fit2.method}_\n")

    (out_dir / "sweep_table.md").write_text("\n".join(lines))
    (out_dir / "sweep_fit.json").write_text(json.dumps(fit.to_dict(), indent=2))
    (out_dir / "sweep_fit_two_feature.json").write_text(
        json.dumps(fit2.to_dict(), indent=2))

    print(f"  wrote {out_dir / 'sweep_table.csv'}")
    print(f"  wrote {out_dir / 'sweep_table.md'}")
    print(f"  wrote {out_dir / 'sweep_fit.json'}")
    print(f"  wrote {out_dir / 'sweep_fit_two_feature.json'}")


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------

LABEL_TOKENS = r"$p_{50}(\mathrm{tokens\_out})$"
LABEL_PER_TOKEN_MS = "per-output-token\nCC overhead  (ms / token)"

def figure_amortization(stats_list: list[CellStats], fit: FitResult,
                         fit2: TwoFeatureFitResult,
                         df: pd.DataFrame, out_dir: Path,
                         ttft_df: pd.DataFrame | None = None) -> None:
    """Headline unified figure with three panels.

      LEFT (full height)
        wall_p50 vs tokens_out_p50 (paired), one curve per CC state, with
        paired-BCa CI whiskers per cell. Single-feature OLS fit lines
        overlaid; Δa and Δb annotated inline (matches the dashed fits).

      TOP RIGHT (rectangular)
        Per-output-token CC overhead (ms/token), paired-median per cell
        with bootstrap 95% CI. Asymptote line at Δb (single-feature fit).
        Flat ≈ per-token tax; falling toward zero ≈ amortisation.

      BOTTOM RIGHT (rectangular, three rows on independent scales)
        Phase-decomposition forest of the two-feature-fit CC deltas (Δa,
        Δb1, Δb2) with paired-bootstrap 95% CIs. Each parameter on its
        own scale (units differ: s, ms/in-tok, ms/out-tok). CIs that
        bracket zero are rendered greyed (no detectable effect); CIs
        that exclude zero are rendered in the CC-on colour.
    """
    fig_dir = out_dir / "figures"
    if not stats_list:
        return
    sl = sorted(stats_list, key=lambda s: s.tokens_out_p50)
    x = np.array([s.tokens_out_p50 for s in sl])
    off_p50 = np.array([s.wall_p50_off for s in sl])
    on_p50  = np.array([s.wall_p50_on  for s in sl])
    # Per-cell CI on wall_p50 (off/on marginal). Compute from paired data
    # for the wall_p50 CI of each arm via plain percentile bootstrap.
    rng = np.random.default_rng(0xBEEF)
    def _marginal_ci(arr: np.ndarray, n_resamples: int = 5000,
                      alpha: float = 0.05) -> tuple[float, float]:
        if len(arr) < 2:
            return (float(np.median(arr)), float(np.median(arr)))
        idx = rng.integers(0, len(arr), size=(n_resamples, len(arr)))
        meds = np.median(arr[idx], axis=1)
        return float(np.quantile(meds, alpha / 2)), float(np.quantile(meds, 1 - alpha / 2))

    off_lo = np.empty_like(off_p50); off_hi = np.empty_like(off_p50)
    on_lo  = np.empty_like(on_p50);  on_hi  = np.empty_like(on_p50)
    for i, s in enumerate(sl):
        sub = df[(df["max_new_tokens"] == s.max_new_tokens) &
                 (df["cc"] == "off")]["wall_seconds"].dropna().to_numpy()
        off_lo[i], off_hi[i] = _marginal_ci(sub)
        sub = df[(df["max_new_tokens"] == s.max_new_tokens) &
                 (df["cc"] == "on")]["wall_seconds"].dropna().to_numpy()
        on_lo[i], on_hi[i] = _marginal_ci(sub)

    # Three-panel layout: left full-height; right column split into two
    # rectangular sub-panels (per-token curve on top, phase-decomposition
    # forest on bottom). The forest itself is a nested 3-row gridspec so
    # each parameter gets its own x-axis scale. Bottom row gets a larger
    # share of vertical space — the forest needs room for three sub-axes,
    # while the per-token curve sits on a tight y-range above.
    fig = plt.figure(figsize=(13.8, 6.6))
    gs_outer = fig.add_gridspec(
        2, 2,
        width_ratios=[1.15, 1.0],
        height_ratios=[0.78, 1.22],
        wspace=0.22, hspace=0.50,
    )
    ax_l  = fig.add_subplot(gs_outer[:, 0])
    ax_tr = fig.add_subplot(gs_outer[0, 1])
    gs_br = gs_outer[1, 1].subgridspec(3, 1, hspace=1.05)
    ax_br_a  = fig.add_subplot(gs_br[0])
    ax_br_b1 = fig.add_subplot(gs_br[1])
    ax_br_b2 = fig.add_subplot(gs_br[2])

    # === LEFT PANEL: wall vs tokens =====================================
    ax_l.fill_between(x, off_lo, off_hi, color=PALETTE["off"], alpha=0.12, zorder=2)
    ax_l.fill_between(x, on_lo, on_hi,   color=PALETTE["on"],  alpha=0.12, zorder=2)
    ax_l.plot(x, off_p50, "-o", color=PALETTE["off"], markersize=8,
              markeredgecolor="white", markeredgewidth=1.2,
              linewidth=2.4, label="CC-off  (paired p50, 95% CI)", zorder=3)
    ax_l.plot(x, on_p50,  "-o", color=PALETTE["on"],  markersize=8,
              markeredgecolor="white", markeredgewidth=1.2,
              linewidth=2.4, label="CC-on  (paired p50, 95% CI)",  zorder=3)

    # Overlay single-feature OLS fits across the full per-pair data range.
    # The two-feature fit can't be drawn as a 1D line here (it depends on
    # tokens_in too) so the dashed lines remain the single-feature visual.
    x_fit = np.linspace(0, x.max() * 1.05, 100)
    ax_l.plot(x_fit, fit.a_off + fit.b_off * x_fit,
              "--", color=PALETTE["off"], linewidth=1.3, alpha=0.85,
              zorder=2,
              label=f"fit off: y = {fit.a_off:.2f} + {fit.b_off * 1000:.1f}·t/1000")
    ax_l.plot(x_fit, fit.a_on + fit.b_on * x_fit,
              "--", color=PALETTE["on"], linewidth=1.3, alpha=0.85,
              zorder=2,
              label=f"fit on: y = {fit.a_on:.2f} + {fit.b_on * 1000:.1f}·t/1000")
    
    # === HarmBench cross-corpus validation point ========================
    # Single off/on pair at max_tokens=512 on HarmBench (n=50 paired,
    # realised tok_out p50 = 82). The two markers land on the per-CC-state
    # fit lines to within 30 ms, demonstrating that the fit (anchored on
    # ToxicChat) generalises to a behaviourally distinct prompt
    # distribution.
    HB_TOK_OUT = 82
    HB_OFF_S, HB_ON_S = 17.52, 22.99
    ax_l.scatter([HB_TOK_OUT], [HB_OFF_S], marker="D", s=80,
                 facecolor="none", edgecolor=PALETTE["off"], linewidth=2.0,
                 zorder=4)
    ax_l.scatter([HB_TOK_OUT], [HB_ON_S], marker="D", s=80,
                 facecolor="none", edgecolor=PALETTE["on"], linewidth=2.0,
                 zorder=4)
    ax_l.annotate("HarmBench\nmax_tokens=512, n=50)",
                  xy=(HB_TOK_OUT, HB_ON_S),
                  xytext=(HB_TOK_OUT + 20, HB_ON_S + 90),
                  fontsize=9.5, color=PALETTE["annotation"],
                  arrowprops=dict(arrowstyle="->",
                                  color=PALETTE["annotation"],
                                  lw=0.9, alpha=0.85,
                                  connectionstyle="arc3,rad=0.15"))

    # Inline annotation of the single-feature fixed CC slice + slope
    ax_l.text(
        0.04, 0.96,
        f"single-feature fit:\n"
        f"  fixed CC slice  Δa = {fit.delta_a:+.2f} s\n"
        f"     95% CI  [{fit.delta_a_ci[0]:+.2f}, {fit.delta_a_ci[1]:+.2f}]\n"
        f"  per-token CC slope  Δb = {fit.delta_b * 1000:+.2f} ms/tok\n"
        f"     95% CI  [{fit.delta_b_ci[0] * 1000:+.2f}, "
        f"{fit.delta_b_ci[1] * 1000:+.2f}]",
        transform=ax_l.transAxes,
        ha="left", va="top", fontsize=10.0, color=PALETTE["annotation"],
        family="DejaVu Sans Mono",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#CCCCCC", linewidth=0.7, alpha=0.95),
    )

    ax_l.set_xlabel(LABEL_TOKENS)
    ax_l.set_ylabel(LABEL_WALL_P50_S)
    ax_l.legend(loc="lower right", fontsize=9.5)
    ax_l.set_xlim(left=0)
    ax_l.set_ylim(bottom=0)
    ax_l.set_title("Wall time vs realised output length", pad=8)

    # === TOP RIGHT: empirical phase split (replaces per-token overhead curve) ===
    # The per-token-overhead curve we used to show here was visually
    # informative but inferentially redundant with the left-panel fits and
    # the bottom-right forest (all three are views of the same regression).
    # The empirical phase split from streaming/concurrent baseline cells
    # carries genuinely independent evidence — direct measurement, not
    # regression — that the CC tax lands on decoding rather than prefill.
    if ttft_df is not None and not ttft_df.empty:
        _render_empirical_phase_bars(ax_tr, ttft_df)
    else:
        # Fall back to the per-token curve if no TTFT data was supplied.
        # Backward-compatible: prior callers see no change.
        pt    = np.array([s.per_token_overhead_ms for s in sl])
        pt_lo = np.array([s.per_token_ci_low_ms for s in sl])
        pt_hi = np.array([s.per_token_ci_high_ms for s in sl])
        ax_tr.fill_between(x, pt_lo, pt_hi, color=PALETTE["on"], alpha=0.18, zorder=2)
        ax_tr.plot(x, pt, "-o", color=PALETTE["on"], markersize=7,
                   markeredgecolor="white", markeredgewidth=1.1,
                   linewidth=2.2, zorder=3,
                   label="paired median (ms/token), 95% CI")
        ax_tr.axhline(fit.delta_b * 1000, color=PALETTE["annotation"],
                      linewidth=1.2, linestyle=":",
                      alpha=0.8, zorder=2,
                      label=fr"linear-fit asymptote  $\Delta b$ = "
                            fr"{fit.delta_b * 1000:+.2f} ms/tok")
        y_pts = np.concatenate([pt_lo, pt_hi, np.array([fit.delta_b * 1000])])
        y_lo, y_hi = float(y_pts.min()), float(y_pts.max())
        span = max(y_hi - y_lo, 1.0)
        ax_tr.set_ylim(y_lo - span * 0.18, y_hi + span * 0.55)
        ax_tr.set_xlabel(LABEL_TOKENS, fontsize=9.5)
        ax_tr.set_ylabel(LABEL_PER_TOKEN_MS, fontsize=9.5)
        ax_tr.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
        ax_tr.set_xlim(left=0)
        ax_tr.set_title("Per-token CC overhead vs realised output length",
                        pad=6, fontsize=10.5)

    # === BOTTOM RIGHT: phase-decomposition forest =======================
    # Three parameters, each on its own row with its own x-scale because
    # units differ (s vs ms/in-tok vs ms/out-tok). Greyed if CI brackets
    # zero (no detectable effect), CC-on colour otherwise. The y-labels
    # name the phase ("fixed slice" / "prefill" / "decode") so the figure
    # is self-contained without the paper's notation; the parameter symbol
    # appears on a second line for cross-reference with the OLS fit.
    forest_rows = [
        {
            "ax": ax_br_a,
            "ylabel": "fixed slice\n($\\Delta a$)",
            "xlabel": "seconds",
            "value": fit2.delta_a,
            "ci":    fit2.delta_a_ci,
        },
        {
            "ax": ax_br_b1,
            "ylabel": "prefill\n($\\Delta b_1$)",
            "xlabel": "ms / input token",
            "value": fit2.delta_b1 * 1000.0,
            "ci":    (fit2.delta_b1_ci[0] * 1000.0,
                      fit2.delta_b1_ci[1] * 1000.0),
        },
        {
            "ax": ax_br_b2,
            "ylabel": "decode\n($\\Delta b_2$)",
            "xlabel": "ms / output token",
            "value": fit2.delta_b2 * 1000.0,
            "ci":    (fit2.delta_b2_ci[0] * 1000.0,
                      fit2.delta_b2_ci[1] * 1000.0),
        },
    ]

    for row in forest_rows:
        ax = row["ax"]
        val = row["value"]
        lo, hi = row["ci"]
        excludes_zero = (lo > 0) or (hi < 0)
        color = PALETTE["on"] if excludes_zero else PALETTE["neutral"]
        face  = PALETTE["on"] if excludes_zero else "#BBBBBB"

        ax.axvline(0, color="#999999", linewidth=1.0, linestyle="--",
                   alpha=0.7, zorder=1)
        ax.errorbar([val], [0], xerr=[[val - lo], [hi - val]],
                    fmt="o", markersize=9, color=color, markerfacecolor=face,
                    markeredgecolor="white", markeredgewidth=1.3,
                    ecolor=color, elinewidth=2.2, capsize=4, capthick=1.6,
                    zorder=3)

        # Inline numeric annotation (value and CI), right-aligned.
        ax.text(0.98, 0.92,
                f"{val:+.3f}  [{lo:+.2f}, {hi:+.2f}]",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9.0, family="DejaVu Sans Mono",
                color=PALETTE["annotation"])

        # Symmetric padding around the data range that includes zero.
        span = hi - lo
        ref = max(abs(lo), abs(hi), abs(val))
        pad = max(span * 0.55, ref * 0.30, 0.05)
        lower = min(lo, 0) - pad
        upper = max(hi, 0) + pad
        ax.set_xlim(lower, upper)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_ylabel(row["ylabel"], rotation=0, ha="right", va="center",
                      fontsize=10.0, labelpad=10)
        ax.set_xlabel(row["xlabel"], fontsize=8.5, labelpad=1)
        ax.tick_params(axis="x", labelsize=8.0)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)

    # Forest header (positioned above the top forest row). Display R² as
    # a lower-bound floored to 2 decimals — for the headline (R²≈0.9999)
    # case this shows "R² > 0.99"; the formulation degrades gracefully
    # to whatever floor the data actually supports.
    r2_min = min(fit2.r2_off, fit2.r2_on)
    r2_floor = float(np.floor(r2_min * 100) / 100.0)
    if r2_floor >= 1.0:
        r2_floor = 0.99   # display cap (R² > 1.00 is impossible)
    ax_br_a.set_title(
        f"Phase-decomposition forest (two-feature fit, R² > {r2_floor:.2f})",
        pad=6, fontsize=10.0,
    )

    fig.suptitle(
        r"CC tax is decode-phase per-token, with no output-length amortisation",
        y=1.00, fontsize=12.5,
    )
    _save(fig, fig_dir / "sweep_amortization")
    print(f"  wrote {fig_dir / 'sweep_amortization.{png,pdf}'}")

def _render_empirical_phase_bars(
    ax: plt.Axes,
    ttft_df: pd.DataFrame,
    regimes: tuple[str, ...] = ("streaming", "concurrent_c8"),
) -> None:
    """Horizontal stacked bars showing prefill (grey) + decode (teal) per
    (regime, CC-state) pair. CC-off is encoded as a hatched fill, CC-on
    as a solid fill — separating the off/on axis from the prefill/decode
    colour axis. Regime headers sit immediately to the left of each
    bar pair; Δprefill / Δdecode annotations sit immediately to the right.
    """
    color_prefill = PALETTE["off"]
    color_decode  = PALETTE["on"]

    rows = []
    for regime in regimes:
        for cc in ("off", "on"):
            sub = ttft_df[(ttft_df["condition"] == "baseline")
                          & (ttft_df["regime"] == regime)
                          & (ttft_df["cc"] == cc)]
            if sub.empty or "ttft_seconds" not in sub.columns:
                continue
            ttft = sub["ttft_seconds"].dropna().to_numpy()
            wall = sub["wall_seconds"].dropna().to_numpy()
            if len(ttft) == 0 or len(wall) == 0:
                continue
            rows.append({
                "regime":    regime,
                "cc":        cc,
                "prefill_s": float(np.median(ttft)),
                "decode_s":  float(np.median(wall) - np.median(ttft)),
                "wall_s":    float(np.median(wall)),
            })
    if not rows:
        ax.text(0.5, 0.5, "(no TTFT data available)",
                ha="center", va="center", transform=ax.transAxes,
                color=PALETTE["neutral"], fontsize=10)
        ax.set_title("Empirical phase split", loc="left", pad=6, fontsize=10.5)
        return
    pdf = pd.DataFrame(rows)
    present = [r for r in regimes if r in pdf["regime"].unique()]

    # Layout — each regime takes 2 rows (off, on), separated by a gap.
    bar_h = 0.62
    intra = 1.00   # spacing within a regime pair (off→on)
    inter = 0.80   # extra gap between regime pairs
    bar_specs: list[dict] = []
    group_centers: list[tuple[str, float]] = []
    y = 0.0
    for regime in present:
        start = y
        for cc in ("off", "on"):
            row = pdf[(pdf["regime"] == regime) & (pdf["cc"] == cc)]
            if row.empty:
                continue
            r = row.iloc[0]
            bar_specs.append({
                "y": y, "regime": regime, "cc": cc,
                "prefill_s": r["prefill_s"], "decode_s": r["decode_s"],
                "wall_s":   r["wall_s"],
            })
            y += intra
        end = y - intra
        group_centers.append((regime, (start + end) / 2))
        y += inter

    # Bars.
    for i, s in enumerate(bar_specs):
        hatch = "////" if s["cc"] == "off" else None
        ax.barh(s["y"], s["prefill_s"], bar_h,
                color=color_prefill, edgecolor=PALETTE["annotation"],
                linewidth=0.7, hatch=hatch, zorder=2,
                label="prefill (TTFT)" if i == 0 else None)
        ax.barh(s["y"], s["decode_s"], bar_h, left=s["prefill_s"],
                color=color_decode, edgecolor=PALETTE["annotation"],
                linewidth=0.7, hatch=hatch, zorder=2,
                label="decoding" if i == 0 else None)

    # Y-tick labels: CC-off / CC-on as the row identity.
    ax.set_yticks([s["y"] for s in bar_specs])
    ax.set_yticklabels([f"CC-{s['cc']}" for s in bar_specs], fontsize=9.5)
    ax.invert_yaxis()    # first regime at top — top-to-bottom reading order
    ax.tick_params(axis="y", length=0, pad=4)

    # Regime headers to the left (outside the data area), bold, at the
    # vertical centre of each bar pair. Drawn in axis-fraction x so they
    # always sit just outside the y-axis regardless of data range.
    for regime, cy in group_centers:
        ax.text(-0.20, cy,
                "streaming" if regime == "streaming" else "concurrent c=8",
                transform=ax.get_yaxis_transform(),
                ha="right", va="center",
                fontsize=10.5, fontweight="bold",
                color=PALETTE["annotation"])

    # Δ annotations to the right of each regime pair, vertically centred.
    for regime, cy in group_centers:
        off_s = next((s for s in bar_specs
                      if s["regime"] == regime and s["cc"] == "off"), None)
        on_s  = next((s for s in bar_specs
                      if s["regime"] == regime and s["cc"] == "on"),  None)
        if off_s is None or on_s is None:
            continue
        d_pref_ms = (on_s["prefill_s"] - off_s["prefill_s"]) * 1000
        d_dec_ms  = (on_s["decode_s"]  - off_s["decode_s"])  * 1000
        ax.text(on_s["wall_s"] + 0.4, cy,
                f"$\\Delta$decode  = {d_dec_ms:+5.0f} ms\n"
                f"$\\Delta$prefill = {d_pref_ms:+5.0f} ms",
                ha="left", va="center", fontsize=8.8,
                color=PALETTE["annotation"],
                family="DejaVu Sans Mono",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#CCCCCC", linewidth=0.6, alpha=0.95))

    # X-axis with headroom for the Δ annotations.
    ax.set_xlabel("wall p50 (s)", fontsize=9.5)
    max_wall = pdf["wall_s"].max()
    ax.set_xlim(0, max_wall * 1.55)

    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95, ncol=2)
    ax.set_title("Empirical phase split  (baseline cells, n=50 each)",
                 fontsize=10.5, loc="left", pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def figure_overhead_pct(stats_list: list[CellStats], out_dir: Path) -> None:
    """The amortisation curve in percent: Δ% vs tokens_out_p50.

    Flat at +33% = per-token tax (bad for the brief).
    Monotone decreasing from +33% to small % = amortisation confirmed (the
    pitch headline).
    """
    fig_dir = out_dir / "figures"
    if not stats_list:
        return
    sl = sorted(stats_list, key=lambda s: s.tokens_out_p50)
    x = np.array([s.tokens_out_p50 for s in sl])
    pct = np.array([s.rel_delta_pct for s in sl])
    lo  = np.array([s.rel_ci_low_pct for s in sl])
    hi  = np.array([s.rel_ci_high_pct for s in sl])

    fig, ax = plt.subplots(figsize=(8.8, 4.4))

    # Reference CC band (33-38%) — the existing Phase 3 finding at 32 tokens.
    # If amortisation holds, the curve enters the band on the left and
    # exits below it on the right.
    ax.axhspan(33.0, 38.0, color=PALETTE["muted"], alpha=0.30, zorder=0,
               label="published Phase 3 CC band (+33-38%)")
    ax.axhline(0, color=PALETTE["neutral"], linewidth=1.0, linestyle="--",
               alpha=0.6, zorder=1)

    ax.fill_between(x, lo, hi, color=PALETTE["on"], alpha=0.22, zorder=2)
    ax.plot(x, pct, "-o", color=PALETTE["on"], markersize=9,
            markeredgecolor="white", markeredgewidth=1.4,
            linewidth=2.6, zorder=3,
            label=r"$\Delta$% (paired BCa 95% CI)")

    # Inline per-point labels
    for xi, yi in zip(x, pct):
        ax.text(xi, yi + 2.5, f"{yi:+.1f}%",
                ha="center", va="bottom", fontsize=10.5,
                color=PALETTE["annotation"])

    ax.set_xlabel(LABEL_TOKENS)
    ax.set_ylabel(LABEL_WALL_P50_PCT)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim(left=0)
    ax.set_title("CC overhead (%) vs realised output length", pad=8)

    _save(fig, fig_dir / "sweep_overhead_pct")
    print(f"  wrote {fig_dir / 'sweep_overhead_pct.{png,pdf}'}")


def figure_eos_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    """Diagnostic: realised tokens_out distribution per max_tokens cap.
    Validates the choice of `tokens_out_p50` as the x-axis (a cap of 2048
    with most prompts EOSing at 200 means the "2048 cell" is really a
    "~200-token cell")."""
    fig_dir = out_dir / "figures"
    caps = sorted(df["max_new_tokens"].unique())
    if not caps:
        return

    n = len(caps)
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n + 0.8, 4.0), sharey=True)
    if n == 1:
        axes = [axes]

    for i, (cap, ax) in enumerate(zip(caps, axes)):
        sub = df[df["max_new_tokens"] == cap]
        off_tok = sub[sub["cc"] == "off"]["tokens_out"].dropna().to_numpy()
        on_tok  = sub[sub["cc"] == "on"]["tokens_out"].dropna().to_numpy()
        all_tok = np.concatenate([off_tok, on_tok])
        if len(all_tok) == 0:
            ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                    transform=ax.transAxes, color=PALETTE["neutral"])
            ax.set_title(f"cap={cap}")
            continue

        # Two side-by-side stacked histograms (CC-off in grey, CC-on in teal).
        bins = np.linspace(0, cap * 1.02, 24)
        ax.hist(off_tok, bins=bins, color=PALETTE["off"], alpha=0.65,
                edgecolor="white", linewidth=0.4, label=f"off (n={len(off_tok)})")
        ax.hist(on_tok,  bins=bins, color=PALETTE["on"],  alpha=0.55,
                edgecolor="white", linewidth=0.4, label=f"on  (n={len(on_tok)})")
        ax.axvline(cap, color=PALETTE["annotation"], linewidth=1.3,
                   linestyle="--", alpha=0.85)
        p50 = float(np.median(all_tok))
        ax.axvline(p50, color=PALETTE["neutral"], linewidth=1.0,
                   linestyle=":", alpha=0.8)
        n_eos = int((all_tok < cap).sum())
        ax.set_title(f"cap={cap}\np50={p50:.0f}, EOS<cap: {n_eos}/{len(all_tok)}",
                     fontsize=10)
        ax.tick_params(axis="x", labelsize=9)
        if i == 0:
            ax.set_ylabel("count")
            ax.legend(loc="upper left", fontsize=8.5)
        ax.set_xlabel("tokens_out", fontsize=9.5)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Realised tokens_out distribution per max_tokens cap\n"
                 "(dashed = cap; dotted = joint median)", y=1.03,
                 fontsize=11.5)
    plt.tight_layout()
    _save(fig, fig_dir / "sweep_eos_distribution")
    print(f"  wrote {fig_dir / 'sweep_eos_distribution.{png,pdf}'}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--data-dir", type=Path, required=True,
                    help="Root with per-iteration subdirs from "
                         "phase3_sweep_max_tokens.py.")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--cell-id-prefix", default="C1",
                    help="Match the prefix you passed to the sweep. Default C1.")
    ap.add_argument("--n-resamples", type=int, default=10_000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument(
        "--ttft-data-dir", type=Path, default=None,
        help="Optional. Directory containing C1-(off|on)-(stream|c8)/requests.parquet "
             "for the empirical phase-split panel that replaces the per-token-overhead "
             "panel in the headline figure. Skipped if omitted.",
    )
    args = ap.parse_args()

    print(f"Discovering sweep cells under {args.data_dir} "
          f"(prefix={args.cell_id_prefix})...")
    cells = discover_cells(args.data_dir, args.cell_id_prefix)
    if not cells:
        sys.exit(f"FATAL: no cells matching {args.cell_id_prefix}-(off|on)-t* "
                 f"found under {args.data_dir}")
    print(f"  found {len(cells)} cells: "
          f"{[c.cell_id for c in cells]}")

    print(f"\nLoading data...")
    df = load_sweep(cells)
    print(f"  loaded {len(df)} rows  "
          f"({df['cc'].value_counts().to_dict()}, "
          f"max_tokens={sorted(df['max_new_tokens'].unique())})")

    print(f"\nComputing per-cell stats (BCa, n_resamples={args.n_resamples})...")
    stats_list = compute_per_cell(df, n_resamples=args.n_resamples,
                                    alpha=args.alpha)
    if len(stats_list) < 2:
        sys.exit(f"FATAL: only {len(stats_list)} cells with paired data; "
                 f"cannot fit a curve")

    print(f"\nFitting linear model (wall = a + b·tokens_out, paired bootstrap)...")
    fit = fit_linear_paired(df, n_resamples=args.n_resamples, alpha=args.alpha)
    print(f"  Δa (fixed CC slice)   = {fit.delta_a:+.3f} s "
          f"[{fit.delta_a_ci[0]:+.3f}, {fit.delta_a_ci[1]:+.3f}]")
    print(f"  Δb (per-token slope)  = {fit.delta_b * 1000:+.3f} ms/tok "
          f"[{fit.delta_b_ci[0] * 1000:+.3f}, {fit.delta_b_ci[1] * 1000:+.3f}]")

    print(f"\nFitting two-feature model (wall = a + b1·tokens_in + b2·tokens_out, "
          f"paired bootstrap)...")
    fit2 = fit_two_feature_paired(df, n_resamples=args.n_resamples,
                                    alpha=args.alpha)
    print(f"  Δa  (fixed CC slice)  = {fit2.delta_a:+.3f} s "
          f"[{fit2.delta_a_ci[0]:+.3f}, {fit2.delta_a_ci[1]:+.3f}]")
    print(f"  Δb1 (prefill slope)   = {fit2.delta_b1 * 1000:+.3f} ms/in-tok "
          f"[{fit2.delta_b1_ci[0] * 1000:+.3f}, {fit2.delta_b1_ci[1] * 1000:+.3f}]")
    print(f"  Δb2 (decode slope)    = {fit2.delta_b2 * 1000:+.3f} ms/out-tok "
          f"[{fit2.delta_b2_ci[0] * 1000:+.3f}, {fit2.delta_b2_ci[1] * 1000:+.3f}]")
    print(f"  R²:  CC-off = {fit2.r2_off:.4f}, CC-on = {fit2.r2_on:.4f}")

    print(f"\nWriting tables to {args.output_dir}...")
    write_tables(stats_list, fit, fit2, args.output_dir)

    print(f"\nGenerating figures...")
    ttft_df = None
    if args.ttft_data_dir is not None:
        ttft_df = load_ttft_cells(args.ttft_data_dir)
        if ttft_df is not None:
            print(f"  loaded {len(ttft_df)} TTFT rows for empirical phase panel")
        else:
            print(f"  [warn] no TTFT cells found under {args.ttft_data_dir}; "
                  f"falling back to per-token overhead panel")

    figure_amortization(stats_list, fit, fit2, df, args.output_dir, ttft_df=ttft_df)
    figure_overhead_pct(stats_list, args.output_dir)
    figure_eos_distribution(df, args.output_dir)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())