#!/usr/bin/env python3
"""
run_m4_anomaly.py — M4 anomaly-trigger run (report-only, offline).

Generates a benign calibration corpus and a LADDER of three adversaries of
increasing sophistication, then runs the report-only detector
(plot_request_distribution.py) over each. The detector carries two divergence
signals:

    marginal KL    — KL(observed plot-type distribution || benign marginal)
    transition KL  — occupancy-weighted KL(observed transitions || benign
                     transitions); catches sequential structure even when the
                     marginal matches benign

The three-tier ladder is the credibility argument: a single easy adversary
would be a strawman. Instead we show exactly which detector signal catches
which class of channel, and name the class that defeats the current detector.

    Tier 1  BURST            — 50 identical-type requests, low-entropy 2-level
                               parameter. The obvious channel. Caught by
                               marginal KL (and run-length, param entropy).
    Tier 2  MARGINAL-MATCH   — preserves the benign plot-type marginal AND the
                               full benign parameter range, but encodes bits in
                               the request ORDER (a non-benign transition
                               structure). Evades marginal KL by construction;
                               caught by transition KL.
    Tier 3  MATCH-BOTH       — preserves benign marginal AND benign first-order
                               transitions, encoding only in second-order /
                               timing structure the detector does not model.
                               Evades BOTH signals. This is the OPEN PROBLEM the
                               pilot's real-corpus, higher-order calibration
                               must address — included so the result is honest
                               about the detector's ceiling, not just its reach.

Report-only: threshold_nats=None, the detector never fires. The deliverable is
the signal-separation MATRIX (which signal separates which tier), not a
committed operating point — threshold calibration from a real benign corpus is
deferred to the pilot (whitepaper §4.3.6).

Fully offline — no Datasite, no network.

Usage:
  python run_m4_anomaly.py --out-dir runs/phase3_pysyft/anomaly
  python run_m4_anomaly.py --out-dir runs/phase3_pysyft/anomaly --bitstring 1011001
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from plot_request_distribution import (
    DEFAULT_PLOT_TYPES,
    AnomalyReport,
    calibrate_baseline,
    corrected_conditional_entropy,
    sliding_windows,
)


# Benign marginal mix: heatmaps + norm curves dominate, steering trace rare.
BENIGN_MIX = {
    "attention_entropy_heatmap": 0.34,
    "residual_norm_curve":       0.30,
    "routing_load_bar":          0.20,
    "logit_lens_panel":          0.12,
    "steering_delta_trace":      0.04,
}
_BENIGN_TYPES = list(BENIGN_MIX.keys())
_BENIGN_PROBS = np.array([BENIGN_MIX[t] for t in _BENIGN_TYPES], dtype=np.float64)
_BENIGN_PROBS /= _BENIGN_PROBS.sum()

# Probe layers used as the benign parameter range (high-entropy when sampled).
_PROBE_LAYERS = np.array([12, 23, 39, 51, 62, 70], dtype=np.float64)


# Benign FIRST-ORDER transition structure. A real auditor's workflow has mild
# sequential correlation (you tend to follow a heatmap with another heatmap or
# a norm curve, not jump randomly). We encode a plausible "sticky" benign chain:
# each type has elevated probability of repeating or moving to an adjacent
# workflow step. Tier 2/3 adversaries are measured against THIS, so a channel
# that randomises order (even with a benign marginal) is detectable, and a
# channel that respects this chain (Tier 3) is not.
def _benign_transition_matrix() -> np.ndarray:
    K = len(_BENIGN_TYPES)
    base = np.tile(_BENIGN_PROBS, (K, 1))
    for i in range(K):
        base[i, i] += 0.35                       # sticky: repeat same plot
        base[i, (i + 1) % K] += 0.15             # workflow drift to next step
    base /= base.sum(axis=1, keepdims=True)
    return base


_BENIGN_TRANS = _benign_transition_matrix()


def gen_benign(n: int, rng: np.random.Generator) -> tuple[list[str], list[float]]:
    """Benign stream: types drawn from the benign first-order CHAIN (not iid),
    so it has the benign transition structure; params iid over the probe set
    (high entropy)."""
    K = len(_BENIGN_TYPES)
    cur = int(rng.choice(K, p=_BENIGN_PROBS))
    seq_idx = [cur]
    for _ in range(n - 1):
        cur = int(rng.choice(K, p=_BENIGN_TRANS[cur]))
        seq_idx.append(cur)
    seq = [_BENIGN_TYPES[i] for i in seq_idx]
    params = list(rng.choice(_PROBE_LAYERS, size=n))
    return seq, params


def gen_burst(
    n: int, bitstring: str, rng: np.random.Generator,
    burst_type: str = "routing_load_bar", burst_len: int = 50,
) -> tuple[list[str], list[float]]:
    """Tier 1: a benign prefix/suffix around a burst of one plot type whose
    parameter encodes the bitstring as a 2-level (low-entropy) sequence."""
    pre = max(0, (n - burst_len) // 2)
    post = n - burst_len - pre
    seq_pre, par_pre = gen_benign(pre, rng)
    seq_post, par_post = gen_benign(post, rng)
    bits = [int(b) for b in bitstring if b in "01"] or [1, 0]
    burst_types = [burst_type] * burst_len
    level0, level1 = 12.0, 70.0
    burst_params = [(level1 if bits[i % len(bits)] else level0)
                    for i in range(burst_len)]
    return seq_pre + burst_types + seq_post, par_pre + burst_params + par_post


def gen_marginal_match(
    n: int, bitstring: str, rng: np.random.Generator,
) -> tuple[list[str], list[float]]:
    """Tier 2: TRCTC analogue (Cabuk 2006 time-replay channel). Replays benign
    symbols so the FIRST-ORDER distribution stays ~ benign (evades the shape
    test), but imposes a REGULAR repeating encoding structure that lowers the
    entropy rate (caught by the CCE regularity test).

    Construction: draw a benign pool (correct marginal + params), then emit it
    in a fixed-period schedule driven by the message bits — a 0-bit draws the
    next symbol from the 'low' half of the benign symbol set, a 1-bit from the
    'high' half, cycling through the bitstring. Because both halves are made of
    benign symbols and we consume the benign pool, the marginal is approximately
    preserved; because the schedule repeats with the bitstring period, the
    sequence is far more REGULAR (lower entropy rate) than benign — exactly the
    TRCTC signature the literature reports first-order tests miss and regularity
    tests catch."""
    bits = [int(b) for b in bitstring if b in "01"] or [1, 0]
    base_seq, base_par = gen_benign(n, rng)
    # Partition benign symbols into two bins by marginal rank (TRCTC's S0/S1).
    order = sorted(set(base_seq), key=lambda t: -BENIGN_MIX.get(t, 0.0))
    half = max(1, len(order) // 2)
    low_set, high_set = set(order[half:]), set(order[:half])
    # Pools of (symbol, param) preserving the benign multiset.
    from collections import deque
    low_pool = deque((t, p) for t, p in zip(base_seq, base_par) if t in low_set)
    high_pool = deque((t, p) for t, p in zip(base_seq, base_par) if t in high_set)
    seq, par = [], []
    bi = 0
    for _ in range(n):
        want_high = bool(bits[bi % len(bits)])
        bi += 1
        pool = high_pool if want_high else low_pool
        other = low_pool if want_high else high_pool
        if not pool:
            pool = other          # fall back if a bin is exhausted
        if not pool:
            break
        t, p = pool.popleft()
        seq.append(t); par.append(p)
    # Top up if pools exhausted early (keeps length = n, marginal ~ benign).
    while len(seq) < n:
        t, p = (low_pool or high_pool).popleft() if (low_pool or high_pool) else (base_seq[len(seq)], base_par[len(seq)])
        seq.append(t); par.append(p)
    return seq[:n], par[:n]


def gen_match_both(
    n: int, rng: np.random.Generator,
) -> tuple[list[str], list[float]]:
    """Tier 3: preserves benign marginal AND benign first-order transitions —
    it draws from the benign chain itself — and is posited to encode in
    SECOND-ORDER / timing structure the detector does not model. To the
    first-order detector it is indistinguishable from benign BY CONSTRUCTION.
    The open-problem anchor: it demonstrates the current detector's ceiling
    (first-order), not a claim that no covert channel exists here."""
    return gen_benign(n, rng)


def summarize_reports(reports: list[AnomalyReport]) -> dict:
    if not reports:
        return {"n_windows": 0}
    kl = np.array([r.kl_nats for r in reports])
    cce = np.array([r.cce_nats for r in reports])
    runs = np.array([r.max_run_length for r in reports])
    ent = np.array([r.param_entropy_bits for r in reports])
    return {
        "n_windows": len(reports),
        "marginal_kl_nats": {"p50": float(np.percentile(kl, 50)),
                             "p95": float(np.percentile(kl, 95)),
                             "max": float(kl.max())},
        "cce_nats": {"min": float(cce.min()),
                     "p50": float(np.percentile(cce, 50)),
                     "max": float(cce.max())},
        "max_run_length": {"max": int(runs.max())},
        "param_entropy_bits": {"min": float(ent.min())},
    }


def _envelope_breaches(reports: list[AnomalyReport],
                       benign_marg_max: float,
                       benign_cce_lo: float,
                       benign_cce_hi: float) -> dict:
    kl = np.array([r.kl_nats for r in reports]) if reports else np.array([])
    cce = np.array([r.cce_nats for r in reports]) if reports else np.array([])
    return {
        "n_windows": int(len(reports)),
        "n_above_marginal_envelope": int(np.sum(kl > benign_marg_max)) if len(kl) else 0,
        # CCE two-sided: below band = over-regular (repeated pattern);
        # above band = over-random.
        "n_below_cce_band": int(np.sum(cce < benign_cce_lo)) if len(cce) else 0,
        "n_above_cce_band": int(np.sum(cce > benign_cce_hi)) if len(cce) else 0,
    }


def make_plots(out_dir: Path, tiers: dict, detection_matrix: dict,
               benign_marg_max: float, benign_cce_lo: float,
               benign_cce_hi: float) -> Path:
    """Two-panel figure styled to the Pour Demain technical-brief visual
    identity:
      (a) shape-vs-regularity scatter: every window as a point in
          (marginal KL, CCE) space, benign band shaded. Tier-1 in the high-KL
          corner, tier-2 in the benign-KL / low-CCE region (evades shape, trips
          regularity), tier-3 inside the band.
      (b) detection matrix: 2 tests x 3 tiers.
    Mirrors how the covert-timing-channel literature presents shape/regularity
    separation (Gianvecchio & Wang 2007)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib import font_manager

    # ---- Pour Demain brand palette (from pourdemain-techbrief.cls) ----------
    PD = {
        "teal":     "#47A5AD",
        "paleteal": "#CAE4E8",
        "stone":    "#D1CFCF",
        "navy":     "#0E2E47",
        "ink":      "#11253A",
        "muted":    "#6B7A87",
        "rule":     "#B8C3CB",
    }
    # Tier encoding: kept in-family but distinguishable. Tier-1 navy (the
    # strongest/most-detected), tier-2 teal, tier-3 muted; benign stone.
    series = {
        "benign":               (PD["muted"], "benign",                 28, 0.40, 2),
        "tier1_burst":          (PD["navy"],  "tier 1 · burst (IPCTC)",  34, 0.85, 4),
        "tier2_marginal_match": (PD["teal"],  "tier 2 · replay (TRCTC)", 34, 0.85, 4),
        "tier3_match_both":     (PD["muted"], "tier 3 · model-match (MBCTC)", 30, 0.8, 3),
    }

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.edgecolor": PD["rule"],
        "axes.linewidth": 0.8,
        "axes.labelcolor": PD["ink"],
        "text.color": PD["ink"],
        "xtick.color": PD["muted"],
        "ytick.color": PD["muted"],
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    fig = plt.figure(figsize=(13.5, 5.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], height_ratios=[1, 1],
                          hspace=0.55, wspace=0.22)
    axA = fig.add_subplot(gs[:, 0])   # scatter spans full height
    axBar = fig.add_subplot(gs[0, 1])  # signal-strength bars (top right)
    axB = fig.add_subplot(gs[1, 1])    # detection matrix (bottom right)

    # ---- (a) scatter --------------------------------------------------------
    ax = axA
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    xmax = max(max((r.kl_nats for r in reps), default=0.5)
               for reps in tiers.values()) * 1.10
    ymax = max(max((r.cce_nats for r in reps), default=1.0)
               for reps in tiers.values()) * 1.08
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)

    # benign band: KL <= marg_max and CCE in [lo, hi]
    ax.add_patch(Rectangle((0, benign_cce_lo), benign_marg_max,
                           benign_cce_hi - benign_cce_lo,
                           facecolor=PD["paleteal"], edgecolor="none",
                           alpha=0.5, zorder=0))
    # threshold lines extend across; labels go in the right margin (whitespace)
    ax.axvline(benign_marg_max, color=PD["teal"], ls="--", lw=1.1, zorder=1)
    ax.axhline(benign_cce_lo, color=PD["rule"], ls=":", lw=1.0, zorder=1)
    ax.axhline(benign_cce_hi, color=PD["rule"], ls=":", lw=1.0, zorder=1)

    # threshold labels at the right edge, in the margin past the data
    xlab = xmax * 0.992
    ax.text(xlab, benign_cce_hi, "benign CCE\nupper bound", color=PD["muted"],
            fontsize=7.5, ha="right", va="bottom", style="italic", zorder=4)
    ax.text(xlab, benign_cce_lo, "benign CCE\nlower bound", color=PD["muted"],
            fontsize=7.5, ha="right", va="top", style="italic", zorder=4)
    ax.text(benign_marg_max + 0.012, ymax * 0.985, "benign KL\nupper bound",
            color=PD["teal"], fontsize=7.5, ha="left", va="top",
            style="italic", zorder=4)

    # region cue: the low-CCE zone below the band is the covert-channel signature
    ax.annotate("low-entropy region\n(covert-channel signature)",
                xy=(xmax * 0.16, benign_cce_lo * 0.42),
                color=PD["navy"], fontsize=8, ha="center", va="center",
                alpha=0.75, style="italic", zorder=1)

    order = ["benign", "tier1_burst", "tier2_marginal_match", "tier3_match_both"]
    for k in order:
        color, lbl, sz, alpha, z = series[k]
        reps = tiers[k]
        xs = [r.kl_nats for r in reps]
        ys = [r.cce_nats for r in reps]
        ax.scatter(xs, ys, s=sz, c=color, alpha=alpha, edgecolors="white",
                   linewidths=0.4, label=lbl, zorder=z)

    ax.set_xlabel("Shape test:  marginal KL from benign (nats)  →", fontsize=9.5,
                  color=PD["ink"])
    ax.set_ylabel("Regularity test:  corrected conditional entropy (nats)",
                  fontsize=9.5, color=PD["ink"])
    ax.set_title("(a)  Per-window shape vs. regularity", color=PD["navy"],
                 loc="left", pad=10)
    ax.tick_params(length=3)
    # legend in the genuinely empty upper-right region (past the KL threshold)
    leg = ax.legend(fontsize=8, loc="upper right",
                    bbox_to_anchor=(0.995, 0.80),
                    framealpha=0.96, edgecolor=PD["rule"], borderpad=0.7,
                    handletextpad=0.5)
    leg.get_frame().set_linewidth(0.6)

    # ---- (b) signal-strength bars: fraction of windows beyond benign band ---
    ax = axBar
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    tier_keys = ["tier1_burst", "tier2_marginal_match", "tier3_match_both"]
    tier_short = ["Tier 1\nburst", "Tier 2\nreplay", "Tier 3\nmodel-match"]
    shape_frac = [detection_matrix[k]["frac_windows_above_marginal"] * 100
                  for k in tier_keys]
    reg_frac = [detection_matrix[k]["frac_windows_outside_cce_band"] * 100
                for k in tier_keys]
    x = np.arange(len(tier_keys))
    w = 0.38
    ax.bar(x - w / 2, shape_frac, w, color=PD["navy"], label="shape (KL)",
           zorder=3)
    ax.bar(x + w / 2, reg_frac, w, color=PD["teal"], label="regularity (CCE)",
           zorder=3)
    # detection threshold line
    thr = 5.0
    ax.axhline(thr, color=PD["muted"], ls="--", lw=1.0, zorder=2)
    ax.text(len(tier_keys) - 0.5, thr + 1.2, "detection threshold",
            color=PD["muted"], fontsize=7, ha="right", va="bottom", style="italic")
    ax.set_xticks(x)
    ax.set_xticklabels(tier_short, fontsize=8, color=PD["ink"])
    ax.set_ylabel("windows beyond\nbenign band (%)", fontsize=8.5, color=PD["ink"])
    ax.set_ylim(0, max(max(shape_frac), max(reg_frac)) * 1.25)
    ax.tick_params(length=3, labelsize=8)
    ax.set_title("(b)  Signal strength per test", color=PD["navy"],
                 loc="left", pad=8)
    leg = ax.legend(fontsize=7.5, loc="upper right", framealpha=0.95,
                    edgecolor=PD["rule"], borderpad=0.5, handlelength=1.2)
    leg.get_frame().set_linewidth(0.6)

    # ---- (c) detection matrix ----------------------------------------------
    ax = axB
    for sp in ax.spines.values():
        sp.set_visible(False)
    tier_keys = ["tier1_burst", "tier2_marginal_match", "tier3_match_both"]
    test_keys = ["caught_by_shape_test_marginal_kl", "caught_by_regularity_test_cce"]
    test_lbl = ["Shape\n(KL)", "Regularity\n(CCE)"]

    n_t, n_s = len(tier_keys), len(test_keys)
    for si, sk in enumerate(test_keys):
        for ti, tk in enumerate(tier_keys):
            detected = detection_matrix[tk][sk]
            face = PD["teal"] if detected else "#EDF0F1"
            txtc = "white" if detected else PD["muted"]
            y = (n_s - 1 - si)
            ax.add_patch(Rectangle((ti - 0.46, y - 0.46), 0.92, 0.92,
                                   facecolor=face, edgecolor="none", zorder=1))
            ax.text(ti, y, "detected" if detected else "evades",
                    ha="center", va="center", fontsize=9.5,
                    color=txtc, weight="bold", zorder=2)
    ax.set_xlim(-0.55, n_t - 0.45)
    ax.set_ylim(-0.6, n_s - 0.4)
    ax.set_xticks(range(n_t))
    ax.set_xticklabels(["Tier 1\nburst\n(IPCTC)", "Tier 2\nreplay\n(TRCTC)",
                        "Tier 3\nmodel-match\n(MBCTC)"],
                       fontsize=8.5, color=PD["ink"])
    ax.set_yticks([n_s - 1, 0])
    ax.set_yticklabels(test_lbl, fontsize=9, color=PD["ink"])
    ax.tick_params(length=0)
    ax.set_title("(c)  Detection outcome", color=PD["navy"], loc="left", pad=10)
    ax.set_aspect("auto")

    fig.suptitle("Egress-channel anomaly detection: shape (KL) + regularity "
                 "(CCE) tests vs. a three-tier adversary ladder",
                 fontsize=12, color=PD["navy"], weight="bold", y=1.01)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = out_dir / "m4_detection.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "m4_detection.pdf", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=300,
                    help="requests per sequence (>=200 for non-degenerate windows)")
    ap.add_argument("--bitstring", default="1011001")
    ap.add_argument("--burst-len", type=int, default=50)
    ap.add_argument("--window", type=int, default=20,
                    help="window for the marginal-KL shape test (small so it "
                         "localises a short burst)")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--cce-window", type=int, default=60,
                    help="window for the CCE regularity test. CCE estimates an "
                         "entropy rate over patterns up to length m_max, so it "
                         "needs enough samples for length-m patterns to recur; "
                         "a 20-request window is too short. Empirically >=60 for "
                         "the 6-symbol plot vocabulary at m_max=4. (The marginal "
                         "KL shape test still uses the smaller --window so it can "
                         "localise a short burst.)")
    ap.add_argument("--cce-m-max", type=int, default=4)
    ap.add_argument("--cal-n", type=int, default=4000,
                    help="benign calibration corpus size; larger sharpens the "
                         "benign CCE band and marginal baseline.")
    ap.add_argument("--seed", type=int, default=20260529)
    ap.add_argument("--plot", action="store_true",
                    help="emit m4_detection.png/.pdf (shape-vs-regularity "
                         "scatter + detection matrix). Requires matplotlib.")
    args = ap.parse_args()

    rng_cal = np.random.default_rng(args.seed)
    rng_t1 = np.random.default_rng(args.seed + 2)
    rng_t2 = np.random.default_rng(args.seed + 3)
    rng_t3 = np.random.default_rng(args.seed + 4)

    # Calibration corpus (larger than test sequences) -> sharp baselines.
    cal_seq, _ = gen_benign(args.cal_n, rng_cal)
    baseline_q = calibrate_baseline(cal_seq, DEFAULT_PLOT_TYPES)

    def run(seq, par):
        # SHAPE test (marginal KL) on the small window so it localises a burst.
        reports = sliding_windows(seq, par, baseline_q,
                                  window=args.window, stride=args.stride,
                                  threshold_nats=None)
        # REGULARITY test (CCE) recomputed per report over a WIDER window
        # centred on the same position, because the entropy-rate estimate needs
        # enough samples for length-m patterns to recur. CCE's correction term
        # handles residual finite-sample bias (that's its whole point).
        half = args.cce_window // 2
        for r in reports:
            centre = (r.window_start + r.window_end) // 2
            lo = max(0, centre - half)
            hi = min(len(seq), lo + args.cce_window)
            lo = max(0, hi - args.cce_window)
            r.cce_nats = corrected_conditional_entropy(
                seq[lo:hi], DEFAULT_PLOT_TYPES, m_max=args.cce_m_max)
        return reports

    # Benign NULL bands from MANY benign sequences. The band percentiles must be
    # stable enough that a FRESH benign draw (tier-3, which is benign by
    # construction) sits inside them; ~28 windows from one n=300 sequence is far
    # too few, so we pool windows across many null sequences. Marginal KL is
    # one-sided (anomaly = high). CCE is TWO-sided: a repeated-pattern channel
    # LOWERS the entropy rate (CCE below band) and a randomising channel RAISES
    # it (above band).
    n_null = 50
    null_marg, null_cce = [], []
    benign_reports = None
    for j in range(n_null):
        bseq, bpar = gen_benign(args.n, np.random.default_rng(args.seed + 1000 + j))
        reps = run(bseq, bpar)
        if benign_reports is None:
            benign_reports = reps          # keep one for per-tier summary table
        null_marg.extend(r.kl_nats for r in reps)
        null_cce.extend(r.cce_nats for r in reps)
    # Use slightly wider-than-extreme percentiles for a stable null envelope.
    benign_marg_max = float(np.percentile(null_marg, 99.5))
    benign_cce_lo = float(np.percentile(null_cce, 0.5))
    benign_cce_hi = float(np.percentile(null_cce, 99.5))

    t1_seq, t1_par = gen_burst(args.n, args.bitstring, rng_t1, burst_len=args.burst_len)
    t2_seq, t2_par = gen_marginal_match(args.n, args.bitstring, rng_t2)
    t3_seq, t3_par = gen_match_both(args.n, rng_t3)
    t1_reports = run(t1_seq, t1_par)
    t2_reports = run(t2_seq, t2_par)
    t3_reports = run(t3_seq, t3_par)

    tiers = {
        "benign":               benign_reports,
        "tier1_burst":          t1_reports,
        "tier2_marginal_match": t2_reports,
        "tier3_match_both":     t3_reports,
    }
    summaries = {k: summarize_reports(v) for k, v in tiers.items()}
    breaches = {k: _envelope_breaches(v, benign_marg_max, benign_cce_lo, benign_cce_hi)
                for k, v in tiers.items()}

    detection_matrix = {}
    # Require a fraction of windows beyond the benign band (not a single tail
    # window, which is noise) to declare detection. A burst/TRCTC produces a
    # run of consecutive breaching windows; benign noise produces isolated ones.
    min_frac = 0.05
    for k in ("tier1_burst", "tier2_marginal_match", "tier3_match_both"):
        b = breaches[k]
        nw = max(1, b["n_windows"])
        shape_frac = b["n_above_marginal_envelope"] / nw
        reg_frac = (b["n_below_cce_band"] + b["n_above_cce_band"]) / nw
        detection_matrix[k] = {
            "caught_by_shape_test_marginal_kl": shape_frac > min_frac,
            "caught_by_regularity_test_cce": reg_frac > min_frac,
            "frac_windows_above_marginal": round(shape_frac, 3),
            "frac_windows_outside_cce_band": round(reg_frac, 3),
            "n_windows_above_marginal": b["n_above_marginal_envelope"],
            "n_windows_below_cce_band": b["n_below_cce_band"],
            "n_windows_above_cce_band": b["n_above_cce_band"],
        }

    out = {
        "schema_version": "m4-anomaly-report-only-v3-cce",
        "mode": "report-only (threshold_nats=None; detector does not fire)",
        "method": "Shape test = first-order relative entropy (KL) vs benign "
                  "marginal; regularity test = corrected conditional entropy "
                  "(CCE), Gianvecchio & Wang 2007 (IEEE TDSC), after Cabuk 2004 "
                  "and Porta et al. 1998. Two-sided CCE band: below = over-"
                  "regular (repeated-pattern channel), above = over-random.",
        "n_per_sequence": args.n, "window": args.window, "stride": args.stride,
        "cce_window": args.cce_window, "cce_m_max": args.cce_m_max,
        "bitstring": args.bitstring, "burst_len": args.burst_len,
        "baseline_marginal": dict(zip(DEFAULT_PLOT_TYPES,
                                      [float(x) for x in baseline_q])),
        "benign_band": {
            "marginal_kl_p99_5": benign_marg_max,
            "cce_p0_5": benign_cce_lo,
            "cce_p99_5": benign_cce_hi,
        },
        "per_tier_summary": summaries,
        "detection_matrix": detection_matrix,
        "adversary_taxonomy": {
            "tier1_burst": "analogue of Cabuk IPCTC (on/off interval channel): "
                           "an over-concentrated burst. Differs in both shape "
                           "and regularity from benign; caught by the shape "
                           "test (and run-length / param entropy).",
            "tier2_marginal_match": "analogue of Cabuk TRCTC (time-replay): "
                                    "replays benign symbols in a bit-determined "
                                    "order, so the first-order distribution ~ "
                                    "benign. Evades the shape test; the repeated "
                                    "encoding structure lowers the entropy rate, "
                                    "caught by the CCE regularity test.",
            "tier3_match_both": "analogue of model-based / Jitterbug stealthy "
                                "channels: matches benign marginal AND benign "
                                "regularity (drawn from the benign chain). "
                                "Evades both tests by construction. Consistent "
                                "with the literature result that model-matching "
                                "channels evade general detectors, and with "
                                "Sellke et al.'s provably-undetectable channels "
                                "over i.i.d. cover. Detecting it needs higher-"
                                "order models and a real benign corpus to "
                                "calibrate — pilot work.",
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "m4_anomaly_report.json").write_text(json.dumps(out, indent=2))
    detail = {k: [asdict(r) for r in v] for k, v in tiers.items()}
    (args.out_dir / "m4_windows.json").write_text(json.dumps(detail, indent=2))

    print("=== M4 anomaly (report-only): shape (KL) + regularity (CCE) tests ===")
    print(f"benign band: marginal_kl_p99={benign_marg_max:.4f}  "
          f"cce_p1={benign_cce_lo:.4f}  cce_p99={benign_cce_hi:.4f}")
    print(f"\n{'tier':<22}{'marg_kl_max':>12}{'cce_min':>10}{'cce_max':>10}"
          f"{'shape':>7}{'regul':>7}")
    for k in ("benign", "tier1_burst", "tier2_marginal_match", "tier3_match_both"):
        s = summaries[k]
        mm = s["marginal_kl_nats"]["max"]
        cmin = s["cce_nats"]["min"]; cmax = s["cce_nats"]["max"]
        if k == "benign":
            print(f"{k:<22}{mm:>12.4f}{cmin:>10.4f}{cmax:>10.4f}{'-':>7}{'-':>7}")
        else:
            d = detection_matrix[k]
            print(f"{k:<22}{mm:>12.4f}{cmin:>10.4f}{cmax:>10.4f}"
                  f"{('Y' if d['caught_by_shape_test_marginal_kl'] else 'n'):>7}"
                  f"{('Y' if d['caught_by_regularity_test_cce'] else 'n'):>7}")
    print("\nadversary taxonomy (covert-timing-channel literature analogues):")
    for k, v in out["adversary_taxonomy"].items():
        print(f"  {k}: {v}")
    print(f"\n[out] {args.out_dir / 'm4_anomaly_report.json'}")
    print(f"[out] {args.out_dir / 'm4_windows.json'}")

    if args.plot:
        try:
            ppath = make_plots(args.out_dir, tiers, detection_matrix,
                               benign_marg_max, benign_cce_lo, benign_cce_hi)
            print(f"[out] {ppath}")
            print(f"[out] {args.out_dir / 'm4_detection.pdf'}")
        except Exception as e:  # noqa: BLE001
            print(f"[plot][warn] could not render: {type(e).__name__}: {e}",
                  file=sys.stderr)

    dm = detection_matrix
    if not dm["tier1_burst"]["caught_by_shape_test_marginal_kl"]:
        print("\n[warn] tier1 burst NOT caught by shape test - unexpected.",
              file=sys.stderr)
    if not dm["tier2_marginal_match"]["caught_by_regularity_test_cce"]:
        print("\n[warn] tier2 (TRCTC-analogue) NOT caught by CCE regularity test "
              "- the regularity feature is not separating; investigate before "
              "citing.", file=sys.stderr)
    if dm["tier3_match_both"]["caught_by_shape_test_marginal_kl"] or \
       dm["tier3_match_both"]["caught_by_regularity_test_cce"]:
        print("\n[note] tier3 (MBCTC-analogue) was flagged by a first-order "
              "signal; it should be indistinguishable from benign by "
              "construction - check the benign chain / band width.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
