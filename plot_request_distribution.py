"""
plot_request_distribution.py — report-only anomaly signal over plot-request
sequences (Phase 1, M4).

WHAT THIS MEASURES
    An auditor who is exfiltrating information through the egress channel can
    encode a bitstring in the *sequence* of plot requests — e.g. 50 requests in
    a row for the same plot type with parameters that vary in a structured way
    (a covert channel). A benign auditor's plot-type usage looks like draws from
    a stable categorical distribution; a steganographic sequence does not (it is
    over-concentrated, or its parameter sequence carries unusual regularity).

    This module computes a per-window KL divergence of the observed plot-type
    distribution from a calibrated baseline categorical, plus a parameter-
    sequence regularity signal. Both are returned as continuous nats; NO fire
    decision is made unless an explicit threshold is set.

REPORT-ONLY (M4 scope reduction, per handoff)
    `threshold_nats=None` (the default) means the detector NEVER fires — it only
    reports the divergence signal. This ships the instrumentation and the
    measured signal separation (steganographic vs benign) WITHOUT committing to
    a calibrated operating point, which is deferred to the pilot (white-paper
    §4.3.6, "calibration deferred"). When the pilot calibrates a threshold from
    a real benign corpus, set threshold_nats to that value and the same code
    path becomes enforcing.

NOT A PYSYFT ENDPOINT BY ITSELF
    This is a pure-Python signal computed over the engagement ledger's recorded
    bundle sequence (endpoint + plot params). It can run (a) offline over a
    completed run's parquet for the writeup, or (b) inside the egress encoder /
    a PySyft endpoint as a post-bundle hook. Phase 1 uses path (a): report-only,
    offline, no enforcement. Keeping it dependency-free (numpy only) means it
    drops into either location unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


# Default plot-type vocabulary for the egress pipeline's plot registry. The
# baseline categorical is over THESE types; an observed type outside the vocab
# is folded into "other" so the KL stays finite.
DEFAULT_PLOT_TYPES = (
    "attention_entropy_heatmap",
    "residual_norm_curve",
    "routing_load_bar",
    "logit_lens_panel",
    "steering_delta_trace",
    "other",
)

# Laplace smoothing so a zero observed/baseline cell never sends KL to inf.
_EPS = 1e-9


@dataclass
class AnomalyReport:
    """Per-window anomaly signal. Report-only: `fired` is None unless a
    threshold was supplied."""
    window_start: int
    window_end: int
    n_requests: int
    kl_nats: float                  # KL(observed || baseline) over plot types (marginal)
    cce_nats: float                 # corrected conditional entropy (regularity; G&W 2007)
    max_run_length: int             # longest run of identical plot type
    param_autocorr: float           # lag-1 autocorrelation of a param sequence
    param_entropy_bits: float       # Shannon entropy of the param sequence
    fired: Optional[bool] = None    # None = report-only; bool only if threshold set
    threshold_nats: Optional[float] = None
    notes: str = ""


def _categorical_from_counts(counts: np.ndarray) -> np.ndarray:
    counts = counts.astype(np.float64) + _EPS
    return counts / counts.sum()


def kl_divergence(observed_p: np.ndarray, baseline_q: np.ndarray) -> float:
    """KL(observed || baseline) in nats. Both must be normalised, same length.
    Smoothed at the caller so neither has hard zeros."""
    p = observed_p
    q = baseline_q
    return float(np.sum(p * np.log(p / q)))


def _max_run_length(seq: Sequence) -> int:
    """Longest run of identical consecutive elements. A steganographic 'N in a
    row' burst shows up directly here."""
    if not len(seq):
        return 0
    best = run = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best


def _lag1_autocorr(x: np.ndarray) -> float:
    """Lag-1 autocorrelation of a numeric parameter sequence. A bitstring
    encoded as smoothly-varying params has higher |autocorr| than benign noise."""
    if len(x) < 3:
        return 0.0
    x = x.astype(np.float64)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom < _EPS:
        return 0.0
    return float(np.dot(x[:-1], x[1:]) / denom)


def _param_entropy_bits(x: np.ndarray, n_bins: int = 16) -> float:
    """Shannon entropy (bits) of a quantised parameter sequence. Low entropy =>
    the parameter takes few distinct values in a structured way (suspicious);
    high entropy => spread out (benign-ish)."""
    if len(x) == 0:
        return 0.0
    x = np.asarray(x, dtype=np.float64)
    if np.ptp(x) < _EPS:
        return 0.0  # constant param: zero entropy
    hist, _ = np.histogram(x, bins=n_bins)
    p = hist.astype(np.float64)
    p = p[p > 0]
    p = p / p.sum()
    return float(-np.sum(p * np.log2(p)))


def calibrate_baseline(
    plot_type_sequence: Sequence[str],
    plot_types: Sequence[str] = DEFAULT_PLOT_TYPES,
) -> np.ndarray:
    """Build a baseline categorical from a (presumed benign) calibration
    sequence. Returns a normalised, smoothed probability vector aligned to
    `plot_types` (unknowns -> 'other')."""
    idx = {t: i for i, t in enumerate(plot_types)}
    other_i = idx.get("other", len(plot_types) - 1)
    counts = np.zeros(len(plot_types), dtype=np.float64)
    for t in plot_type_sequence:
        counts[idx.get(t, other_i)] += 1
    return _categorical_from_counts(counts)


def _entropy_nats(counts: np.ndarray) -> float:
    """Shannon entropy (nats) of a count vector."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-np.sum(p * np.log(p)))


def corrected_conditional_entropy(
    plot_type_sequence: Sequence[str],
    plot_types: Sequence[str] = DEFAULT_PLOT_TYPES,
    m_max: int = 4,
) -> float:
    """Corrected conditional entropy (CCE) of a symbol sequence, in nats — the
    entropy-rate estimate of Gianvecchio & Wang (2007), itself adapting Porta
    et al. (1998). This is the standard *regularity* test from the covert-
    timing-channel literature, here applied to the plot-request symbol stream.

    Why this, not a bespoke transition-KL: entropy rate is the established
    second/higher-order regularity measure, and CCE is specifically the finite-
    sample-corrected estimator — it solves the small-window problem (a naive
    conditional entropy collapses to zero once length-m patterns stop repeating,
    even for i.i.d. data) via a correction term, WITHOUT us having to hand-tune
    a window size.

    Definition (G&W Eq. 6):
        CCE(X_m | X_1..X_{m-1}) = CE(X_m | X_1..X_{m-1}) + perc(X_m)·EN(X_1)
    where CE is the empirical conditional entropy at pattern length m,
    perc(X_m) is the fraction of length-m patterns that occur exactly once, and
    EN(X_1) is the first-order entropy. The entropy-rate estimate is the
    MINIMUM of CCE over m (CE decreases in m while the correction term rises,
    so a minimum exists).

    Interpretation against a benign CCE band (the detector's job, done in the
    driver, not here):
      - a channel that injects a REPEATED encoding pattern (e.g. an on/off
        burst, IPCTC-style) lowers the entropy rate -> CCE BELOW the benign band;
      - a channel that injects randomness raises it -> CCE ABOVE the band.
    A single distance-from-benign scalar (my earlier transition-KL) could not
    distinguish these two directions; CCE-vs-band does.
    """
    idx = {t: i for i, t in enumerate(plot_types)}
    other_i = idx.get("other", len(plot_types) - 1)
    seq = [idx.get(t, other_i) for t in plot_type_sequence]
    n = len(seq)
    if n < 2:
        return 0.0
    en1 = _entropy_nats(np.bincount([s for s in seq], minlength=len(plot_types))
                        .astype(np.float64))  # first-order entropy EN(X_1)

    def joint_entropy(m: int) -> float:
        if n < m:
            return 0.0
        patterns = {}
        for i in range(n - m + 1):
            key = tuple(seq[i:i + m])
            patterns[key] = patterns.get(key, 0) + 1
        counts = np.array(list(patterns.values()), dtype=np.float64)
        return _entropy_nats(counts)

    def perc_unique(m: int) -> float:
        if n < m:
            return 1.0
        patterns = {}
        for i in range(n - m + 1):
            key = tuple(seq[i:i + m])
            patterns[key] = patterns.get(key, 0) + 1
        n_pat = sum(patterns.values())
        n_unique = sum(1 for v in patterns.values() if v == 1)
        return (n_unique / n_pat) if n_pat else 1.0

    # CE(X_m | X_1..X_{m-1}) = H(X_1..X_m) - H(X_1..X_{m-1}); CCE = CE + corr.
    cce_values = []
    h_prev = joint_entropy(1)
    for m in range(2, m_max + 1):
        h_m = joint_entropy(m)
        ce = h_m - h_prev
        cce = ce + perc_unique(m) * en1
        cce_values.append(cce)
        h_prev = h_m
    if not cce_values:
        return en1
    return float(max(0.0, min(cce_values)))


def analyze_window(
    plot_type_sequence: Sequence[str],
    param_sequence: Optional[Sequence[float]],
    baseline_q: np.ndarray,
    *,
    plot_types: Sequence[str] = DEFAULT_PLOT_TYPES,
    window_start: int = 0,
    threshold_nats: Optional[float] = None,
    cce_m_max: int = 4,
) -> AnomalyReport:
    """Compute the anomaly signal over one window of plot requests.

    Two divergence signals:
      kl_nats  — SHAPE test: KL(observed plot-type distribution || benign
                 marginal). Catches distribution-shape anomalies (e.g. an
                 over-concentrated burst).
      cce_nats — REGULARITY test: corrected conditional entropy (Gianvecchio
                 & Wang 2007). Catches sequential-regularity anomalies that
                 leave the first-order shape intact. Compared against a benign
                 CCE band by the caller (below the band => over-regular /
                 repeated-pattern channel; above => over-random).

    threshold_nats=None  -> report-only: `fired` stays None.
    threshold_nats=float -> `fired` = (kl_nats >= threshold_nats). Phase 1 does
                            NOT set this (calibration deferred).
    """
    idx = {t: i for i, t in enumerate(plot_types)}
    other_i = idx.get("other", len(plot_types) - 1)
    counts = np.zeros(len(plot_types), dtype=np.float64)
    for t in plot_type_sequence:
        counts[idx.get(t, other_i)] += 1
    observed_p = _categorical_from_counts(counts)

    kl = kl_divergence(observed_p, baseline_q)
    cce = corrected_conditional_entropy(plot_type_sequence, plot_types, m_max=cce_m_max)
    max_run = _max_run_length(list(plot_type_sequence))

    if param_sequence is not None and len(param_sequence):
        params = np.asarray(param_sequence, dtype=np.float64)
        autocorr = _lag1_autocorr(params)
        ent = _param_entropy_bits(params)
    else:
        autocorr = 0.0
        ent = 0.0

    fired = None
    if threshold_nats is not None:
        fired = bool(kl >= threshold_nats)

    return AnomalyReport(
        window_start=window_start,
        window_end=window_start + len(plot_type_sequence),
        n_requests=len(plot_type_sequence),
        kl_nats=kl,
        cce_nats=cce,
        max_run_length=max_run,
        param_autocorr=autocorr,
        param_entropy_bits=ent,
        fired=fired,
        threshold_nats=threshold_nats,
        notes="report-only" if threshold_nats is None else "enforcing",
    )


def sliding_windows(
    plot_type_sequence: Sequence[str],
    param_sequence: Optional[Sequence[float]],
    baseline_q: np.ndarray,
    *,
    window: int = 20,
    stride: int = 10,
    plot_types: Sequence[str] = DEFAULT_PLOT_TYPES,
    threshold_nats: Optional[float] = None,
    cce_m_max: int = 4,
) -> list[AnomalyReport]:
    """Run analyze_window over sliding windows. A localized over-concentrated
    burst is caught by the window whose marginal KL spikes (shape test); a
    sequential-regularity anomaly that preserves the first-order shape is caught
    by the window's CCE departing from the benign CCE band (regularity test,
    compared in the caller)."""
    n = len(plot_type_sequence)
    reports: list[AnomalyReport] = []
    if n == 0:
        return reports
    start = 0
    while start < n:
        end = min(start + window, n)
        pseq = plot_type_sequence[start:end]
        qseq = (param_sequence[start:end] if param_sequence is not None else None)
        reports.append(analyze_window(
            pseq, qseq, baseline_q,
            plot_types=plot_types, window_start=start,
            threshold_nats=threshold_nats,
            cce_m_max=cce_m_max,
        ))
        if end == n:
            break
        start += stride
    return reports
