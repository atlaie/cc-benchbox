#!/usr/bin/env python3
"""
analyze_interdeploy.py — Inter-deploy robustness analysis for Experiment #1.

Consumes 5 cells from runs/phase3_robustness/:
    R1-D1-off, R1-D1-on   (deploy generation A, debug mode)
    R1-D2-off, R1-D2-on   (deploy generation B, debug mode)
    R1-D3-on-attest        (deploy generation C, attestation-on / production)

Produces:
    1. Within-deploy paired Δ% with BCa CIs (D1 pair, D2 pair)
    2. Inter-deploy comparison: D1 vs D2 on each CC arm
    3. Attestation gap: D1-on (debug) vs D3-on-attest (production), paired
    4. Tail percentiles: p50, p95, p99, p99.9 per cell
    5. Pooled inter-deploy Δ% with variance propagated

Usage:
    python analyze_interdeploy.py \
        --data-dir runs/phase3_robustness \
        --output-dir runs/phase3_robustness/analysis
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Reuse the BCa machinery from the main analyzer
from analyze_cc_deltas import (
    bootstrap_paired_delta,
    bootstrap_unpaired_delta,
    setup_style,
    _save,
    PALETTE,
)

setup_style()

# ── Cell registry ────────────────────────────────────────────────────────────

CELLS = {
    "R1-D1-off":       {"deploy": "D1", "cc": "off", "debug": True,  "generation": "A"},
    "R1-D1-on":        {"deploy": "D1", "cc": "on",  "debug": True,  "generation": "A"},
    "R1-D2-off":       {"deploy": "D2", "cc": "off", "debug": True,  "generation": "B"},
    "R1-D2-on":        {"deploy": "D2", "cc": "on",  "debug": True,  "generation": "B"},
    "R1-D3-on-attest": {"deploy": "D3", "cc": "on",  "debug": False, "generation": "C"},
}

PERCENTILES = [50, 95, 99, 99.9]


# ── Data loading ─────────────────────────────────────────────────────────────

def load_all(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load requests.parquet for each cell. Returns {cell_id: DataFrame}."""
    out = {}
    for cell_id in CELLS:
        p = data_dir / cell_id / "requests.parquet"
        if not p.exists():
            print(f"  [warn] missing {p}", file=sys.stderr)
            continue
        df = pd.read_parquet(p)
        # Filter to successful requests only
        if "error" in df.columns:
            df = df[df["error"].isna()].copy()
        out[cell_id] = df
    return out


# ── Per-cell summary ─────────────────────────────────────────────────────────

def cell_summary(cell_id: str, df: pd.DataFrame) -> dict:
    w = df["wall_seconds"].to_numpy()
    row = {"cell_id": cell_id, "n": len(w)}
    for p in PERCENTILES:
        row[f"p{p}"] = float(np.percentile(w, p))
    row["mean"] = float(np.mean(w))
    row.update(CELLS[cell_id])
    return row


# ── Paired CC delta ──────────────────────────────────────────────────────────

def paired_cc_delta(
    off_df: pd.DataFrame, on_df: pd.DataFrame,
    label: str, n_resamples: int = 10_000, alpha: float = 0.05,
) -> dict:
    """Paired BCa delta on wall_seconds p50, aligned by (pair_id, prompt_class)."""
    off_idx = off_df.set_index(["pair_id", "prompt_class"])["wall_seconds"]
    on_idx = on_df.set_index(["pair_id", "prompt_class"])["wall_seconds"]
    common = sorted(set(off_idx.index) & set(on_idx.index))
    if len(common) < 2:
        return {"label": label, "error": "insufficient paired data"}

    off_arr = off_idx.loc[common].to_numpy(dtype=float)
    on_arr = on_idx.loc[common].to_numpy(dtype=float)

    results = {"label": label, "n_paired": len(common)}

    for stat_name, stat_fn in [
        ("p50", lambda x: float(np.median(x))),
        ("p95", lambda x: float(np.percentile(x, 95))),
        ("p99", lambda x: float(np.percentile(x, 99))),
        ("mean", lambda x: float(np.mean(x))),
    ]:
        off_val = stat_fn(off_arr)
        on_val = stat_fn(on_arr)
        point, lo, hi = bootstrap_paired_delta(
            off_arr, on_arr, stat_fn,
            n_resamples=n_resamples, alpha=alpha,
        )
        rel = 100 * point / off_val if abs(off_val) > 1e-12 else float("nan")
        rel_lo = 100 * lo / off_val if abs(off_val) > 1e-12 else float("nan")
        rel_hi = 100 * hi / off_val if abs(off_val) > 1e-12 else float("nan")
        results[f"{stat_name}_off"] = off_val
        results[f"{stat_name}_on"] = on_val
        results[f"{stat_name}_abs_delta"] = point
        results[f"{stat_name}_rel_pct"] = rel
        results[f"{stat_name}_rel_ci"] = [rel_lo, rel_hi]

    return results


# ── Attestation gap ──────────────────────────────────────────────────────────

def attestation_gap(
    debug_on_df: pd.DataFrame, attest_on_df: pd.DataFrame,
    n_resamples: int = 10_000, alpha: float = 0.05,
) -> dict:
    """Paired comparison: debug CC-on vs attestation-on CC-on.
    Both are CC-on; the difference is attestation. Paired by (pair_id, prompt_class)."""
    d_idx = debug_on_df.set_index(["pair_id", "prompt_class"])["wall_seconds"]
    a_idx = attest_on_df.set_index(["pair_id", "prompt_class"])["wall_seconds"]
    common = sorted(set(d_idx.index) & set(a_idx.index))
    if len(common) < 2:
        return {"error": "insufficient paired data"}

    d_arr = d_idx.loc[common].to_numpy(dtype=float)
    a_arr = a_idx.loc[common].to_numpy(dtype=float)

    stat_fn = lambda x: float(np.median(x))
    d_p50 = stat_fn(d_arr)
    a_p50 = stat_fn(a_arr)
    point, lo, hi = bootstrap_paired_delta(
        d_arr, a_arr, stat_fn,
        n_resamples=n_resamples, alpha=alpha,
    )
    rel = 100 * point / d_p50 if abs(d_p50) > 1e-12 else float("nan")
    rel_lo = 100 * lo / d_p50 if abs(d_p50) > 1e-12 else float("nan")
    rel_hi = 100 * hi / d_p50 if abs(d_p50) > 1e-12 else float("nan")

    return {
        "label": "attestation gap (D1-on debug vs D3-on-attest production)",
        "n_paired": len(common),
        "debug_p50": d_p50,
        "attest_p50": a_p50,
        "abs_delta_s": point,
        "abs_ci_s": [lo, hi],
        "rel_delta_pct": rel,
        "rel_ci_pct": [rel_lo, rel_hi],
    }


# ── Inter-deploy drift ───────────────────────────────────────────────────────

def interdeploy_drift(
    d1_df: pd.DataFrame, d2_df: pd.DataFrame,
    cc_state: str, n_resamples: int = 10_000, alpha: float = 0.05,
) -> dict:
    """Unpaired comparison of the same CC arm across two deploy generations."""
    d1_w = d1_df["wall_seconds"].to_numpy(dtype=float)
    d2_w = d2_df["wall_seconds"].to_numpy(dtype=float)

    stat_fn = lambda x: float(np.median(x))
    d1_p50 = stat_fn(d1_w)
    d2_p50 = stat_fn(d2_w)
    point, lo, hi = bootstrap_unpaired_delta(
        d1_w, d2_w, stat_fn,
        n_resamples=n_resamples, alpha=alpha,
    )
    return {
        "label": f"inter-deploy drift CC-{cc_state} (D2 − D1)",
        "D1_p50": d1_p50,
        "D2_p50": d2_p50,
        "abs_delta_s": point,
        "abs_ci_s": [lo, hi],
        "n_D1": len(d1_w),
        "n_D2": len(d2_w),
    }


# ── Pooled inter-deploy Δ% ──────────────────────────────────────────────────

def pooled_interdeploy_delta(
    data: dict[str, pd.DataFrame],
    n_resamples: int = 10_000, alpha: float = 0.05,
) -> dict:
    """Pool D1 and D2 pairs, compute Δ% with inter-deploy variance folded in.

    Strategy: concatenate D1-off+D2-off as the off arm, D1-on+D2-on as the
    on arm. Pair by (pair_id, prompt_class) — since both deploys use the same
    prompt set, every prompt appears twice (once per deploy). We DON'T pair
    across deploys; instead we bootstrap at the prompt level from the pooled
    set, which naturally includes the inter-deploy variance in the CI.
    """
    off_frames = []
    on_frames = []
    for cell_id, df in data.items():
        meta = CELLS[cell_id]
        if meta["deploy"] not in ("D1", "D2"):
            continue
        tagged = df.copy()
        tagged["deploy"] = meta["deploy"]
        if meta["cc"] == "off":
            off_frames.append(tagged)
        elif meta["cc"] == "on" and meta["debug"]:
            on_frames.append(tagged)

    if not off_frames or not on_frames:
        return {"error": "missing deploy data"}

    off_all = pd.concat(off_frames, ignore_index=True)
    on_all = pd.concat(on_frames, ignore_index=True)

    # Unpaired bootstrap (can't pair across deploys — same pair_id appears
    # in both D1 and D2, so pairing would be ambiguous)
    stat_fn = lambda x: float(np.median(x))
    off_w = off_all["wall_seconds"].to_numpy(dtype=float)
    on_w = on_all["wall_seconds"].to_numpy(dtype=float)

    off_p50 = stat_fn(off_w)
    on_p50 = stat_fn(on_w)
    point, lo, hi = bootstrap_unpaired_delta(
        off_w, on_w, stat_fn,
        n_resamples=n_resamples, alpha=alpha,
    )
    rel = 100 * point / off_p50 if abs(off_p50) > 1e-12 else float("nan")
    rel_lo = 100 * lo / off_p50 if abs(off_p50) > 1e-12 else float("nan")
    rel_hi = 100 * hi / off_p50 if abs(off_p50) > 1e-12 else float("nan")

    return {
        "label": "pooled D1+D2 CC delta (unpaired bootstrap, inter-deploy variance included)",
        "n_off": len(off_w),
        "n_on": len(on_w),
        "off_p50": off_p50,
        "on_p50": on_p50,
        "abs_delta_s": point,
        "abs_ci_s": [lo, hi],
        "rel_delta_pct": rel,
        "rel_ci_pct": [rel_lo, rel_hi],
    }


# ── Figure: forest comparing D1, D2, pooled, attestation ────────────────────

def figure_interdeploy_forest(results: dict, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    # D1 pair
    d1 = results["cc_deltas"]["D1"]
    rows.append(("D1 (gen A, debug)", d1["p50_rel_pct"], d1["p50_rel_ci"]))
    # D2 pair
    d2 = results["cc_deltas"]["D2"]
    rows.append(("D2 (gen B, debug)", d2["p50_rel_pct"], d2["p50_rel_ci"]))
    # Pooled
    pooled = results["pooled_delta"]
    if "rel_delta_pct" in pooled:
        rows.append(("pooled D1+D2", pooled["rel_delta_pct"], pooled["rel_ci_pct"]))
    # Attestation — D3 vs D1-on as the CC delta
    # We need the CC delta for D3, which pairs D3-on-attest against D1-off (or D2-off)
    d3 = results.get("cc_delta_attest")
    if d3 and "p50_rel_pct" in d3:
        rows.append(("D3 (attest-on)", d3["p50_rel_pct"], d3["p50_rel_ci"]))
    # Primary matrix reference
    rows.append(("primary matrix (n=100)", 33.4, [33.0, 34.0]))

    fig, ax = plt.subplots(figsize=(10, 0.6 * len(rows) + 1.8))
    ax.axvspan(30.0, 38.0, color=PALETTE["muted"], alpha=0.28, zorder=0)
    ax.axvline(0, color=PALETTE["neutral"], linewidth=1.0, linestyle="--", alpha=0.5)

    y = np.arange(len(rows))
    colors = [PALETTE["on"]] * (len(rows) - 1) + [PALETTE["off"]]

    for i, (label, val, ci) in enumerate(rows):
        lo, hi = ci
        ax.errorbar([val], [y[i]], xerr=[[val - lo], [hi - val]],
                    fmt="o", capsize=5, color=colors[i], ecolor=colors[i],
                    markersize=8, linewidth=2.4, capthick=2.0, zorder=3)
        ax.text(hi + 0.8, y[i],
                f"{val:+.1f}% [{lo:+.1f}, {hi:+.1f}]",
                va="center", ha="left", fontsize=10.5, color=colors[i])

    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=11)
    ax.set_xlabel("CC overhead on wall p50 (%)", fontsize=11)
    ax.set_title("Inter-deploy robustness: CC delta across independent deploy generations\n"
                 "(paired BCa 95% CIs; shaded = +30–38% platform band)",
                 fontsize=12, pad=10)
    ax.tick_params(axis="y", length=0)
    plt.tight_layout()
    _save(fig, fig_dir / "interdeploy_forest")
    print(f"  wrote {fig_dir / 'interdeploy_forest.{{png,pdf}}'}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--n-resamples", type=int, default=10_000)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    print("Loading data...")
    data = load_all(args.data_dir)
    if len(data) < 4:
        print(f"FATAL: need at least 4 cells, found {len(data)}", file=sys.stderr)
        return 2

    results: dict = {}

    # ── 1. Per-cell summary ──────────────────────────────────────────────
    print("\n1. Per-cell summary (tail percentiles)")
    summaries = [cell_summary(cid, df) for cid, df in sorted(data.items())]
    results["cell_summaries"] = summaries
    for s in summaries:
        print(f"  {s['cell_id']:<22s}  n={s['n']:>4d}  "
              f"p50={s['p50']:.2f}  p95={s['p95']:.2f}  "
              f"p99={s['p99']:.2f}  p99.9={s['p99.9']:.2f}  "
              f"mean={s['mean']:.2f}")

    # ── 2. Within-deploy CC deltas ───────────────────────────────────────
    print("\n2. Within-deploy CC deltas (paired BCa)")
    cc_deltas = {}
    for deploy, off_id, on_id in [("D1", "R1-D1-off", "R1-D1-on"),
                                   ("D2", "R1-D2-off", "R1-D2-on")]:
        if off_id not in data or on_id not in data:
            print(f"  [skip] {deploy}: missing arm")
            continue
        d = paired_cc_delta(data[off_id], data[on_id],
                           label=f"{deploy} CC delta",
                           n_resamples=args.n_resamples, alpha=args.alpha)
        cc_deltas[deploy] = d
        ci = d["p50_rel_ci"]
        print(f"  {deploy}: Δp50 = {d['p50_rel_pct']:+.1f}% [{ci[0]:+.1f}, {ci[1]:+.1f}]  "
              f"(off={d['p50_off']:.2f}s, on={d['p50_on']:.2f}s, n={d['n_paired']})")
    results["cc_deltas"] = cc_deltas

    # ── 3. Inter-deploy drift ────────────────────────────────────────────
    print("\n3. Inter-deploy drift (same CC arm, different deploys)")
    drifts = {}
    for cc, d1_id, d2_id in [("off", "R1-D1-off", "R1-D2-off"),
                              ("on", "R1-D1-on", "R1-D2-on")]:
        if d1_id not in data or d2_id not in data:
            continue
        d = interdeploy_drift(data[d1_id], data[d2_id], cc,
                              n_resamples=args.n_resamples, alpha=args.alpha)
        drifts[cc] = d
        ci = d["abs_ci_s"]
        print(f"  CC-{cc}: D1 p50={d['D1_p50']:.3f}s, D2 p50={d['D2_p50']:.3f}s, "
              f"Δ={d['abs_delta_s']:+.3f}s [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    results["interdeploy_drift"] = drifts

    # ── 4. Attestation gap ───────────────────────────────────────────────
    print("\n4. Attestation gap (debug CC-on vs production CC-on)")
    if "R1-D1-on" in data and "R1-D3-on-attest" in data:
        att = attestation_gap(data["R1-D1-on"], data["R1-D3-on-attest"],
                              n_resamples=args.n_resamples, alpha=args.alpha)
        results["attestation_gap"] = att
        if "error" not in att:
            ci = att["rel_ci_pct"]
            print(f"  debug p50={att['debug_p50']:.3f}s, attest p50={att['attest_p50']:.3f}s")
            print(f"  Δ = {att['rel_delta_pct']:+.2f}% [{ci[0]:+.2f}, {ci[1]:+.2f}]  "
                  f"(abs={att['abs_delta_s']:+.3f}s [{att['abs_ci_s'][0]:+.3f}, {att['abs_ci_s'][1]:+.3f}])")
        else:
            print(f"  {att['error']}")

    # Also compute the CC delta for the attestation cell (D3-on vs D1-off)
    if "R1-D1-off" in data and "R1-D3-on-attest" in data:
        d3_cc = paired_cc_delta(data["R1-D1-off"], data["R1-D3-on-attest"],
                                label="D3 attest CC delta (vs D1-off)",
                                n_resamples=args.n_resamples, alpha=args.alpha)
        results["cc_delta_attest"] = d3_cc
        ci = d3_cc["p50_rel_ci"]
        print(f"  D3 attest CC Δp50 = {d3_cc['p50_rel_pct']:+.1f}% [{ci[0]:+.1f}, {ci[1]:+.1f}]")

    # ── 5. Pooled inter-deploy delta ─────────────────────────────────────
    print("\n5. Pooled D1+D2 CC delta (inter-deploy variance in CI)")
    pooled = pooled_interdeploy_delta(data,
                                      n_resamples=args.n_resamples, alpha=args.alpha)
    results["pooled_delta"] = pooled
    if "error" not in pooled:
        ci = pooled["rel_ci_pct"]
        print(f"  pooled Δp50 = {pooled['rel_delta_pct']:+.1f}% [{ci[0]:+.1f}, {ci[1]:+.1f}]  "
              f"(n_off={pooled['n_off']}, n_on={pooled['n_on']})")

    # ── Write outputs ────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "interdeploy_results.json"
    # Convert numpy types for JSON serialization
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    out_path.write_text(json.dumps(results, indent=2, default=_convert))
    print(f"\nWrote {out_path}")

    # ── Figure ───────────────────────────────────────────────────────────
    print("\nGenerating figure...")
    figure_interdeploy_forest(results, args.output_dir)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
