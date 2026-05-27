#!/usr/bin/env python3
"""
phase3_aggregate_egress.py — egress-pipeline aggregation extensions.

Companion to phase3_aggregate.py.  Adds the egress-cost decomposition that
pairs E1/E2/E3 cells against E0 (the existing C2-on cell) and reports per-stage
timing deltas.

This module is import-safe — adding the splice below to phase3_aggregate.py
does not change the behaviour for runs that don't include any E-cells.

Splice into phase3_aggregate.py (3 changes, ~5 lines total):

    # 1. Add import near the top with the other module imports:
    from phase3_aggregate_egress import (
        egress_cell_summary, egress_breakdown, render_egress_section,
    )

    # 2. In cell_summary(), after the existing `out["streaming"] = ...` line,
    #    add:
    out["egress"] = egress_cell_summary(cell)

    # 3. In main(), after the existing cross-cell analyses
    #    (cc_overhead_table, phase2_sanity, bench_comparison,
    #    concurrent_comparison), add one line:
    egress_rows = egress_breakdown(cell_summaries)

    #    Then add it to the agg dict:
    agg["egress_breakdown"] = egress_rows

    # 4. In render_markdown(agg), before the "Phase 2 sanity check" section,
    #    add:
    lines.append(render_egress_section(agg))

The new section in aggregate.md looks like:

    ## Egress-pipeline overhead (E1/E2/E3 vs E0)

    | Cell | Stages | wall p50 (s) | deser p50 (ms) | enc p50 (ms) |
    | aggregate p50 (ms) | plot p50 (ms) | bundle p50 (ms) | bundle p50 (KB) |
    ...

    ### Stage deltas vs E0 (existing C2-on)

    | Cell | Δ wall (s) | Δ deser (ms) | Δ enc (ms) | Δ% total |
    ...

    ### Session totals (E3-on only)

    tokens_out: ..., aggregate bytes: ..., bundle bytes: ..., within tier-1 caps: ...
"""
from __future__ import annotations

from typing import Any, Optional


# ====================================================================
# Per-cell egress summary
# ====================================================================

def egress_cell_summary(cell: dict[str, Any]) -> dict[str, Any]:
    """Extract egress-specific stats from an E-cell's requests dataframe.

    Falls back to ``{"available": False}`` for non-egress cells, so calling
    this on every cell from cell_summary() is safe.

    Looks for the egress-driver-only columns (``deserialize_seconds``,
    ``aggregate_seconds``, etc.).  If absent the cell didn't go through
    phase3_egress_driver.py and there's nothing to report.
    """
    if "requests" not in cell:
        return {"available": False}

    df = cell["requests"]
    if df is None or df.empty:
        return {"available": False}

    egress_cols = (
        "server_deserialize_seconds", "server_encoder_total_seconds",
        "server_aggregate_seconds", "server_plot_seconds", "server_bundle_seconds", "server_ledger_seconds",
        "aggregate_bytes", "bundle_bytes", "n_plots", "stages_run",
    )
    missing = [c for c in egress_cols if c not in df.columns]
    if missing:
        return {"available": False, "missing_cols": missing}

    ok = df[df["error"].isna() | (df["error"] == "")]
    if ok.empty:
        return {"available": False, "note": "no successful requests"}

    def _p(col: str, q: float) -> Optional[float]:
        if col not in ok.columns:
            return None
        clean = ok[col].dropna()
        if clean.empty:
            return None
        return float(clean.quantile(q))

    stages_run = ok["stages_run"].iloc[0] if "stages_run" in ok.columns else ""

    summary = cell.get("summary") or {}
    session_totals = summary.get("session_totals")

    return {
        "available": True,
        "n": int(len(ok)),
        "stages_run": stages_run,
        # Stage timings (seconds).
        "wall_p50_s": _p("wall_seconds", 0.50),
        "wall_p95_s": _p("wall_seconds", 0.95),
        "deserialize_p50_s": _p("server_deserialize_seconds", 0.50),
        "deserialize_p95_s": _p("server_deserialize_seconds", 0.95),
        "encoder_total_p50_s": _p("server_encoder_total_seconds", 0.50),
        "encoder_total_p95_s": _p("server_encoder_total_seconds", 0.95),
        "aggregate_p50_s": _p("server_aggregate_seconds", 0.50),
        "plot_p50_s": _p("server_plot_seconds", 0.50),
        "bundle_p50_s": _p("server_bundle_seconds", 0.50),
        "ledger_p50_s": _p("server_ledger_seconds", 0.50),
        # Sizing.
        "payload_p50_B": _p("payload_bytes", 0.50),
        "aggregate_p50_B": _p("aggregate_bytes", 0.50),
        "bundle_p50_B": _p("bundle_bytes", 0.50),
        # Session-level (ledger).
        "session_totals": session_totals,
    }


# ====================================================================
# Cross-cell egress breakdown
# ====================================================================

def _safe_pct(num: Optional[float], denom: Optional[float]) -> Optional[float]:
    if num is None or denom is None or denom == 0:
        return None
    return 100.0 * num / denom


def egress_breakdown(cell_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Pair E1/E2/E3 cells against an E0 reference (a baseline cell with the
    same condition + cc_state that did NOT run through the egress driver).

    Returns a dict with two keys:
      ``rows``     per-E-cell stage decomposition (one row per E-cell)
      ``deltas``   per-E-cell delta vs E0 (when an E0 reference exists)

    Heuristic for pairing:
      - The "E0" reference is a non-egress cell (``egress.available == False``)
        with matching ``condition`` and ``cc_state``.
      - When multiple candidates exist, prefer cells whose ``cell_id`` ends in
        ``-via-egress``. These are run against the SAME deployment as the
        E-cells, so the delta reflects only the in-TEE encoder pipeline cost.
        Cross-deployment baselines (e.g. ``C2-on`` from an earlier deploy)
        confound the delta with host-load drift and should not be used when a
        same-deployment alternative is present.
      - If no candidate exists, deltas[<cell>] = {"reference": None} and only
        ``rows`` is populated. Useful when running the E-cell sweep in
        isolation.
    """
    # Partition.
    egress_cells: list[dict[str, Any]] = []
    reference_cells_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for cs in cell_summaries:
        eg = cs.get("egress") or {}
        if eg.get("available"):
            egress_cells.append(cs)
            continue

        key = (cs.get("condition"), cs.get("cc_state"))
        existing = reference_cells_by_key.get(key)
        cell_id = cs.get("cell_id") or ""
        is_via_egress = cell_id.endswith("-via-egress")

        if existing is None:
            # First candidate wins by default.
            reference_cells_by_key[key] = cs
        elif is_via_egress and not (existing.get("cell_id") or "").endswith("-via-egress"):
            # Same-deployment baseline takes precedence over a cross-deployment one.
            reference_cells_by_key[key] = cs
        # Otherwise: keep what we have.

    rows: list[dict[str, Any]] = []
    deltas: list[dict[str, Any]] = []

    for cs in egress_cells:
        eg = cs["egress"]
        row = {
            "cell_id": cs["cell_id"],
            "condition": cs.get("condition"),
            "cc_state": cs.get("cc_state"),
            "stages_run": eg.get("stages_run") or "∅",
            "n": eg.get("n"),
            **{
                k: eg.get(k)
                for k in (
                    "wall_p50_s", "wall_p95_s",
                    "deserialize_p50_s", "encoder_total_p50_s",
                    "aggregate_p50_s", "plot_p50_s",
                    "bundle_p50_s", "ledger_p50_s",
                    "payload_p50_B", "aggregate_p50_B", "bundle_p50_B",
                )
            },
            "session_totals": eg.get("session_totals"),
        }
        rows.append(row)

        # Delta vs reference.
        key = (cs.get("condition"), cs.get("cc_state"))
        ref = reference_cells_by_key.get(key)
        if ref is None:
            deltas.append({"cell_id": cs["cell_id"], "reference": None})
            continue

        ref_wall = ref.get("wall_p50_s")
        our_wall = eg.get("wall_p50_s")
        delta_wall = (our_wall - ref_wall) if (our_wall is not None and ref_wall is not None) else None
        # Encoder slice is what the egress pipeline added on top of the existing
        # request — deserialize + encoder_total are the new costs.
        added_total = None
        if eg.get("deserialize_p50_s") is not None and eg.get("encoder_total_p50_s") is not None:
            added_total = eg["deserialize_p50_s"] + eg["encoder_total_p50_s"]

        deltas.append({
            "cell_id": cs["cell_id"],
            "reference_cell_id": ref["cell_id"],
            "ref_wall_p50_s": ref_wall,
            "our_wall_p50_s": our_wall,
            "wall_delta_s": delta_wall,
            "wall_delta_pct": _safe_pct(delta_wall, ref_wall),
            "added_deserialize_s": eg.get("deserialize_p50_s"),
            "added_encoder_total_s": eg.get("encoder_total_p50_s"),
            "added_total_s": added_total,
            "added_total_pct_of_ref": _safe_pct(added_total, ref_wall),
        })

    return {"rows": rows, "deltas": deltas}

# ====================================================================
# Markdown rendering
# ====================================================================

def _fmt(v: Any, suffix: str = "", decimals: int = 2) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "✅" if v else "❌"
    if isinstance(v, (int, float)):
        return f"{v:.{decimals}f}{suffix}"
    return str(v)


def render_egress_section(agg: dict[str, Any]) -> str:
    """Render the egress section of aggregate.md."""
    eg = agg.get("egress_breakdown") or {}
    rows = eg.get("rows") or []
    deltas = eg.get("deltas") or []

    if not rows:
        return ""  # Skip the section entirely if no E-cells in the matrix.

    lines: list[str] = []
    lines.append("## Egress-pipeline overhead (E1/E2/E3)")
    lines.append("")
    lines.append("Per-stage timing breakdown for the white-paper Tier-1 evidence "
                 "pipeline running on top of the existing CC + instrumentation "
                 "cost.  Wall = HTTP round-trip; deserialize = "
                 "`captures.extract_*` time; encoder = `EgressPipeline.run()` "
                 "(aggregate + plot + bundle + ledger).")
    lines.append("")

    # Per-cell table.
    lines.append("| Cell | Cond | CC | Stages | n | wall p50 (s) | deser p50 (ms) | "
                 "enc p50 (ms) | agg p50 (ms) | plot p50 (ms) | bundle p50 (ms) | "
                 "payload p50 (KB) | bundle p50 (KB) |")
    lines.append("|------|------|----|--------|---|--------------|----------------|"
                 "--------------|--------------|---------------|-----------------|"
                 "------------------|-----------------|")
    for r in rows:
        lines.append(
            f"| `{r['cell_id']}` | {r.get('condition') or '—'} "
            f"| {r.get('cc_state') or '—'} | `{r['stages_run'] or '∅'}` "
            f"| {r.get('n') or 0} "
            f"| {_fmt(r.get('wall_p50_s'), decimals=3)} "
            f"| {_fmt((r.get('deserialize_p50_s') or 0) * 1000, decimals=1)} "
            f"| {_fmt((r.get('encoder_total_p50_s') or 0) * 1000, decimals=1)} "
            f"| {_fmt((r.get('aggregate_p50_s') or 0) * 1000, decimals=1)} "
            f"| {_fmt((r.get('plot_p50_s') or 0) * 1000, decimals=1)} "
            f"| {_fmt((r.get('bundle_p50_s') or 0) * 1000, decimals=1)} "
            f"| {_fmt((r.get('payload_p50_B') or 0) / 1024, decimals=0)} "
            f"| {_fmt((r.get('bundle_p50_B') or 0) / 1024, decimals=1)} |"
        )
    lines.append("")

    # Delta table.
    delta_rows = [d for d in deltas if d.get("reference_cell_id")]
    if delta_rows:
        lines.append("### Stage deltas vs reference (E0)")
        lines.append("")
        lines.append("Wall delta = full request-time delta (includes anything that "
                     "differs between the E-cell and the reference, not just "
                     "encoder cost).  Added deser + encoder = the pipeline's "
                     "additional CPU cost in isolation.")
        lines.append("")
        lines.append("| Cell | Reference | Ref wall p50 (s) | Our wall p50 (s) | "
                     "Δ wall (s) | Δ wall % | Added deser (ms) | Added enc (ms) | "
                     "Added total (ms) | Added % of ref |")
        lines.append("|------|-----------|------------------|------------------|"
                     "------------|----------|------------------|----------------|"
                     "------------------|----------------|")
        for d in delta_rows:
            lines.append(
                f"| `{d['cell_id']}` | `{d['reference_cell_id']}` "
                f"| {_fmt(d.get('ref_wall_p50_s'), decimals=3)} "
                f"| {_fmt(d.get('our_wall_p50_s'), decimals=3)} "
                f"| {_fmt(d.get('wall_delta_s'), decimals=3)} "
                f"| {_fmt(d.get('wall_delta_pct'), suffix='%', decimals=1)} "
                f"| {_fmt((d.get('added_deserialize_s') or 0) * 1000, decimals=1)} "
                f"| {_fmt((d.get('added_encoder_total_s') or 0) * 1000, decimals=1)} "
                f"| {_fmt((d.get('added_total_s') or 0) * 1000, decimals=1)} "
                f"| {_fmt(d.get('added_total_pct_of_ref'), suffix='%', decimals=2)} |"
            )
        lines.append("")

    # Session totals (E3 specifically — that's the only stage set that
    # populates the ledger).
    e3_with_ledger = [r for r in rows if r.get("session_totals")]
    if e3_with_ledger:
        lines.append("### Session totals (ledger-enabled cells)")
        lines.append("")
        lines.append("Cumulative export across the cell run.  White-paper Tier-1 "
                     "illustrative caps: 20,000 generated tokens, 50 plots per session.")
        lines.append("")
        lines.append("| Cell | tokens_in | tokens_out | aggregate bytes | "
                     "bundle bytes | n plots | n records |")
        lines.append("|------|-----------|------------|-----------------|"
                     "--------------|---------|-----------|")
        for r in e3_with_ledger:
            t = r["session_totals"]
            lines.append(
                f"| `{r['cell_id']}` "
                f"| {t.get('tokens_in', '—')} "
                f"| {t.get('tokens_out', '—')} "
                f"| {t.get('aggregate_bytes', '—')} "
                f"| {t.get('bundle_bytes', '—')} "
                f"| {t.get('n_plots', '—')} "
                f"| {t.get('n_records', '—')} |"
            )
        lines.append("")

    return "\n".join(lines)
