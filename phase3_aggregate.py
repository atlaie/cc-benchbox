#!/usr/bin/env python3
"""
phase3_aggregate.py — Phase 3 results aggregator (laptop-side).

Pulls all 8 cells' artifacts from R2 (caches locally), reads driver outputs
(requests.parquet + summary.json) plus optional metrics.parquet + vllm-bench
reference JSON, computes:

  - Per-cell summary table (n_ok, wall p50/p95, payload p50, throughput).
  - CC-overhead deltas per condition (paired CC-on / CC-off).
  - Phase 2 sanity check: CC-off cells vs Phase 2 50-pair aggregates
    (±5% on payload_bytes, ±20% on wall_seconds — matches PHASE3_PLAN §12).
  - vllm bench reference cross-check for C1 cells (where bench JSON exists).
  - /metrics rollup (final histogram sums/counts, final gauge values),
    when Prometheus text is present per Q10 resolution.

Outputs:
  <out-dir>/aggregate.md     — human-readable, drops into the report.
  <out-dir>/aggregate.json   — structured, programmatically re-aggregatable.

Usage:

    # Default: pull from R2 (with cache), aggregate, write to runs/phase3/
    python phase3_aggregate.py --matrix phase3-matrix.yaml

    # Local mode: skip R2 entirely, use whatever's in <cache-dir>
    python phase3_aggregate.py --matrix phase3-matrix.yaml --local-only

    # Force re-pull (ignore cache freshness)
    python phase3_aggregate.py --matrix phase3-matrix.yaml --refresh

R2 access uses these env vars (same as the orchestrator):
  S3_BUCKET, R2_ENDPOINT_URL (or R2_ENDPOINT), AWS_ACCESS_KEY_ID,
  AWS_SECRET_ACCESS_KEY.

Exit codes:
  0   aggregation completed (some cells may be missing/failed; check stdout)
  2   user error (bad YAML, no cells, etc.)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import pandas as pd  # type: ignore
except ImportError:
    print("ERROR: pandas required. `pip install pandas pyarrow`", file=sys.stderr)
    sys.exit(2)

try:
    import yaml  # type: ignore
except ImportError:
    print("ERROR: PyYAML required. `pip install pyyaml`", file=sys.stderr)
    sys.exit(2)

try:
    import boto3  # type: ignore
except ImportError:
    boto3 = None

# Optional Prometheus parser; if missing, we skip metrics rollup gracefully.
try:
    from prometheus_client.parser import text_string_to_metric_families  # type: ignore
    PROM_AVAILABLE = True
except ImportError:
    PROM_AVAILABLE = False


SCHEMA_VERSION = "phase3-aggregate-v1"

# Sanity tolerances per PHASE3_PLAN §12.
PHASE2_WALL_TOLERANCE = 0.20      # ±20% on wall p50
PHASE2_PAYLOAD_TOLERANCE = 0.05   # ±5% on payload p50


# ===== R2 / cache management =================================================

def _r2_client() -> Optional[Any]:
    if boto3 is None:
        return None
    endpoint_url = os.environ.get("R2_ENDPOINT_URL") or os.environ.get("R2_ENDPOINT")
    kwargs = {"endpoint_url": endpoint_url} if endpoint_url else {}
    return boto3.client("s3", **kwargs)


def pull_cell_from_r2(cell_id: str, cache_dir: Path, refresh: bool = False) -> dict[str, Path]:
    """Download all objects under phase3/<cell_id>/ to <cache_dir>/<cell_id>/.
    Returns a dict of {basename: local_path}. Skips files already present
    unless refresh=True.
    """
    bucket = os.environ.get("S3_BUCKET")
    if not bucket:
        print(f"  [r2] S3_BUCKET not set; skipping pull for {cell_id}")
        return {}
    s3 = _r2_client()
    if s3 is None:
        print(f"  [r2] boto3 not installed; skipping pull for {cell_id}")
        return {}

    prefix = f"phase3/{cell_id}/"
    local_dir = cache_dir / cell_id
    local_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}

    try:
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    except Exception as e:
        print(f"  [r2] list FAILED for {prefix}: {type(e).__name__}: {e}", file=sys.stderr)
        return {}

    for obj in resp.get("Contents", []):
        key = obj["Key"]
        basename = key[len(prefix):]
        if not basename:
            continue
        local_path = local_dir / basename
        if local_path.exists() and not refresh:
            out[basename] = local_path
            continue
        try:
            s3.download_file(bucket, key, str(local_path))
            out[basename] = local_path
        except Exception as e:
            print(f"  [r2] FAILED to pull {key}: {type(e).__name__}: {e}", file=sys.stderr)
    return out


def discover_local_cell(cell_id: str, cache_dir: Path) -> dict[str, Path]:
    """Return whatever's already in cache_dir/<cell_id>/."""
    local_dir = cache_dir / cell_id
    if not local_dir.exists():
        return {}
    return {p.name: p for p in local_dir.iterdir() if p.is_file()}


# ===== per-cell loading ======================================================

def load_cell(cell_id: str, files: dict[str, Path]) -> dict[str, Any]:
    """Read a cell's artifacts into memory."""
    out: dict[str, Any] = {"cell_id": cell_id, "files": list(files.keys())}

    # summary.json — always required
    summary_path = files.get("summary.json")
    if summary_path:
        try:
            out["summary"] = json.loads(summary_path.read_text())
        except Exception as e:
            out["summary_error"] = f"{type(e).__name__}: {e}"
    else:
        out["summary_error"] = "summary.json missing"

    # requests.parquet (or .jsonl fallback)
    req_path = files.get("requests.parquet")
    if req_path:
        try:
            out["requests"] = pd.read_parquet(req_path)
        except Exception as e:
            out["requests_error"] = f"{type(e).__name__}: {e}"
    elif files.get("requests.jsonl"):
        try:
            jsonl = files["requests.jsonl"].read_text()
            rows = [json.loads(line) for line in jsonl.splitlines() if line.strip()]
            out["requests"] = pd.DataFrame(rows)
        except Exception as e:
            out["requests_error"] = f"{type(e).__name__}: {e}"

    # metrics.parquet (optional, Q10-dependent)
    metrics_path = files.get("metrics.parquet")
    if metrics_path:
        try:
            out["metrics"] = pd.read_parquet(metrics_path)
        except Exception as e:
            out["metrics_error"] = f"{type(e).__name__}: {e}"

    # NEW: gpu_memory.parquet (optional, sugg 1)
    gpu_path = files.get("gpu_memory.parquet")
    if gpu_path:
        try:
            out["gpu_memory"] = pd.read_parquet(gpu_path)
        except Exception as e:
            out["gpu_memory_error"] = f"{type(e).__name__}: {e}"
    elif files.get("gpu_memory.jsonl"):
        try:
            jsonl = files["gpu_memory.jsonl"].read_text()
            rows = [json.loads(line) for line in jsonl.splitlines() if line.strip()]
            out["gpu_memory"] = pd.DataFrame(rows)
        except Exception as e:
            out["gpu_memory_error"] = f"{type(e).__name__}: {e}"
    # vllm-bench reference — directory of JSON file(s) inside cache
    bench_dir = files.get("vllm-bench-reference")
    # Tinfoil's `s3.list_objects_v2` won't list pseudo-directories; we glob
    # for bench files separately in the loader.
    return out


def load_vllm_bench(cell_id: str, cache_dir: Path) -> Optional[dict]:
    """vllm bench writes its result JSON under
    phase3/<cell>/vllm-bench-reference/<file>.json. R2 list returns it as
    a flat key; we look under that path in the cache."""
    bench_dir = cache_dir / cell_id / "vllm-bench-reference"
    if not bench_dir.exists():
        return None
    json_files = sorted(bench_dir.glob("*.json"))
    if not json_files:
        return None
    try:
        return json.loads(json_files[0].read_text())
    except Exception as e:
        print(f"  [bench] failed to read {json_files[0]}: {type(e).__name__}: {e}",
              file=sys.stderr)
        return None


# ===== Prometheus parsing ====================================================

def parse_prometheus_snapshot(text: str) -> dict[tuple[str, frozenset], float]:
    """One /metrics dump → {(metric_name, labels_frozenset): value}."""
    out: dict[tuple[str, frozenset], float] = {}
    if not PROM_AVAILABLE:
        return out
    try:
        for fam in text_string_to_metric_families(text):
            for sample in fam.samples:
                labels = frozenset(sample.labels.items())
                out[(sample.name, labels)] = sample.value
    except Exception:
        # Malformed snapshot; skip.
        pass
    return out


def summarize_metrics(metrics_df: pd.DataFrame) -> dict[str, Any]:
    """Reduce a metrics.parquet sample stream to a flat summary dict.
    Strategy: take the *final* successful snapshot and pull a curated set
    of vLLM metric values. Falls back to "all gauges/sums" if curated set
    is absent."""
    if metrics_df is None or metrics_df.empty:
        return {"note": "no metrics samples"}
    ok = metrics_df[metrics_df["http_status"] == 200]
    if ok.empty:
        return {
            "note": "no successful /metrics responses",
            "n_samples": int(len(metrics_df)),
            "n_ok": 0,
        }
    if not PROM_AVAILABLE:
        return {
            "note": "prometheus_client not installed; skipping parse",
            "n_samples": int(len(metrics_df)),
            "n_ok": int(len(ok)),
        }
    last = ok.iloc[-1]
    snapshot = parse_prometheus_snapshot(last["text"])

    # Curated metric prefixes we care about for the headline report.
    # If vLLM 0.20.0 uses different names, the "all_metric_names" field
    # below lets us discover them post-hoc.
    interesting_prefixes = (
        "vllm:time_to_first_token_seconds",
        "vllm:time_per_output_token_seconds",
        "vllm:e2e_request_latency_seconds",
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:num_requests_swapped",
        "vllm:gpu_cache_usage_perc",
        "vllm:cpu_cache_usage_perc",
        "vllm:prompt_tokens_total",
        "vllm:generation_tokens_total",
    )

    curated: dict[str, float] = {}
    for (name, _labels), value in snapshot.items():
        if any(name.startswith(p) for p in interesting_prefixes):
            # Use plain name; labels collapsed (we expect single-instance).
            curated[name] = float(value)

    all_metric_names = sorted({name for (name, _) in snapshot.keys()})

    return {
        "n_samples": int(len(metrics_df)),
        "n_ok": int(len(ok)),
        "t_first_sample": float(metrics_df["t_sample"].min()),
        "t_last_sample": float(metrics_df["t_sample"].max()),
        "curated_final": curated,
        "all_metric_names_seen": all_metric_names,
    }

def burst_write_profile(df_ok: "pd.DataFrame") -> dict:
    """Per-cell I/O footprint. Pure aggregation over existing parquet."""
    if df_ok is None or df_ok.empty:
        return {"available": False}
    if "t_send" not in df_ok.columns or "t_complete" not in df_ok.columns:
        return {"available": False, "note": "missing time columns"}

    t_send = df_ok["t_send"].astype(float)
    t_end = df_ok["t_complete"].astype(float)
    wall_total_s = float(t_end.max() - t_send.min())
    if wall_total_s <= 0:
        return {"available": False, "note": "non-positive wall window"}

    bytes_total = float(df_ok["payload_bytes"].sum())
    per_req_bytes = df_ok["payload_bytes"].astype(float)
    per_req_wall = df_ok["wall_seconds"].astype(float).clip(lower=1e-6)
    peak_burst_p95 = float((per_req_bytes / per_req_wall).quantile(0.95))

    return {
        "available": True,
        "wall_total_s": wall_total_s,
        "payload_bytes_total": bytes_total,
        "payload_mb_total": bytes_total / 1e6,
        "payload_mbps_sustained": bytes_total / wall_total_s / 1e6,
        "payload_mb_per_req_p50": float(per_req_bytes.median()) / 1e6,
        "payload_mb_per_req_p95": float(per_req_bytes.quantile(0.95)) / 1e6,
        "peak_burst_mbps_p95": peak_burst_p95 / 1e6,
    }


def gpu_memory_summary(gpu_mem_df: "pd.DataFrame") -> dict:
    """Aggregate GPU/CPU metrics from `tinfoil container metrics -o json`.

    Schema (per data_point row, from CLI):
      avg_gpu_mem_util, max_gpu_mem_util  — pct of total
      gpu_mem_total                       — GiB summed across 8 GPUs
      avg_gpu_util, max_gpu_util          — compute pct
      avg_cpu_mem_util, max_cpu_mem_util  — pct
      cpu_mem_total                       — GiB
      time                                — ISO timestamp

    CLI emits ~one real sample per 60s; intermediate buckets show all zeros.
    We filter to rows with gpu_mem_total > 0 for peak computation.
    No per-GPU breakdown — CLI limitation.
    """
    if gpu_mem_df is None or gpu_mem_df.empty:
        return {"available": False}
    required = {"max_gpu_mem_util", "gpu_mem_total"}
    if not required.issubset(set(gpu_mem_df.columns)):
        return {"available": False, "note": "schema mismatch"}

    ok = gpu_mem_df[gpu_mem_df["gpu_mem_total"] > 0]
    if ok.empty:
        return {"available": False,
                "note": f"no non-zero samples in {len(gpu_mem_df)} rows (idle target?)"}

    total_gib = float(ok["gpu_mem_total"].max())
    peak_pct = float(ok["max_gpu_mem_util"].max())
    peak_gib = peak_pct * total_gib / 100.0
    mean_pct = float(ok["avg_gpu_mem_util"].mean())

    return {
        "available": True,
        "n_samples": int(len(gpu_mem_df)),
        "n_nonzero_samples": int(len(ok)),
        "aggregate_total_gib": total_gib,
        "peak_pct": peak_pct,
        "peak_gib": peak_gib,
        "mean_pct": mean_pct,
        "gpu_util_peak_pct": float(ok["max_gpu_util"].max())
            if "max_gpu_util" in ok.columns else None,
        "gpu_util_mean_pct": float(ok["avg_gpu_util"].mean())
            if "avg_gpu_util" in ok.columns else None,
        "note": "aggregate across 8 GPUs; per-GPU detail unavailable from CLI",
    }


def streaming_summary(df_ok: "pd.DataFrame") -> dict:
    """TTFT/ITL aggregates; safe no-op if columns absent."""
    if df_ok is None or df_ok.empty or "ttft_seconds" not in df_ok.columns:
        return {"available": False}
    ttft = df_ok["ttft_seconds"].dropna()
    if ttft.empty:
        return {"available": False, "note": "ttft column present but all NaN"}
    itl50 = df_ok["itl_p50_seconds"].dropna() if "itl_p50_seconds" in df_ok.columns else None
    itl95 = df_ok["itl_p95_seconds"].dropna() if "itl_p95_seconds" in df_ok.columns else None
    n_chunks = df_ok["n_chunks"].dropna() if "n_chunks" in df_ok.columns else None
    return {
        "available": True,
        "n": int(len(ttft)),
        "ttft_p50_s": float(ttft.quantile(0.50)),
        "ttft_p95_s": float(ttft.quantile(0.95)),
        "ttft_max_s": float(ttft.max()),
        "itl_median_of_per_req_p50_s": float(itl50.median()) if itl50 is not None and not itl50.empty else None,
        "itl_median_of_per_req_p95_s": float(itl95.median()) if itl95 is not None and not itl95.empty else None,
        "n_chunks_p50": float(n_chunks.median()) if n_chunks is not None and not n_chunks.empty else None,
    }


def concurrent_comparison(cell_summaries: list[dict]) -> list[dict]:
    """For each (condition, cc_state) pair, compare concurrent vs sequential cells."""
    by_key: dict[tuple, dict] = {}
    for c in cell_summaries:
        if c.get("missing"):
            continue
        key = (c.get("condition"), c.get("cc_state"), int(c.get("concurrency") or 1))
        by_key[key] = c

    rows: list[dict] = []
    for (cond, cc, conc), c in by_key.items():
        if conc <= 1:
            continue
        seq = by_key.get((cond, cc, 1))
        if not seq:
            continue
        seq_stream = (seq.get("streaming") or {})
        c_stream = (c.get("streaming") or {})
        rows.append({
            "condition": cond,
            "cc_state": cc,
            "concurrency": conc,
            "seq_wall_p50_s": seq.get("wall_p50_s"),
            "conc_wall_p50_s": c.get("wall_p50_s"),
            "seq_throughput_rps": seq.get("request_throughput_req_per_s"),
            "conc_throughput_rps": c.get("request_throughput_req_per_s"),
            "seq_ttft_p50_s": seq_stream.get("ttft_p50_s"),
            "conc_ttft_p50_s": c_stream.get("ttft_p50_s"),
        })
    return rows

# ===== per-cell summary ======================================================

def _percentile(series: pd.Series, p: float) -> Optional[float]:
    if series is None or series.empty:
        return None
    clean = series.dropna()
    if clean.empty:                              # NEW guard
        return None
    return float(np.percentile(clean, p))


def cell_summary(cell: dict[str, Any]) -> dict[str, Any]:
    """One-row-per-cell stats. Tolerates union schema (gradient vs vllm,
    sequential vs streaming vs concurrent)."""
    out: dict[str, Any] = {
        "cell_id": cell["cell_id"],
        "available": "requests" in cell,
    }

    summary = cell.get("summary") or {}
    out["condition"] = summary.get("condition")
    out["cc_state"] = summary.get("cc_state")
    out["image_digest"] = summary.get("image_digest")
    out["n_total"] = summary.get("n_total")
    out["n_success"] = summary.get("n_success")
    out["success_rate"] = summary.get("success_rate")

    # NEW: driver mode and concurrency (defaults preserve existing-cell behavior)
    out["driver"] = summary.get("driver", "sequential")
    out["concurrency"] = int(summary.get("concurrency") or 1)
    out["streaming"] = {"available": False}  # overwritten below if present

    # vllm-bench-aligned derived fields are added by phase3_run_cell.py.
    aligned = summary.get("vllm_bench_aligned") or {}
    out["mean_e2el_ms"] = aligned.get("mean_e2el_ms")
    out["median_e2el_ms"] = aligned.get("median_e2el_ms")
    out["p99_e2el_ms"] = aligned.get("p99_e2el_ms")
    out["request_throughput_req_per_s"] = aligned.get("request_throughput_req_per_s")
    out["output_token_throughput_tok_per_s"] = aligned.get("output_token_throughput_tok_per_s")

    if "requests" not in cell:
        # No request data; still try to surface gpu_memory if it landed.
        out["burst_write"] = {"available": False}
        out["gpu_memory"] = gpu_memory_summary(cell.get("gpu_memory"))
        return out

    df = cell["requests"]
    ok = df[df["error"].isna() | (df["error"] == "")]
    if ok.empty:
        out["burst_write"] = {"available": False}
        out["gpu_memory"] = gpu_memory_summary(cell.get("gpu_memory"))
        return out

    out["wall_p50_s"] = _percentile(ok["wall_seconds"], 50)
    out["wall_p95_s"] = _percentile(ok["wall_seconds"], 95)
    out["wall_max_s"] = float(ok["wall_seconds"].max()) if "wall_seconds" in ok else None
    out["payload_p50_B"] = _percentile(ok["payload_bytes"], 50)
    out["payload_p95_B"] = _percentile(ok["payload_bytes"], 95)
    out["tokens_in_p50"] = _percentile(ok["tokens_in"], 50)
    out["tokens_out_p50"] = _percentile(ok["tokens_out"], 50)

    # Gradient-only fields (None if absent).
    for col in ("fwd_seconds", "bwd_seconds", "loss"):
        if col in ok.columns:
            out[f"{col}_p50"] = _percentile(ok[col], 50)
            out[f"{col}_p95"] = _percentile(ok[col], 95)

    # Per-class split (toxic vs benign).
    if "prompt_class" in ok.columns:
        per_class: dict[str, dict[str, Any]] = {}
        for cls, sub in ok.groupby("prompt_class"):
            per_class[str(cls)] = {
                "n": int(len(sub)),
                "wall_p50_s": _percentile(sub["wall_seconds"], 50),
                "wall_p95_s": _percentile(sub["wall_seconds"], 95),
                "payload_p50_B": _percentile(sub["payload_bytes"], 50),
            }
        out["per_class"] = per_class

    # NEW: burst-write, streaming, gpu memory sub-blocks.
    out["burst_write"] = burst_write_profile(ok)
    out["streaming"] = streaming_summary(ok)
    out["gpu_memory"] = gpu_memory_summary(cell.get("gpu_memory"))

    return out


# ===== CC overhead deltas ====================================================

def _safe_pct(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return 100.0 * numerator / denominator


def cc_overhead_table(cell_summaries: list[dict]) -> list[dict]:
    """Pair CC-on / CC-off cells by condition, compute deltas on headline metrics."""
    by_condition: dict[str, dict[str, dict]] = {}
    for cs in cell_summaries:
        cond = cs.get("condition")
        cc = cs.get("cc_state")
        if cond is None or cc not in ("on", "off"):
            continue
        by_condition.setdefault(cond, {})[cc] = cs

    rows: list[dict] = []
    metrics = [
        ("wall_p50_s", "wall p50 (s)"),
        ("wall_p95_s", "wall p95 (s)"),
        ("payload_p50_B", "payload p50 (B)"),
        ("request_throughput_req_per_s", "throughput (req/s)"),
    ]
    for cond, pair in sorted(by_condition.items()):
        off = pair.get("off") or {}
        on = pair.get("on") or {}
        for key, label in metrics:
            v_off = off.get(key)
            v_on = on.get(key)
            delta = None
            delta_pct = None
            if v_off is not None and v_on is not None:
                delta = v_on - v_off
                delta_pct = _safe_pct(delta, v_off)
            rows.append({
                "condition": cond,
                "metric": label,
                "metric_key": key,
                "cc_off": v_off,
                "cc_on": v_on,
                "delta_abs": delta,
                "delta_pct": delta_pct,
            })
    return rows


# ===== Phase 2 sanity check ==================================================

def phase2_sanity(cell_summaries: list[dict], phase2_dir: Path) -> list[dict]:
    """Compare CC-off cells to Phase 2's 50-pair aggregates."""
    if not phase2_dir.exists():
        print(f"[phase2] dir not found: {phase2_dir}; skipping sanity check")
        return []
    out: list[dict] = []
    for cs in cell_summaries:
        if cs.get("cc_state") != "off":
            continue
        cond = cs.get("condition")
        if cond is None:
            continue
        p2_agg_path = phase2_dir / cond / "aggregate.json"
        if not p2_agg_path.exists():
            out.append({
                "cell_id": cs["cell_id"], "condition": cond,
                "note": f"no phase2 aggregate at {p2_agg_path}",
            })
            continue
        try:
            p2 = json.loads(p2_agg_path.read_text())
        except Exception as e:
            out.append({
                "cell_id": cs["cell_id"], "condition": cond,
                "error": f"failed to read phase2 aggregate: {e}",
            })
            continue

        # Phase 2 aggregates report per-class with median (Phase 2's term) under
        # `toxic.wall_seconds.median` and `payload_bytes.median`.
        # Combine toxic + benign by averaging their medians (rough but OK as
        # a drift check; matches Phase 3's overall p50 within noise when prompts
        # are interleaved).
        p2_toxic_wall = (p2.get("toxic") or {}).get("wall_seconds", {}).get("median")
        p2_benign_wall = (p2.get("benign") or {}).get("wall_seconds", {}).get("median")
        p2_toxic_pl = (p2.get("toxic") or {}).get("payload_bytes", {}).get("median")
        p2_benign_pl = (p2.get("benign") or {}).get("payload_bytes", {}).get("median")

        def _avg_or_none(a: Optional[float], b: Optional[float]) -> Optional[float]:
            xs = [x for x in (a, b) if x is not None]
            return float(np.mean(xs)) if xs else None

        p2_wall = _avg_or_none(p2_toxic_wall, p2_benign_wall)
        p2_pl = _avg_or_none(p2_toxic_pl, p2_benign_pl)
        p3_wall = cs.get("wall_p50_s")
        p3_pl = cs.get("payload_p50_B")

        wall_delta_pct = _safe_pct((p3_wall - p2_wall) if (p3_wall is not None and p2_wall is not None) else None,
                                    p2_wall)
        pl_delta_pct = _safe_pct((p3_pl - p2_pl) if (p3_pl is not None and p2_pl is not None) else None,
                                  p2_pl)

        wall_ok = (wall_delta_pct is not None and abs(wall_delta_pct) <= 100 * PHASE2_WALL_TOLERANCE)
        pl_ok = (pl_delta_pct is not None and abs(pl_delta_pct) <= 100 * PHASE2_PAYLOAD_TOLERANCE)

        out.append({
            "cell_id": cs["cell_id"],
            "condition": cond,
            "phase2_wall_median_s": p2_wall,
            "phase3_wall_p50_s": p3_wall,
            "wall_delta_pct": wall_delta_pct,
            "wall_within_tolerance": wall_ok,
            "phase2_payload_median_B": p2_pl,
            "phase3_payload_p50_B": p3_pl,
            "payload_delta_pct": pl_delta_pct,
            "payload_within_tolerance": pl_ok,
        })
    return out


# ===== vllm bench reference comparison =======================================

def _extract_bench_metrics(bench: dict) -> dict[str, Optional[float]]:
    """vllm bench save-result schema uses ms for latencies, req/s for throughputs."""
    return {
        "mean_ttft_ms": bench.get("mean_ttft_ms"),
        "median_ttft_ms": bench.get("median_ttft_ms"),
        "p99_ttft_ms": bench.get("p99_ttft_ms"),
        "mean_e2el_ms": bench.get("mean_e2el_ms") or bench.get("median_e2el_ms"),
        "median_e2el_ms": bench.get("median_e2el_ms"),
        "request_throughput": bench.get("request_throughput"),
        "output_throughput": bench.get("output_throughput"),
        "successful_requests": bench.get("completed") or bench.get("successful_requests"),
    }


def bench_comparison(cell_summaries: list[dict], cells_data: dict[str, dict],
                     cache_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for cs in cell_summaries:
        cell_id = cs["cell_id"]
        bench = load_vllm_bench(cell_id, cache_dir)
        if bench is None:
            continue
        bench_metrics = _extract_bench_metrics(bench)
        driver_e2el = cs.get("mean_e2el_ms")
        driver_thr = cs.get("request_throughput_req_per_s")
        delta_e2el_pct = None
        delta_thr_pct = None
        if driver_e2el is not None and bench_metrics["mean_e2el_ms"]:
            delta_e2el_pct = _safe_pct(
                driver_e2el - bench_metrics["mean_e2el_ms"],
                bench_metrics["mean_e2el_ms"],
            )
        if driver_thr is not None and bench_metrics["request_throughput"]:
            delta_thr_pct = _safe_pct(
                driver_thr - bench_metrics["request_throughput"],
                bench_metrics["request_throughput"],
            )
        rows.append({
            "cell_id": cell_id,
            "driver_mean_e2el_ms": driver_e2el,
            "bench_mean_e2el_ms": bench_metrics["mean_e2el_ms"],
            "e2el_delta_pct": delta_e2el_pct,
            "driver_throughput_req_per_s": driver_thr,
            "bench_throughput_req_per_s": bench_metrics["request_throughput"],
            "throughput_delta_pct": delta_thr_pct,
            "bench_ttft_ms_median": bench_metrics["median_ttft_ms"],
        })
    return rows


# ===== markdown rendering ====================================================

def _fmt(v: Any, suffix: str = "", decimals: int = 2) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "✅" if v else "❌"
    if isinstance(v, (int, float)):
        return f"{v:.{decimals}f}{suffix}"
    return str(v)


def render_markdown(agg: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Phase 3 — Aggregate Results")
    lines.append("")
    lines.append(f"**Generated:** {agg['generated']}")
    lines.append(f"**Matrix file:** `{agg['matrix_file']}`")
    lines.append(f"**Schema version:** `{agg['schema_version']}`")
    lines.append("")

    # --- Summary header ---
    n_cells = len(agg["cells"])
    n_ok = sum(1 for c in agg["cells"] if c.get("n_success", 0) and c.get("n_success") == c.get("n_total"))
    lines.append(f"## Summary")
    lines.append("")
    lines.append(f"- Cells processed: **{n_cells}**")
    lines.append(f"- Cells with 100% success rate: **{n_ok}/{n_cells}**")
    lines.append("")

    # --- Per-cell summary table ---
    lines.append("## Per-cell summary")
    lines.append("")
    lines.append("| Cell | Cond | CC | Driver | Conc | n_ok/n | wall p50 (s) | wall p95 (s) | payload p50 (B) | throughput (req/s) |")
    lines.append("|------|------|----|--------|------|--------|--------------|--------------|-----------------|--------------------|")
    for c in agg["cells"]:
        lines.append(
            f"| `{c['cell_id']}` | {c.get('condition') or '—'} | {c.get('cc_state') or '—'} "
            f"| {c.get('driver') or 'sequential'} | {c.get('concurrency') or 1} "
            f"| {c.get('n_success') or 0}/{c.get('n_total') or 0} "
            f"| {_fmt(c.get('wall_p50_s'), decimals=3)} "
            f"| {_fmt(c.get('wall_p95_s'), decimals=3)} "
            f"| {_fmt(c.get('payload_p50_B'), decimals=0)} "
            f"| {_fmt(c.get('request_throughput_req_per_s'), decimals=3)} |"
        )
    lines.append("")

    # --- CC overhead deltas (unchanged) ---
    lines.append("## CC overhead per condition")
    lines.append("")
    if agg["cc_overhead"]:
        lines.append("| Condition | Metric | CC-off | CC-on | Δ abs | Δ % |")
        lines.append("|-----------|--------|--------|-------|-------|-----|")
        for row in agg["cc_overhead"]:
            lines.append(
                f"| {row['condition']} | {row['metric']} "
                f"| {_fmt(row['cc_off'], decimals=3)} "
                f"| {_fmt(row['cc_on'], decimals=3)} "
                f"| {_fmt(row['delta_abs'], decimals=3)} "
                f"| {_fmt(row['delta_pct'], suffix='%', decimals=1)} |"
            )
    else:
        lines.append("*No CC-on/CC-off pairs available.*")
    lines.append("")

    # --- NEW §3.5: Burst-write I/O profile ---
    burst_rows = [c for c in agg["cells"] if (c.get("burst_write") or {}).get("available")]
    if burst_rows:
        lines.append("## I/O burst-write profile (sugg 3)")
        lines.append("")
        lines.append("Sustained and peak payload rates per cell. Sustained = "
                     "total payload bytes / cell wall window (min-send to max-complete). "
                     "Peak burst p95 = 95th-pct of per-request `bytes / wall_seconds`. "
                     "Project upward to size facility burst-buffer tiers.")
        lines.append("")
        lines.append("| Cell | Cond | CC | sustained (MB/s) | p50 MB/req | p95 MB/req | peak burst p95 (MB/s) | total payload (MB) |")
        lines.append("|------|------|----|------------------|------------|------------|-----------------------|--------------------|")
        for c in burst_rows:
            bw = c["burst_write"]
            lines.append(
                f"| `{c['cell_id']}` | {c.get('condition') or '—'} | {c.get('cc_state') or '—'} "
                f"| {_fmt(bw.get('payload_mbps_sustained'), decimals=3)} "
                f"| {_fmt(bw.get('payload_mb_per_req_p50'), decimals=3)} "
                f"| {_fmt(bw.get('payload_mb_per_req_p95'), decimals=3)} "
                f"| {_fmt(bw.get('peak_burst_mbps_p95'), decimals=3)} "
                f"| {_fmt(bw.get('payload_mb_total'), decimals=1)} |"
            )
        lines.append("")

    # --- NEW §3.6: GPU memory peaks (sugg 1) ---
    mem_rows = [c for c in agg["cells"] if (c.get("gpu_memory") or {}).get("available")]
    if mem_rows:
        lines.append("## GPU memory under instrumentation (sugg 1)")
        lines.append("")
        lines.append("Aggregate across 8 GPUs (per-GPU detail not exposed by "
                    "`tinfoil container metrics`). Peak GiB = max(`max_gpu_mem_util`) "
                    "× `gpu_mem_total` / 100 over the cell drive window.")
        lines.append("")
        # Build pairs for CC delta
        by_pair: dict[tuple, dict[str, dict]] = {}
        for c in mem_rows:
            key = (c.get("condition"), c.get("driver"), c.get("concurrency"))
            by_pair.setdefault(key, {})[c.get("cc_state")] = c

        lines.append("| Cell | Cond | CC | peak (GiB) | peak (%) | mean (%) | "
                    "gpu_util peak (%) | n samples | Δ peak vs CC-off (GiB) |")
        lines.append("|------|------|----|------------|----------|----------|"
                    "-------------------|-----------|------------------------|")
        for c in mem_rows:
            gm = c["gpu_memory"]
            key = (c.get("condition"), c.get("driver"), c.get("concurrency"))
            off_peak = ((by_pair.get(key, {}).get("off") or {}).get("gpu_memory") or {}).get("peak_gib")
            peak_gib = gm.get("peak_gib")
            delta_gib = None
            if c.get("cc_state") == "on" and peak_gib is not None and off_peak is not None:
                delta_gib = peak_gib - off_peak
            lines.append(
                f"| `{c['cell_id']}` | {c.get('condition') or '—'} | {c.get('cc_state') or '—'} "
                f"| {_fmt(peak_gib, decimals=1)} "
                f"| {_fmt(gm.get('peak_pct'), decimals=1)} "
                f"| {_fmt(gm.get('mean_pct'), decimals=1)} "
                f"| {_fmt(gm.get('gpu_util_peak_pct'), decimals=1)} "
                f"| {gm.get('n_nonzero_samples', '—')}/{gm.get('n_samples', '—')} "
                f"| {_fmt(delta_gib, decimals=2)} |"
            )
        lines.append("")

    # --- NEW §3.7: TTFT / ITL for streaming cells (sugg 2) ---
    stream_rows = [c for c in agg["cells"] if (c.get("streaming") or {}).get("available")]
    if stream_rows:
        lines.append("## TTFT and inter-token latency (sugg 2)")
        lines.append("")
        lines.append("Per-cell streaming statistics. TTFT = wall time from request send "
                     "to first SSE chunk with non-empty `delta.content`. ITL = inter-chunk "
                     "gap after the first. Only streaming cells populate these columns.")
        lines.append("")
        lines.append("| Cell | Cond | CC | n | TTFT p50 (s) | TTFT p95 (s) | ITL p50 (s) | ITL p95 (s) | chunks p50 |")
        lines.append("|------|------|----|---|--------------|--------------|-------------|-------------|------------|")
        for c in stream_rows:
            s = c["streaming"]
            lines.append(
                f"| `{c['cell_id']}` | {c.get('condition') or '—'} | {c.get('cc_state') or '—'} "
                f"| {s.get('n', 0)} "
                f"| {_fmt(s.get('ttft_p50_s'), decimals=3)} "
                f"| {_fmt(s.get('ttft_p95_s'), decimals=3)} "
                f"| {_fmt(s.get('itl_median_of_per_req_p50_s'), decimals=3)} "
                f"| {_fmt(s.get('itl_median_of_per_req_p95_s'), decimals=3)} "
                f"| {_fmt(s.get('n_chunks_p50'), decimals=0)} |"
            )
        lines.append("")
        # Paired CC-on vs CC-off delta on TTFT specifically.
        by_cond: dict[str, dict[str, dict]] = {}
        for c in stream_rows:
            cond = c.get("condition")
            cc = c.get("cc_state")
            if cond and cc in ("on", "off"):
                by_cond.setdefault(cond, {})[cc] = c
        pair_rows = []
        for cond, pair in by_cond.items():
            off = pair.get("off")
            on = pair.get("on")
            if not off or not on:
                continue
            off_ttft = (off.get("streaming") or {}).get("ttft_p50_s")
            on_ttft = (on.get("streaming") or {}).get("ttft_p50_s")
            delta = (on_ttft - off_ttft) if (off_ttft is not None and on_ttft is not None) else None
            delta_pct = _safe_pct(delta, off_ttft) if delta is not None else None
            pair_rows.append((cond, off_ttft, on_ttft, delta, delta_pct))
        if pair_rows:
            lines.append("### TTFT CC delta")
            lines.append("")
            lines.append("| Condition | TTFT p50 CC-off (s) | TTFT p50 CC-on (s) | Δ abs (s) | Δ % |")
            lines.append("|-----------|---------------------|--------------------|-----------|-----|")
            for cond, off_ttft, on_ttft, delta, delta_pct in pair_rows:
                lines.append(
                    f"| {cond} "
                    f"| {_fmt(off_ttft, decimals=3)} "
                    f"| {_fmt(on_ttft, decimals=3)} "
                    f"| {_fmt(delta, decimals=3)} "
                    f"| {_fmt(delta_pct, suffix='%', decimals=1)} |"
                )
            lines.append("")

    # --- NEW §3.8: Concurrent vs sequential (sugg 5) ---
    conc_rows = agg.get("concurrent_comparison") or []
    if conc_rows:
        lines.append("## Concurrent vs sequential CC delta (sugg 5)")
        lines.append("")
        lines.append("Tests whether the sequential CC tax (notably C1's +33.4% in the main matrix) "
                     "survives realistic continuous-batching load. Concurrent cells use vLLM's "
                     "continuous batching with a bounded semaphore on the client.")
        lines.append("")
        lines.append("| Condition | CC | Conc | Seq wall p50 (s) | Conc wall p50 (s) | "
                     "Seq thr (r/s) | Conc thr (r/s) | Seq TTFT p50 | Conc TTFT p50 |")
        lines.append("|-----------|----|------|------------------|-------------------|"
                     "---------------|----------------|--------------|---------------|")
        for r in conc_rows:
            lines.append(
                f"| {r['condition']} | {r['cc_state']} | {r['concurrency']} "
                f"| {_fmt(r.get('seq_wall_p50_s'), decimals=3)} "
                f"| {_fmt(r.get('conc_wall_p50_s'), decimals=3)} "
                f"| {_fmt(r.get('seq_throughput_rps'), decimals=3)} "
                f"| {_fmt(r.get('conc_throughput_rps'), decimals=3)} "
                f"| {_fmt(r.get('seq_ttft_p50_s'), decimals=3)} "
                f"| {_fmt(r.get('conc_ttft_p50_s'), decimals=3)} |"
            )
        lines.append("")

    # --- Phase 2 sanity check (unchanged) ---
    lines.append("## Phase 2 sanity check (CC-off cells)")
    lines.append("")
    lines.append(f"Tolerances per `PHASE3_PLAN §12`: ±{int(PHASE2_WALL_TOLERANCE*100)}% on "
                 f"wall p50, ±{int(PHASE2_PAYLOAD_TOLERANCE*100)}% on payload p50.")
    lines.append("")
    if agg["phase2_check"]:
        lines.append("| Cell | Cond | P2 wall median | P3 wall p50 | wall Δ% | wall OK | "
                     "P2 payload median | P3 payload p50 | payload Δ% | payload OK |")
        lines.append("|------|------|----------------|-------------|---------|---------|"
                     "-------------------|----------------|-----------|------------|")
        for row in agg["phase2_check"]:
            if "note" in row or "error" in row:
                note = row.get("note") or row.get("error")
                lines.append(f"| `{row['cell_id']}` | {row.get('condition', '—')} "
                             f"| *{note}* | | | | | | | |")
                continue
            lines.append(
                f"| `{row['cell_id']}` | {row['condition']} "
                f"| {_fmt(row.get('phase2_wall_median_s'), decimals=3)} "
                f"| {_fmt(row.get('phase3_wall_p50_s'), decimals=3)} "
                f"| {_fmt(row.get('wall_delta_pct'), suffix='%', decimals=1)} "
                f"| {_fmt(row.get('wall_within_tolerance'))} "
                f"| {_fmt(row.get('phase2_payload_median_B'), decimals=0)} "
                f"| {_fmt(row.get('phase3_payload_p50_B'), decimals=0)} "
                f"| {_fmt(row.get('payload_delta_pct'), suffix='%', decimals=2)} "
                f"| {_fmt(row.get('payload_within_tolerance'))} |"
            )
    else:
        lines.append("*No Phase 2 aggregates available for comparison.*")
    lines.append("")

    # --- vllm bench reference (unchanged) ---
    lines.append("## `vllm bench` reference cross-check")
    lines.append("")
    if agg["bench_comparison"]:
        lines.append("| Cell | Driver mean e2el (ms) | bench mean e2el (ms) | e2el Δ% | "
                     "Driver thr (req/s) | bench thr (req/s) | thr Δ% | bench median TTFT (ms) |")
        lines.append("|------|----------------------|---------------------|---------|"
                     "--------------------|-------------------|--------|------------------------|")
        for row in agg["bench_comparison"]:
            lines.append(
                f"| `{row['cell_id']}` "
                f"| {_fmt(row['driver_mean_e2el_ms'], decimals=1)} "
                f"| {_fmt(row['bench_mean_e2el_ms'], decimals=1)} "
                f"| {_fmt(row['e2el_delta_pct'], suffix='%', decimals=1)} "
                f"| {_fmt(row['driver_throughput_req_per_s'], decimals=3)} "
                f"| {_fmt(row['bench_throughput_req_per_s'], decimals=3)} "
                f"| {_fmt(row['throughput_delta_pct'], suffix='%', decimals=1)} "
                f"| {_fmt(row['bench_ttft_ms_median'], decimals=1)} |"
            )
    else:
        lines.append("*No vllm-bench reference data captured.*")
    lines.append("")

    # --- /metrics rollup (unchanged) ---
    lines.append("## `/metrics` rollup (where available)")
    lines.append("")
    any_metrics = any(c.get("metrics_summary") for c in agg["cells"])
    if not any_metrics:
        lines.append("*No `/metrics` samples captured. Q10 status: target enclave egress "
                     "likely blocked, or `--metrics-url` not configured.*")
    elif not PROM_AVAILABLE:
        lines.append("*`prometheus_client` not installed on laptop. Install with "
                     "`pip install prometheus-client` and rerun.*")
    else:
        lines.append("Final-snapshot values from the last successful `/metrics` poll per cell.")
        lines.append("")
        for c in agg["cells"]:
            ms = c.get("metrics_summary")
            if not ms or not ms.get("curated_final"):
                continue
            lines.append(f"### `{c['cell_id']}` ({c.get('n_ok', '?')}/{ms.get('n_samples', '?')} samples ok)")
            lines.append("")
            lines.append("| Metric | Final value |")
            lines.append("|--------|-------------|")
            for name, v in sorted(ms["curated_final"].items()):
                lines.append(f"| `{name}` | {_fmt(v, decimals=4)} |")
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("Generated by `phase3_aggregate.py` "
                 "(see `PHASE3_PLAN.md §7.2` and `PHASE3_REFERENCE.md`).")
    return "\n".join(lines) + "\n"


# ===== main ==================================================================

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--matrix", type=Path, required=True,
                   help="Path to phase3-matrix.yaml (defines which cells to expect).")
    p.add_argument("--cache-dir", type=Path, default=Path("runs/phase3"),
               help="...")
    p.add_argument("--phase2-dir", type=Path, default=Path("runs/phase2_validation"),
                   help="Directory containing Phase 2 per-condition aggregate.json files.")
    p.add_argument("--out-dir", type=Path, default=Path("runs/phase3"),
                   help="Where to write aggregate.md + aggregate.json.")
    p.add_argument("--local-only", action="store_true",
                   help="Skip R2 entirely; use only what's in --cache-dir.")
    p.add_argument("--refresh", action="store_true",
                   help="Force re-pull from R2 even if cache files exist.")
    
    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        matrix = yaml.safe_load(args.matrix.read_text())
    except Exception as e:
        print(f"[error] failed to read matrix: {e}", file=sys.stderr)
        return 2
    cell_ids = [c["cell_id"] for c in matrix.get("cells") or []]
    if not cell_ids:
        print("[error] no cells in matrix", file=sys.stderr)
        return 2

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cells_data: dict[str, dict] = {}
    for cid in cell_ids:
        print(f"[load] {cid}")
        if args.local_only:
            files = discover_local_cell(cid, args.cache_dir)
        else:
            files = pull_cell_from_r2(cid, args.cache_dir, refresh=args.refresh)
            # Even after a pull, fall back to local discovery for anything
            # the listing missed (e.g. legacy files).
            local = discover_local_cell(cid, args.cache_dir)
            for k, v in local.items():
                files.setdefault(k, v)
        if not files:
            print(f"  no artifacts found for {cid}")
            cells_data[cid] = {"cell_id": cid, "missing": True}
            continue
        cells_data[cid] = load_cell(cid, files)

    # Per-cell summaries.
    cell_summaries: list[dict] = []
    for cid in cell_ids:
        cd = cells_data[cid]
        if cd.get("missing"):
            cell_summaries.append({"cell_id": cid, "missing": True})
            continue
        cs = cell_summary(cd)
        # Attach metrics summary.
        if "metrics" in cd:
            cs["metrics_summary"] = summarize_metrics(cd["metrics"])
            cs["n_ok"] = cs["metrics_summary"].get("n_ok")
        cell_summaries.append(cs)

    # Cross-cell analyses.
    cc_rows = cc_overhead_table(cell_summaries)
    p2_rows = phase2_sanity(cell_summaries, args.phase2_dir)
    bench_rows = bench_comparison(cell_summaries, cells_data, args.cache_dir)
    conc_rows = concurrent_comparison(cell_summaries)

    agg = {
        "schema_version": SCHEMA_VERSION,
        "generated": now_iso(),
        "matrix_file": str(args.matrix),
        "cells": cell_summaries,
        "cc_overhead": cc_rows,
        "phase2_check": p2_rows,
        "bench_comparison": bench_rows,
        "concurrent_comparison": conc_rows,   # NEW
    }

    (args.out_dir / "aggregate.json").write_text(json.dumps(agg, indent=2, default=str))
    (args.out_dir / "aggregate.md").write_text(render_markdown(agg))

    print(f"\n[out] {args.out_dir / 'aggregate.md'}")
    print(f"[out] {args.out_dir / 'aggregate.json'}")

    # Summary stats to stdout.
    n_ok_cells = sum(1 for c in cell_summaries
                     if c.get("n_success") and c.get("n_success") == c.get("n_total"))
    print(f"[summary] {n_ok_cells}/{len(cell_summaries)} cells at 100% success rate")
    if p2_rows:
        n_wall_ok = sum(1 for r in p2_rows if r.get("wall_within_tolerance"))
        n_pl_ok = sum(1 for r in p2_rows if r.get("payload_within_tolerance"))
        n_checked = sum(1 for r in p2_rows if "wall_within_tolerance" in r)
        print(f"[summary] Phase 2 check: wall {n_wall_ok}/{n_checked} within ±{int(PHASE2_WALL_TOLERANCE*100)}%, "
              f"payload {n_pl_ok}/{n_checked} within ±{int(PHASE2_PAYLOAD_TOLERANCE*100)}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
