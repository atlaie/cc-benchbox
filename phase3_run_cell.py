#!/usr/bin/env python3
"""
phase3_run_cell.py — per-cell orchestrator for Phase 3.

Runs inside the benchbox over SSH from phase3_run_matrix.py. Dispatches to
phase3_grad_driver or phase3_vllm_driver based on --condition, optionally
polls the target's /metrics endpoint in parallel, post-processes the driver's
summary.json with vllm-bench-aligned derived fields, and uploads all
artifacts to R2 in a single batch at the end.

The drivers are imported as modules (same Python process) rather than
subprocessed, so the /metrics poller can run as a background thread without
inter-process coordination.

Usage (inside benchbox):

  python /workspace/scripts/phase3_run_cell.py \\
      --cell-id C1-off --condition baseline --cc-state off \\
      --target-base-url https://c1-off-target.<org>.containers.tinfoil.dev/v1 \\
      --pairs-json /workspace/data/pairs/baseline_pairs.json \\
      --out-dir /mnt/ramdisk/phase3 \\
      --image-digest sha256:<vllm-image-digest> \\
      --metrics-url https://c1-off-target.<org>.containers.tinfoil.dev/metrics

For the gradient cell, --target-base-url has no /v1 suffix:

  python /workspace/scripts/phase3_run_cell.py \\
      --cell-id C3-off --condition gradient --cc-state off \\
      --target-base-url https://c3-off-target.<org>.containers.tinfoil.dev \\
      --pairs-json /workspace/data/pairs/gradient_pairs.json \\
      --out-dir /mnt/ramdisk/phase3 \\
      --image-digest sha256:<grad-image-digest>

R2 upload requires the following env vars (typically Tinfoil org secrets):
  S3_BUCKET           — target bucket name
  R2_ENDPOINT_URL     — Cloudflare R2 endpoint (omit for AWS S3)
  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY  — credentials (boto3 picks up)

Exit codes:
  0   success
  2   user error (bad CLI args, pairs.json malformed, etc.)
  3   driver completed but produced zero successful requests
  4   driver dispatch failed (health check timeout, etc.)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Optional

import httpx
import numpy as np

# Optional deps.
try:
    import pandas as pd  # type: ignore
except ImportError:
    pd = None

try:
    import boto3  # type: ignore
except ImportError:
    boto3 = None

# Ensure sibling scripts/ files (drivers + captures) are importable.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import phase3_grad_driver as grad_drv  # noqa: E402
import phase3_vllm_driver as vllm_drv  # noqa: E402
import phase3_vllm_driver_stream as stream_drv  # noqa: E402
import phase3_vllm_driver_concurrent as conc_drv  # noqa: E402
import phase3_egress_driver_v2 as egress_drv  # noqa: E402
from openai import OpenAI  # noqa: E402


ORCHESTRATOR_SCHEMA_VERSION = "phase3-run-cell-v1"

# Default req-rate fallback when not set via CLI: gradient → 0.05, others → driver lookup.
GRADIENT_DEFAULT_REQ_RATE = 0.05


# ===== /metrics poller =======================================================

def poll_metrics(
    url: str,
    api_key: str,
    interval: float,
    stop_event: threading.Event,
    samples: list[dict],
    timeout: float = 5.0,
) -> None:
    """Background thread. Hits `url` every `interval` seconds until stop_event
    is set, appends each result to `samples`.

    Stores raw Prometheus text + timestamp. Parsing is deferred to
    phase3_aggregate.py to avoid baking metric-name assumptions in here.
    """
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    while not stop_event.is_set():
        t_sample = time.time()
        text: Optional[str] = None
        http_status = 0
        err: Optional[str] = None
        try:
            r = httpx.get(url, headers=headers, timeout=timeout)
            http_status = r.status_code
            if r.status_code == 200:
                text = r.text
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        samples.append({
            "t_sample": t_sample,
            "http_status": http_status,
            "text": text,
            "error": err,
        })
        # stop_event.wait returns True when set; allows clean early exit.
        if stop_event.wait(interval):
            break

def _nearest_metrics_window(elapsed_s: float) -> str:
    """Smallest supported `tinfoil container metrics --time` window that
    fully covers `elapsed_s`. The CLI only accepts the fixed set
    {5m, 15m, 30m, 1h, 24h, today, 7d, 30d, 60d, 90d, 180d, 365d, 3mo,
    6mo, 12mo, all}; raw `<N>s` values 400."""
    if elapsed_s <= 300:   return "5m"
    if elapsed_s <= 900:   return "15m"
    if elapsed_s <= 1800:  return "30m"
    if elapsed_s <= 3600:  return "1h"
    return "24h"

def capture_gpu_memory_window(
    target_name: str,
    cell_duration_s: float,
    out_dir: Path,
) -> Optional[Path]:
    """One-shot at cell teardown. Pulls aggregate GPU/CPU metrics for a
    window covering the drive period plus a 2-minute buffer (since the
    backend emits samples ~once per 60s).

    Returns path to gpu_memory.parquet, or None on failure. Safe to call
    even if the CLI is missing — degrades to None and logs.
    """
    window_str = _nearest_metrics_window(cell_duration_s + 120)
    cmd = [
        "tinfoil", "container", "metrics",
        "--debug-mode", target_name,
        "--time", window_str,
        "-o", "json",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            print(f"[gpu-mem] CLI failed (rc={r.returncode}): "
                  f"{r.stderr[:300]}", file=sys.stderr)
            return None
        data = json.loads(r.stdout)
        data_points = data.get("data_points", [])
        if not data_points:
            print(f"[gpu-mem] no data points returned for window {window_str}")
            return None
        if pd is not None:
            df = pd.DataFrame(data_points)
            path = out_dir / "gpu_memory.parquet"
            df.to_parquet(path, index=False)
        else:
            path = out_dir / "gpu_memory.json"
            path.write_text(json.dumps(data, indent=2))
        n_nonzero = sum(1 for dp in data_points if dp.get("gpu_mem_total", 0) > 0)
        print(f"[gpu-mem] captured {len(data_points)} samples "
              f"({n_nonzero} non-zero) over {window_str}; wrote {path}")
        return path
    except Exception as e:
        print(f"[gpu-mem] error: {type(e).__name__}: {e}", file=sys.stderr)
        return None

def write_gpu_memory_artifact(samples: list[dict], out_dir: Path) -> Optional[Path]:
    if not samples:
        return None
    if pd is not None:
        path = out_dir / "gpu_memory.parquet"
        pd.DataFrame(samples).to_parquet(path, index=False)
    else:
        path = out_dir / "gpu_memory.jsonl"
        path.write_text("\n".join(json.dumps(s) for s in samples))
    return path

def write_metrics_artifact(samples: list[dict], out_dir: Path) -> Optional[Path]:
    if not samples:
        return None
    if pd is not None:
        path = out_dir / "metrics.parquet"
        pd.DataFrame(samples).to_parquet(path, index=False)
    else:
        path = out_dir / "metrics.jsonl"
        path.write_text("\n".join(json.dumps(s) for s in samples))
    return path


# ===== driver dispatch =======================================================

def resolve_req_rate(condition: str, override: Optional[float]) -> float:
    if override is not None:
        return override
    if condition == "gradient":
        return GRADIENT_DEFAULT_REQ_RATE
    if condition == "steer":
        return 0.15   # similar weight class to baseline; steer is per-token but lightweight
    return vllm_drv.DEFAULT_REQ_RATES[condition]


def run_gradient_cell(args: argparse.Namespace, out_dir: Path) -> tuple[int, dict]:
    """Returns (exit_code, summary_dict). 0 = success, 3 = zero successes, 4 = dispatch fail."""
    req_rate = resolve_req_rate("gradient", args.req_rate)
    print(f"[grad] base_url={args.target_base_url} req_rate={req_rate}")

    if not args.skip_health:
        print(f"[health] polling {args.target_base_url}/health "
              f"(max {args.health_timeout:.0f}s)...")
        try:
            h = grad_drv.wait_for_health(
                args.target_base_url, args.api_key,
                timeout=args.health_timeout,
                poll_interval=args.health_poll_interval,
            )
            print(f"[health] ready: {h}")
        except TimeoutError as e:
            print(f"[error] {e}", file=sys.stderr)
            return 4, {}

    try:
        pairs = grad_drv.load_pairs(args.pairs_json)
    except Exception as e:
        print(f"[error] failed to load --pairs-json {args.pairs_json}: {e}", file=sys.stderr)
        return 2, {}
    prompts = grad_drv.interleave(pairs, args.n_requests)
    if not prompts:
        print("[error] no prompts after interleave (empty pairs.json?)", file=sys.stderr)
        return 2, {}

    print(f"[grad] {len(prompts)} requests @ {req_rate} req/s")
    t0 = time.monotonic()
    rows = grad_drv.run_cell(
        args.target_base_url, args.api_key, prompts,
        req_rate=req_rate, timeout=args.timeout,
    )
    elapsed_min = (time.monotonic() - t0) / 60.0
    print(f"[grad] complete in {elapsed_min:.1f} min")

    summary = grad_drv.summarize(
        rows,
        cell_id=args.cell_id,
        condition="gradient",
        cc_state=args.cc_state,
        base_url=args.target_base_url,
        image_digest=args.image_digest,
        req_rate=req_rate,
        n_requests_target=args.n_requests,
    )
    grad_drv.write_outputs(rows, summary, out_dir)
    return (0 if summary["n_success"] > 0 else 3), summary

def run_vllm_egress_cell(args: argparse.Namespace, out_dir: Path) -> tuple[int, dict]:
    """Egress-pipeline cell. Hits the in-TEE /v1/egress_eval endpoint."""
    stages = egress_drv._parse_stages(args.egress_stages or "")
    preset = vllm_drv.CONDITION_PRESETS[args.condition]
    xargs = vllm_drv.preset_to_xargs(preset)
    if not xargs:
        print(f"[error] condition={args.condition} produces no captures", file=sys.stderr)
        return 2, {}
    req_rate = (args.req_rate if args.req_rate is not None
                else vllm_drv.DEFAULT_REQ_RATES[args.condition])
    session_id = args.session_id or args.cell_id

    if not args.skip_health:
        try:
            egress_drv.wait_for_health(
                args.target_base_url, args.api_key,
                timeout=args.health_timeout,
                poll_interval=args.health_poll_interval,
            )
        except TimeoutError as e:
            print(f"[error] {e}", file=sys.stderr); return 4, {}

    try:
        pairs = vllm_drv.load_pairs(args.pairs_json)
    except Exception as e:
        print(f"[error] failed to load --pairs-json: {e}", file=sys.stderr); return 2, {}
    prompts = vllm_drv.interleave(pairs, args.n_requests)
    if not prompts:
        return 2, {}

    print(f"[egress] {len(prompts)} requests @ {req_rate} req/s; stages={sorted(stages) or '∅'}")
    t0 = time.monotonic()
    rows = egress_drv.run_cell(
        base_url=args.target_base_url, api_key=args.api_key, model=args.model,
        prompts=prompts, xargs=xargs, max_new_tokens=args.max_new_tokens,
        stages=sorted(stages), session_id=session_id,
        include_bundle=not args.no_include_bundle,
        req_rate=req_rate, timeout=args.timeout,
    )
    print(f"[egress] complete in {(time.monotonic()-t0)/60:.1f} min")

    summary = egress_drv.summarize(
        rows=rows, cell_id=args.cell_id, condition=args.condition,
        cc_state=args.cc_state, base_url=args.target_base_url,
        image_digest=args.image_digest, req_rate=req_rate,
        n_requests_target=args.n_requests, xargs=xargs,
        stages_run=sorted(stages), session_id=session_id,
    )
    egress_drv.write_outputs(rows, summary, out_dir)
    return (0 if summary["n_success"] > 0 else 3), summary

def run_vllm_cell(args: argparse.Namespace, out_dir: Path) -> tuple[int, dict]:
    """Returns (exit_code, summary_dict)."""
    # === STEER SPECIAL CASE ===
    if args.condition == "steer":
        from phase3_steering import build_steer_xargs, summarize_steer_config
        if not args.steer_direction or not args.steer_direction.exists():
            print(f"[error] --steer-direction required and must exist for "
                  f"steer condition (got: {args.steer_direction})", file=sys.stderr)
            return 2, {}
        xargs = build_steer_xargs(args.steer_direction)
        xargs_log = summarize_steer_config(xargs)
    else:
        preset = vllm_drv.CONDITION_PRESETS[args.condition]
        xargs = vllm_drv.preset_to_xargs(preset)
        xargs_log = xargs

    req_rate = resolve_req_rate(args.condition, args.req_rate)
    print(f"[vllm] base_url={args.target_base_url} condition={args.condition} "
          f"xargs={xargs_log} req_rate={req_rate}")

    if not args.skip_health:
        print(f"[health] polling (max {args.health_timeout:.0f}s)...")
        try:
            h = vllm_drv.wait_for_health(
                args.target_base_url, args.api_key,
                timeout=args.health_timeout,
                poll_interval=args.health_poll_interval,
            )
            print(f"[health] ready: {h}")
        except TimeoutError as e:
            print(f"[error] {e}", file=sys.stderr)
            return 4, {}

    try:
        pairs = vllm_drv.load_pairs(args.pairs_json)
    except Exception as e:
        print(f"[error] failed to load --pairs-json {args.pairs_json}: {e}", file=sys.stderr)
        return 2, {}
    prompts = vllm_drv.interleave(pairs, args.n_requests)
    if not prompts:
        print("[error] no prompts after interleave (empty pairs.json?)", file=sys.stderr)
        return 2, {}

    client = OpenAI(
        base_url=args.target_base_url,
        api_key=args.api_key,
        max_retries=0,
        timeout=args.timeout,
    )

    print(f"[vllm] {len(prompts)} requests @ {req_rate} req/s")
    t0 = time.monotonic()
    # ----- Tier 1C: pre-load steering payload if requested ---------
    apply_steering_str = None
    if args.apply_steering_json is not None:
        payload = json.loads(args.apply_steering_json.read_text())
        steering_vectors = payload["steering_vectors"]
        # vllm_xargs typed Dict[str, Union[str, int, float, List[scalars]]];
        # list-of-dicts is rejected at FastAPI boundary, must json.dumps.
        # See PHASE2_REFERENCE §5.2, §11.1.
        apply_steering_str = json.dumps(steering_vectors)
        print(f"[steering] loaded from {args.apply_steering_json}: "
              f"layer={steering_vectors[0]['layer_indices']}, "
              f"scale={steering_vectors[0]['scale']}, "
              f"norm_match={steering_vectors[0]['norm_match']}")

    rows = vllm_drv.run_cell(
        client, args.model, prompts,
        xargs=xargs, max_new_tokens=args.max_new_tokens,
        req_rate=req_rate,
        enable_thinking=args.enable_thinking,        # NEW
        apply_steering_str=apply_steering_str,       # NEW
    )
    elapsed_min = (time.monotonic() - t0) / 60.0
    print(f"[vllm] complete in {elapsed_min:.1f} min")

    summary = vllm_drv.summarize(
        rows,
        cell_id=args.cell_id,
        condition=args.condition,
        cc_state=args.cc_state,
        base_url=args.target_base_url,
        image_digest=args.image_digest,
        req_rate=req_rate,
        n_requests_target=args.n_requests,
        xargs=xargs,
        enable_thinking=args.enable_thinking,                                # NEW
        apply_steering_path=str(args.apply_steering_json) if args.apply_steering_json else None,  # NEW
    )
    vllm_drv.write_outputs(rows, summary, out_dir)
    return (0 if summary["n_success"] > 0 else 3), summary


def run_vllm_stream_cell(args: argparse.Namespace, out_dir: Path) -> tuple[int, dict]:
    # === STEER SPECIAL CASE ===
    if args.condition == "steer":
        from phase3_steering import build_steer_xargs, summarize_steer_config
        if not args.steer_direction or not args.steer_direction.exists():
            print(f"[error] --steer-direction required and must exist for "
                  f"steer condition (got: {args.steer_direction})", file=sys.stderr)
            return 2, {}
        xargs = build_steer_xargs(args.steer_direction)
        xargs_log = summarize_steer_config(xargs)
    else:
        preset = vllm_drv.CONDITION_PRESETS[args.condition]
        xargs = vllm_drv.preset_to_xargs(preset)
        xargs_log = xargs

    req_rate = resolve_req_rate(args.condition, args.req_rate)
    print(f"[vllm] base_url={args.target_base_url} condition={args.condition} "
          f"xargs={xargs_log} req_rate={req_rate}")

    if not args.skip_health:
        print(f"[health] polling (max {args.health_timeout:.0f}s)...")
        try:
            stream_drv.wait_for_health(
                args.target_base_url, args.api_key,
                timeout=args.health_timeout,
                poll_interval=args.health_poll_interval,
            )
        except TimeoutError as e:
            print(f"[error] {e}", file=sys.stderr)
            return 4, {}

    try:
        pairs = stream_drv.load_pairs(args.pairs_json)
    except Exception as e:
        print(f"[error] failed to load --pairs-json: {e}", file=sys.stderr)
        return 2, {}
    prompts = stream_drv.interleave(pairs, args.n_requests)
    if not prompts:
        return 2, {}

    print(f"[vllm-stream] {len(prompts)} requests @ {req_rate} req/s")
    t0 = time.monotonic()
    rows = stream_drv.run_cell(
        args.target_base_url, args.api_key, args.model, prompts,
        xargs=xargs, max_new_tokens=args.max_new_tokens,
        req_rate=req_rate, timeout=args.timeout,
    )
    print(f"[vllm-stream] complete in {(time.monotonic()-t0)/60:.1f} min")

    summary = stream_drv.summarize(
        rows, cell_id=args.cell_id, condition=args.condition, cc_state=args.cc_state,
        base_url=args.target_base_url, image_digest=args.image_digest,
        req_rate=req_rate, n_requests_target=args.n_requests, xargs=xargs,
    )
    stream_drv.write_outputs(rows, summary, out_dir)
    return (0 if summary["n_success"] > 0 else 3), summary


def run_vllm_concurrent_cell(args: argparse.Namespace, out_dir: Path) -> tuple[int, dict]:
    preset = conc_drv.CONDITION_PRESETS[args.condition]
    xargs = conc_drv.preset_to_xargs(preset)
    req_rate = (args.req_rate if args.req_rate is not None
                else float(args.concurrency))
    use_streaming = bool(args.stream)
    print(f"[vllm-concurrent] base_url={args.target_base_url} c={args.concurrency} "
          f"stream={use_streaming} req_rate={req_rate}")

    if not args.skip_health:
        try:
            conc_drv.wait_for_health(
                args.target_base_url, args.api_key,
                timeout=args.health_timeout,
                poll_interval=args.health_poll_interval,
            )
        except TimeoutError as e:
            print(f"[error] {e}", file=sys.stderr)
            return 4, {}

    try:
        pairs = conc_drv.load_pairs(args.pairs_json)
    except Exception as e:
        print(f"[error] failed to load --pairs-json: {e}", file=sys.stderr)
        return 2, {}
    prompts = conc_drv.interleave(pairs, args.n_requests)
    if not prompts:
        return 2, {}

    t0 = time.monotonic()
    rows = conc_drv.run_cell(
        args.target_base_url, args.api_key, args.model, prompts,
        xargs=xargs, max_new_tokens=args.max_new_tokens,
        req_rate=req_rate, concurrency=args.concurrency,
        timeout=args.timeout, use_streaming=use_streaming,
    )
    print(f"[vllm-concurrent] complete in {(time.monotonic()-t0)/60:.1f} min")

    summary = conc_drv.summarize(
        rows, cell_id=args.cell_id, condition=args.condition, cc_state=args.cc_state,
        base_url=args.target_base_url, image_digest=args.image_digest,
        req_rate=req_rate, concurrency=args.concurrency,
        n_requests_target=args.n_requests, xargs=xargs, use_streaming=use_streaming,
    )
    conc_drv.write_outputs(rows, summary, out_dir)
    return (0 if summary["n_success"] > 0 else 3), summary

# ===== vllm-bench-aligned derived fields =====================================

def _load_request_rows(out_dir: Path) -> list[dict]:
    parquet = out_dir / "requests.parquet"
    jsonl = out_dir / "requests.jsonl"
    if parquet.exists() and pd is not None:
        return pd.read_parquet(parquet).to_dict(orient="records")
    if jsonl.exists():
        return [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
    return []


def _safe_get(row: Any, key: str) -> Any:
    """Tolerate dict rows (jsonl path) or pandas Series (parquet path)."""
    if isinstance(row, dict):
        return row.get(key)
    return getattr(row, key, None)


def add_vllm_bench_aligned_fields(summary: dict, rows: list[dict]) -> dict:
    """Compute vllm-bench-style derived aggregates from per-request rows.
    Returns the additions as a dict (also mutates summary in-place under
    `vllm_bench_aligned`).
    """
    # Truthiness check on `error` handles None, "", and pandas NaN uniformly.
    ok = [r for r in rows if not _safe_get(r, "error")]
    if not ok:
        derived = {"n_success": 0, "note": "no successful requests; no aggregates"}
        summary["vllm_bench_aligned"] = derived
        return derived

    t_sends = [float(_safe_get(r, "t_send") or 0.0) for r in ok]
    t_ends = [float(_safe_get(r, "t_complete") or 0.0) for r in ok]
    run_wall = max(t_ends) - min(t_sends) if t_sends and t_ends else 0.0

    walls = [float(_safe_get(r, "wall_seconds") or 0.0) for r in ok]
    tok_in = [int(_safe_get(r, "tokens_in") or 0) for r in ok]
    tok_out = [int(_safe_get(r, "tokens_out") or 0) for r in ok]

    walls_ms = np.asarray(walls, dtype=np.float64) * 1000.0
    derived: dict[str, Any] = {
        "schema_version": ORCHESTRATOR_SCHEMA_VERSION,
        "n_success": len(ok),
        "run_wall_seconds": run_wall,
        "mean_e2el_ms": float(walls_ms.mean()),
        "median_e2el_ms": float(np.median(walls_ms)),
        "p99_e2el_ms": float(np.percentile(walls_ms, 99)),
        "max_e2el_ms": float(walls_ms.max()),
        "total_input_tokens": int(sum(tok_in)),
        "total_generated_tokens": int(sum(tok_out)),
        # TTFT/ITL require streaming — populated when Q11 is answered. Until
        # then, leave the keys present so downstream code doesn't have to
        # special-case missing fields.
        "mean_ttft_ms": None,
        "median_ttft_ms": None,
        "p99_ttft_ms": None,
        "mean_itl_ms": None,
    }
    if run_wall > 0:
        derived["request_throughput_req_per_s"] = len(ok) / run_wall
        derived["output_token_throughput_tok_per_s"] = sum(tok_out) / run_wall
        derived["total_token_throughput_tok_per_s"] = (sum(tok_in) + sum(tok_out)) / run_wall
    else:
        derived["request_throughput_req_per_s"] = None
        derived["output_token_throughput_tok_per_s"] = None
        derived["total_token_throughput_tok_per_s"] = None

    summary["vllm_bench_aligned"] = derived
    return derived


# ===== R2 upload =============================================================

def upload_cell_to_r2(out_dir: Path, cell_id: str) -> None:
    """Upload every file in out_dir to s3://$S3_BUCKET/phase3/<cell_id>/."""
    bucket = os.environ.get("S3_BUCKET")
    endpoint_url = os.environ.get("R2_ENDPOINT_URL") or os.environ.get("R2_ENDPOINT")
    if not bucket:
        print("[upload] S3_BUCKET not set; skipping upload")
        return
    if boto3 is None:
        print("[upload] boto3 not installed; skipping upload")
        return
    client_kwargs: dict[str, str] = {}
    if endpoint_url:
        client_kwargs["endpoint_url"] = endpoint_url
    s3 = boto3.client("s3", **client_kwargs)
    backend = "r2" if endpoint_url else "s3"
    uploaded = 0
    for p in sorted(out_dir.iterdir()):
        if not p.is_file():
            continue
        key = f"phase3/{cell_id}/{p.name}"
        try:
            s3.upload_file(str(p), bucket, key)
            print(f"[upload] {backend}://{bucket}/{key}")
            uploaded += 1
        except Exception as e:
            print(f"[upload] FAILED {key}: {type(e).__name__}: {e}", file=sys.stderr)
    print(f"[upload] {uploaded} file(s) uploaded to {backend}://{bucket}/phase3/{cell_id}/")


# ===== CLI ===================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # --- identity ---
    p.add_argument("--cell-id", required=True,
                   help="Phase 3 cell identifier (e.g. C1-off, C3-on).")
    p.add_argument("--condition", required=True,
               choices=["baseline", "repe_bundle", "routing", "gradient", "steer"])
    p.add_argument("--cc-state", required=True, choices=["on", "off"])

    # --- target ---
    p.add_argument("--target-base-url", required=True,
                   help="vLLM cells: base URL ending in /v1. Gradient cell: base URL "
                        "(no /v1 suffix).")
    p.add_argument("--api-key",
                   default=os.environ.get("VLLM_API_KEY", "EMPTY"),
                   help="Bearer token. Default $VLLM_API_KEY or 'EMPTY'.")
    p.add_argument("--model", default="glm-5-1-fp8",
                   help="vLLM cells only: served-model name.")

    # --- workload ---
    p.add_argument("--pairs-json", type=Path, required=True,
                   help="Path to pairs.json. Phase 2-generated, reused for comparability.")
    p.add_argument("--n-requests", type=int, default=100,
                   help="Total requests (default 100 = 50 toxic + 50 benign interleaved).")
    p.add_argument("--req-rate", type=float, default=None,
                   help="Override default req-rate (gradient: 0.05, baseline: 1.0, "
                        "repe_bundle/routing: 0.2).")
    p.add_argument("--max-new-tokens", type=int, default=32,
                   help="vLLM cells only: completion token cap.")
    p.add_argument("--steer-direction", type=Path, default=None,
               help="Path to precomputed RepE direction .npy (required for "
                    "--condition steer; ignored otherwise).")
    # ----- Tier 3G / Tier 1C extension flags -----------------------
    p.add_argument(
        "--enable-thinking",
        action="store_true",
        default=False,
        help="Forward to phase3_vllm_driver: enables "
             "chat_template_kwargs.enable_thinking=True. Tier 3G.",
    )
    p.add_argument(
        "--apply-steering-json",
        type=Path,
        default=None,
        help="Forward to phase3_vllm_driver: path to a JSON file "
             "built by make_steering_payload.py. Tier 1C "
             "C_H-on-steer. Independent from --steer-direction "
             "(the older .npy-based path via phase3_steering).",
    )
    p.add_argument("--timeout", type=float, default=600.0,
                   help="Per-request HTTP timeout (sec).")

    # --- health check ---
    p.add_argument("--health-timeout", type=float, default=1800.0)
    p.add_argument("--health-poll-interval", type=float, default=10.0)
    p.add_argument("--skip-health", action="store_true")

    # --- /metrics poller ---
    p.add_argument("--metrics-url", default=None,
                   help="Target Prometheus /metrics URL. If set, polled in parallel "
                        "and written as metrics.parquet. If absent, no metrics capture.")
    p.add_argument("--metrics-interval", type=float, default=1.0,
                   help="Seconds between /metrics polls.")
    p.add_argument("--metrics-timeout", type=float, default=5.0,
                   help="Per-poll HTTP timeout (sec).")
    p.add_argument("--driver",
                   choices=["sequential", "stream", "concurrent", "egress"],
                   default="sequential")
    p.add_argument("--egress-stages", type=str, default=None,
                   help="Comma-separated subset of {aggregate,plot,bundle,ledger}.")
    p.add_argument("--session-id", type=str, default=None,
                   help="Ledger session id; defaults to cell_id.")
    p.add_argument("--no-include-bundle", action="store_true",
                   help="Don't ship bundle bytes back over the wire (size measurement only).")
    p.add_argument("--stream", action="store_true",
                help="Streaming mode (forced on for --driver stream).")
    p.add_argument("--concurrency", type=int, default=1,
                help="Concurrent in-flight requests (concurrent driver only).")
    p.add_argument("--tinfoil-target-name", default=None,
                help="Tinfoil container name for SSH polling (e.g. c1-off-target).")

    # --- output + upload ---
    p.add_argument("--out-dir", type=Path, default=Path("/mnt/ramdisk/phase3"),
                   help="Base directory; a subdir named <cell-id> is created under it.")
    p.add_argument("--image-digest", default=None,
                   help="Target image digest (sha256:...) for reproducibility metadata.")
    p.add_argument("--no-upload", action="store_true",
                   help="Skip R2 upload. Use for local development or dry runs.")

    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir / args.cell_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[cell] {args.cell_id} (condition={args.condition}, cc={args.cc_state})")
    print(f"[cell] out_dir={out_dir}")
    print(f"[cell] target={args.target_base_url}")
    gpu_capture_t0 = time.monotonic()

    # ---- start /metrics poller (optional) ----
    metrics_samples: list[dict] = []
    stop_event: Optional[threading.Event] = None
    metrics_thread: Optional[threading.Thread] = None
    if args.metrics_url:
        stop_event = threading.Event()
        metrics_thread = threading.Thread(
            target=poll_metrics,
            args=(
                args.metrics_url, args.api_key, args.metrics_interval,
                stop_event, metrics_samples, args.metrics_timeout,
            ),
            daemon=True,
            name="metrics-poller",
        )
        metrics_thread.start()
        print(f"[metrics] polling {args.metrics_url} every {args.metrics_interval}s")

    # ---- dispatch to driver ----
    try:
        if args.condition == "gradient":
            rc, _summary = run_gradient_cell(args, out_dir)
        elif args.driver == "stream":
            rc, _summary = run_vllm_stream_cell(args, out_dir)
        elif args.driver == "concurrent":
            rc, _summary = run_vllm_concurrent_cell(args, out_dir)
        elif args.driver == "egress":
            rc, _summary = run_vllm_egress_cell(args, out_dir)
        else:
            rc, _summary = run_vllm_cell(args, out_dir)
    finally:
        # ---- stop /metrics poller and write artifact ----
        if stop_event is not None:
            stop_event.set()
            if metrics_thread is not None:
                metrics_thread.join(timeout=args.metrics_interval + args.metrics_timeout + 2.0)
            artifact = write_metrics_artifact(metrics_samples, out_dir)
            n_ok = sum(1 for s in metrics_samples if s["http_status"] == 200)
            print(f"[metrics] captured {len(metrics_samples)} samples "
                  f"({n_ok} HTTP 200); wrote {artifact}")

    if args.tinfoil_target_name:  # name retained; semantically "tinfoil container name"
        cell_duration_s = time.monotonic() - gpu_capture_t0
        capture_gpu_memory_window(args.tinfoil_target_name, cell_duration_s, out_dir)
    if args.tinfoil_target_name:
        cell_duration_s = time.monotonic() - gpu_capture_t0
        capture_gpu_memory_window(args.tinfoil_target_name, cell_duration_s, out_dir)
    # ---- post-process summary.json with vllm-bench-aligned fields ----
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
            rows = _load_request_rows(out_dir)
            derived = add_vllm_bench_aligned_fields(summary, rows)
            summary_path.write_text(json.dumps(summary, indent=2))
            ttf = derived.get("mean_e2el_ms")
            thr = derived.get("request_throughput_req_per_s")
            print(f"[summary] vllm-bench-aligned added: "
                  f"mean_e2el={ttf:.1f}ms thr={thr:.3f}req/s" if ttf and thr else
                  "[summary] vllm-bench-aligned added (zero successes)")
        except Exception as e:
            print(f"[warn] failed to add vllm-bench-aligned fields: "
                  f"{type(e).__name__}: {e}", file=sys.stderr)

    
    # ---- upload all artifacts to R2 ----
    if not args.no_upload:
        upload_cell_to_r2(out_dir, args.cell_id)
    else:
        print("[upload] --no-upload set; artifacts remain only in out-dir")

    return rc


if __name__ == "__main__":
    sys.exit(main())
