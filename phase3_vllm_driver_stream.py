#!/usr/bin/env python3
"""
phase3_vllm_driver_stream.py — streaming sibling of phase3_vllm_driver.py.

Captures per-request TTFT (time to first token, in seconds) and ITL
(inter-token latency) by parsing vLLM's SSE response stream directly via
httpx. The OpenAI SDK supports streaming, but raw httpx gives tight
control over "first SSE event observed" timing — critical for TTFT.

Schema additions on RequestRow (None for rows where streaming was off
or the field is irrelevant):
  ttft_seconds       wall from request send to first delta.content
  itl_p50_seconds    median inter-chunk gap (skip first chunk)
  itl_p95_seconds    95th-pct inter-chunk gap
  n_chunks           total SSE `data:` events received

Wire format identical to phase3_vllm_driver: same extra_body, same
vllm_xargs, same chat_template_kwargs. Differences:
  - "stream": true
  - "stream_options": {"include_usage": true} so the final chunk carries
    the usage block (tokens_in / tokens_out).

CAUTION (verify before relying on instrumented streaming cells): vLLM's
behavior under stream=true with vllm_xargs payloads (residual stream,
attention stats, routing) is not empirically validated. The xargs blobs
may collapse into the final chunk only, arrive piecemeal, or be dropped
entirely. payload_bytes here is computed over the *final* chunk's raw
JSON, which captures the collapse-into-final-chunk case. Smoke
C2-off-stream against an existing warm deploy before trusting payload
numbers on instrumented cells.

Smoke (5 requests, ~30 s):

  python phase3_vllm_driver_stream.py \\
      --condition baseline \\
      --base-url https://<vllm>.debug.<org>.containers.tinfoil.dev/v1 \\
      --api-key "$VLLM_API_KEY" \\
      --pairs-json pairs.json \\
      --n-requests 5 --req-rate 1.0 \\
      --skip-health \\
      --out-dir runs/phase3_smoke/C1-off-stream \\
      --cell-id C1-off-stream --cc-state off

Full streaming cell (50 requests):

  python phase3_vllm_driver_stream.py \\
      --condition baseline \\
      --base-url ... --api-key "$VLLM_API_KEY" \\
      --pairs-json pairs.json \\
      --n-requests 50 \\
      --out-dir runs/phase3/C1-off-stream \\
      --cell-id C1-off-stream --cc-state off \\
      --image-digest sha256:<digest>

Exit codes:
  0  success
  2  user error
  4  zero successful requests
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import httpx
import numpy as np

try:
    import pandas as pd  # type: ignore
except ImportError:
    pd = None

try:
    import boto3  # type: ignore
except ImportError:
    boto3 = None

from captures import _estimate_payload_bytes

# Reuse canonical preset/xargs/health/load_pairs/interleave from the
# sequential driver to keep a single source of truth.
from phase3_vllm_driver import (
    CONDITION_PRESETS,
    DEFAULT_REQ_RATES,
    preset_to_xargs,
    wait_for_health,
    load_pairs,
    interleave,
)


SCHEMA_VERSION = "phase3-vllm-stream-driver-v1"


# ===== per-request row =======================================================

@dataclass
class RequestRow:
    """Union schema with phase3_vllm_driver.RequestRow plus streaming fields.
    Gradient-only fields kept as None defaults for parquet compat."""
    request_id: int
    pair_id: int
    prompt_class: str
    t_send: float
    t_first_token: float
    t_complete: float
    wall_seconds: float
    tokens_in: int
    tokens_out: int
    payload_bytes: int
    completion_text: Optional[str]
    # Streaming-specific:
    ttft_seconds: Optional[float] = None
    itl_p50_seconds: Optional[float] = None
    itl_p95_seconds: Optional[float] = None
    n_chunks: Optional[int] = None
    # Gradient-only (always None for vllm cells):
    fwd_seconds: Optional[float] = None
    bwd_seconds: Optional[float] = None
    server_total_seconds: Optional[float] = None
    loss: Optional[float] = None
    target_token_id: Optional[int] = None
    target_token: Optional[str] = None
    # Both:
    http_status: int = 0
    error: Optional[str] = None


# ===== single request ========================================================

def send_one_stream(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    xargs: dict,
    max_new_tokens: int,
    request_id: int,
    pair_id: int,
    prompt_class: str,
    timeout: float = 600.0,
) -> RequestRow:
    """One streaming /v1/chat/completions call.

    Timing semantics:
      t_send       wall-clock epoch at HTTP POST initiation
      t_perf_start monotonic baseline at the same moment
      ttft         perf_counter delta to first non-empty delta.content
      chunk_times  perf_counter at each non-empty content chunk
      t_complete   wall-clock when stream loop exits
      wall_seconds perf_counter delta to t_complete
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_new_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "vllm_xargs": xargs,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    url = f"{base_url.rstrip('/')}/chat/completions"

    t_send = time.time()
    t_perf_start = time.perf_counter()
    ttft: Optional[float] = None
    chunk_times: list[float] = []
    completion_text = ""
    usage: dict = {}
    final_chunk: dict = {}
    http_status = 0
    error: Optional[str] = None
    n_chunks = 0
    payload_bytes_accum = 0

    try:
        with httpx.Client(timeout=timeout) as client:
            with client.stream("POST", url, json=body, headers=headers) as resp:
                http_status = resp.status_code
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    if isinstance(line, bytes):
                        line = line.decode("utf-8", errors="replace")
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        evt = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    n_chunks += 1
                    t_chunk = time.perf_counter()
                    final_chunk = evt
                    payload_bytes_accum += _estimate_payload_bytes(evt)

                    choices = evt.get("choices") or []
                    if choices:
                        delta = (choices[0] or {}).get("delta") or {}
                        content_chunk = delta.get("content") or ""
                        if content_chunk:
                            if ttft is None:
                                ttft = t_chunk - t_perf_start
                            chunk_times.append(t_chunk)
                            completion_text += content_chunk
                    if evt.get("usage"):
                        usage = evt["usage"]
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        http_status = getattr(getattr(e, "response", None), "status_code", 0) or http_status

    t_complete = time.time()
    wall = time.perf_counter() - t_perf_start

    itl_p50: Optional[float] = None
    itl_p95: Optional[float] = None
    if len(chunk_times) >= 3:
        gaps = np.diff(chunk_times)
        itl_p50 = float(np.percentile(gaps, 50))
        itl_p95 = float(np.percentile(gaps, 95))

    payload_bytes = payload_bytes_accum
    t_first_token_abs = (
        t_send + ttft if ttft is not None
        else t_complete
    )

    return RequestRow(
        request_id=request_id,
        pair_id=pair_id,
        prompt_class=prompt_class,
        t_send=t_send,
        t_first_token=t_first_token_abs,
        t_complete=t_complete,
        wall_seconds=wall,
        tokens_in=int(usage.get("prompt_tokens") or 0),
        tokens_out=int(usage.get("completion_tokens") or 0),
        payload_bytes=payload_bytes,
        completion_text=completion_text or None,
        ttft_seconds=ttft,
        itl_p50_seconds=itl_p50,
        itl_p95_seconds=itl_p95,
        n_chunks=n_chunks,
        http_status=http_status,
        error=error,
    )


# ===== driver loop ===========================================================

def run_cell(
    base_url: str,
    api_key: str,
    model: str,
    prompts: Sequence[tuple[int, str, str]],
    xargs: dict,
    max_new_tokens: int,
    req_rate: float,
    timeout: float = 600.0,
    warmup_requests: int = 2,                # NEW
) -> list[RequestRow]:
    # Warmup: fire `warmup_requests` requests with the first prompt(s) and
    # discard. Quenches first-request cold-cache penalty.
    if warmup_requests > 0 and prompts:
        print(f"[warmup] firing {warmup_requests} throwaway requests")
        for w in range(warmup_requests):
            warm_prompt = prompts[w % len(prompts)][2]
            warm_row = send_one_stream(
                base_url, api_key, model, warm_prompt, xargs, max_new_tokens,
                request_id=-1 - w, pair_id=-1, prompt_class="warmup",
                timeout=timeout,
            )
            ttft_str = (f"ttft={warm_row.ttft_seconds:.3f}s"
                        if warm_row.ttft_seconds is not None else "ttft=—")
            print(f"  [warmup {w+1}/{warmup_requests}] "
                  f"wall={warm_row.wall_seconds:.2f}s {ttft_str}")
    min_interval = 1.0 / req_rate if req_rate > 0 else 0.0
    rows: list[RequestRow] = []
    last_send = 0.0
    for i, (pair_id, prompt_class, prompt) in enumerate(prompts):
        wait = min_interval - (time.monotonic() - last_send)
        if wait > 0:
            time.sleep(wait)
        last_send = time.monotonic()
        row = send_one_stream(
            base_url, api_key, model, prompt, xargs, max_new_tokens,
            request_id=i, pair_id=pair_id, prompt_class=prompt_class,
            timeout=timeout,
        )
        rows.append(row)
        if row.error:
            print(f"  [{i+1}/{len(prompts)}] {prompt_class} pair={pair_id} "
                  f"ERROR http={row.http_status} {row.error}")
        else:
            ttft_str = (f"ttft={row.ttft_seconds:.3f}s"
                        if row.ttft_seconds is not None else "ttft=—")
            print(f"  [{i+1}/{len(prompts)}] {prompt_class} pair={pair_id} "
                  f"wall={row.wall_seconds:.2f}s {ttft_str} "
                  f"tok_out={row.tokens_out} chunks={row.n_chunks} "
                  f"payload={(row.payload_bytes or 0) // 1024}KB")
    return rows


# ===== aggregation ===========================================================

def _percentiles(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    a = np.asarray(xs, dtype=np.float64)
    return {
        "n": int(a.size),
        "min": float(a.min()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "stdev": float(a.std(ddof=1)) if a.size > 1 else 0.0,
    }


def _stats_block(rows: list[RequestRow]) -> dict:
    return {
        "wall_seconds":  _percentiles([r.wall_seconds for r in rows]),
        "ttft_seconds":  _percentiles([r.ttft_seconds for r in rows if r.ttft_seconds is not None]),
        "itl_p50_seconds": _percentiles(
            [r.itl_p50_seconds for r in rows if r.itl_p50_seconds is not None]),
        "payload_bytes": _percentiles([float(r.payload_bytes) for r in rows]),
        "tokens_in":     _percentiles([float(r.tokens_in) for r in rows]),
        "tokens_out":    _percentiles([float(r.tokens_out) for r in rows]),
        "n_chunks":      _percentiles(
            [float(r.n_chunks) for r in rows if r.n_chunks is not None]),
    }


def summarize(
    rows: list[RequestRow],
    cell_id: str,
    condition: str,
    cc_state: str,
    base_url: str,
    image_digest: Optional[str],
    req_rate: float,
    n_requests_target: int,
    xargs: dict,
) -> dict:
    ok = [r for r in rows if r.error is None]
    err = [r for r in rows if r.error is not None]
    toxic_ok = [r for r in ok if r.prompt_class == "toxic"]
    benign_ok = [r for r in ok if r.prompt_class == "benign"]
    return {
        "schema_version": SCHEMA_VERSION,
        "cell_id": cell_id,
        "condition": condition,
        "cc_state": cc_state,
        "driver": "stream",
        "concurrency": 1,
        "streaming": True,
        "base_url": base_url,
        "image_digest": image_digest,
        "endpoint": "/v1/chat/completions",
        "vllm_xargs": xargs,
        "request_options": {
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        "req_rate": req_rate,
        "n_requests_target": n_requests_target,
        "n_total": len(rows),
        "n_success": len(ok),
        "n_error": len(err),
        "success_rate": len(ok) / max(1, len(rows)),
        "errors_sample": [
            {
                "request_id": r.request_id,
                "pair_id": r.pair_id,
                "prompt_class": r.prompt_class,
                "http_status": r.http_status,
                "error": r.error,
            }
            for r in err[:5]
        ],
        "per_class": {
            "toxic":  {"n": len(toxic_ok),  **_stats_block(toxic_ok)},
            "benign": {"n": len(benign_ok), **_stats_block(benign_ok)},
        },
        "overall": _stats_block(ok),
    }


# ===== output ================================================================

def write_outputs(rows: list[RequestRow], summary: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_dict = [asdict(r) for r in rows]
    if pd is not None:
        req_path = out_dir / "requests.parquet"
        pd.DataFrame(rows_dict).to_parquet(req_path, index=False)
    else:
        req_path = out_dir / "requests.jsonl"
        req_path.write_text("\n".join(json.dumps(r) for r in rows_dict))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return req_path


def maybe_upload(out_dir: Path, cell_id: str) -> None:
    bucket = os.environ.get("S3_BUCKET")
    endpoint_url = os.environ.get("R2_ENDPOINT_URL") or os.environ.get("R2_ENDPOINT")
    if not bucket:
        return
    if boto3 is None:
        print("[warn] S3_BUCKET set but boto3 not installed; skipping upload")
        return
    client_kwargs = {"endpoint_url": endpoint_url} if endpoint_url else {}
    s3 = boto3.client("s3", **client_kwargs)
    for fname in ("requests.parquet", "requests.jsonl", "summary.json"):
        p = out_dir / fname
        if not p.exists():
            continue
        key = f"phase3/{cell_id}/{fname}"
        s3.upload_file(str(p), bucket, key)
        backend = "r2" if endpoint_url else "s3"
        print(f"[upload] {backend}://{bucket}/{key}")


# ===== CLI ===================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--condition", required=True, choices=list(CONDITION_PRESETS.keys()))
    p.add_argument("--base-url", required=True,
                   help="vLLM OpenAI-compatible base URL, ending in /v1.")
    p.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    p.add_argument("--model", default="glm-5-1-fp8")
    p.add_argument("--pairs-json", type=Path, required=True)
    p.add_argument("--n-requests", type=int, default=50,
                   help="Total requests (default 50 = 25 toxic + 25 benign).")
    p.add_argument("--req-rate", type=float, default=None,
                   help=f"Target requests/sec. Default per condition: {DEFAULT_REQ_RATES}")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--cell-id", required=True)
    p.add_argument("--cc-state", choices=["on", "off"], required=True)
    p.add_argument("--image-digest", default=None)
    p.add_argument("--health-timeout", type=float, default=1800.0)
    p.add_argument("--health-poll-interval", type=float, default=10.0)
    p.add_argument("--skip-health", action="store_true")
    p.add_argument("--warmup-requests", type=int, default=2,
               help="Throwaway requests before measurement to quench cold-cache penalty.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    preset = CONDITION_PRESETS[args.condition]
    xargs = preset_to_xargs(preset)
    req_rate = args.req_rate if args.req_rate is not None else DEFAULT_REQ_RATES[args.condition]

    try:
        pairs = load_pairs(args.pairs_json)
    except Exception as e:
        print(f"[error] failed to load --pairs-json {args.pairs_json}: {e}", file=sys.stderr)
        return 2
    prompts = interleave(pairs, args.n_requests)
    if not prompts:
        print("[error] no prompts after interleave", file=sys.stderr)
        return 2

    print(f"[cell] {args.cell_id} (cc={args.cc_state}) condition={args.condition} STREAMING")
    print(f"[cell] target={args.base_url} model={args.model}")
    print(f"[cell] xargs={xargs}")
    print(f"[cell] {len(prompts)} requests @ {req_rate} req/s "
          f"(min wall ~{len(prompts) / req_rate / 60:.1f} min)")

    if not args.skip_health:
        print(f"[health] polling (max {args.health_timeout:.0f}s)...")
        try:
            h = wait_for_health(args.base_url, args.api_key,
                                timeout=args.health_timeout,
                                poll_interval=args.health_poll_interval)
            print(f"[health] ready: {h}")
        except TimeoutError as e:
            print(f"[error] {e}", file=sys.stderr)
            return 4

    t0 = time.monotonic()
    rows = run_cell(
        args.base_url, args.api_key, args.model, prompts,
        xargs=xargs, max_new_tokens=args.max_new_tokens,
        req_rate=req_rate, timeout=args.timeout,
        warmup_requests=args.warmup_requests,
    )
    elapsed_min = (time.monotonic() - t0) / 60.0
    print(f"[run] complete in {elapsed_min:.1f} min")

    summary = summarize(
        rows,
        cell_id=args.cell_id,
        condition=args.condition,
        cc_state=args.cc_state,
        base_url=args.base_url,
        image_digest=args.image_digest,
        req_rate=req_rate,
        n_requests_target=args.n_requests,
        xargs=xargs,
    )

    req_path = write_outputs(rows, summary, args.out_dir)
    print(f"[out] {req_path}")
    print(f"[out] {args.out_dir / 'summary.json'}")
    wall_p50 = summary["overall"]["wall_seconds"].get("p50", float("nan"))
    ttft_block = summary["overall"]["ttft_seconds"]
    ttft_p50 = ttft_block.get("p50") if ttft_block.get("n") else None
    print(f"[summary] n_success={summary['n_success']}/{summary['n_total']} "
          f"wall_p50={wall_p50:.2f}s "
          f"ttft_p50={(ttft_p50 or float('nan')):.3f}s")

    maybe_upload(args.out_dir, args.cell_id)
    return 0 if summary["n_success"] > 0 else 4


if __name__ == "__main__":
    sys.exit(main())