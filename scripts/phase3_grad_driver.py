#!/usr/bin/env python3
"""
phase3_grad_driver.py — Phase 3 C3 cell driver: rate-limited /v1/saliency load.

Drives the prepilot-vllm-lens-grad sidecar for the gradient cell (C3) of the
Phase 3 measurement matrix. Wire format is byte-for-byte identical to
captures_gradients.call_with_gradient_capture; the only differences are:

  - Rate-limited, single-stream dispatch (send-to-send min interval).
  - Per-request row schema matches PHASE3_PLAN §7.1.
  - Aggregates to summary.json with p50/p95/max per metric, per class.
  - Health-check polls for {"status": "ok"}, not just HTTP 200.
  - Optional S3/R2 upload (boto3) gated on S3_BUCKET / R2_ENDPOINT env vars.

Depends on captures.py being importable (for _estimate_payload_bytes — keeps
payload_bytes comparable to runs/phase2_validation/gradient/aggregate.json).

Smoke run against the live (already-warm) sidecar, ~2 min wall:

  python phase3_grad_driver.py \\
      --base-url https://glm-5-1-prepilot-grad.debug.pour-demain.containers.tinfoil.dev \\
      --api-key "$VLLM_API_KEY" \\
      --pairs-json runs/phase2_validation/gradient/pairs.json \\
      --n-requests 10 --req-rate 0.5 \\
      --skip-health \\
      --out-dir runs/phase3_smoke/C3-off \\
      --cell-id C3-off --cc-state off

Full C3-off cell (matches Phase 2's 50-pair size), ~33 min wall at 0.05 req/s:

  python phase3_grad_driver.py \\
      --base-url ... --api-key "$VLLM_API_KEY" \\
      --pairs-json runs/phase2_validation/gradient/pairs.json \\
      --n-requests 100 --req-rate 0.05 \\
      --out-dir runs/phase3/C3-off \\
      --cell-id C3-off --cc-state off \\
      --image-digest sha256:41b1d219b967e73683f78601f0b039e88ba91e77567a25885aab4a1162e00710

Exit codes:
  0  success
  2  user error (bad CLI args, pairs.json malformed, etc.)
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

# Optional deps — degrade gracefully so the script runs on a bare laptop.
try:
    import pandas as pd  # type: ignore
except ImportError:
    pd = None

try:
    import boto3  # type: ignore
except ImportError:
    boto3 = None

# Phase 2 utility — single source of truth for payload-size accounting.
from captures import _estimate_payload_bytes


SCHEMA_VERSION = "phase3-grad-driver-v1"


# ===== per-request row =======================================================

@dataclass
class RequestRow:
    """One row of requests.parquet. Schema = PHASE3_PLAN §7.1 + gradient-cell fields.

    Time fields:
      t_send / t_complete  — wall-clock epoch seconds (for joining with nvidia-smi).
      t_first_token        — = t_complete (this endpoint is non-streaming).
      wall_seconds         — monotonic delta from t_send to t_complete.

    Server-side timings come from response 'diagnostics' block.
    """
    request_id: int
    pair_id: int
    prompt_class: str          # "toxic" | "benign"
    t_send: float
    t_first_token: float
    t_complete: float
    wall_seconds: float
    tokens_in: int             # diagnostics.prompt_tokens
    tokens_out: int            # always 0 — gradient cell has no completion
    payload_bytes: int         # _estimate_payload_bytes(response_json)
    fwd_seconds: Optional[float]
    bwd_seconds: Optional[float]
    server_total_seconds: Optional[float]
    loss: Optional[float]
    target_token_id: Optional[int]
    target_token: Optional[str]
    http_status: int
    error: Optional[str]


# ===== health check ==========================================================

def wait_for_health(
    base_url: str,
    api_key: str,
    timeout: float = 1800.0,
    poll_interval: float = 10.0,
) -> dict:
    """Poll /health until status == 'ok'. HTTP 200 alone is insufficient: the
    route returns 200 with status='loading' during the ~16 min model load.
    """
    deadline = time.monotonic() + timeout
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    last_body: dict = {}
    while time.monotonic() < deadline:
        try:
            r = httpx.get(
                f"{base_url.rstrip('/')}/health",
                headers=headers,
                timeout=30.0,
            )
            if r.status_code == 200:
                last_body = r.json()
                if last_body.get("status") == "ok":
                    return last_body
                print(f"  [health] status={last_body.get('status')!r}; waiting...")
            else:
                print(f"  [health] HTTP {r.status_code}; waiting...")
        except Exception as e:
            print(f"  [health] {type(e).__name__}: {e}")
        time.sleep(poll_interval)
    raise TimeoutError(
        f"/health did not return status=ok within {timeout:.0f}s "
        f"(last body: {last_body})"
    )


# ===== single request ========================================================

def send_one(
    base_url: str,
    api_key: str,
    prompt: str,
    request_id: int,
    pair_id: int,
    prompt_class: str,
    timeout: float = 600.0,
) -> RequestRow:
    """One /v1/saliency call. Wire format identical to
    captures_gradients.call_with_gradient_capture — preserves payload_bytes
    comparability with Phase 2 aggregate.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {"messages": [{"role": "user", "content": prompt}]}
    # Omit target_token_id → server defaults to argmax of last-position logits
    # (matches Phase 2; argmax is deterministic per prompt → clean CC delta).

    t_send = time.time()
    t_perf_start = time.perf_counter()
    http_status = 0
    error: Optional[str] = None
    raw: dict = {}
    try:
        resp = httpx.post(
            f"{base_url.rstrip('/')}/v1/saliency",
            json=body,
            headers=headers,
            timeout=timeout,
        )
        http_status = resp.status_code
        resp.raise_for_status()
        raw = resp.json()
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    t_complete = time.time()
    wall = time.perf_counter() - t_perf_start

    diag = (raw.get("diagnostics") or {}) if not error else {}

    def _maybe_float(k: str) -> Optional[float]:
        v = diag.get(k)
        return float(v) if v is not None else None

    def _maybe_int(k: str) -> Optional[int]:
        v = diag.get(k)
        return int(v) if v is not None else None

    return RequestRow(
        request_id=request_id,
        pair_id=pair_id,
        prompt_class=prompt_class,
        t_send=t_send,
        t_first_token=t_complete,   # non-streaming endpoint
        t_complete=t_complete,
        wall_seconds=wall,
        tokens_in=int(diag.get("prompt_tokens") or 0),
        tokens_out=0,
        payload_bytes=_estimate_payload_bytes(raw),
        fwd_seconds=_maybe_float("fwd_seconds"),
        bwd_seconds=_maybe_float("bwd_seconds"),
        server_total_seconds=_maybe_float("total_seconds"),
        loss=_maybe_float("loss"),
        target_token_id=_maybe_int("target_token_id"),
        target_token=str(diag.get("target_token") or ""),
        http_status=http_status,
        error=error,
    )


# ===== prompt loading + ordering ============================================

def load_pairs(pairs_json: Path) -> list[dict]:
    pairs = json.loads(pairs_json.read_text())
    if not isinstance(pairs, list) or not pairs:
        raise ValueError(f"{pairs_json} did not contain a non-empty list")
    for p in pairs:
        for k in ("pair_id", "toxic", "benign"):
            if k not in p:
                raise ValueError(f"pair missing required field {k!r}: {p}")
    return pairs


def interleave(pairs: list[dict], n_requests: int) -> list[tuple[int, str, str]]:
    """Alternate toxic/benign by pair index, cap at n_requests.

    For pair_id i, emit (i, 'toxic', pair.toxic) then (i, 'benign', pair.benign).
    Result length = min(2 * len(pairs), n_requests).
    """
    seq: list[tuple[int, str, str]] = []
    for p in pairs:
        seq.append((int(p["pair_id"]), "toxic", p["toxic"]))
        seq.append((int(p["pair_id"]), "benign", p["benign"]))
    if n_requests < len(seq):
        seq = seq[:n_requests]
    return seq


# ===== driver loop ===========================================================

def run_cell(
    base_url: str,
    api_key: str,
    prompts: Sequence[tuple[int, str, str]],
    req_rate: float,
    timeout: float,
) -> list[RequestRow]:
    """Sequential, throttled by send-to-send interval. 'Single-stream' per
    PHASE3_PLAN §4.1 — no concurrency, no client-side queueing.

    Effective rate = min(req_rate, 1 / per_request_wall). If the backward pass
    exceeds the interval, the cell runs at the latency floor (no queue forms).
    """
    min_interval = 1.0 / req_rate if req_rate > 0 else 0.0
    rows: list[RequestRow] = []
    last_send = 0.0
    for i, (pair_id, prompt_class, prompt) in enumerate(prompts):
        wait = min_interval - (time.monotonic() - last_send)
        if wait > 0:
            time.sleep(wait)
        last_send = time.monotonic()
        row = send_one(
            base_url, api_key, prompt,
            request_id=i, pair_id=pair_id, prompt_class=prompt_class,
            timeout=timeout,
        )
        rows.append(row)
        if row.error:
            print(
                f"  [{i + 1}/{len(prompts)}] {prompt_class} pair={pair_id} "
                f"ERROR http={row.http_status} {row.error}"
            )
        else:
            fwd = f"{row.fwd_seconds:.2f}" if row.fwd_seconds is not None else "—"
            bwd = f"{row.bwd_seconds:.2f}" if row.bwd_seconds is not None else "—"
            print(
                f"  [{i + 1}/{len(prompts)}] {prompt_class} pair={pair_id} "
                f"wall={row.wall_seconds:.2f}s fwd={fwd}s bwd={bwd}s "
                f"payload={row.payload_bytes // 1024}KB"
            )
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
        "wall_seconds":   _percentiles([r.wall_seconds for r in rows]),
        "fwd_seconds":    _percentiles([r.fwd_seconds for r in rows if r.fwd_seconds is not None]),
        "bwd_seconds":    _percentiles([r.bwd_seconds for r in rows if r.bwd_seconds is not None]),
        "server_total_seconds": _percentiles(
            [r.server_total_seconds for r in rows if r.server_total_seconds is not None]
        ),
        "loss":           _percentiles([r.loss for r in rows if r.loss is not None]),
        "payload_bytes":  _percentiles([float(r.payload_bytes) for r in rows]),
        "tokens_in":      _percentiles([float(r.tokens_in) for r in rows]),
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
        "base_url": base_url,
        "image_digest": image_digest,
        "endpoint": "/v1/saliency",
        "request_body_schema": {"messages": "[{role,content}]", "target_token_id": "omitted"},
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
    """Write requests.parquet (or .jsonl fallback) and summary.json. Returns
    the path to the requests file actually written."""
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
    """Upload requests.{parquet,jsonl} + summary.json to S3/R2 if env-configured."""
    bucket = os.environ.get("S3_BUCKET")
    endpoint_url = os.environ.get("R2_ENDPOINT")  # set for Cloudflare R2; omit for AWS S3
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
    p.add_argument("--base-url", required=True,
                   help="Gradient sidecar base URL, e.g. "
                        "https://glm-5-1-prepilot-grad.debug.<org>.containers.tinfoil.dev")
    p.add_argument("--api-key",
                   default=os.environ.get("VLLM_API_KEY", "EMPTY"),
                   help="Bearer token. Default $VLLM_API_KEY or 'EMPTY' (debug deployments).")
    p.add_argument("--pairs-json", type=Path, required=True,
                   help="Path to Phase 2 pairs.json (reuse for CC-on/off comparability).")
    p.add_argument("--n-requests", type=int, default=100,
                   help="Total requests. Default 100 = 50 toxic + 50 benign interleaved.")
    p.add_argument("--req-rate", type=float, default=0.05,
                   help="Target requests/sec. Default 0.05 (PHASE3_PLAN §4.1 for gradient).")
    p.add_argument("--timeout", type=float, default=600.0,
                   help="Per-request HTTP timeout (sec). Default 600.")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Directory to write requests.parquet/jsonl + summary.json.")
    p.add_argument("--cell-id", default="C3-off",
                   help="Phase 3 cell identifier (e.g. C3-off, C3-on). Used in summary + S3 key.")
    p.add_argument("--cc-state", choices=["on", "off"], default="off",
                   help="Confidential Compute state of the target deployment. Metadata only.")
    p.add_argument("--image-digest", default=None,
                   help="ghcr image digest of the gradient sidecar (sha256:...) for reproducibility.")
    p.add_argument("--health-timeout", type=float, default=1800.0,
                   help="Max seconds to wait for /health → status=ok. Default 1800 (model load ~16 min).")
    p.add_argument("--health-poll-interval", type=float, default=10.0,
                   help="Seconds between /health polls.")
    p.add_argument("--skip-health", action="store_true",
                   help="Skip /health poll. Use only when target is known to be warm.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        pairs = load_pairs(args.pairs_json)
    except Exception as e:
        print(f"[error] failed to load --pairs-json {args.pairs_json}: {e}", file=sys.stderr)
        return 2
    prompts = interleave(pairs, args.n_requests)
    if not prompts:
        print("[error] no prompts after interleave (empty pairs.json?)", file=sys.stderr)
        return 2

    print(f"[cell] {args.cell_id} (cc={args.cc_state}) "
          f"target={args.base_url}")
    print(f"[cell] {len(prompts)} requests @ {args.req_rate} req/s "
          f"(min wall ~{len(prompts) / args.req_rate / 60:.1f} min)")

    if not args.skip_health:
        print(f"[health] polling {args.base_url}/health "
              f"(max {args.health_timeout:.0f}s, every {args.health_poll_interval:.0f}s)...")
        try:
            h = wait_for_health(
                args.base_url, args.api_key,
                timeout=args.health_timeout,
                poll_interval=args.health_poll_interval,
            )
            print(f"[health] ready: {h}")
        except TimeoutError as e:
            print(f"[error] {e}", file=sys.stderr)
            return 4

    t0 = time.monotonic()
    rows = run_cell(
        args.base_url, args.api_key, prompts,
        req_rate=args.req_rate, timeout=args.timeout,
    )
    elapsed_min = (time.monotonic() - t0) / 60.0
    print(f"[run] complete in {elapsed_min:.1f} min")

    summary = summarize(
        rows,
        cell_id=args.cell_id,
        condition="gradient",
        cc_state=args.cc_state,
        base_url=args.base_url,
        image_digest=args.image_digest,
        req_rate=args.req_rate,
        n_requests_target=args.n_requests,
    )

    req_path = write_outputs(rows, summary, args.out_dir)
    print(f"[out] {req_path}")
    print(f"[out] {args.out_dir / 'summary.json'}")
    print(f"[summary] n_success={summary['n_success']}/{summary['n_total']} "
          f"wall_p50={summary['overall']['wall_seconds'].get('p50', float('nan')):.2f}s "
          f"bwd_p50={summary['overall']['bwd_seconds'].get('p50', float('nan')):.2f}s")

    maybe_upload(args.out_dir, args.cell_id)

    if summary["n_success"] == 0:
        print("[fail] zero successful requests — check --base-url / --api-key / sidecar status.",
              file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
