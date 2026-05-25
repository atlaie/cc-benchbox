#!/usr/bin/env python3
"""
phase3_egress_driver_v2.py — laptop-side driver for TEE-hosted egress pipeline.

Replaces phase3_egress_driver.py (which ran the encoder client-side).  With
egress_service.py running inside the Tinfoil CC container, the encoder now
executes in the TEE; this driver is a thin client that records:

    wall_seconds              — end-to-end HTTP, laptop-side perf_counter
    server_*_seconds          — per-stage timings reported by egress_service
                                (deserialize, encoder_total, aggregate, plot,
                                 bundle, ledger)
    network_seconds           — wall_seconds − server_encoder_total
                                − server_deserialize − server_vllm_estimate

The network slice is a derived quantity: it tells you how much of the
laptop's end-to-end wall is HTTPS + TLS + CC TEE boundary, vs how much is
server-side compute.  Useful when comparing this driver's wall_seconds to
the existing C2-on baseline (same vLLM compute path, same CC, no encoder).

Bundle bytes return base64-encoded in the response when egress.include_bundle
is True (default).  We record bundle_bytes (the size of the actual artifact
that crossed the egress boundary) but do not persist the bundle locally
unless --save-bundles is set — for the cost measurement we only need the size.

Usage mirrors phase3_egress_driver.py:

  python phase3_egress_driver_v2.py \\
      --condition repe_bundle \\
      --base-url https://<egress-deployment>.containers.tinfoil.dev \\
      --api-key "$VLLM_API_KEY" \\
      --pairs-json runs/phase2_validation/repe_bundle/pairs.json \\
      --n-requests 100 \\
      --out-dir runs/phase3/E3-on \\
      --cell-id E3-on --cc-state on \\
      --egress-stages aggregate,plot,bundle,ledger \\
      --session-id E3-on \\
      --image-digest sha256:<egress-image-digest>

Note: --base-url should NOT end in /v1 — egress_service serves at the root.
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

# Reuse condition presets, default rates, pair handling, health check.
from phase3_vllm_driver import (
    CONDITION_PRESETS,
    DEFAULT_REQ_RATES,
    interleave,
    load_pairs,
    preset_to_xargs,
)


SCHEMA_VERSION = "phase3-egress-driver-v2"

ALL_STAGES = ("aggregate", "plot", "bundle", "ledger")


# ===== per-request row =======================================================

@dataclass
class EgressRowV2:
    """All timings except wall_seconds come from the server response.  wall is
    laptop perf_counter around the HTTP call."""

    request_id: int
    pair_id: int
    prompt_class: str

    t_send: float
    t_complete: float
    wall_seconds: float                   # end-to-end (laptop perf_counter)

    # Authoritative server-side timings.
    server_deserialize_seconds: float
    server_encoder_total_seconds: float
    server_aggregate_seconds: float
    server_plot_seconds: float
    server_bundle_seconds: float
    server_ledger_seconds: float

    # Derived: laptop network + TEE boundary slice.
    network_seconds: float                # wall − (deser + encoder)

    tokens_in: int
    tokens_out: int
    raw_payload_bytes: int                # what the loopback saw, never crossed boundary
    bundle_bytes: int                     # what ACTUALLY crossed the egress boundary
    aggregate_bytes: int
    n_plots: int

    stages_run: str                       # comma-joined for parquet

    completion_text: Optional[str]
    bundle_sha256: Optional[str]
    bundle_signature_hex: Optional[str]

    http_status: int = 0
    error: Optional[str] = None


# ===== health check (egress_service has /health that combines both) =========

def wait_for_health(
    base_url: str, api_key: str,
    timeout: float = 1800.0, poll_interval: float = 10.0,
) -> dict:
    base = base_url.rstrip("/")
    deadline = time.monotonic() + timeout
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{base}/health", headers=headers, timeout=30.0)
            if r.status_code == 200:
                body = r.json()
                if body.get("status") == "ok":
                    return body
                print(f"  [health] status={body.get('status')!r}; waiting...")
            else:
                print(f"  [health] HTTP {r.status_code}; waiting...")
        except Exception as e:
            print(f"  [health] {type(e).__name__}: {e}")
        time.sleep(poll_interval)
    raise TimeoutError(f"/health did not return status=ok within {timeout:.0f}s")


# ===== single request ========================================================

def send_one(
    client: httpx.Client,
    base_url: str,
    model: str,
    prompt: str,
    xargs: dict,
    max_new_tokens: int,
    stages: list[str],
    session_id: str,
    include_bundle: bool,
    request_id: int,
    pair_id: int,
    prompt_class: str,
) -> EgressRowV2:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_new_tokens,
        "vllm_xargs": xargs,
        "chat_template_kwargs": {"enable_thinking": False},
        "egress": {
            "stages": stages,
            "request_id": request_id,
            "pair_id": pair_id,
            "prompt_class": prompt_class,
            "session_id": session_id,
            "include_bundle": include_bundle,
        },
    }

    t_send = time.time()
    t_perf_start = time.perf_counter()
    http_status = 0
    error: Optional[str] = None
    payload: dict = {}

    try:
        r = client.post(f"{base_url.rstrip('/')}/v1/egress_eval", json=body)
        http_status = r.status_code
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    t_complete = time.time()
    wall = time.perf_counter() - t_perf_start

    if error is not None:
        return EgressRowV2(
            request_id=request_id, pair_id=pair_id, prompt_class=prompt_class,
            t_send=t_send, t_complete=t_complete, wall_seconds=wall,
            server_deserialize_seconds=0.0, server_encoder_total_seconds=0.0,
            server_aggregate_seconds=0.0, server_plot_seconds=0.0,
            server_bundle_seconds=0.0, server_ledger_seconds=0.0,
            network_seconds=wall,
            tokens_in=0, tokens_out=0,
            raw_payload_bytes=0, bundle_bytes=0, aggregate_bytes=0, n_plots=0,
            stages_run=",".join(stages),
            completion_text=None, bundle_sha256=None, bundle_signature_hex=None,
            http_status=http_status, error=error,
        )

    completion = payload.get("completion") or {}
    eg = payload.get("egress") or {}
    usage = completion.get("usage") or {}
    choices = completion.get("choices") or [{}]
    message = (choices[0] or {}).get("message") or {}
    completion_text = message.get("content") or ""

    server_deser = float(eg.get("deserialize_seconds") or 0.0)
    server_encoder = float(eg.get("encoder_total_seconds") or 0.0)
    network = max(0.0, wall - server_deser - server_encoder)

    return EgressRowV2(
        request_id=request_id, pair_id=pair_id, prompt_class=prompt_class,
        t_send=t_send, t_complete=t_complete, wall_seconds=wall,
        server_deserialize_seconds=server_deser,
        server_encoder_total_seconds=server_encoder,
        server_aggregate_seconds=float(eg.get("aggregate_seconds") or 0.0),
        server_plot_seconds=float(eg.get("plot_seconds") or 0.0),
        server_bundle_seconds=float(eg.get("bundle_seconds") or 0.0),
        server_ledger_seconds=float(eg.get("ledger_seconds") or 0.0),
        network_seconds=network,
        tokens_in=int(usage.get("prompt_tokens") or 0),
        tokens_out=int(usage.get("completion_tokens") or 0),
        raw_payload_bytes=int(eg.get("raw_payload_bytes") or 0),
        bundle_bytes=int(eg.get("bundle_bytes") or 0),
        aggregate_bytes=int(eg.get("aggregate_bytes") or 0),
        n_plots=int(eg.get("n_plots") or 0),
        stages_run=",".join(eg.get("stages_run") or stages),
        completion_text=completion_text,
        bundle_sha256=eg.get("bundle_sha256"),
        bundle_signature_hex=eg.get("bundle_signature_hex"),
        http_status=http_status,
        error=None,
    )


# ===== driver loop ===========================================================

def run_cell(
    base_url: str,
    api_key: str,
    model: str,
    prompts: Sequence[tuple[int, str, str]],
    xargs: dict,
    max_new_tokens: int,
    stages: list[str],
    session_id: str,
    include_bundle: bool,
    req_rate: float,
    timeout: float,
    save_bundles_dir: Optional[Path] = None,
) -> list[EgressRowV2]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    min_interval = 1.0 / req_rate if req_rate > 0 else 0.0
    rows: list[EgressRowV2] = []
    last_send = 0.0

    with httpx.Client(headers=headers, timeout=timeout) as client:
        for i, (pair_id, prompt_class, prompt) in enumerate(prompts):
            wait = min_interval - (time.monotonic() - last_send)
            if wait > 0:
                time.sleep(wait)
            last_send = time.monotonic()
            row = send_one(
                client=client, base_url=base_url, model=model, prompt=prompt,
                xargs=xargs, max_new_tokens=max_new_tokens,
                stages=stages, session_id=session_id,
                include_bundle=include_bundle,
                request_id=i, pair_id=pair_id, prompt_class=prompt_class,
            )
            rows.append(row)
            if row.error:
                print(
                    f"  [{i + 1}/{len(prompts)}] {prompt_class} pair={pair_id} "
                    f"ERROR http={row.http_status} {row.error}"
                )
            else:
                print(
                    f"  [{i + 1}/{len(prompts)}] {prompt_class} pair={pair_id} "
                    f"wall={row.wall_seconds:.2f}s "
                    f"net={row.network_seconds:.2f}s "
                    f"deser={row.server_deserialize_seconds*1000:.0f}ms "
                    f"enc={row.server_encoder_total_seconds*1000:.0f}ms "
                    f"raw={row.raw_payload_bytes//1024}KB→"
                    f"bundle={row.bundle_bytes//1024}KB"
                )
            # Optional: persist the bundle locally for verification.  The
            # bundle bytes come back base64 in the response if
            # include_bundle=True; we'd need to extract from the response,
            # not the row.  Skipped here for size — turn on with --save-bundles
            # if you want a local copy for signature verification spot-checks.
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


def summarize(
    rows: list[EgressRowV2],
    cell_id: str, condition: str, cc_state: str,
    base_url: str, image_digest: Optional[str],
    req_rate: float, n_requests_target: int,
    xargs: dict, stages_run: list[str], session_id: str,
) -> dict:
    ok = [r for r in rows if r.error is None]
    err = [r for r in rows if r.error is not None]
    return {
        "schema_version": SCHEMA_VERSION,
        "cell_id": cell_id, "condition": condition, "cc_state": cc_state,
        "base_url": base_url, "image_digest": image_digest,
        "endpoint": "/v1/egress_eval", "vllm_xargs": xargs,
        "egress_stages": stages_run, "session_id": session_id,
        "req_rate": req_rate, "n_requests_target": n_requests_target,
        "n_total": len(rows), "n_success": len(ok), "n_error": len(err),
        "success_rate": len(ok) / max(1, len(rows)),
        "errors_sample": [
            {"request_id": r.request_id, "pair_id": r.pair_id,
             "http_status": r.http_status, "error": r.error}
            for r in err[:5]
        ],
        "overall": {
            "wall_seconds":         _percentiles([r.wall_seconds for r in ok]),
            "network_seconds":      _percentiles([r.network_seconds for r in ok]),
            "server_deserialize_seconds":   _percentiles([r.server_deserialize_seconds for r in ok]),
            "server_encoder_total_seconds": _percentiles([r.server_encoder_total_seconds for r in ok]),
            "server_aggregate_seconds":     _percentiles([r.server_aggregate_seconds for r in ok]),
            "server_plot_seconds":          _percentiles([r.server_plot_seconds for r in ok]),
            "server_bundle_seconds":        _percentiles([r.server_bundle_seconds for r in ok]),
            "server_ledger_seconds":        _percentiles([r.server_ledger_seconds for r in ok]),
            "raw_payload_bytes":    _percentiles([float(r.raw_payload_bytes) for r in ok]),
            "bundle_bytes":         _percentiles([float(r.bundle_bytes) for r in ok]),
            "aggregate_bytes":      _percentiles([float(r.aggregate_bytes) for r in ok]),
            "tokens_in":            _percentiles([float(r.tokens_in) for r in ok]),
            "tokens_out":           _percentiles([float(r.tokens_out) for r in ok]),
        },
    }


# ===== output ================================================================

def write_outputs(rows: list[EgressRowV2], summary: dict, out_dir: Path) -> Path:
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
    endpoint_url = os.environ.get("R2_ENDPOINT")
    if not bucket or boto3 is None:
        return
    client_kwargs = {"endpoint_url": endpoint_url} if endpoint_url else {}
    s3 = boto3.client("s3", **client_kwargs)
    for fname in ("requests.parquet", "requests.jsonl", "summary.json"):
        p = out_dir / fname
        if not p.exists():
            continue
        key = f"phase3/{cell_id}/{fname}"
        s3.upload_file(str(p), bucket, key)
        print(f"[upload] {'r2' if endpoint_url else 's3'}://{bucket}/{key}")


# ===== CLI ===================================================================

def _parse_stages(s: str) -> list[str]:
    if not s.strip():
        return []
    stages = [x.strip() for x in s.split(",") if x.strip()]
    unknown = set(stages) - set(ALL_STAGES)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown stages: {unknown}; valid={ALL_STAGES}")
    return stages


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--condition", required=True,
                   choices=list(CONDITION_PRESETS.keys()))
    p.add_argument("--base-url", required=True,
                   help="egress_service URL, e.g. https://<deploy>.containers.tinfoil.dev "
                        "(no /v1 suffix — service serves /v1/egress_eval at root).")
    p.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    p.add_argument("--model", default="glm-5-1-fp8")
    p.add_argument("--pairs-json", type=Path, required=True)
    p.add_argument("--n-requests", type=int, default=100)
    p.add_argument("--req-rate", type=float, default=None)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--cell-id", required=True)
    p.add_argument("--cc-state", choices=["on", "off"], required=True)
    p.add_argument("--image-digest", default=None)
    p.add_argument("--health-timeout", type=float, default=1800.0)
    p.add_argument("--health-poll-interval", type=float, default=10.0)
    p.add_argument("--skip-health", action="store_true")
    p.add_argument("--egress-stages", type=str, default="",
                   help=f"Comma-separated subset of {ALL_STAGES}. Empty = deserialize only.")
    p.add_argument("--session-id", type=str, default=None,
                   help="Ledger session id. Default: cell-id.")
    p.add_argument("--no-include-bundle", action="store_true",
                   help="Set egress.include_bundle=False — bundle persisted in TEE only, "
                        "not returned over the wire.  Use to isolate generation cost from "
                        "egress-transit cost.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        stages = _parse_stages(args.egress_stages)
    except argparse.ArgumentTypeError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2

    preset = CONDITION_PRESETS[args.condition]
    xargs = preset_to_xargs(preset)
    if not xargs:
        print(f"[error] condition={args.condition} produces no captures.", file=sys.stderr)
        return 2

    req_rate = args.req_rate if args.req_rate is not None else DEFAULT_REQ_RATES[args.condition]
    session_id = args.session_id or args.cell_id

    try:
        pairs = load_pairs(args.pairs_json)
    except Exception as e:
        print(f"[error] failed to load --pairs-json: {e}", file=sys.stderr)
        return 2
    prompts = interleave(pairs, args.n_requests)
    if not prompts:
        print("[error] empty prompt set", file=sys.stderr)
        return 2

    print(f"[cell] {args.cell_id} (cc={args.cc_state}) condition={args.condition}")
    print(f"[cell] target={args.base_url} model={args.model}")
    print(f"[cell] egress stages: {stages if stages else '∅ (deserialize only)'}")
    print(f"[cell] session_id={session_id} include_bundle={not args.no_include_bundle}")
    print(f"[cell] {len(prompts)} requests @ {req_rate} req/s")

    if not args.skip_health:
        try:
            h = wait_for_health(args.base_url, args.api_key,
                                timeout=args.health_timeout,
                                poll_interval=args.health_poll_interval)
            print(f"[health] {h}")
        except TimeoutError as e:
            print(f"[error] {e}", file=sys.stderr)
            return 4

    t0 = time.monotonic()
    rows = run_cell(
        base_url=args.base_url, api_key=args.api_key, model=args.model,
        prompts=prompts, xargs=xargs, max_new_tokens=args.max_new_tokens,
        stages=stages, session_id=session_id,
        include_bundle=not args.no_include_bundle,
        req_rate=req_rate, timeout=args.timeout,
    )
    print(f"[run] complete in {(time.monotonic()-t0)/60:.1f} min")

    summary = summarize(
        rows=rows, cell_id=args.cell_id, condition=args.condition,
        cc_state=args.cc_state, base_url=args.base_url,
        image_digest=args.image_digest, req_rate=req_rate,
        n_requests_target=args.n_requests, xargs=xargs,
        stages_run=stages, session_id=session_id,
    )
    req_path = write_outputs(rows, summary, args.out_dir)
    print(f"[out] {req_path}")
    print(f"[out] {args.out_dir / 'summary.json'}")
    if summary["n_success"]:
        ovr = summary["overall"]
        print(f"[summary] n_success={summary['n_success']}/{summary['n_total']} "
              f"wall_p50={ovr['wall_seconds']['p50']:.2f}s "
              f"net_p50={ovr['network_seconds']['p50']:.2f}s "
              f"server_enc_p50={ovr['server_encoder_total_seconds']['p50']*1000:.0f}ms")

    maybe_upload(args.out_dir, args.cell_id)
    return 0 if summary["n_success"] else 4


if __name__ == "__main__":
    sys.exit(main())
