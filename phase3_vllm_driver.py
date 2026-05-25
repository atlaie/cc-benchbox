#!/usr/bin/env python3
"""
phase3_vllm_driver.py — Phase 3 driver for the three vLLM-deployment cells:

    C1  baseline      no instrumentation
    C2  repe_bundle   residual + attention_stats at probe layers
    C4  routing       MoE routing at all 75 MoE layers

(C3 / gradient is driven by phase3_grad_driver.py — different deployment,
different endpoint, different request schema.)

Wire format byte-for-byte identical to captures.call_with_capture so
payload_bytes and wall_seconds are directly comparable to
runs/phase2_validation/<condition>/aggregate.json.

Phase 2 → Phase 3 deltas:

  - Rate-limited single-stream dispatch (send-to-send min interval).
  - Per-request row schema is the union of vLLM- and gradient-cell fields,
    so phase3_aggregate.py can union both parquets.
  - openai client configured with max_retries=0; retries silently double
    wall-time on transients and would contaminate the CC delta. (Phase 2
    left the default 3 retries on; this is a deliberate measurement
    decision, not a bug-compat issue.)
  - Non-streaming. t_first_token = t_complete. Streaming + xargs-payload
    survival is unverified; add via --stream once tested. TODO above.

Smoke run (10 requests, fast, against an already-warm vLLM):

  python phase3_vllm_driver.py \\
      --condition baseline \\
      --base-url https://<vllm-deployment>.containers.tinfoil.dev/v1 \\
      --api-key "$VLLM_API_KEY" \\
      --pairs-json runs/phase2_validation/baseline/pairs.json \\
      --n-requests 10 --req-rate 1.0 \\
      --skip-health \\
      --out-dir runs/phase3_smoke/C1-off \\
      --cell-id C1-off --cc-state off

Full cell run (100 requests at the per-condition default rate):

  python phase3_vllm_driver.py \\
      --condition repe_bundle \\
      --base-url ... --api-key "$VLLM_API_KEY" \\
      --pairs-json runs/phase2_validation/repe_bundle/pairs.json \\
      --n-requests 100 \\
      --out-dir runs/phase3/C2-off \\
      --cell-id C2-off --cc-state off \\
      --image-digest sha256:<digest of the prepilot-vllm-lens image>

Exit codes:
  0  success
  2  user error (bad CLI args, pairs.json malformed, unknown condition)
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
from openai import OpenAI

# Optional deps — degrade gracefully so the script runs on a bare laptop.
try:
    import pandas as pd  # type: ignore
except ImportError:
    pd = None

try:
    import boto3  # type: ignore
except ImportError:
    boto3 = None

# Phase 2 utilities — single source of truth.
from captures import (
    DEFAULT_PROBE_LAYERS,
    DEFAULT_ROUTING_LAYERS,
    _estimate_payload_bytes,
)


SCHEMA_VERSION = "phase3-vllm-driver-v1"


# ===== condition presets =====================================================
#
# Lifted verbatim from phase2_capture.CONDITION_PRESETS (the three non-sidecar
# entries). Single source of truth for the cell-to-vllm_xargs mapping.
# When phase3-matrix.yaml lands, this dict moves to YAML and gets read.

CONDITION_PRESETS: dict[str, dict] = {
    "baseline": {
        "description": "No instrumentation. Phase 1 image, no vllm_xargs.",
        "residual_layers": None,
        "routing_layers": None,
        "attention_layers": None,
    },
    "repe_bundle": {
        "description": "EII-1+3+4: residual stream + attention stats on probe layers.",
        "residual_layers": DEFAULT_PROBE_LAYERS,
        "routing_layers": None,
        "attention_layers": DEFAULT_PROBE_LAYERS,
    },
    "routing": {
        "description": "MoE routing capture across all 75 MoE layers.",
        "residual_layers": None,
        "routing_layers": DEFAULT_ROUTING_LAYERS,
        "attention_layers": None,
    },
}

# Per-condition default request rates. Picked from Phase 2 wall medians ×
# safety margin to keep the system in the no-queue regime. User can override
# with --req-rate.
DEFAULT_REQ_RATES: dict[str, float] = {
    "baseline":    1.0,   # Phase 2 baseline walls are sub-second to low seconds
    "repe_bundle": 0.2,   # Heavier serialization payload, conservative
    "routing":     0.2,   # 75-layer routing payload, conservative
}


def preset_to_xargs(preset: dict) -> dict:
    """Mirror of CaptureRequest.to_xargs from captures.py — keep aligned."""
    x: dict = {}
    if preset["residual_layers"] is not None:
        x["output_residual_stream"] = preset["residual_layers"]
    if preset["routing_layers"] is not None:
        x["output_routing"] = preset["routing_layers"]
    if preset["attention_layers"] is not None:
        x["output_attention_stats"] = preset["attention_layers"]
    return x


# ===== per-request row =======================================================

@dataclass
class RequestRow:
    """Union schema with phase3_grad_driver.RequestRow. Gradient-only fields
    (fwd_seconds, bwd_seconds, loss, target_token_id, target_token) are
    None in this driver; vLLM-only fields (tokens_out, completion_text) are
    None in the gradient driver. phase3_aggregate.py unions both."""
    request_id: int
    pair_id: int
    prompt_class: str           # "toxic" | "benign"
    t_send: float               # wall-clock epoch seconds, for joining with nvidia-smi
    t_first_token: float        # = t_complete in non-streaming mode (see header)
    t_complete: float
    wall_seconds: float         # monotonic delta
    tokens_in: int              # usage.prompt_tokens
    tokens_out: int             # usage.completion_tokens
    payload_bytes: int          # _estimate_payload_bytes(response_dict)
    completion_text: Optional[str]
    # Gradient-only — always None here:
    fwd_seconds: Optional[float] = None
    bwd_seconds: Optional[float] = None
    server_total_seconds: Optional[float] = None
    loss: Optional[float] = None
    target_token_id: Optional[int] = None
    target_token: Optional[str] = None
    # Both:
    http_status: int = 0
    error: Optional[str] = None


# ===== health check ==========================================================

def wait_for_health(
    base_url: str,
    api_key: str,
    timeout: float = 1800.0,
    poll_interval: float = 10.0,
) -> dict:
    """Poll /health until HTTP 200. vLLM's default /health returns 200 only
    once the model is loaded and the server is accepting completions.

    Note: --base-url should be the OpenAI-compatible base (ends with /v1).
    Health probes one level up.
    """
    health_url = base_url.rstrip("/")
    if health_url.endswith("/v1"):
        health_url = health_url[:-3]
    health_url = f"{health_url.rstrip('/')}/health"

    deadline = time.monotonic() + timeout
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    while time.monotonic() < deadline:
        try:
            r = httpx.get(health_url, headers=headers, timeout=30.0)
            if r.status_code == 200:
                try:
                    return r.json()
                except Exception:
                    return {"status": "ok"}  # body may be empty for vLLM
            print(f"  [health] HTTP {r.status_code}; waiting...")
        except Exception as e:
            print(f"  [health] {type(e).__name__}: {e}")
        time.sleep(poll_interval)
    raise TimeoutError(f"/health did not return 200 within {timeout:.0f}s")


# ===== single request ========================================================

def send_one(
    client: OpenAI,
    model: str,
    prompt: str,
    xargs: dict,
    max_new_tokens: int,
    request_id: int,
    pair_id: int,
    prompt_class: str,
    enable_thinking: bool = False,             # NEW
    apply_steering_str: Optional[str] = None,  # NEW
) -> RequestRow:
    """One /v1/chat/completions call. extra_body matches
    captures.call_with_capture line-for-line so the response payload
    layout — including vllm_xargs-injected fields — is identical.

    Tier 3G / 1C extensions:
      enable_thinking      sets chat_template_kwargs.enable_thinking
                           (default False preserves prior behavior).
      apply_steering_str   pre-built JSON-stringified list of
                           SteeringVector dicts; injected into
                           vllm_xargs as `apply_steering_vectors`.
                           Caller is responsible for json.dumps(...)
                           per PHASE2_REFERENCE §5.2 / §11.1.
    """
    t_send = time.time()
    t_perf_start = time.perf_counter()
    http_status = 0
    error: Optional[str] = None
    raw: dict = {}

    # ----- Tier 3G/1C: build vllm_xargs with optional steering ----------
    # Shallow copy so we never mutate the caller's xargs (the
    # CONDITION_PRESETS entry is shared across the run).
    request_xargs = dict(xargs)
    if apply_steering_str is not None:
        request_xargs["apply_steering_vectors"] = apply_steering_str

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_new_tokens,
            extra_body={
                "vllm_xargs": request_xargs,
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
            },
        )
        http_status = 200
        raw = response.model_dump()
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        http_status = getattr(getattr(e, "response", None), "status_code", 0) or 0

    t_complete = time.time()
    wall = time.perf_counter() - t_perf_start

    if error is None:
        usage = raw.get("usage") or {}
        choices = raw.get("choices") or [{}]
        message = (choices[0] or {}).get("message") or {}
        completion_text = message.get("content") or ""
    else:
        usage, completion_text = {}, None

    return RequestRow(
        request_id=request_id,
        pair_id=pair_id,
        prompt_class=prompt_class,
        t_send=t_send,
        t_first_token=t_complete,
        t_complete=t_complete,
        wall_seconds=wall,
        tokens_in=int(usage.get("prompt_tokens") or 0),
        tokens_out=int(usage.get("completion_tokens") or 0),
        payload_bytes=_estimate_payload_bytes(raw),
        completion_text=completion_text,
        http_status=http_status,
        error=error,
    )

# ===== prompt loading + ordering ============================================
# (Identical to phase3_grad_driver — duplicated for now, see header note.)

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
    """Alternate toxic/benign by pair index, cap at n_requests."""
    seq: list[tuple[int, str, str]] = []
    for p in pairs:
        seq.append((int(p["pair_id"]), "toxic", p["toxic"]))
        seq.append((int(p["pair_id"]), "benign", p["benign"]))
    if n_requests < len(seq):
        seq = seq[:n_requests]
    return seq


# ===== driver loop ===========================================================

def run_cell(
    client: OpenAI,
    model: str,
    prompts: Sequence[tuple[int, str, str]],
    xargs: dict,
    max_new_tokens: int,
    req_rate: float,
    enable_thinking: bool = False,             # NEW
    apply_steering_str: Optional[str] = None,  # NEW
) -> list[RequestRow]:
    """Sequential, throttled by send-to-send interval."""
    min_interval = 1.0 / req_rate if req_rate > 0 else 0.0
    rows: list[RequestRow] = []
    last_send = 0.0
    for i, (pair_id, prompt_class, prompt) in enumerate(prompts):
        wait = min_interval - (time.monotonic() - last_send)
        if wait > 0:
            time.sleep(wait)
        last_send = time.monotonic()
        row = send_one(
            client, model, prompt, xargs, max_new_tokens,
            request_id=i, pair_id=pair_id, prompt_class=prompt_class,
            enable_thinking=enable_thinking,           # NEW
            apply_steering_str=apply_steering_str,     # NEW
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
                f"tok_in={row.tokens_in} tok_out={row.tokens_out} "
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
        "wall_seconds":  _percentiles([r.wall_seconds for r in rows]),
        "payload_bytes": _percentiles([float(r.payload_bytes) for r in rows]),
        "tokens_in":     _percentiles([float(r.tokens_in) for r in rows]),
        "tokens_out":    _percentiles([float(r.tokens_out) for r in rows]),
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
    enable_thinking: bool = False,                # NEW
    apply_steering_path: Optional[str] = None,    # NEW
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
        "endpoint": "/v1/chat/completions",
        "vllm_xargs": xargs,
        "request_options": {
            "stream": False,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},  # was hardcoded False
            "max_retries": 0,
            "apply_steering_payload": apply_steering_path,                 # NEW
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
    endpoint_url = os.environ.get("R2_ENDPOINT")
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
    p.add_argument("--condition", required=True,
                   choices=list(CONDITION_PRESETS.keys()),
                   help="vLLM cell to exercise: baseline (C1), repe_bundle (C2), routing (C4).")
    p.add_argument("--base-url", required=True,
                   help="vLLM OpenAI-compatible base URL, ending in /v1.")
    p.add_argument("--api-key",
                   default=os.environ.get("VLLM_API_KEY", "EMPTY"),
                   help="Bearer token. Default $VLLM_API_KEY or 'EMPTY' (debug deployments).")
    p.add_argument("--model", default="glm-5-1-fp8",
                   help="vLLM served-model name. Default matches Phase 2.")
    p.add_argument("--pairs-json", type=Path, required=True,
                   help="Path to Phase 2 pairs.json (reuse for CC-on/off comparability).")
    p.add_argument("--n-requests", type=int, default=100,
                   help="Total requests. Default 100 = 50 toxic + 50 benign interleaved.")
    p.add_argument("--req-rate", type=float, default=None,
                   help="Target requests/sec. Default: per-condition lookup "
                        f"({DEFAULT_REQ_RATES}).")
    p.add_argument("--max-new-tokens", type=int, default=32,
                   help="Max completion tokens per request. Matches Phase 2 default.")
    # ----- Tier 3G / Tier 1C extension flags ----------------------------
    # Defaults preserve existing behavior — every flag here is optional
    # and additive. PHASE3_REFERENCE §5.1 unchanged when none are set.

    p.add_argument(
        "--enable-thinking",
        action="store_true",
        default=False,
        help=(
            "Set chat_template_kwargs.enable_thinking=True. Default "
            "False matches the hardcoded value used by the 18-cell "
            "primary matrix. Enable for Tier 3G reasoning cells."
        ),
    )
    p.add_argument(
        "--apply-steering-json",
        type=Path,
        default=None,
        help=(
            "Path to a JSON file built by make_steering_payload.py. "
            "Adds `apply_steering_vectors` to vllm_xargs (JSON-"
            "stringified per PHASE2_REFERENCE §5.2). Required for "
            "Tier 1C C_H-on-steer."
        ),
    )
    p.add_argument(
        "--save-responses",
        action="store_true",
        default=False,
        help=(
            "Write per-request response text + metadata to "
            "responses.jsonl alongside requests.parquet. Required "
            "for Tier 1C refusal classifier; harmless otherwise."
        ),
    )
    p.add_argument("--timeout", type=float, default=120.0,
                   help="Per-request HTTP timeout (sec). Default 120.")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Directory to write requests.parquet/jsonl + summary.json.")
    p.add_argument("--cell-id", required=True,
                   help="Phase 3 cell identifier (e.g. C1-off, C2-on). Used in summary + S3 key.")
    p.add_argument("--cc-state", choices=["on", "off"], required=True,
                   help="Confidential Compute state of the target deployment. Metadata only.")
    p.add_argument("--image-digest", default=None,
                   help="ghcr image digest of the vLLM deployment (sha256:...) for reproducibility.")
    p.add_argument("--health-timeout", type=float, default=1800.0,
                   help="Max seconds to wait for /health → 200. Default 1800.")
    p.add_argument("--health-poll-interval", type=float, default=10.0,
                   help="Seconds between /health polls.")
    p.add_argument("--skip-health", action="store_true",
                   help="Skip /health poll. Use only when target is known to be warm.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # ----- Tier 1C: pre-load steering payload if requested --------------
    apply_steering_str = None  # JSON-stringified list, or None
    if args.apply_steering_json is not None:
        payload = json.loads(args.apply_steering_json.read_text())
        steering_vectors = payload["steering_vectors"]
        # vllm_xargs is typed Dict[str, Union[str, int, float, List[scalars]]]
        # → list-of-dicts is rejected at FastAPI boundary; must json.dumps
        # (PHASE2_REFERENCE §5.2, §11.1). vllm-lens server-side parser
        # checks isinstance(str) and json.loads back.
        apply_steering_str = json.dumps(steering_vectors)
        print(
            f"[steering] loaded {len(steering_vectors)} vector(s) from "
            f"{args.apply_steering_json}: layer="
            f"{steering_vectors[0]['layer_indices']}, scale="
            f"{steering_vectors[0]['scale']}, "
            f"norm_match={steering_vectors[0]['norm_match']}",
            flush=True,
        )

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
        print("[error] no prompts after interleave (empty pairs.json?)", file=sys.stderr)
        return 2

    print(f"[cell] {args.cell_id} (cc={args.cc_state}) condition={args.condition}")
    print(f"[cell] target={args.base_url} model={args.model}")
    print(f"[cell] xargs={xargs}")
    print(f"[cell] {len(prompts)} requests @ {req_rate} req/s "
          f"(min wall ~{len(prompts) / req_rate / 60:.1f} min)")

    if not args.skip_health:
        print(f"[health] polling (max {args.health_timeout:.0f}s, "
              f"every {args.health_poll_interval:.0f}s)...")
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

    # max_retries=0: don't double-charge wall on transients. Tradeoff explained
    # in the file header.
    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        max_retries=0,
        timeout=args.timeout,
    )

    t0 = time.monotonic()
    rows = run_cell(
        client, args.model, prompts,
        xargs=xargs,
        max_new_tokens=args.max_new_tokens,
        req_rate=req_rate,
        enable_thinking=args.enable_thinking,        # NEW
        apply_steering_str=apply_steering_str,       # NEW (already loaded above)
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
        enable_thinking=args.enable_thinking,                                 # NEW
        apply_steering_path=str(args.apply_steering_json) if args.apply_steering_json else None,  # NEW
    )

    req_path = write_outputs(rows, summary, args.out_dir)
    print(f"[out] {req_path}")
    print(f"[out] {args.out_dir / 'summary.json'}")
    wall_p50 = summary["overall"]["wall_seconds"].get("p50", float("nan"))
    payload_p50 = summary["overall"]["payload_bytes"].get("p50", float("nan"))
    print(f"[summary] n_success={summary['n_success']}/{summary['n_total']} "
          f"wall_p50={wall_p50:.2f}s payload_p50={payload_p50:.0f}B")

    maybe_upload(args.out_dir, args.cell_id)

    if summary["n_success"] == 0:
        print("[fail] zero successful requests — check --base-url / --api-key / vLLM status.",
              file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
