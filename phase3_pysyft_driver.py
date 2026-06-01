#!/usr/bin/env python3
"""
phase3_pysyft_driver.py - laptop-side driver for the PySyft build.

Drives the four Mode B TwinAPIEndpoints registered by pysyft_datasite_server.py
inside the Phase 1 image. Per-request row schema is the union of
phase3_egress_driver_v2.EgressRowV2 fields plus five pysyft_* timing columns
that decompose the PySyft layer's overhead into workflow / approval / encoder
/ ledger / response_assembly stages (matching the brief's Table 9 split).

ATTRIBUTION (corrected): the server-side PySyft stages are
workflow/approval/encoder/ledger/response_assembly. The fifth was previously
named `bundle_return` and implied it captured response serialisation; it does
not — PySyft serialises AFTER the endpoint returns, unobservable server-side.
The real serialise+transport cost is `transport_serialize_seconds`, measured
laptop-side as (wall - pysyft_total). Table 9 shows it as a separate line, not
folded into a PySyft governance stage.

Auth transport:
  CC-on (production):   subprocess `tinfoil container connect --debug-mode
                        --port N`; client points at http://127.0.0.1:N.
                        Verified proxy handles attestation + bearer.
  Non-CC debug:         --no-proxy --bearer ... mode; client hits the
                        public URL directly with Authorization: Bearer.
                        Verified proxy refuses dummy attestation, so this
                        is the only viable path for --disable-cc-mode
                        deploys. The shim must be authenticated=false OR
                        the bearer must match the deploy's runtime key.

Smoke run (10 requests, baseline endpoint only, against a debug deploy):

  python phase3_pysyft_driver.py \\
      --endpoints capture_residual_stream \\
      --datasite-url https://pysyft.debug.pour-demain.containers.tinfoil.dev \\
      --bearer "$VLLM_API_KEY" \\
      --no-proxy \\
      --pairs-json runs/phase2_validation/repe_bundle/pairs.json \\
      --n-requests 10 --req-rate 0.15 \\
      --out-dir runs/pysyft_smoke \\
      --cell-id PS1-off --cc-state off

Full cell (CC-on, all 4 endpoints, 50 paired prompts per endpoint = 200 total):

  python phase3_pysyft_driver.py \\
      --endpoints capture_residual_stream,capture_routing,capture_attention_stats,apply_steering \\
      --datasite-url https://pysyft.containers.tinfoil.dev \\
      --pairs-json runs/phase2_validation/repe_bundle/pairs.json \\
      --steering-direction runs/phase2_d6_steering/direction_L62.npy \\
      --n-requests 200 --req-rate 0.15 \\
      --out-dir runs/pysyft/PS1-on \\
      --cell-id PS1-on --cc-state on \\
      --image-digest sha256:<digest>

Exit codes:
  0  success
  2  user error
  3  tinfoil container connect proxy failed to come up
  4  zero successful requests
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence
from urllib.parse import urlparse

import httpx
import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import boto3
except ImportError:
    boto3 = None

try:
    import syft as sy
except ImportError as e:
    print(f"[fatal] syft import failed: {e}\n  pip install 'syft>=0.9.5,<0.9.6'",
          file=sys.stderr)
    sys.exit(2)

# Reuse the egress driver's pair-handling helpers - same pairs.json format.
from phase3_vllm_driver import interleave, load_pairs


def flatten_pairs(pairs, sides=("toxic", "benign"), limit=None):
    """Flat prompt list from pairs.json, no pairing semantics.

    Returns [(pair_id, prompt_class, prompt), ...] walking every requested side
    of every pair in order. Used for the governed-egress mechanism run, where
    there is no paired CC-on/off delta to preserve — we just want N realistic
    prompts through the capture endpoint. `limit` caps the total count.

    Default sends BOTH sides (toxic + benign) so the workload spans the corpus;
    pass sides=("toxic",) or ("benign",) to restrict.
    """
    out = []
    for p in pairs:
        pid = p.get("pair_id", len(out))
        for side in sides:
            text = p.get(side)
            if text:
                out.append((pid, side, text))
    if limit is not None:
        out = out[:limit]
    return out


SCHEMA_VERSION = "phase3-pysyft-driver-v2"

ENDPOINTS = (
    "capture_residual_stream",
    "capture_routing",
    "capture_attention_stats",
    "apply_steering",
)

# Maps each PySyft endpoint to the egress-driver `condition` whose wall is the
# correct paired baseline for the PySyft-layer delta. Used by
# analyze_pysyft_overhead.py to compare like-for-like instead of pooling all
# four endpoints into one median. capture_residual_stream and
# capture_attention_stats both correspond to the repe_bundle egress condition
# (residual + attention stats are the two halves of that bundle); routing →
# routing; apply_steering → steer.
ENDPOINT_TO_EGRESS_CONDITION = {
    "capture_residual_stream":  "repe_bundle",
    "capture_attention_stats":  "repe_bundle",
    "capture_routing":          "routing",
    "apply_steering":           "steer",
}


# ===== per-request row =======================================================

@dataclass
class PysyftRow:
    """Union of phase3_egress_driver_v2.EgressRowV2 + PySyft 5-stage timings.

    Wall and server timings come from the encoder's response embedded in the
    PySyft endpoint return value. PySyft timings come from `pysyft_timings`
    in the same response. wall_seconds is laptop perf_counter around the
    entire client.api.services... call.

    cell_id and cc_state are carried on every row so multiple cells can be
    concatenated into one parquet without losing the grouping keys (the
    analysis script keys paired deltas on cc_state)."""

    request_id: int
    pair_id: int
    prompt_class: str           # "toxic" | "benign" | "auditor"
    endpoint: str               # short name without prepilot. prefix
    cell_id: str
    cc_state: str               # "on" | "off"
    t_send: float
    t_complete: float
    wall_seconds: float                          # laptop perf_counter, end-to-end

    # PySyft 5-stage decomposition (from the endpoint response, server-measured).
    pysyft_workflow_seconds: float
    pysyft_approval_seconds: float
    pysyft_encoder_seconds: float
    pysyft_ledger_seconds: float
    pysyft_response_assembly_seconds: float      # was bundle_return; see header
    pysyft_total_seconds: float

    # Inherited from EgressRowV2 - encoder's own stage timings.
    server_deserialize_seconds: float
    server_encoder_total_seconds: float
    server_aggregate_seconds: float
    server_plot_seconds: float
    server_bundle_seconds: float
    server_ledger_seconds: float

    # Derived: the serialise+transport slice NOT accounted for server-side.
    # = wall - pysyft_total. Captures PySyft's own response serialisation
    # (which happens after the endpoint returns) plus laptop<->worker network.
    # This is the line that was previously mis-attributed to a PySyft stage.
    transport_serialize_seconds: float

    tokens_in: int
    tokens_out: int
    raw_payload_bytes: int
    bundle_bytes: int
    aggregate_bytes: int
    n_plots: int

    completion_text: Optional[str]
    bundle_sha256: Optional[str]

    engagement_id: Optional[str] = None
    session_id: Optional[str] = None
    auditor_id: Optional[str] = None

    error: Optional[str] = None


# ===== verified proxy subprocess =============================================

def start_verified_proxy(
    container_name: str,
    local_port: int,
    debug_mode: bool,
) -> subprocess.Popen:
    """Launch `tinfoil container connect` and wait for it to bind.

    Returns the Popen handle so the caller can shut it down on exit. Raises
    RuntimeError if the proxy doesn't bind within 60s.
    """
    cmd = ["tinfoil", "container", "connect", container_name,
           "--port", str(local_port)]
    if debug_mode:
        cmd.append("--debug-mode")
    print(f"[proxy] starting: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            output = proc.stdout.read().decode("utf-8", errors="replace") if proc.stdout else ""
            raise RuntimeError(
                f"tinfoil container connect exited early (rc={proc.returncode}):\n{output}"
            )
        # Try to connect to the local port.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            try:
                s.connect(("127.0.0.1", local_port))
                print(f"[proxy] verified proxy bound to 127.0.0.1:{local_port}",
                      flush=True)
                return proc
            except (ConnectionRefusedError, socket.timeout):
                pass
        time.sleep(1.0)

    proc.terminate()
    raise RuntimeError(f"verified proxy did not bind 127.0.0.1:{local_port} within 60s")


# ===== Datasite client setup =================================================

def login_via_proxy(local_port: int, email: str, password: str):
    """sy.login against the verified proxy on localhost."""
    url = f"http://127.0.0.1:{local_port}"
    return sy.login(url=url, port=local_port, email=email, password=password)


def login_via_bearer(datasite_url: str, bearer: str, email: str, password: str):
    """sy.login direct to the deploy URL with bearer in the header.

    Only viable when the shim is `authenticated: false` (debug deploys)
    OR a patched syft client carries the bearer on every call. Default
    syft 0.9.5 does NOT carry it - documented limitation in SPIKE_PLAN.md.
    """
    # Use the same parser the spike used. PySyft will fail on /api/v2/api_call
    # if the shim still requires auth; that's the known transport issue.
    parsed = urlparse(datasite_url)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    # Quick liveness probe with bearer; if this succeeds login *may* still 401.
    r = httpx.get(f"{datasite_url.rstrip('/')}/api/v2/metadata",
                  headers={"Authorization": f"Bearer {bearer}"} if bearer else {},
                  timeout=30.0)
    r.raise_for_status()
    return sy.login(url=datasite_url, port=port, email=email, password=password)


# ===== endpoint dispatch =====================================================

def call_endpoint(
    client,
    endpoint: str,
    prompt: str,
    *,
    steering_direction: Optional[np.ndarray] = None,
    max_new_tokens: int = 32,
    routing_layers: Optional[list] = None,
) -> dict:
    """Dispatch one call. Returns the raw endpoint result dict."""
    svc = client.api.services.prepilot
    if endpoint == "capture_residual_stream":
        return svc.capture_residual_stream(
            prompt=prompt, max_new_tokens=max_new_tokens,
        )
    if endpoint == "capture_routing":
        # routing across all 75 MoE layers exceeds the debug shim's ~60s
        # result-return ceiling; passing a layer subset keeps the call under it.
        kwargs = {"prompt": prompt, "max_new_tokens": max_new_tokens}
        if routing_layers:
            kwargs["layers"] = routing_layers
        return svc.capture_routing(**kwargs)
    if endpoint == "capture_attention_stats":
        return svc.capture_attention_stats(
            prompt=prompt, max_new_tokens=max_new_tokens,
        )
    if endpoint == "apply_steering":
        if steering_direction is None:
            raise ValueError("apply_steering requires --steering-direction")
        # PySyft serialises Python lists transparently. The endpoint validates
        # length == 6144 server-side.
        return svc.apply_steering(
            prompt=prompt,
            direction=steering_direction.astype(np.float32).tolist(),
            layer=62, alpha=1.0, sign=-1, norm_match=True,
            max_new_tokens=max_new_tokens,
        )
    raise ValueError(f"unknown endpoint: {endpoint}")


def _unwrap_pysyft_result(raw) -> dict:
    """sy.api.services... returns the endpoint's dict wrapped in a PySyft
    ActionObject (e.g. AnyActionObject) or a SyftSuccess/SyftError. Pull the
    dict out.

    Order: plain dict → ActionObject.get() (resolves to the underlying value,
    proven path) → .syft_action_data → legacy attribute paths."""
    if isinstance(raw, dict):
        return raw
    # ActionObject: .get() resolves the wrapped payload to a Python value.
    getter = getattr(raw, "get", None)
    if callable(getter):
        try:
            v = getter()
            if isinstance(v, dict):
                return v
        except Exception:
            pass
    # ActionObject stored payload, no resolution call.
    v = getattr(raw, "syft_action_data", None)
    if isinstance(v, dict):
        return v
    for attr in ("message", "value", "result", "_message"):
        v = getattr(raw, attr, None)
        if isinstance(v, dict):
            return v
    # Fallback: render to str so the driver records something usable.
    return {"error": "unwrap_failed", "detail": f"{type(raw).__name__}: {str(raw)[:200]}"}


def send_one(
    client,
    request_id: int,
    pair_id: int,
    prompt_class: str,
    prompt: str,
    endpoint: str,
    cell_id: str,
    cc_state: str,
    *,
    steering_direction: Optional[np.ndarray] = None,
    max_new_tokens: int = 32,
    routing_layers: Optional[list] = None,
) -> PysyftRow:
    t_send = time.time()
    t_perf_start = time.perf_counter()
    error: Optional[str] = None
    raw: dict = {}
    try:
        raw = _unwrap_pysyft_result(
            call_endpoint(client, endpoint, prompt,
                          steering_direction=steering_direction,
                          max_new_tokens=max_new_tokens,
                          routing_layers=routing_layers)
        )
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    t_complete = time.time()
    wall = time.perf_counter() - t_perf_start

    if error is not None or raw.get("error"):
        # Surface the typed error from the endpoint (cap-exceeded, encoder_failed, etc.)
        if not error:
            error = f"endpoint_error:{raw.get('error')}: {raw.get('detail', '')}"
        return PysyftRow(
            request_id=request_id, pair_id=pair_id, prompt_class=prompt_class,
            endpoint=endpoint, cell_id=cell_id, cc_state=cc_state,
            t_send=t_send, t_complete=t_complete,
            wall_seconds=wall,
            pysyft_workflow_seconds=0.0, pysyft_approval_seconds=0.0,
            pysyft_encoder_seconds=0.0, pysyft_ledger_seconds=0.0,
            pysyft_response_assembly_seconds=0.0, pysyft_total_seconds=0.0,
            server_deserialize_seconds=0.0, server_encoder_total_seconds=0.0,
            server_aggregate_seconds=0.0, server_plot_seconds=0.0,
            server_bundle_seconds=0.0, server_ledger_seconds=0.0,
            transport_serialize_seconds=wall,
            tokens_in=0, tokens_out=0,
            raw_payload_bytes=0, bundle_bytes=0, aggregate_bytes=0, n_plots=0,
            completion_text=None, bundle_sha256=None,
            error=error,
        )

    completion = raw.get("completion") or {}
    eg = raw.get("egress") or {}
    pt = raw.get("pysyft_timings") or {}
    usage = completion.get("usage") or {}
    choices = completion.get("choices") or [{}]
    message = (choices[0] or {}).get("message") or {}

    server_deser = float(eg.get("deserialize_seconds") or 0.0)
    server_encoder = float(eg.get("encoder_total_seconds") or 0.0)
    pysyft_total = float(pt.get("total_seconds") or 0.0)
    # transport_serialize = laptop wall - everything measured server-side.
    # pysyft_total is the server-measured sum of the five PySyft stages
    # (which includes the loopback encoder call). What remains is PySyft's own
    # response serialisation (post-return, unobservable server-side) plus the
    # laptop<->worker network round-trip. This is the slice that was formerly
    # mis-attributed to the `bundle_return` stage.
    transport_serialize = max(0.0, wall - pysyft_total)

    return PysyftRow(
        request_id=request_id, pair_id=pair_id, prompt_class=prompt_class,
        endpoint=endpoint, cell_id=cell_id, cc_state=cc_state,
        t_send=t_send, t_complete=t_complete,
        wall_seconds=wall,
        pysyft_workflow_seconds=float(pt.get("workflow_seconds") or 0.0),
        pysyft_approval_seconds=float(pt.get("approval_seconds") or 0.0),
        pysyft_encoder_seconds=float(pt.get("encoder_seconds") or 0.0),
        pysyft_ledger_seconds=float(pt.get("ledger_seconds") or 0.0),
        # Accept either the new or the legacy key so the driver works against a
        # container still running the pre-patch _common.py (e.g. the live
        # v0.0.5 deploy before the convergence rebuild).
        pysyft_response_assembly_seconds=float(
            pt.get("response_assembly_seconds")
            if pt.get("response_assembly_seconds") is not None
            else (pt.get("bundle_return_seconds") or 0.0)
        ),
        pysyft_total_seconds=pysyft_total,
        server_deserialize_seconds=server_deser,
        server_encoder_total_seconds=server_encoder,
        server_aggregate_seconds=float(eg.get("aggregate_seconds") or 0.0),
        server_plot_seconds=float(eg.get("plot_seconds") or 0.0),
        server_bundle_seconds=float(eg.get("bundle_seconds") or 0.0),
        server_ledger_seconds=float(eg.get("ledger_seconds") or 0.0),
        transport_serialize_seconds=transport_serialize,
        tokens_in=int(usage.get("prompt_tokens") or 0),
        tokens_out=int(usage.get("completion_tokens") or 0),
        raw_payload_bytes=int(eg.get("raw_payload_bytes") or 0),
        bundle_bytes=int(eg.get("bundle_bytes") or 0),
        aggregate_bytes=int(eg.get("aggregate_bytes") or 0),
        n_plots=int(eg.get("n_plots") or 0),
        completion_text=message.get("content"),
        bundle_sha256=eg.get("bundle_sha256"),
        engagement_id=raw.get("engagement_id"),
        session_id=raw.get("session_id"),
        auditor_id=raw.get("auditor_id"),
        error=None,
    )


# ===== driver loop ===========================================================

def run_cell(
    client,
    endpoints: Sequence[str],
    prompts: Sequence[tuple[int, str, str]],
    req_rate: float,
    cell_id: str,
    cc_state: str,
    *,
    steering_direction: Optional[np.ndarray] = None,
    max_new_tokens: int = 32,
    routing_layers: Optional[list] = None,
) -> list[PysyftRow]:
    """Round-robin through endpoints x prompts. Throttled by send-to-send
    interval. Total calls = len(endpoints) * len(prompts) when prompts is
    the full list; the caller pre-trims to n_requests if needed."""
    min_interval = 1.0 / req_rate if req_rate > 0 else 0.0
    rows: list[PysyftRow] = []
    last_send = 0.0
    request_id = 0
    for endpoint in endpoints:
        for pair_id, prompt_class, prompt in prompts:
            wait = min_interval - (time.monotonic() - last_send)
            if wait > 0:
                time.sleep(wait)
            last_send = time.monotonic()
            row = send_one(
                client, request_id, pair_id, prompt_class, prompt, endpoint,
                cell_id, cc_state,
                steering_direction=steering_direction,
                max_new_tokens=max_new_tokens,
                routing_layers=routing_layers,
            )
            rows.append(row)
            tag = f"{endpoint} pair={pair_id} {prompt_class}"
            if row.error:
                print(f"  [{request_id+1}] {tag} ERROR {row.error}", flush=True)
            else:
                print(
                    f"  [{request_id+1}] {tag} wall={row.wall_seconds:.2f}s "
                    f"pys={row.pysyft_total_seconds:.2f}s "
                    f"(wf={row.pysyft_workflow_seconds*1000:.0f}ms "
                    f"appr={row.pysyft_approval_seconds*1000:.0f}ms "
                    f"enc={row.pysyft_encoder_seconds*1000:.0f}ms "
                    f"ldg={row.pysyft_ledger_seconds*1000:.0f}ms "
                    f"asm={row.pysyft_response_assembly_seconds*1000:.1f}ms) "
                    f"tx+ser={row.transport_serialize_seconds:.2f}s",
                    flush=True,
                )
            request_id += 1
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


def summarize(rows: list[PysyftRow], **meta) -> dict:
    ok = [r for r in rows if r.error is None]
    err = [r for r in rows if r.error is not None]
    by_endpoint = {}
    for ep in sorted({r.endpoint for r in rows}):
        sub_ok = [r for r in ok if r.endpoint == ep]
        by_endpoint[ep] = {
            "n_success": len(sub_ok),
            "n_error": sum(1 for r in err if r.endpoint == ep),
            "wall_seconds":             _percentiles([r.wall_seconds for r in sub_ok]),
            "pysyft_total_seconds":     _percentiles([r.pysyft_total_seconds for r in sub_ok]),
            "pysyft_workflow_seconds":  _percentiles([r.pysyft_workflow_seconds for r in sub_ok]),
            "pysyft_approval_seconds":  _percentiles([r.pysyft_approval_seconds for r in sub_ok]),
            "pysyft_encoder_seconds":   _percentiles([r.pysyft_encoder_seconds for r in sub_ok]),
            "pysyft_ledger_seconds":    _percentiles([r.pysyft_ledger_seconds for r in sub_ok]),
            "pysyft_response_assembly_seconds": _percentiles([r.pysyft_response_assembly_seconds for r in sub_ok]),
            "transport_serialize_seconds": _percentiles([r.transport_serialize_seconds for r in sub_ok]),
            "bundle_bytes":             _percentiles([float(r.bundle_bytes) for r in sub_ok]),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        **meta,
        "n_total": len(rows),
        "n_success": len(ok),
        "n_error": len(err),
        "success_rate": len(ok) / max(1, len(rows)),
        "errors_sample": [
            {"request_id": r.request_id, "endpoint": r.endpoint, "error": r.error}
            for r in err[:5]
        ],
        "by_endpoint": by_endpoint,
    }


def write_outputs(rows: list[PysyftRow], summary: dict, out_dir: Path) -> Path:
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
    endpoint_url = os.environ.get("R2_ENDPOINT") or os.environ.get("R2_ENDPOINT_URL")
    if not bucket or boto3 is None:
        return
    kwargs = {"endpoint_url": endpoint_url} if endpoint_url else {}
    s3 = boto3.client("s3", **kwargs)
    for fname in ("requests.parquet", "requests.jsonl", "summary.json"):
        p = out_dir / fname
        if not p.exists():
            continue
        key = f"phase3_pysyft/{cell_id}/{fname}"
        s3.upload_file(str(p), bucket, key)
        print(f"[upload] {'r2' if endpoint_url else 's3'}://{bucket}/{key}")


# ===== CLI ===================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--endpoints", required=True,
                   help=f"comma-separated subset of {','.join(ENDPOINTS)}")
    p.add_argument("--datasite-url", required=True,
                   help="public URL of the deployed Datasite (used by --no-proxy "
                        "OR to resolve the container name for the verified proxy)")
    p.add_argument("--container-name", default=None,
                   help="Tinfoil container name for `tinfoil container connect`. "
                        "Defaults to the hostname's leftmost label.")
    p.add_argument("--proxy-port", type=int, default=8080,
                   help="local port for the verified proxy")
    p.add_argument("--no-proxy", action="store_true",
                   help="bypass tinfoil container connect; hit datasite-url directly. "
                        "Only viable for non-CC deploys with shim authenticated=false.")
    p.add_argument("--debug-mode-proxy", action="store_true", default=True,
                   help="pass --debug-mode to `tinfoil container connect`")
    p.add_argument("--bearer", default=os.environ.get("VLLM_API_KEY"),
                   help="Shim bearer for --no-proxy mode")

    p.add_argument("--register-endpoints", action="store_true",
                   help="Register the 4 TwinAPIEndpoints from the client as admin "
                        "before running. Needed on every fresh deploy "
                        "(reset=True wipes them; server-side startup registration "
                        "does not persist). Imports build_endpoint() from the "
                        "cc-deep-eval repo on PYTHONPATH.")
    p.add_argument("--admin-email", default="info@openmined.org")
    p.add_argument("--admin-password", default="changethis")

    p.add_argument("--auditor-email", default=None,
                   help="If set, dispatch runs as this auditor identity "
                        "(DATA_SCIENTIST role) instead of admin. Admin creds "
                        "are still used for --register-endpoints and "
                        "--register-v2-endpoint. Requires the identity-gated "
                        "endpoint (capture_residual_stream_v2) to be registered "
                        "with this email in authorized_auditors.")
    p.add_argument("--auditor-password", default=None,
                   help="Password for --auditor-email.")
    p.add_argument("--register-v2-endpoint", action="store_true",
                   help="Delete the old TwinAPIEndpoint for "
                        "capture_residual_stream and register the identity-gated "
                        "v2 endpoint. Requires capture_residual_stream_v2.py "
                        "on PYTHONPATH or in cwd.")

    p.add_argument("--pairs-json", type=Path, required=True)
    p.add_argument("--n-requests", type=int, default=50,
                   help="Total paired prompts (toxic+benign interleaved) sent to "
                        "EACH endpoint. Total calls = n-requests * len(endpoints).")
    p.add_argument("--flatten", action="store_true",
                   help="Bypass paired interleave; send a FLAT list of prompts "
                        "(both sides by default) from pairs.json. For the "
                        "governed-egress mechanism run (no paired delta). "
                        "Capped by --n-requests.")
    p.add_argument("--flatten-sides", default="toxic,benign",
                   help="Comma list of pair sides to include when --flatten "
                        "(e.g. 'toxic' or 'benign' or 'toxic,benign').")
    p.add_argument("--req-rate", type=float, default=0.15,
                   help="Target requests/sec across all endpoints, sequential.")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--routing-layers", default=None,
                   help="Comma-separated MoE layer indices (3..77) for "
                        "capture_routing, e.g. '12,23,39,51,62,70'. Default (None) "
                        "captures all 75 MoE layers, which can exceed the debug "
                        "shim's ~60s result-return ceiling; pass a subset to stay "
                        "under it. Ignored for non-routing endpoints.")
    p.add_argument("--steering-direction", type=Path, default=None,
                   help="Path to direction_L62.npy. Required if apply_steering is "
                        "in --endpoints.")

    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--cell-id", required=True)
    p.add_argument("--cc-state", choices=["on", "off"], required=True)
    p.add_argument("--image-digest", default=None)
    return p.parse_args()


def _register_endpoints(client) -> None:
    """Register the 4 TwinAPIEndpoints from the client as admin. Idempotent-ish:
    if an endpoint already exists, custom_api.add returns an error which we log
    and continue (re-running after a fresh deploy is the normal case).

    Requires the cc-deep-eval repo on PYTHONPATH so the endpoint modules import.
    """
    try:
        from pysyft_endpoints.endpoints import (
            apply_steering, capture_attention_stats,
            capture_residual_stream, capture_routing,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[register][error] cannot import endpoint modules: {e}\n"
              f"  ensure the cc-deep-eval repo is on PYTHONPATH "
              f"(PYTHONPATH=/path/to/cc-deep-eval)", file=sys.stderr)
        raise
    mods = [
        ("capture_residual_stream", capture_residual_stream),
        ("capture_routing",         capture_routing),
        ("capture_attention_stats", capture_attention_stats),
        ("apply_steering",          apply_steering),
    ]
    for label, mod in mods:
        try:
            res = client.custom_api.add(endpoint=mod.build_endpoint())
            print(f"[register] {label}: {type(res).__name__}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[register] {label}: EXCEPTION {type(e).__name__}: {e}",
                  flush=True)
    try:
        n = len(client.custom_api.api_endpoints())
        print(f"[register] endpoints now visible: {n}", flush=True)
    except Exception:
        pass


def _register_v2_endpoint(client) -> None:
    """Delete old TwinAPIEndpoint for capture_residual_stream and register
    the identity-gated v2 endpoint. Uses positional delete arg to avoid
    PySyft's path kwarg collision."""
    try:
        from capture_residual_stream_v2 import build_endpoint
    except ImportError:
        try:
            # Try cc-deep-eval path
            from pysyft_endpoints.endpoints.capture_residual_stream_v2 import build_endpoint
        except ImportError as e:
            print(f"[register-v2][error] cannot import capture_residual_stream_v2: {e}\n"
                  f"  place capture_residual_stream_v2.py in cwd or on PYTHONPATH",
                  file=sys.stderr)
            raise

    # Delete old (positional arg — PySyft kwarg collision on 'path')
    try:
        client.custom_api.delete("prepilot.capture_residual_stream")
        print("[register-v2] deleted old TwinAPIEndpoint", flush=True)
    except Exception as e:
        print(f"[register-v2] delete old: {type(e).__name__}: {e} (continuing)",
              flush=True)

    # Register new
    try:
        ep = build_endpoint()
        res = client.custom_api.add(endpoint=ep)
        print(f"[register-v2] registered identity-gated endpoint: "
              f"{type(res).__name__}", flush=True)
    except Exception as e:
        print(f"[register-v2] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        raise

    try:
        eps = client.custom_api.api_endpoints()
        print(f"[register-v2] endpoints now visible: {len(eps)}", flush=True)
    except Exception:
        pass


def main() -> int:
    args = parse_args()
    endpoints = [e.strip() for e in args.endpoints.split(",") if e.strip()]
    unknown = set(endpoints) - set(ENDPOINTS)
    if unknown:
        print(f"[error] unknown endpoints: {unknown}; valid={ENDPOINTS}",
              file=sys.stderr)
        return 2

    # Parse --routing-layers "12,23,..." -> [12,23,...]; validate 3..77.
    if isinstance(args.routing_layers, str):
        try:
            args.routing_layers = [int(x) for x in args.routing_layers.split(",") if x.strip()]
        except ValueError:
            print(f"[error] --routing-layers must be comma-separated ints",
                  file=sys.stderr)
            return 2
        bad = [L for L in args.routing_layers if not (3 <= L <= 77)]
        if bad:
            print(f"[error] --routing-layers out of MoE range [3,77]: {bad}",
                  file=sys.stderr)
            return 2

    steering_direction = None
    if "apply_steering" in endpoints:
        if args.steering_direction is None or not args.steering_direction.exists():
            print("[error] --steering-direction is required and must exist when "
                  "apply_steering is included", file=sys.stderr)
            return 2
        steering_direction = np.load(args.steering_direction).astype(np.float32)
        if steering_direction.shape != (6144,):
            print(f"[error] steering direction shape {steering_direction.shape} "
                  f"!= (6144,); GLM-5.1 hidden_size mismatch", file=sys.stderr)
            return 2

    try:
        pairs = load_pairs(args.pairs_json)
    except Exception as e:
        print(f"[error] failed to load --pairs-json: {e}", file=sys.stderr)
        return 2
    if args.flatten:
        sides = tuple(s.strip() for s in args.flatten_sides.split(",") if s.strip())
        prompts = flatten_pairs(pairs, sides=sides, limit=args.n_requests)
        print(f"[cell] FLATTEN sides={sides}: {len(prompts)} prompts "
              f"(cap n_requests={args.n_requests})")
    else:
        prompts = interleave(pairs, args.n_requests)

    print(f"[cell] {args.cell_id} cc={args.cc_state} endpoints={endpoints}")
    print(f"[cell] {len(prompts)} prompts x {len(endpoints)} endpoints = "
          f"{len(prompts) * len(endpoints)} total calls @ {args.req_rate} req/s")

    proxy_proc: Optional[subprocess.Popen] = None
    try:
        if args.no_proxy:
            print(f"[cell] --no-proxy: hitting {args.datasite_url} directly with bearer")
            client = login_via_bearer(args.datasite_url, args.bearer or "",
                                      args.admin_email, args.admin_password)
        else:
            container_name = args.container_name or urlparse(args.datasite_url).hostname.split(".")[0]
            try:
                proxy_proc = start_verified_proxy(
                    container_name, args.proxy_port, args.debug_mode_proxy,
                )
            except RuntimeError as e:
                print(f"[error] {e}", file=sys.stderr)
                return 3
            client = login_via_proxy(args.proxy_port, args.admin_email, args.admin_password)

        client.refresh()    # populate api.services tree

        if args.register_endpoints:
            _register_endpoints(client)
            client.refresh()

        if args.register_v2_endpoint:
            _register_v2_endpoint(client)
            client.refresh()

        # If --auditor-email is set, login as auditor for dispatch.
        # Admin client was used for registration above; auditor client
        # is used for the actual measurement sweep.
        dispatch_client = client  # default: admin
        if args.auditor_email:
            if not args.auditor_password:
                print("[error] --auditor-password required with --auditor-email",
                      file=sys.stderr)
                return 2
            print(f"[cell] logging in as auditor: {args.auditor_email}")
            if args.no_proxy:
                dispatch_client = login_via_bearer(
                    args.datasite_url, args.bearer or "",
                    args.auditor_email, args.auditor_password)
            else:
                dispatch_client = login_via_proxy(
                    args.proxy_port, args.auditor_email, args.auditor_password)
            dispatch_client.refresh()
            print(f"[cell] dispatch will run as: {args.auditor_email}")

        t0 = time.monotonic()
        rows = run_cell(
            dispatch_client, endpoints, prompts, args.req_rate,
            args.cell_id, args.cc_state,
            steering_direction=steering_direction,
            max_new_tokens=args.max_new_tokens,
            routing_layers=args.routing_layers,
        )
        print(f"[run] complete in {(time.monotonic()-t0)/60:.1f} min")

        summary = summarize(
            rows,
            cell_id=args.cell_id, cc_state=args.cc_state,
            datasite_url=args.datasite_url, image_digest=args.image_digest,
            dispatch_identity=args.auditor_email or args.admin_email,
            transport=("verified_proxy" if not args.no_proxy else "direct_bearer"),
            endpoints=endpoints, req_rate=args.req_rate,
            n_requests_per_endpoint=args.n_requests,
        )
        req_path = write_outputs(rows, summary, args.out_dir)
        print(f"[out] {req_path}")
        print(f"[out] {args.out_dir / 'summary.json'}")
        if summary["n_success"]:
            for ep, s in summary["by_endpoint"].items():
                wall_p50 = s["wall_seconds"].get("p50", float("nan"))
                pys_p50  = s["pysyft_total_seconds"].get("p50", float("nan"))
                print(f"[summary] {ep}: n={s['n_success']} "
                      f"wall_p50={wall_p50:.2f}s pysyft_p50={pys_p50:.2f}s")
        maybe_upload(args.out_dir, args.cell_id)
        return 0 if summary["n_success"] else 4
    finally:
        if proxy_proc is not None:
            print("[proxy] terminating verified proxy")
            proxy_proc.terminate()
            try:
                proxy_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proxy_proc.kill()


if __name__ == "__main__":
    sys.exit(main())