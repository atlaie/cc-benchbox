"""
probe_common.py — shared plumbing for the wedge-triangulation probes (T1/T2/T3).

These probes reuse the LIVE driver path (phase3_pysyft_driver) so they exercise
exactly the request route that fails — the verified-proxy subprocess, sy.login,
client.api.services.prepilot.<endpoint>, and the same result unwrap. Nothing
about the transport is re-implemented, so a probe result transfers directly to
the real driver.

Goal: discriminate among the hypotheses for the endpoint-execution wedge:
  1B  consumers die on first execution and don't respawn (logic exception)
  1C  resource exhaustion on first heavy execution (OOM / fd / shm)
  1D  the verified proxy / shim closes the connection (idle timeout or
      response-size limit), PySyft consumers healthy

The probes do NOT assume which is true. They record, per call, a fine-grained
outcome and full timing so the pattern across calls reveals the cause.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Optional

# Reuse the live driver plumbing verbatim — same path that fails in production.
import phase3_pysyft_driver as drv


# Reuse the driver/m5 disconnect classifier semantics (kept local to avoid a
# hard dependency on m5_concurrent, but identical predicate).
def is_remote_disconnect(err: Optional[str]) -> bool:
    if not err:
        return False
    e = err.lower()
    return ("remotedisconnected" in e or "connection aborted" in e
            or "connection reset" in e or "server disconnected" in e
            or "peer closed" in e or "incompleteread" in e)


def classify_failure(err: Optional[str]) -> str:
    """Fine-grained bucket for triangulation. The buckets are chosen to
    separate the hypotheses:
      - 'ok'                : success
      - 'remote_disconnect' : connection closed before response (1B/1C/1D all
                              surface here client-side; timing/size pattern
                              discriminates)
      - 'timeout'           : client-side read timeout (points at a slow/hung
                              consumer or proxy hold, leans 1D/1C over 1B)
      - 'http_5xx'          : server returned an error response (consumer ALIVE
                              and answered — argues AGAINST a dead pool; a typed
                              app error or a proxy 502/504)
      - 'typed_endpoint_err': structured endpoint error payload (cap exceeded,
                              invalid layers) — consumer alive and executed
      - 'other'             : anything else (record raw)
    """
    if not err:
        return "ok"
    e = err.lower()
    if is_remote_disconnect(err):
        return "remote_disconnect"
    if "timeout" in e or "timed out" in e or "readtimeout" in e:
        return "timeout"
    if any(c in e for c in ("502", "503", "504", "500", "bad gateway",
                            "gateway timeout", "service unavailable")):
        return "http_5xx"
    if any(c in e for c in ("engagement_cap_exceeded", "invalid_layers",
                            "invalid_direction", "cap exceeded")):
        return "typed_endpoint_err"
    return "other"


@dataclass
class ProbeCall:
    """One probe request with everything needed to spot a pattern."""
    seq: int
    endpoint: str
    payload_hint: str            # which size class: '0B' | '129KB' | '4.2MB'
    gap_before_s: float          # idle gap since previous call ended
    t_start: float
    wall_s: float                # request wall time (perf_counter)
    outcome: str                 # classify_failure bucket
    error: Optional[str] = None
    result_bytes: Optional[int] = None   # size of returned payload if success
    notes: str = ""


# Endpoint → payload size class (from Table 3 of the brief).
ENDPOINT_PAYLOAD = {
    "steering":                 ("apply_steering",            "0B"),
    "apply_steering":           ("apply_steering",            "0B"),
    "routing":                  ("capture_routing",           "129KB"),
    "capture_routing":          ("capture_routing",           "129KB"),
    "residual":                 ("capture_residual_stream",   "4.2MB"),
    "capture_residual_stream":  ("capture_residual_stream",   "4.2MB"),
    "attention":                ("capture_attention_stats",   "4.2MB"),
    "capture_attention_stats":  ("capture_attention_stats",   "4.2MB"),
}

DEFAULT_PROMPT = "Summarise the safety properties of confidential computing."


def resolve_endpoint(name: str) -> tuple[str, str]:
    """Map a friendly endpoint name to (real_endpoint, payload_hint)."""
    if name not in ENDPOINT_PAYLOAD:
        raise ValueError(f"unknown endpoint '{name}'; "
                         f"choose from {sorted(set(k for k in ENDPOINT_PAYLOAD))}")
    return ENDPOINT_PAYLOAD[name]


def do_call(client, seq: int, endpoint_name: str, gap_before_s: float,
            *, steering_direction=None, max_new_tokens: int = 32,
            read_timeout_s: Optional[float] = None) -> ProbeCall:
    """Execute one endpoint call via the live driver path, fully classified.

    read_timeout_s: if set, used to distinguish a hung consumer (timeout) from a
    closed connection (remote_disconnect). The driver's own client manages the
    socket; we wrap timing and classification around call_endpoint.
    """
    real_ep, payload_hint = resolve_endpoint(endpoint_name)
    t_start = time.time()
    t0 = time.perf_counter()
    err: Optional[str] = None
    result_bytes: Optional[int] = None
    try:
        raw = drv.call_endpoint(
            client, real_ep, DEFAULT_PROMPT,
            steering_direction=steering_direction,
            max_new_tokens=max_new_tokens,
        )
        result = drv._unwrap_pysyft_result(raw)
        if isinstance(result, dict) and result.get("error"):
            # endpoint returned a structured error payload (consumer ran!)
            err = f"{result.get('error')}: {str(result)[:160]}"
        else:
            try:
                import json as _json
                result_bytes = len(_json.dumps(result).encode("utf-8"))
            except Exception:
                result_bytes = None
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    wall = time.perf_counter() - t0
    return ProbeCall(
        seq=seq, endpoint=real_ep, payload_hint=payload_hint,
        gap_before_s=round(gap_before_s, 2), t_start=t_start,
        wall_s=round(wall, 3), outcome=classify_failure(err),
        error=err, result_bytes=result_bytes,
    )


def connect(args):
    """Bring up the verified proxy + login, reusing the driver helpers.
    Returns (client, proxy_proc). Caller must terminate proxy_proc."""
    proxy_proc = None
    if getattr(args, "container", None):
        proxy_proc = drv.launch_verified_proxy(
            args.container, args.port, debug_mode=getattr(args, "debug", True))
        client = drv.login_via_proxy(args.port, args.email, args.password)
    else:
        # direct URL path (debug shim authenticated:false)
        client = drv.login_via_bearer(
            args.datasite_url, getattr(args, "bearer", "") or "",
            args.email, args.password)
    return client, proxy_proc


def add_connect_args(ap):
    ap.add_argument("--container", default=None,
                    help="Tinfoil container name (launches verified proxy). "
                         "Omit to connect directly to --datasite-url.")
    ap.add_argument("--datasite-url", default=None,
                    help="direct Datasite URL (debug shim, authenticated:false)")
    ap.add_argument("--port", type=int, default=8081,
                    help="local port for the verified proxy")
    ap.add_argument("--bearer", default=None, help="bearer token if needed")
    ap.add_argument("--email", default="auditor_a@example.com")
    ap.add_argument("--password", default="auditor_password")
    ap.add_argument("--debug", action="store_true", default=True,
                    help="proxy --debug-mode (default true for debug deploys)")
    ap.add_argument("--max-new-tokens", type=int, default=32)
