"""
streaming_client.py — work around / fix the ~60s synchronous-response deadline
on the PySyft execution path (diagnosed 2026-05-29; see REDEPLOY_NOTES.md).

THE PROBLEM
    A TwinAPIEndpoint call --
        client.api.services.prepilot.capture_residual_stream(prompt=...)
    -- is SYNCHRONOUS: one HTTP request is held open for the entire server-side
    execution. The laptop->TEE path closes any connection at a fixed ~60s, so
    every execution that runs longer than ~60s dies with RemoteDisconnected,
    independent of payload size or inter-request gap. Measured: 60.3-61.4s
    across steering (0B) and routing (129KB).

TWO FIXES, in increasing order of what they need:

  (2a) BOUNDED-SYNC  -- needs NOTHING from Tinfoil.
       Keep each synchronous call's server-side work UNDER the ~60s wall by
       capping max_new_tokens (and, where supported, capture scope). This is a
       workaround, not a true fix, but it is testable today and yields the key
       number: how much work fits under the deadline. `find_token_budget()`
       below binary-searches that threshold.

  (2b) JOB-POLL  -- needs Tinfoil to PREFIX-allowlist the PySyft job/stream
       path (`/api/v2/stream/{peer_uid}/{url_path}/`), which SPIKE_PLAN.md line
       31 flagged as un-allowlistable under exact matching. With that path
       open, submit the endpoint call NON-BLOCKING (returns a Job handle in one
       fast request) and POLL the job status (many fast requests), so no single
       HTTP request spans the execution. `call_via_job()` below implements the
       client side; it is GATED behind a capability check so it fails loudly
       with the exact escalation needed, rather than silently.

Both reuse the live driver path (phase3_pysyft_driver) so behaviour transfers.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Callable

import phase3_pysyft_driver as drv


# The observed hard deadline. Keep bounded-sync work comfortably under it.
RESPONSE_DEADLINE_S = 60.0
SAFETY_MARGIN_S = 8.0          # target finishing by ~52s, not 59.9s


@dataclass
class CallResult:
    ok: bool
    wall_s: float
    result: Optional[dict] = None
    error: Optional[str] = None
    mode: str = ""             # 'bounded_sync' | 'job_poll'
    attempts: int = 1


# ---------------------------------------------------------------------------
# (2a) Bounded synchronous call + threshold finder
# ---------------------------------------------------------------------------

def call_bounded_sync(client, endpoint: str, prompt: str, *,
                      max_new_tokens: int,
                      steering_direction=None) -> CallResult:
    """One ordinary synchronous endpoint call, timed. Succeeds iff the
    server-side execution finishes before the path's ~60s cutoff."""
    t0 = time.perf_counter()
    try:
        raw = drv.call_endpoint(client, endpoint, prompt,
                                steering_direction=steering_direction,
                                max_new_tokens=max_new_tokens)
        result = drv._unwrap_pysyft_result(raw)
        wall = time.perf_counter() - t0
        if isinstance(result, dict) and result.get("error"):
            return CallResult(False, wall, error=str(result)[:200],
                              mode="bounded_sync")
        return CallResult(True, wall, result=result, mode="bounded_sync")
    except Exception as e:  # noqa: BLE001
        return CallResult(False, time.perf_counter() - t0,
                          error=f"{type(e).__name__}: {e}", mode="bounded_sync")


def find_token_budget(client, endpoint: str, prompt: str, *,
                      lo: int = 1, hi: int = 512,
                      steering_direction=None,
                      log: Callable[[str], None] = print) -> dict:
    """Binary-search the largest max_new_tokens whose synchronous call finishes
    under the ~60s deadline. This is the key number for fix 2a and for sizing
    the real fix: it tells you how much generation fits in one request.

    Returns {'budget': int, 'trials': [...]}. budget=0 means even the smallest
    request (lo) exceeded the deadline -> bounded-sync is NOT viable, you need
    the job-poll path (2b)."""
    trials = []
    best = 0
    # First confirm the floor: does the smallest request even fit?
    r_lo = call_bounded_sync(client, endpoint, prompt, max_new_tokens=lo,
                             steering_direction=steering_direction)
    trials.append({"max_new_tokens": lo, "ok": r_lo.ok, "wall_s": round(r_lo.wall_s, 1)})
    log(f"[budget] max_new_tokens={lo}: ok={r_lo.ok} wall={r_lo.wall_s:.1f}s")
    if not r_lo.ok:
        log("[budget] even the smallest request exceeded the deadline -> "
            "bounded-sync NOT viable; the real fix (job-poll, 2b) is required.")
        return {"budget": 0, "trials": trials}
    best = lo
    a, b = lo, hi
    while a < b:
        mid = (a + b + 1) // 2
        r = call_bounded_sync(client, endpoint, prompt, max_new_tokens=mid,
                              steering_direction=steering_direction)
        trials.append({"max_new_tokens": mid, "ok": r.ok,
                       "wall_s": round(r.wall_s, 1)})
        log(f"[budget] max_new_tokens={mid}: ok={r.ok} wall={r.wall_s:.1f}s")
        if r.ok:
            best = mid
            a = mid
        else:
            b = mid - 1
    log(f"[budget] largest max_new_tokens under deadline: {best}")
    return {"budget": best, "trials": trials}


# ---------------------------------------------------------------------------
# (2b) Non-blocking job submission + poll  (needs Tinfoil prefix-allowlist)
# ---------------------------------------------------------------------------

def job_poll_capability(client) -> dict:
    """Probe whether the non-blocking job path is reachable through the shim.

    PySyft 0.9.5 runs endpoints as jobs when invoked non-blocking; the result
    is fetched by polling the job, which routes through
    /api/v2/stream/{peer_uid}/{url_path}/. The Tinfoil shim allowlists EXACT
    paths, so this path is (per SPIKE_PLAN.md) not reachable without prefix
    matching. This check tries to see the jobs service and returns a verdict +
    the precise escalation string if it's blocked."""
    info = {"jobs_service": False, "submit_nonblocking": False,
            "stream_path_reachable": None, "verdict": "", "escalation": ""}
    try:
        # Is the jobs API even exposed to the client?
        _ = client.api.services.job
        info["jobs_service"] = True
    except Exception as e:  # noqa: BLE001
        info["verdict"] = f"jobs service not visible to client: {e}"
        return info
    # We cannot fully verify the stream path from here without firing a real
    # long job; the definitive test is call_via_job() against the live deploy.
    info["verdict"] = ("jobs service visible; the binding question is whether "
                       "/api/v2/stream/{peer_uid}/{url_path}/ is allowlisted by "
                       "the shim. Run call_via_job() to test end-to-end.")
    info["escalation"] = ("Tinfoil shim must PREFIX-allowlist "
                          "/api/v2/stream/  (FastAPI path params can't be "
                          "exact-matched). Without it, non-blocking job result "
                          "polling cannot return through the proxy.")
    return info


def call_via_job(client, endpoint: str, prompt: str, *,
                 max_new_tokens: int = 256,
                 steering_direction=None,
                 poll_interval_s: float = 3.0,
                 max_wait_s: float = 1800.0,
                 log: Callable[[str], None] = print) -> CallResult:
    """Submit the endpoint call NON-BLOCKING and poll the resulting job until it
    completes. Each HTTP request (submit, each poll) is short, so none spans the
    ~60s deadline.

    NOTE: the exact non-blocking invocation and job-fetch API surface of PySyft
    0.9.5 must be confirmed against the live client (sandbox has no syft). The
    structure below is the standard submit->poll->fetch; adjust the three TODO
    lines to the real 0.9.5 calls once verified. It FAILS LOUDLY (not silently)
    if the path is blocked, surfacing the escalation."""
    t0 = time.perf_counter()
    try:
        svc = client.api.services.prepilot
        method = getattr(svc, endpoint, None) or getattr(svc, _friendly(endpoint))
        kwargs = dict(prompt=prompt, max_new_tokens=max_new_tokens)
        if endpoint in ("apply_steering",) and steering_direction is not None:
            kwargs.update(direction=steering_direction.astype("float32").tolist(),
                          layer=62, alpha=1.0, sign=-1, norm_match=True)
        # TODO(verify on live 0.9.5): non-blocking submission.
        # In 0.9.5 this is typically `method(..., blocking=False)` returning a
        # Job, OR client.api.services.job after a non-blocking call. Confirm.
        job = method(**kwargs, blocking=False)
        log(f"[job] submitted: {getattr(job, 'id', job)}")

        deadline = time.perf_counter() + max_wait_s
        polls = 0
        while time.perf_counter() < deadline:
            time.sleep(poll_interval_s)
            polls += 1
            # TODO(verify): refresh job status. Often job.fetch()/job.wait() or
            # re-getting the job by id from client.api.services.job.
            status = getattr(job, "status", None)
            log(f"[job] poll {polls}: status={status}")
            if status in ("completed", "COMPLETED", "done", "DONE"):
                # TODO(verify): result accessor on a completed 0.9.5 Job.
                raw = job.result if hasattr(job, "result") else job.wait()
                result = drv._unwrap_pysyft_result(raw)
                wall = time.perf_counter() - t0
                return CallResult(True, wall, result=result, mode="job_poll",
                                  attempts=polls)
            if status in ("errored", "ERRORED", "error", "failed", "FAILED"):
                return CallResult(False, time.perf_counter() - t0,
                                  error=f"job failed: status={status}",
                                  mode="job_poll", attempts=polls)
        return CallResult(False, time.perf_counter() - t0,
                          error=f"job did not complete within {max_wait_s}s",
                          mode="job_poll", attempts=polls)
    except Exception as e:  # noqa: BLE001
        wall = time.perf_counter() - t0
        # If this failed near ~60s with a disconnect, the job RESULT fetch is
        # itself crossing the synchronous deadline -> the stream path is NOT
        # allowlisted; escalate.
        hint = ""
        if "remotedisconnected" in str(e).lower() and 55 <= wall <= 70:
            hint = (" | LIKELY the job poll/result path crossed the ~60s "
                    "deadline -> /api/v2/stream/ not allowlisted; escalate to "
                    "Tinfoil for prefix matching (SPIKE_PLAN.md line 31).")
        return CallResult(False, wall, error=f"{type(e).__name__}: {e}{hint}",
                          mode="job_poll")


def _friendly(name: str) -> str:
    m = {"residual": "capture_residual_stream", "routing": "capture_routing",
         "attention": "capture_attention_stats", "steering": "apply_steering"}
    return m.get(name, name)
