#!/usr/bin/env python3
"""
probe_t1_fresh_single.py — T1: does a SINGLE endpoint execution succeed on a
freshly-deployed, never-concurrently-hit Datasite?

This is the highest-value triangulation test and it needs nothing from Tinfoil
support — just a fresh deploy you trigger yourself.

DISCRIMINATES:
  - If metadata 200 + a single execution SUCCEEDS  -> the pool starts HEALTHY.
    Whatever wedges it is triggered by load/repetition (hypotheses 1B/1C:
    consumers die after first/under concurrency). The fix is in the worker
    path or resource provisioning, NOT a static config/proxy issue.
  - If metadata 200 but the single execution FAILS (remote_disconnect / 5xx /
    timeout) on a pristine deploy with zero prior load -> execution NEVER
    worked through this path. That points at the proxy/shim (1D) or a config
    problem, NOT a death-under-load story. A consumer bump would be irrelevant.

It also runs a tiny escalation (1 -> 2 -> 3 sequential calls, generous gaps) to
see whether failure onset is at call #1 (config/proxy) or after #1 (death on
first execution).

USAGE (fresh deploy, debug shim):
  python probe_t1_fresh_single.py --datasite-url https://<fqdn> --email ... \
      --password ... --endpoint steering
  # or via verified proxy:
  python probe_t1_fresh_single.py --container cc-deep-eval-pysyft --endpoint steering

Pick --endpoint steering first (0B payload) to isolate the CONTROL path from any
payload-size effect; then re-run with --endpoint residual (4.2MB) for T3-style
contrast if call #1 succeeds.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict

import httpx

import probe_common as pc
import phase3_pysyft_driver as drv


def check_metadata(datasite_url: str | None, port: int) -> dict:
    """Pre-auth liveness: /api/v2/metadata. Fast + small — if THIS fails the
    Datasite isn't up at all (not a wedge)."""
    url = (datasite_url.rstrip("/") if datasite_url
           else f"http://127.0.0.1:{port}")
    t0 = time.perf_counter()
    try:
        r = httpx.get(f"{url}/api/v2/metadata", timeout=30.0)
        dt = time.perf_counter() - t0
        return {"ok": r.status_code == 200, "status": r.status_code,
                "wall_s": round(dt, 3)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}",
                "wall_s": round(time.perf_counter() - t0, 3)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    pc.add_connect_args(ap)
    ap.add_argument("--endpoint", default="steering",
                    help="friendly endpoint name (steering|routing|residual|attention)")
    ap.add_argument("--n-escalate", type=int, default=3,
                    help="sequential calls after the first, generous gap, to see "
                         "if failure onset is at call #1 or later")
    ap.add_argument("--gap-s", type=float, default=8.0,
                    help="idle gap between the escalation calls")
    ap.add_argument("--steering-direction", default=None,
                    help="path to a .npy/.json 6144-vector for apply_steering")
    ap.add_argument("--out", default="probe_t1_result.json")
    args = ap.parse_args()

    steer = None
    if args.endpoint in ("steering", "apply_steering"):
        steer = _load_direction(args.steering_direction)

    print("=== T1: single execution on a fresh Datasite ===", flush=True)

    # Step 1: metadata liveness (must be a FRESH deploy; do this first, before
    # any execution, so we know the box is up and unpoisoned).
    meta = check_metadata(args.datasite_url, args.port)
    print(f"[meta] /api/v2/metadata: {meta}", flush=True)
    if not meta["ok"]:
        print("[verdict] Datasite not reachable at metadata — this is not a "
              "wedge, the deploy isn't up. Check deploy state via SSH docker ps.",
              file=sys.stderr)
        _dump(args.out, {"metadata": meta, "verdict": "datasite_down"})
        return 2

    # Step 2: connect (proxy + login) and fire the FIRST execution.
    client, proxy = pc.connect(args)
    calls = []
    try:
        first = pc.do_call(client, 0, args.endpoint, gap_before_s=0.0,
                           steering_direction=steer,
                           max_new_tokens=args.max_new_tokens)
        calls.append(first)
        print(f"[call 0] outcome={first.outcome} wall={first.wall_s}s "
              f"bytes={first.result_bytes} err={first.error}", flush=True)

        # Step 3: escalate a few more, generous gaps, to locate failure onset.
        last_end = time.perf_counter()
        for i in range(1, args.n_escalate):
            gap = args.gap_s
            time.sleep(gap)
            c = pc.do_call(client, i, args.endpoint, gap_before_s=gap,
                          steering_direction=steer,
                          max_new_tokens=args.max_new_tokens)
            calls.append(c)
            print(f"[call {i}] outcome={c.outcome} wall={c.wall_s}s "
                  f"gap={gap}s bytes={c.result_bytes} err={c.error}", flush=True)
    finally:
        if proxy is not None:
            print("[proxy] terminating verified proxy", flush=True)
            proxy.terminate()

    # ---- verdict logic -----------------------------------------------------
    outcomes = [c.outcome for c in calls]
    first_ok = outcomes[0] == "ok"
    any_ok = any(o == "ok" for o in outcomes)
    all_fail = all(o != "ok" for o in outcomes)
    first_fail_bucket = outcomes[0] if not first_ok else None

    if first_ok and all(o == "ok" for o in outcomes):
        verdict = ("HEALTHY_SINGLE_AND_LOW_N: a fresh deploy serves single + a "
                   "few sequential executions fine. The wedge is triggered by "
                   "CONCURRENCY or sustained load, not by the first execution. "
                   "=> hypotheses 1B/1C under concurrency; NOT a static "
                   "proxy/config failure of the execution path. Next: run T3 "
                   "(payload size) and a small concurrency step.")
    elif first_ok and not all(o == "ok" for o in outcomes):
        verdict = ("DIES_AFTER_FIRST: call #0 succeeded but a later low-rate "
                   "call failed (onset at call "
                   f"#{outcomes.index(next(o for o in outcomes if o!='ok'))}). "
                   "Consumer(s) die after the first execution and don't "
                   "respawn => hypothesis 1B (or 1C if the first heavy call "
                   "exhausted a resource). NOT pure saturation. Pull worker "
                   "logs / check /dev/shm.")
    elif all_fail and first_fail_bucket == "remote_disconnect":
        verdict = ("EXECUTION_NEVER_WORKED (remote_disconnect on call #0, fresh "
                   "deploy, zero prior load): the execution path never "
                   "completed a single request. Strongly points at the "
                   "verified proxy / shim closing the connection on the "
                   "slow/large execution response (hypothesis 1D), or a config "
                   "problem — NOT death-under-load. A consumer bump is "
                   "IRRELEVANT. Next: run T2 (idle-gap) and T3 (payload size).")
    elif all_fail and first_fail_bucket in ("http_5xx", "timeout"):
        verdict = (f"EXECUTION_NEVER_WORKED ({first_fail_bucket} on call #0): the "
                   "server/proxy answered with an error or hung rather than "
                   "dropping the socket. A 5xx means something is ALIVE and "
                   "erroring (proxy 502/504 or app error) — pull the response "
                   "body and worker logs. Leans 1D (proxy) or a startup/"
                   "dependency fault, away from silent consumer death.")
    elif all_fail and first_fail_bucket == "typed_endpoint_err":
        verdict = ("ENDPOINT_RAN_BUT_REJECTED: the consumer executed and "
                   "returned a typed error (e.g. cap/validation). This is NOT a "
                   "wedge at all — the governance path worked. Check the "
                   "payload (budget? direction length?).")
    else:
        verdict = (f"MIXED/UNCLEAR: outcomes={outcomes}. Inspect per-call errors "
                   "below; if call #0 succeeded the path can work, so focus on "
                   "what differs in the failing calls (timing, size).")

    print("\n[verdict]", verdict, flush=True)
    _dump(args.out, {
        "metadata": meta,
        "endpoint": args.endpoint,
        "calls": [asdict(c) for c in calls],
        "outcomes": outcomes,
        "verdict": verdict,
    })
    print(f"[out] {args.out}", flush=True)
    # exit 0 if the path can work at all (first_ok or any_ok), 1 if execution
    # never worked — lets you script "did T1 prove the path works?".
    return 0 if any_ok else 1


def _load_direction(path):
    if not path:
        # apply_steering needs a 6144 vector; use zeros so the call exercises
        # the path without asserting a real steering effect (length is what the
        # endpoint validates). A zero vector is a valid no-op direction.
        import numpy as np
        return np.zeros(6144, dtype="float32")
    import numpy as np
    if path.endswith(".npy"):
        return np.load(path)
    with open(path) as f:
        return np.asarray(json.load(f), dtype="float32")


def _dump(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
