#!/usr/bin/env python3
"""
probe_t3_payload_size.py — T3: does failure depend on RESPONSE PAYLOAD SIZE?

Holds dispatch rate FIXED (low, sequential, generous gaps so idle-timeout and
concurrency are NOT in play) and varies the endpoint's response size across the
three size classes from the brief:
    steering / apply_steering   -> 0 B      (control: no capture payload)
    routing / capture_routing   -> ~129 KB
    residual / capture_residual -> ~4.2 MB  (and attention ~4.2 MB)

DISCRIMINATES:
  - failure rate RISES with payload size (0B ok, 4.2MB fails)
    -> a verified-proxy / shim RESPONSE-SIZE limit (1D variant), or memory
       pressure on the worker building/serialising the large capture (1C).
       Distinguish the two: a size LIMIT tends to fail cleanly at a threshold
       (e.g. 129KB ok, 4.2MB drops); MEMORY pressure tends to correlate with a
       worker death that ALSO shows in logs / leaves the pool dead for the next
       (even small) call.
  - 0 B (steering) ALSO fails
    -> NOT a payload-size effect. The control path with no capture payload
       can't be a response-size or capture-memory issue, so look at 1B (logic
       death) or 1D (idle/connection) via T1/T2.
  - everything succeeds at low rate regardless of size
    -> the wedge needs CONCURRENCY; none of the single-stream hypotheses fire.
       Run a 2-evaluator concurrency step.

CROSS-CHECK (key): after a large-payload FAILURE, this probe immediately fires a
small (0B steering) call. If the small call now ALSO fails, the large call
killed the pool (1C memory death) rather than merely hitting a per-response size
cap (1D, which would leave the pool able to serve the next small call). This
"canary after the big one" is the cleanest 1C-vs-1D separator.

USAGE:
  python probe_t3_payload_size.py --datasite-url https://<fqdn> \
      --sizes steering,routing,residual --reps 3 --gap-s 8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict

import probe_common as pc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    pc.add_connect_args(ap)
    ap.add_argument("--sizes", default="steering,routing,residual",
                    help="comma list of endpoints, small->large payload")
    ap.add_argument("--reps", type=int, default=3, help="calls per size class")
    ap.add_argument("--gap-s", type=float, default=8.0,
                    help="idle gap between calls (keep modest so idle-timeout "
                         "and concurrency are out of play)")
    ap.add_argument("--canary-after-large", action="store_true", default=True,
                    help="after each large-payload call, fire a 0B steering "
                         "canary to test whether a large call killed the pool")
    ap.add_argument("--steering-direction", default=None)
    ap.add_argument("--out", default="probe_t3_result.json")
    args = ap.parse_args()

    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    from probe_t1_fresh_single import _load_direction
    steer = _load_direction(args.steering_direction)

    # rank payload classes so we can detect monotonicity
    rank = {"0B": 0, "129KB": 1, "4.2MB": 2}

    print("=== T3: payload-size sweep (fixed low rate) ===", flush=True)
    print(f"sizes={sizes} reps={args.reps} gap={args.gap_s}s", flush=True)

    client, proxy = pc.connect(args)
    calls = []
    seq = 0
    try:
        for ep in sizes:
            _, hint = pc.resolve_endpoint(ep)
            for r in range(args.reps):
                if seq > 0:
                    time.sleep(args.gap_s)
                c = pc.do_call(client, seq, ep, gap_before_s=(0.0 if seq == 0 else args.gap_s),
                              steering_direction=steer,
                              max_new_tokens=args.max_new_tokens)
                c.notes = f"size={hint}"
                calls.append(c); seq += 1
                print(f"[{hint:>6} #{r}] outcome={c.outcome} wall={c.wall_s}s "
                      f"bytes={c.result_bytes} err={c.error}", flush=True)

                # canary: after a large-payload call, fire a 0B steering call
                if (args.canary_after_large and hint == "4.2MB"):
                    time.sleep(2.0)
                    can = pc.do_call(client, seq, "steering", gap_before_s=2.0,
                                    steering_direction=steer,
                                    max_new_tokens=args.max_new_tokens)
                    can.notes = "canary_after_large"
                    calls.append(can); seq += 1
                    print(f"   [canary 0B] outcome={can.outcome} "
                          f"wall={can.wall_s}s err={can.error}", flush=True)
    finally:
        if proxy is not None:
            print("[proxy] terminating verified proxy", flush=True)
            proxy.terminate()

    # ---- per-size failure rate ---------------------------------------------
    by_size = {}
    for c in calls:
        if c.notes.startswith("size="):
            h = c.notes.split("=")[1]
            by_size.setdefault(h, []).append(c.outcome)
    size_rates = {h: {"n": len(v),
                      "fail": sum(1 for o in v if o != "ok"),
                      "fail_rate": round(sum(1 for o in v if o != "ok") / len(v), 3)}
                  for h, v in by_size.items()}
    ordered = sorted(size_rates, key=lambda h: rank.get(h, 9))

    canaries = [c.outcome for c in calls if c.notes == "canary_after_large"]
    canary_failed = any(o != "ok" for o in canaries)

    # ---- verdict -----------------------------------------------------------
    rates = [size_rates[h]["fail_rate"] for h in ordered]
    monotone_up = (len(rates) >= 2 and rates[-1] > rates[0]
                   and all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1)))
    zero_b_fails = size_rates.get("0B", {}).get("fail_rate", 0.0) > 0.0
    all_ok = all(r == 0.0 for r in rates) if rates else False

    if zero_b_fails:
        verdict = ("0B CONTROL FAILS: the steering path (no capture payload) "
                   "fails too, so this is NOT a response-size or capture-memory "
                   "effect. Look at connection/idle (1D, run T2) or logic death "
                   "(1B, run T1 + worker logs).")
    elif monotone_up and canary_failed:
        verdict = ("SIZE-CORRELATED + CANARY DIED: large payloads fail AND a 0B "
                   "call right after a large one also fails. The large capture "
                   "is KILLING the worker (memory/resource death, hypothesis "
                   "1C) rather than merely hitting a per-response size cap. FIX: "
                   "raise the CVM memory ceiling and/or cap capture size before "
                   "serialise; check /dev/shm and worker logs for OOM. NOT a "
                   "consumer-count issue.")
    elif monotone_up and not canary_failed:
        verdict = ("SIZE-CORRELATED, POOL SURVIVES: large payloads fail but a 0B "
                   "call right after succeeds — the pool is alive, the large "
                   "RESPONSE itself is being dropped. Points at a verified-proxy "
                   "/ shim response-SIZE limit (hypothesis 1D). FIX: raise the "
                   "proxy body-size limit, or stream/paginate the large bundle. "
                   "NB: the production governed-egress pipeline returns a 60KB "
                   "bundle, NOT the 4.2MB raw capture, so this may not affect the "
                   "real Tier-1 path — confirm which payload the failing calls "
                   "used.")
    elif all_ok:
        verdict = ("SIZE-INDEPENDENT, ALL OK at low rate: no single-stream size "
                   "effect. The wedge needs CONCURRENCY. Run a 2-evaluator step.")
    else:
        verdict = (f"NO CLEAR SIZE TREND: per-size fail rates={size_rates}, "
                   f"canaries={canaries}. Inspect per-call detail.")

    print("\n[per-size]", json.dumps(size_rates), flush=True)
    print("[canaries after large]", canaries, flush=True)
    print("\n[verdict]", verdict, flush=True)

    with open(args.out, "w") as f:
        json.dump({"sizes": sizes, "size_rates": size_rates,
                   "canaries": canaries, "calls": [asdict(c) for c in calls],
                   "verdict": verdict}, f, indent=2)
    print(f"[out] {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
