#!/usr/bin/env python3
"""
probe_t2_idle_gap.py — T2: does failure depend on the IDLE GAP between calls?

This is the cheapest test that can overturn the "worker pool wedged" framing.
It holds endpoint and payload size FIXED and varies only the idle gap between
consecutive calls. A verified-proxy / shim idle-connection timeout (hypothesis
1D) makes failures appear past a gap threshold; a dead-consumer story (1B/1C)
is indifferent to the gap.

Independent motivation: the brief already documents a ~60s idle-connection
limit on the laptop-to-TEE path (TLS-terminating edge proxy / NAT closing
non-streaming HTTPS). If endpoint executions hold the connection across that
window, or if the inter-request gap itself trips a keep-alive timeout, that
alone explains the symptom — including why THROTTLING MADE IT WORSE (a 0.15/s
throttle = ~6.7s gaps; longer gaps = more idle-timeout exposure).

DESIGN:
  For each gap in --gaps, send a small burst of identical calls spaced by that
  gap. Compare failure rate vs gap. Also issues one "long single idle" probe:
  open the client, wait --long-idle seconds, then fire one call — this tests a
  per-connection keep-alive timeout directly.

DISCRIMINATES:
  - failure rate RISES with gap (esp. a cliff near a round number like 60s)
    -> 1D idle timeout. Fix is proxy keep-alive / switch endpoints to
       streaming or polling. NOT a consumer issue.
  - failure rate FLAT across gaps (incl. short gaps fail too)
    -> gap-independent: 1B/1C (consumer death) or saturation. Run T1/T3.
  - SHORT gaps fail but LONG gaps succeed -> the opposite of idle timeout;
    suggests overlapping/queued execution contention, not idle close.

USAGE:
  python probe_t2_idle_gap.py --datasite-url https://<fqdn> --endpoint steering \
      --gaps 2,10,30,70 --burst 3 --long-idle 90
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
    ap.add_argument("--endpoint", default="steering",
                    help="friendly endpoint name; keep FIXED across the sweep")
    ap.add_argument("--gaps", default="2,10,30,70",
                    help="comma list of idle gaps (s) to test, low to high")
    ap.add_argument("--burst", type=int, default=3,
                    help="calls per gap level")
    ap.add_argument("--long-idle", type=float, default=90.0,
                    help="single long idle (s) then one call — direct keep-alive "
                         "timeout probe. 0 to skip.")
    ap.add_argument("--steering-direction", default=None)
    ap.add_argument("--out", default="probe_t2_result.json")
    args = ap.parse_args()

    gaps = [float(g) for g in args.gaps.split(",") if g.strip()]
    steer = None
    if args.endpoint in ("steering", "apply_steering"):
        from probe_t1_fresh_single import _load_direction
        steer = _load_direction(args.steering_direction)

    print("=== T2: idle-gap sweep ===", flush=True)
    print(f"endpoint={args.endpoint} (fixed)  gaps={gaps}  burst={args.burst}",
          flush=True)

    client, proxy = pc.connect(args)
    calls = []
    seq = 0
    try:
        # Warm: one call to establish the path. Recorded but excluded from the
        # per-gap rate (its job is to confirm the path can work at all).
        warm = pc.do_call(client, seq, args.endpoint, gap_before_s=0.0,
                          steering_direction=steer,
                          max_new_tokens=args.max_new_tokens)
        warm.notes = "warmup"
        calls.append(warm); seq += 1
        print(f"[warm] outcome={warm.outcome} wall={warm.wall_s}s err={warm.error}",
              flush=True)

        for gap in gaps:
            for b in range(args.burst):
                time.sleep(gap)
                c = pc.do_call(client, seq, args.endpoint, gap_before_s=gap,
                              steering_direction=steer,
                              max_new_tokens=args.max_new_tokens)
                c.notes = f"gap_level={gap}"
                calls.append(c); seq += 1
                print(f"[gap={gap:>4}s #{b}] outcome={c.outcome} "
                      f"wall={c.wall_s}s err={c.error}", flush=True)

        # Direct keep-alive probe: one long idle then a single call.
        if args.long_idle and args.long_idle > 0:
            print(f"[long-idle] sleeping {args.long_idle}s then one call...",
                  flush=True)
            time.sleep(args.long_idle)
            c = pc.do_call(client, seq, args.endpoint,
                          gap_before_s=args.long_idle,
                          steering_direction=steer,
                          max_new_tokens=args.max_new_tokens)
            c.notes = "long_idle"
            calls.append(c); seq += 1
            print(f"[long-idle={args.long_idle}s] outcome={c.outcome} "
                  f"wall={c.wall_s}s err={c.error}", flush=True)
    finally:
        if proxy is not None:
            print("[proxy] terminating verified proxy", flush=True)
            proxy.terminate()

    # ---- per-gap failure rate ----------------------------------------------
    by_gap = {}
    for c in calls:
        if c.notes.startswith("gap_level="):
            g = float(c.notes.split("=")[1])
            by_gap.setdefault(g, []).append(c.outcome)
    gap_rates = {g: {"n": len(v),
                     "fail": sum(1 for o in v if o != "ok"),
                     "fail_rate": round(sum(1 for o in v if o != "ok") / len(v), 3)}
                 for g, v in sorted(by_gap.items())}

    long_idle_outcome = next((c.outcome for c in calls if c.notes == "long_idle"),
                             None)

    # ---- verdict -----------------------------------------------------------
    rates_sorted = [gap_rates[g]["fail_rate"] for g in sorted(gap_rates)]
    gaps_sorted = sorted(gap_rates)
    monotone_up = all(rates_sorted[i] <= rates_sorted[i + 1]
                      for i in range(len(rates_sorted) - 1)) and \
        len(rates_sorted) >= 2 and rates_sorted[-1] > rates_sorted[0]
    all_fail = all(r >= 0.99 for r in rates_sorted) if rates_sorted else False
    all_ok = all(r == 0.0 for r in rates_sorted) if rates_sorted else False

    if monotone_up or (long_idle_outcome and long_idle_outcome != "ok" and
                       rates_sorted and rates_sorted[0] < 0.5):
        verdict = ("IDLE-GAP DEPENDENT: failure rate rises with the idle gap"
                   + (f" and the {args.long_idle}s single-idle call FAILED"
                      if long_idle_outcome and long_idle_outcome != "ok" else "")
                   + ". Strong signal for a verified-proxy / shim idle-connection "
                   "timeout (hypothesis 1D). This also explains why throttling "
                   "made the production run WORSE (longer gaps = more idle "
                   "exposure). FIX is transport-side: proxy keep-alive / idle "
                   "timeout, or switch endpoints to a streaming/polling response "
                   "so they don't hold a long synchronous connection. A consumer "
                   "bump will NOT help.")
    elif all_fail:
        verdict = ("GAP-INDEPENDENT, ALL FAIL: every gap level fails. Not an "
                   "idle timeout. Consistent with consumer death (1B/1C) or a "
                   "config/proxy failure of the execution path regardless of "
                   "timing. Run T1 (fresh single) to see if it EVER works, and "
                   "pull worker logs.")
    elif all_ok:
        verdict = ("GAP-INDEPENDENT, ALL OK: no failures at any gap on this "
                   "endpoint/size. The wedge is not reproduced by idle gaps "
                   "alone — it likely needs CONCURRENCY. Run a small concurrency "
                   "step (2 simultaneous evaluators) and T3 (payload size).")
    else:
        verdict = (f"NO CLEAR GAP TREND: per-gap fail rates={gap_rates}. "
                   "If short gaps fail but long succeed, suspect overlapping "
                   "execution contention rather than idle close. Inspect "
                   "per-call timing below.")

    print("\n[per-gap]", json.dumps(gap_rates), flush=True)
    print("[long-idle outcome]", long_idle_outcome, flush=True)
    print("\n[verdict]", verdict, flush=True)

    with open(args.out, "w") as f:
        json.dump({"endpoint": args.endpoint, "gaps": gaps,
                   "gap_rates": gap_rates, "long_idle_outcome": long_idle_outcome,
                   "calls": [asdict(c) for c in calls], "verdict": verdict},
                  f, indent=2)
    print(f"[out] {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
