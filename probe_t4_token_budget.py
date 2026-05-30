#!/usr/bin/env python3
"""
probe_t4_token_budget.py — find the largest max_new_tokens whose SYNCHRONOUS
endpoint call finishes under the ~60s response deadline (fix 2a).

This needs nothing from Tinfoil. It yields the key number for the bounded-sync
workaround AND for sizing the real job-poll fix: how much generation fits in one
sub-60s request.

It also runs the job-poll CAPABILITY check (read-only) so you learn whether the
non-blocking path is even visible to the client before asking Tinfoil to
prefix-allowlist the stream route.

USAGE:
  python probe_t4_token_budget.py \
      --datasite-url https://pysyft-m1.debug.pour-demain.containers.tinfoil.dev \
      --endpoint steering --lo 1 --hi 512
"""
from __future__ import annotations

import argparse
import json
import sys

import probe_common as pc
import streaming_client as sc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    pc.add_connect_args(ap)
    ap.add_argument("--endpoint", default="steering",
                    help="friendly endpoint name (steering|routing|residual|attention)")
    ap.add_argument("--lo", type=int, default=1)
    ap.add_argument("--hi", type=int, default=512)
    ap.add_argument("--prompt", default=pc.DEFAULT_PROMPT)
    ap.add_argument("--steering-direction", default=None)
    ap.add_argument("--out", default="probe_t4_result.json")
    args = ap.parse_args()

    real_ep, payload_hint = pc.resolve_endpoint(args.endpoint)
    steer = None
    if args.endpoint in ("steering", "apply_steering"):
        from probe_t1_fresh_single import _load_direction
        steer = _load_direction(args.steering_direction)

    print(f"=== T4: max_new_tokens budget under ~{int(sc.RESPONSE_DEADLINE_S)}s "
          f"deadline ({args.endpoint}, payload {payload_hint}) ===", flush=True)

    client, proxy = pc.connect(args)
    try:
        # capability check for the real fix (read-only, fast)
        cap = sc.job_poll_capability(client)
        print(f"[jobs] {cap['verdict']}", flush=True)
        if cap.get("escalation"):
            print(f"[jobs] escalation if needed: {cap['escalation']}", flush=True)

        budget = sc.find_token_budget(
            client, real_ep, args.prompt,
            lo=args.lo, hi=args.hi, steering_direction=steer)
    finally:
        if proxy is not None:
            proxy.terminate()

    verdict = _verdict(budget["budget"], args.endpoint)
    print(f"\n[verdict] {verdict}", flush=True)

    out = {"endpoint": real_ep, "payload_hint": payload_hint,
           "deadline_s": sc.RESPONSE_DEADLINE_S,
           "token_budget_under_deadline": budget["budget"],
           "trials": budget["trials"], "jobs_capability": cap,
           "verdict": verdict}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[out] {args.out}", flush=True)
    return 0


def _verdict(budget: int, endpoint: str) -> str:
    if budget == 0:
        return ("ZERO BUDGET: even the smallest request exceeds ~60s. The "
                "bounded-sync workaround (2a) is NOT viable for this endpoint; "
                "the real fix is the non-blocking job-poll path (2b), which "
                "needs Tinfoil to prefix-allowlist /api/v2/stream/. The fact "
                "that even minimal generation can't return in 60s also strongly "
                "suggests the server-side execution is genuinely slow (or "
                "hung), not merely a tight timeout on fast work.")
    return (f"BUDGET={budget} tokens fit under the ~60s deadline for "
            f"'{endpoint}'. Bounded-sync (2a) is viable up to here: cap "
            f"max_new_tokens at <= {budget} for synchronous calls. Anything "
            f"longer needs the job-poll fix (2b). Record this number — it sizes "
            f"both the workaround and the real fix, and belongs in the brief's "
            f"governed-egress limitations.")


if __name__ == "__main__":
    sys.exit(main())
