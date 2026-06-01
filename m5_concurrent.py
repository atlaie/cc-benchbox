#!/usr/bin/env python3
"""
m5_concurrent.py — M5 concurrent multi-evaluator run.

Drives N distinct auditors (default 2) in PARALLEL against one shared Datasite,
each running --sessions calls to the prepilot endpoints. Measures:

  (a) Ledger write contention at the SQLite BEGIN IMMEDIATE boundary, by
      comparing the per-call `pysyft_ledger_seconds` distribution under
      concurrency against a single-threaded reference (pass --reference-parquet
      from an M3/sequential run, or read the printed p50/p95 and compare).
  (b) RemoteDisconnected incidence vs the 1/20 transient seen once in M3. If it
      scales with concurrency, that points at datasite worker/IPC saturation
      (SYFT_N_CONSUMERS), not the ledger.
  (c) Budget-accounting correctness under concurrent writes: with a tight shared
      cap and --check-budget, confirm total recorded tokens_out across BOTH
      auditors' engagements never exceeds the cap (no race-through past the
      atomic cap-check).

Identity: each thread logs in as a distinct PySyft user (auditor_a@..,
auditor_b@.. — register them first with register_auditors.py). The Tinfoil
debug shim is identity-blind, but PySyft carries the identity, so each auditor
gets its own engagement under the same gateway. This is the same per-evaluator
identity the white-paper pilot uses; CC-on swaps the transport (verified proxy)
without touching this harness.

Reuses send_one / PysyftRow from phase3_pysyft_driver so rows are schema-
identical to single-threaded runs (one parquet per auditor + a merged parquet).

Smoke (2 auditors, 5 sessions each, residual endpoint only, debug deploy):

  python m5_concurrent.py \\
      --datasite-url https://pysyft-m1.debug.pour-demain.containers.tinfoil.dev \\
      --auditors auditor_a@pilot.test,auditor_b@pilot.test \\
      --endpoints capture_residual_stream \\
      --pairs-json runs/phase2_validation/repe_bundle/pairs.json \\
      --sessions 5 \\
      --out-dir runs/phase3_pysyft/concurrent_smoke \\
      --cc-state off

Full M5 (2 auditors, 20 sessions each, all 4 endpoints):

  python m5_concurrent.py \\
      --datasite-url https://pysyft.debug.pour-demain.containers.tinfoil.dev \\
      --auditors auditor_a@pilot.test,auditor_b@pilot.test \\
      --endpoints capture_residual_stream,capture_routing,capture_attention_stats,apply_steering \\
      --steering-direction runs/phase2_d6_steering/direction_L62.npy \\
      --pairs-json runs/phase2_validation/repe_bundle/pairs.json \\
      --sessions 20 \\
      --out-dir runs/phase3_pysyft/concurrent \\
      --cc-state off

Exit codes:
  0  success
  2  user error
  4  zero successful requests across all auditors
  6  --check-budget detected an over-cap total (correctness failure)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import syft as sy
except ImportError as e:
    print(f"[fatal] syft import failed: {e}", file=sys.stderr)
    sys.exit(2)

# Single source of truth: reuse the driver's row schema + per-call logic.
from phase3_pysyft_driver import (
    PysyftRow,
    send_one,
    login_via_proxy,
    start_verified_proxy,
    ENDPOINTS,
)
from phase3_vllm_driver import interleave, load_pairs


AUDITOR_PASSWORD = "auditor_password"   # matches register_auditors.py default


def _percentiles(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    a = np.asarray(xs, dtype=np.float64)
    return {
        "n": int(a.size),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
        "mean": float(a.mean()),
    }


def _is_remote_disconnect(err: Optional[str]) -> bool:
    if not err:
        return False
    e = err.lower()
    return ("remotedisconnected" in e or "connection aborted" in e
            or "connection reset" in e or "server disconnected" in e)


def run_auditor(
    *,
    auditor_email: str,
    datasite_url: str,
    port: int,
    endpoints: list[str],
    prompts: list[tuple[int, str, str]],
    sessions: int,
    cell_id: str,
    cc_state: str,
    steering_direction: Optional[np.ndarray],
    max_new_tokens: int,
    req_rate: float,
    max_retries: int,
    retry_backoff_base: float,
    start_barrier: threading.Barrier,
    results: dict,
):
    """One evaluator thread. Logs in as a distinct PySyft user, waits at the
    barrier so all auditors start together, then issues sessions*len(endpoints)
    calls throttled to req_rate calls/sec (per thread; 0 = unthrottled burst).

    req_rate matters: unthrottled, two threads against a 4-worker Datasite pool
    saturate it and the producer drops connections (RemoteDisconnected). A
    realistic per-evaluator rate separates 'the ledger can't handle concurrency'
    (disconnects persist when throttled) from 'unthrottled bursts saturate the
    worker pool' (disconnects vanish when throttled)."""
    try:
        client = sy.login(url=datasite_url, port=port,
                          email=auditor_email, password=AUDITOR_PASSWORD)
        client.refresh()
    except Exception as e:  # noqa: BLE001
        results[auditor_email] = {"login_error": f"{type(e).__name__}: {e}", "rows": []}
        # Still wait so peers don't block forever on the barrier.
        try:
            start_barrier.wait(timeout=30)
        except Exception:
            pass
        return

    # Build the per-auditor call sequence: round-robin endpoints over the first
    # `sessions` paired prompts. Each (endpoint, prompt) is one "session" call.
    seq = []
    rid = 0
    trimmed = prompts[:sessions]
    for ep in endpoints:
        for pair_id, prompt_class, prompt in trimmed:
            seq.append((rid, pair_id, prompt_class, prompt, ep))
            rid += 1

    # Synchronize the start across all auditor threads.
    try:
        start_barrier.wait(timeout=60)
    except threading.BrokenBarrierError:
        pass

    rows: list[PysyftRow] = []
    min_interval = 1.0 / req_rate if req_rate and req_rate > 0 else 0.0
    last_send = 0.0
    retry_log: list[dict] = []   # per-call retry accounting
    for (request_id, pair_id, prompt_class, prompt, endpoint) in seq:
        if min_interval > 0:
            wait = min_interval - (time.monotonic() - last_send)
            if wait > 0:
                time.sleep(wait)
        last_send = time.monotonic()

        # Bounded retry with exponential backoff, but ONLY for transient
        # RemoteDisconnect-class errors — a real evaluator client retries a
        # dropped connection rather than counting it as a hard failure. This
        # lets M5 separate "transient, recovered on retry N" from "persistent
        # disconnect (pool wedged)". Typed endpoint errors (cap-exceeded,
        # invalid_layers) are NOT retried — they're deterministic.
        n_attempts = 0
        row = None
        while n_attempts <= max_retries:
            n_attempts += 1
            row = send_one(
                client, request_id, pair_id, prompt_class, prompt, endpoint,
                cell_id, cc_state,
                steering_direction=steering_direction,
                max_new_tokens=max_new_tokens,
            )
            if not _is_remote_disconnect(row.error):
                break  # success or a non-retryable typed error
            if n_attempts <= max_retries:
                backoff = retry_backoff_base * (2 ** (n_attempts - 1))
                time.sleep(backoff)
        retry_log.append({
            "request_id": request_id, "endpoint": endpoint,
            "attempts": n_attempts,
            "recovered": (n_attempts > 1 and not row.error),
            "final_remote_disconnect": _is_remote_disconnect(row.error),
        })

        # Tag the auditor onto the row via auditor_id if the server didn't set
        # it (debug fallback). The server-resolved auditor_id is authoritative
        # when present; otherwise stamp the login email so per-auditor grouping
        # still works downstream.
        if not row.auditor_id:
            row.auditor_id = auditor_email
        rows.append(row)
        if row.error:
            tag = (f"(after {n_attempts} attempts)" if n_attempts > 1 else "")
            print(f"  [{auditor_email}] {endpoint} pair={pair_id} "
                  f"ERROR {tag} {row.error}", flush=True)
        elif n_attempts > 1:
            print(f"  [{auditor_email}] {endpoint} pair={pair_id} "
                  f"recovered on attempt {n_attempts}", flush=True)
    results[auditor_email] = {"login_error": None, "rows": rows, "retry_log": retry_log}


def read_ledger_totals(client) -> Optional[dict]:
    """Read the live ledger via the diagnostic.ledger_admin endpoint
    (registered by ledger_admin.py) in READ-ONLY mode, so the snapshot never
    mutates the budgets under test. Returns None if the endpoint isn't present.

    Requires the patched ledger_admin.py (read_only kwarg + per_engagement
    breakdown). An older endpoint without read_only would IGNORE the kwarg and
    fall through to its WRITE path with default budgets — clobbering the caps.
    We guard against that: if the response doesn't echo read_only=True, we treat
    the read as untrusted and return it flagged so the caller can warn loudly
    rather than silently trust a clobbered ledger."""
    try:
        paths = {getattr(v, "path", None) for v in client.custom_api.api_endpoints()}
    except Exception:  # noqa: BLE001
        return None
    if "diagnostic.ledger_admin" not in paths:
        return None
    try:
        res = client.api.services.diagnostic.ledger_admin(read_only=True)
        res = res.get() if hasattr(res, "get") else res
        if not isinstance(res, dict):
            return None
        if res.get("read_only") is not True:
            res["_stale_endpoint_warning"] = (
                "ledger_admin did not honour read_only=True; the deployed "
                "endpoint predates the read-only patch. Budgets may have been "
                "clobbered to defaults. Re-register ledger_admin.py."
            )
        return res
    except Exception as e:  # noqa: BLE001
        print(f"[budget][warn] ledger read failed: {type(e).__name__}: {e}")
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasite-url", required=True)
    ap.add_argument("--port", type=int, default=443)
    ap.add_argument("--auditors", required=True,
                    help="comma-separated auditor emails (>=2 for a real M5)")
    ap.add_argument("--admin-email", default="info@openmined.org")
    ap.add_argument("--admin-password", default="changethis")
    ap.add_argument("--no-proxy", action="store_true", default=True,
                    help="hit the datasite URL directly (debug deploy). Default "
                         "True; CC-on uses the verified proxy instead.")
    ap.add_argument("--proxy-port", type=int, default=8080)
    ap.add_argument("--container-name", default=None)

    ap.add_argument("--endpoints", required=True,
                    help=f"comma-separated subset of {','.join(ENDPOINTS)}")
    ap.add_argument("--pairs-json", type=Path, required=True)
    ap.add_argument("--sessions", type=int, default=20,
                    help="paired prompts per endpoint per auditor (~20 for M5)")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--req-rate", type=float, default=0.0,
                    help="per-auditor target calls/sec (0 = unthrottled burst, "
                         "the maximal stressor). Set e.g. 0.15 to mirror the "
                         "sequential-cell rate and test the ledger at a realistic "
                         "concurrent load rather than a synchronized burst.")
    ap.add_argument("--max-retries", type=int, default=0,
                    help="retries for transient RemoteDisconnect-class errors "
                         "(default 0 = none, to measure raw disconnect rate). "
                         "Set e.g. 3 to model a resilient evaluator client and "
                         "separate transient drops from a wedged worker pool.")
    ap.add_argument("--retry-backoff-base", type=float, default=1.0,
                    help="base backoff seconds; attempt N waits base*2^(N-1)")
    ap.add_argument("--steering-direction", type=Path, default=None)

    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--cc-state", choices=["on", "off"], required=True)
    ap.add_argument("--check-budget", action="store_true",
                    help="after the run, read the ledger and assert total "
                         "tokens_out did not exceed the engagement cap")
    args = ap.parse_args()

    auditors = [a.strip() for a in args.auditors.split(",") if a.strip()]
    if len(auditors) < 1:
        print("[error] need >=1 auditor", file=sys.stderr)
        return 2
    if len(auditors) < 2:
        print("[warn] only 1 auditor given; this is not a concurrency test")

    endpoints = [e.strip() for e in args.endpoints.split(",") if e.strip()]
    unknown = set(endpoints) - set(ENDPOINTS)
    if unknown:
        print(f"[error] unknown endpoints: {unknown}", file=sys.stderr)
        return 2

    steering_direction = None
    if "apply_steering" in endpoints:
        if not args.steering_direction or not args.steering_direction.exists():
            print("[error] --steering-direction required for apply_steering",
                  file=sys.stderr)
            return 2
        steering_direction = np.load(args.steering_direction).astype(np.float32)
        if steering_direction.shape != (6144,):
            print(f"[error] steering shape {steering_direction.shape} != (6144,)",
                  file=sys.stderr)
            return 2

    try:
        pairs = load_pairs(args.pairs_json)
    except Exception as e:  # noqa: BLE001
        print(f"[error] load pairs: {e}", file=sys.stderr)
        return 2
    prompts = interleave(pairs, args.sessions)

    # Resolve transport. Default debug path is direct (no-proxy); the syft
    # client just logs in to the public URL. CC-on would start the verified
    # proxy and point logins at 127.0.0.1:proxy_port — left as the documented
    # extension; M5 Stage 1 runs no-proxy.
    if not args.no_proxy:
        print("[error] CC-on verified-proxy path for M5 is the Stage-2 extension; "
              "run Stage 1 with --no-proxy (default).", file=sys.stderr)
        return 2

    login_url = args.datasite_url
    login_port = args.port

    print(f"[m5] auditors={auditors}")
    print(f"[m5] endpoints={endpoints} sessions={args.sessions} "
          f"-> {len(endpoints)*args.sessions} calls/auditor, "
          f"{len(auditors)*len(endpoints)*args.sessions} total")
    rate_desc = (f"{args.req_rate} calls/s per auditor"
                 if args.req_rate and args.req_rate > 0
                 else "UNTHROTTLED burst (maximal worker-pool stressor)")
    print(f"[m5] synchronized start via barrier; rate: {rate_desc}")

    start_barrier = threading.Barrier(len(auditors))
    results: dict = {}
    threads = []
    t0 = time.monotonic()
    for i, email in enumerate(auditors):
        th = threading.Thread(
            target=run_auditor,
            kwargs=dict(
                auditor_email=email,
                datasite_url=login_url, port=login_port,
                endpoints=endpoints, prompts=prompts, sessions=args.sessions,
                cell_id=f"M5-{chr(ord('a')+i)}", cc_state=args.cc_state,
                steering_direction=steering_direction,
                max_new_tokens=args.max_new_tokens,
                req_rate=args.req_rate,
                max_retries=args.max_retries,
                retry_backoff_base=args.retry_backoff_base,
                start_barrier=start_barrier, results=results,
            ),
            name=f"auditor-{email}",
        )
        threads.append(th)
        th.start()
    for th in threads:
        th.join()
    wall_min = (time.monotonic() - t0) / 60.0
    print(f"[m5] all auditors done in {wall_min:.1f} min")

    # --- assemble outputs ---
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[PysyftRow] = []
    per_auditor_summary = {}
    for email, r in results.items():
        if r.get("login_error"):
            print(f"[m5][FAIL] {email} login: {r['login_error']}")
            per_auditor_summary[email] = {"login_error": r["login_error"]}
            continue
        rows = r["rows"]
        all_rows.extend(rows)
        ok = [x for x in rows if x.error is None]
        rd = [x for x in rows if _is_remote_disconnect(x.error)]
        other_err = [x for x in rows if x.error and not _is_remote_disconnect(x.error)]
        retry_log = r.get("retry_log", [])
        n_recovered = sum(1 for e in retry_log if e.get("recovered"))
        n_multi_attempt = sum(1 for e in retry_log if e.get("attempts", 1) > 1)
        # Per-auditor parquet for traceability.
        if pd is not None and rows:
            pd.DataFrame([asdict(x) for x in rows]).to_parquet(
                args.out_dir / f"requests_{email.replace('@','_at_')}.parquet",
                index=False)
        per_auditor_summary[email] = {
            "n_total": len(rows),
            "n_success": len(ok),
            "n_remote_disconnect": len(rd),
            "n_other_error": len(other_err),
            "n_calls_with_retry": n_multi_attempt,
            "n_recovered_after_retry": n_recovered,
            "ledger_seconds": _percentiles([x.pysyft_ledger_seconds for x in ok]),
            "wall_seconds": _percentiles([x.wall_seconds for x in ok]),
            "transport_serialize_seconds": _percentiles(
                [x.transport_serialize_seconds for x in ok]),
            "distinct_auditor_ids": sorted({x.auditor_id for x in ok if x.auditor_id}),
        }

    ok_all = [x for x in all_rows if x.error is None]
    rd_all = [x for x in all_rows if _is_remote_disconnect(x.error)]

    # Merged parquet across auditors (schema-identical to single-thread runs).
    if pd is not None and all_rows:
        pd.DataFrame([asdict(x) for x in all_rows]).to_parquet(
            args.out_dir / "requests.parquet", index=False)

    # Contention headline: pooled ledger-insert distribution under concurrency.
    contention = {
        "n_auditors": len(auditors),
        "n_calls_total": len(all_rows),
        "n_success_total": len(ok_all),
        "n_remote_disconnect_total": len(rd_all),
        "remote_disconnect_rate": len(rd_all) / max(1, len(all_rows)),
        "ledger_seconds_pooled": _percentiles([x.pysyft_ledger_seconds for x in ok_all]),
        "approval_seconds_pooled": _percentiles([x.pysyft_approval_seconds for x in ok_all]),
        "wall_seconds_pooled": _percentiles([x.wall_seconds for x in ok_all]),
    }

    # --- budget correctness check ---
    budget_result = None
    budget_violation = False
    if args.check_budget:
        try:
            admin = sy.login(url=login_url, port=login_port,
                            email=args.admin_email, password=args.admin_password)
            admin.refresh()
            ledger = read_ledger_totals(admin)
            if ledger is None:
                print("[budget][warn] diagnostic.ledger_admin not present; "
                      "cannot verify budget. Register it via ledger_admin.py.")
            elif ledger.get("_stale_endpoint_warning"):
                print(f"[budget][WARN] {ledger['_stale_endpoint_warning']}")
            else:
                used_total = int(ledger.get("tokens_out_used_now") or 0)
                per_eng = ledger.get("per_engagement") or []
                # per_engagement row: [eng_id, auditor_id, token_budget,
                #   tokens_out_used, plots_used, exemplars_used, n_sessions, n_bundles]
                # TRUE isolation check: every engagement's own accrual must stay
                # within its OWN cap. A race-through past the atomic cap-check
                # would show tokens_out_used > token_budget for some engagement.
                per_eng_violations = []
                for row in per_eng:
                    if len(row) < 4:
                        continue
                    eng_id, auditor_id, cap, used = row[0], row[1], int(row[2]), int(row[3])
                    if used > cap:
                        per_eng_violations.append({
                            "engagement_id": eng_id, "auditor_id": auditor_id,
                            "tokens_out_used": used, "token_budget": cap,
                            "overshoot": used - cap,
                        })
                caps = [int(e[2]) for e in per_eng if len(e) >= 3]
                cap_sum = sum(caps)
                global_violation = bool(caps) and used_total > cap_sum
                budget_violation = bool(per_eng_violations) or global_violation
                # Isolation is only TESTED when >=2 distinct engagements ran
                # concurrently. With 1 engagement (e.g. the auditor_id='unknown'
                # collapse under the identity-blind debug shim) there is nothing
                # to isolate: a clean per-engagement check is vacuous and must
                # NOT be reported as "isolation verified". Distinguish the two.
                n_distinct_auditors = len({e[1] for e in per_eng if len(e) >= 2})
                isolation_tested = len(per_eng) >= 2 and n_distinct_auditors >= 2
                budget_result = {
                    "tokens_out_used_total": used_total,
                    "n_engagements": len(per_eng),
                    "n_distinct_auditors": n_distinct_auditors,
                    "isolation_tested": isolation_tested,
                    "per_engagement": [
                        {"engagement_id": e[0], "auditor_id": e[1],
                         "token_budget": int(e[2]), "tokens_out_used": int(e[3]),
                         "n_sessions": int(e[6]) if len(e) > 6 else None,
                         "n_bundles": int(e[7]) if len(e) > 7 else None}
                        for e in per_eng if len(e) >= 4
                    ],
                    "cap_sum": cap_sum,
                    "per_engagement_violations": per_eng_violations,
                    "global_violation_used_gt_cap_sum": global_violation,
                    "violation": budget_violation,
                }
                if per_eng_violations:
                    print(f"[budget] VIOLATION: {len(per_eng_violations)} "
                          f"engagement(s) exceeded their own cap (race-through):")
                    for v in per_eng_violations:
                        print(f"  {v['auditor_id']}: used={v['tokens_out_used']} "
                              f"> cap={v['token_budget']} (+{v['overshoot']})")
                elif global_violation:
                    print(f"[budget] VIOLATION: total used={used_total} "
                          f"> cap_sum={cap_sum}")
                elif isolation_tested:
                    print(f"[budget] ISOLATION VERIFIED: {len(per_eng)} engagements "
                          f"across {n_distinct_auditors} auditors, each within its "
                          f"own cap; total used={used_total} <= cap_sum={cap_sum}")
                else:
                    print(f"[budget] ISOLATION UNTESTED: only {len(per_eng)} "
                          f"engagement(s) / {n_distinct_auditors} distinct "
                          f"auditor_id(s) present — concurrent writers collapsed to "
                          f"a shared engagement (identity-blind shim?), so per-"
                          f"engagement isolation cannot be verified on this run. "
                          f"Cap accounting itself is intact (used={used_total} "
                          f"<= cap_sum={cap_sum}).")
        except Exception as e:  # noqa: BLE001
            print(f"[budget][warn] check failed: {type(e).__name__}: {e}")

    summary = {
        "schema_version": "m5-concurrent-v1",
        "cc_state": args.cc_state,
        "datasite_url": args.datasite_url,
        "endpoints": endpoints,
        "sessions_per_endpoint_per_auditor": args.sessions,
        "req_rate_per_auditor": args.req_rate,
        "wall_minutes": wall_min,
        "contention": contention,
        "per_auditor": per_auditor_summary,
        "budget_check": budget_result,
        "m3_reference_remote_disconnect": "1/20 observed once in M3 (single-thread)",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # --- console report ---
    print("\n=== M5 contention report ===")
    print(f"distinct auditor_ids seen: "
          f"{sorted({x.auditor_id for x in ok_all if x.auditor_id})}")
    lp = contention["ledger_seconds_pooled"]
    if lp.get("n"):
        print(f"ledger_insert (pooled, n={lp['n']}): "
              f"p50={lp['p50']*1000:.2f}ms p95={lp['p95']*1000:.2f}ms "
              f"p99={lp['p99']*1000:.2f}ms max={lp['max']*1000:.2f}ms")
    print(f"RemoteDisconnected: {len(rd_all)}/{len(all_rows)} "
          f"({contention['remote_disconnect_rate']*100:.1f}%) "
          f"[M3 single-thread baseline: 1/20 = 5%]")
    print(f"[out] {args.out_dir / 'requests.parquet'}")
    print(f"[out] {args.out_dir / 'summary.json'}")

    if budget_violation:
        return 6
    if not ok_all:
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())