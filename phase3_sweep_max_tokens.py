#!/usr/bin/env python3
"""
phase3_sweep_max_tokens.py — single-deploy max_tokens sweep (Tier 1A).

Amortises model-load cost (~25 min on GLM-5.1-FP8) across N max_tokens
values by keeping ONE deploy alive for the whole sweep. Two invocations
(one CC-off, one CC-on) produce the full 10-cell dataset behind the
"CC overhead vs output length" plot.

The matrix orchestrator (phase3_run_matrix.py) deploys-and-tears-down per
cell, so running the sweep through it would pay model load 10 times for
the same data. This script pays it twice (once per CC state). Net saving
at the default 5-point sweep: ~3-4 hours.

Each (max_tokens) iteration writes a per-virtual-cell artifact set to
`<out-dir>/<cell-id>/{requests.parquet, summary.json}`, with the same
schema phase3_run_cell.py produces, so phase3_aggregate.py can read each
as a virtual cell with no code changes.

For cross-iteration analysis a sweep-level manifest also lands at
`<out-dir>/<sweep_run_id>.json` summarising all iterations.

Coupling note: this script is debug-mode-only and has zero dependencies on
the Tier 1B patches. It deploys with the existing hardcoded --debug flag
in phase3_run_matrix.deploy_target and constructs URLs via the local
_target_base_url_debug() helper below (matches the existing
phase3_run_matrix._target_endpoint format). If you later apply Tier 1B
and want to sweep against an attestation-enabled deploy, switch the
helper to orch.target_base_url(..., debug=False) and reintroduce a
--debug/--no-debug CLI flag.

Usage (CC-off branch — run first):

  python phase3_sweep_max_tokens.py \\
      --matrix phase3-matrix.yaml \\
      --cc-state off \\
      --max-tokens 32 128 512 1024 2048 \\
      --n-requests 100 100 50 50 50 \\
      --out-dir runs/phase3 \\
      --cell-id-prefix C1

  # Then CC-on branch (separate invocation = separate deploy = required by
  # PHASE2_REFERENCE §9: CC mode is per-deployment, can't be flipped on a
  # running container).
  python phase3_sweep_max_tokens.py \\
      --matrix phase3-matrix.yaml \\
      --cc-state on \\
      --max-tokens 32 128 512 1024 2048 \\
      --n-requests 100 100 50 50 50 \\
      --out-dir runs/phase3 \\
      --cell-id-prefix C1

Exit codes:
  0  all iterations completed successfully
  1  partial: at least one iteration produced zero successes; others ok
  2  user error (bad CLI, bad YAML, length mismatch)
  4  deploy / status / health failed (no measurements possible)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pandas as pd  # type: ignore
except ImportError:
    pd = None

try:
    import boto3  # type: ignore
except ImportError:
    boto3 = None

# Resolve sibling imports the same way phase3_run_cell.py does.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import phase3_run_matrix as orch     # noqa: E402
import phase3_vllm_driver as vllm_drv  # noqa: E402
from openai import OpenAI            # noqa: E402


SWEEP_SCHEMA_VERSION = "phase3-sweep-max-tokens-v1"


# ===== local URL helper =====================================================
# Inlined to keep this script independent of Tier 1B patches in
# phase3_run_matrix.py. Matches the existing _target_endpoint() format
# verified against docs.tinfoil.sh/containers/debug-mode (2026-05-20):
# debug-mode FQDNs use the `.debug.` subdomain.
# If/when Tier 1B lands and you want a non-debug sweep, switch the call
# sites to orch.target_base_url(..., debug=False).

def _target_base_url_debug(cell_id: str, org_subdomain: str, has_v1: bool) -> str:
    base = f"https://{cell_id.lower()}-target.debug.{org_subdomain}.containers.tinfoil.dev"
    return f"{base}/v1" if has_v1 else base


# ===== vllm-bench-aligned aggregates ========================================
# Lifted verbatim from phase3_run_cell.add_vllm_bench_aligned_fields. Copied
# rather than imported so this script doesn't pull in the four driver modules
# that phase3_run_cell imports at top level (gradient sidecar deps etc.).
# Any change to the canonical version in phase3_run_cell.py should be mirrored
# here, but the surface is small and stable.

def _safe_get(row: Any, key: str) -> Any:
    if isinstance(row, dict):
        return row.get(key)
    return getattr(row, key, None)


def add_vllm_bench_aligned_fields(summary: dict, rows: list[dict]) -> dict:
    ok = [r for r in rows if not _safe_get(r, "error")]
    if not ok:
        derived = {"n_success": 0, "note": "no successful requests; no aggregates"}
        summary["vllm_bench_aligned"] = derived
        return derived

    t_sends = [float(_safe_get(r, "t_send") or 0.0) for r in ok]
    t_ends  = [float(_safe_get(r, "t_complete") or 0.0) for r in ok]
    run_wall = max(t_ends) - min(t_sends) if t_sends and t_ends else 0.0
    walls   = [float(_safe_get(r, "wall_seconds") or 0.0) for r in ok]
    tok_in  = [int(_safe_get(r, "tokens_in") or 0) for r in ok]
    tok_out = [int(_safe_get(r, "tokens_out") or 0) for r in ok]
    walls_ms = np.asarray(walls, dtype=np.float64) * 1000.0

    derived: dict[str, Any] = {
        "schema_version": SWEEP_SCHEMA_VERSION,
        "n_success": len(ok),
        "run_wall_seconds": run_wall,
        "mean_e2el_ms":   float(walls_ms.mean()),
        "median_e2el_ms": float(np.median(walls_ms)),
        "p99_e2el_ms":    float(np.percentile(walls_ms, 99)),
        "max_e2el_ms":    float(walls_ms.max()),
        "total_input_tokens":     int(sum(tok_in)),
        "total_generated_tokens": int(sum(tok_out)),
        # TTFT/ITL require streaming — left None for parity with sequential cells.
        "mean_ttft_ms": None,
        "median_ttft_ms": None,
        "p99_ttft_ms": None,
        "mean_itl_ms": None,
    }
    if run_wall > 0:
        derived["request_throughput_req_per_s"] = len(ok) / run_wall
        derived["output_token_throughput_tok_per_s"] = sum(tok_out) / run_wall
        derived["total_token_throughput_tok_per_s"] = (sum(tok_in) + sum(tok_out)) / run_wall
    else:
        derived["request_throughput_req_per_s"] = None
        derived["output_token_throughput_tok_per_s"] = None
        derived["total_token_throughput_tok_per_s"] = None
    summary["vllm_bench_aligned"] = derived
    return derived


# ===== single iteration =====================================================

def run_one_iteration(
    client: OpenAI,
    matrix: dict,
    cell: dict,
    max_new_tokens: int,
    n_requests: int,
    cell_id: str,
    pairs: list[dict],
    out_dir: Path,
    req_rate: float,
) -> tuple[int, dict]:
    """Drive ONE max_tokens point against an already-warm deploy.
    Returns (n_success, summary_dict)."""
    img = matrix["images"][cell["image"]]
    prompts = vllm_drv.interleave(pairs, n_requests)
    # Sweep is baseline-only (no instrumentation payload); xargs is {}.
    preset = vllm_drv.CONDITION_PRESETS["baseline"]
    xargs = vllm_drv.preset_to_xargs(preset)

    iter_out = out_dir / cell_id
    iter_out.mkdir(parents=True, exist_ok=True)

    print(f"\n--- {cell_id}: max_tokens={max_new_tokens}, N={len(prompts)}, "
          f"req_rate={req_rate} ---")
    t0 = time.monotonic()
    rows = vllm_drv.run_cell(
        client, img["model"], prompts,
        xargs=xargs, max_new_tokens=max_new_tokens, req_rate=req_rate,
    )
    elapsed_min = (time.monotonic() - t0) / 60.0
    print(f"--- {cell_id}: done in {elapsed_min:.1f} min ---")

    target_base = _target_base_url_debug(
        cell["cell_id"], matrix["org_subdomain"], img["has_v1"],
    )
    summary = vllm_drv.summarize(
        rows,
        cell_id=cell_id,
        condition="baseline",
        cc_state=cell["cc_state"],
        base_url=target_base,
        image_digest=img["digest"],
        req_rate=req_rate,
        n_requests_target=n_requests,
        xargs=xargs,
    )
    # Sweep-specific metadata. max_new_tokens is the lever; tokens_out_p50
    # (already inside summary["overall"]["tokens_out"]["p50"]) is the
    # realised x-axis for the headline plot.
    summary["max_new_tokens"] = max_new_tokens
    summary["sweep_run_id"]   = cell["sweep_run_id"]
    summary["image"]          = cell["image"]
    summary["driver"]         = "sequential"

    vllm_drv.write_outputs(rows, summary, iter_out)

    # vllm-bench-aligned post-processing (matches phase3_run_cell behaviour).
    rows_dict = [asdict(r) for r in rows]
    add_vllm_bench_aligned_fields(summary, rows_dict)
    (iter_out / "summary.json").write_text(json.dumps(summary, indent=2))

    wall_p50    = summary["overall"]["wall_seconds"].get("p50", float("nan"))
    tok_out_p50 = summary["overall"]["tokens_out"].get("p50", float("nan"))
    print(f"[{cell_id}] n_success={summary['n_success']}/{summary['n_total']} "
          f"wall_p50={wall_p50:.2f}s  tokens_out_p50={tok_out_p50:.0f}")
    return summary["n_success"], summary


def upload_iteration(out_dir: Path, cell_id: str) -> None:
    """R2/S3 upload for one iteration's artifacts. No-op if creds absent."""
    bucket = os.environ.get("S3_BUCKET")
    endpoint_url = os.environ.get("R2_ENDPOINT_URL") or os.environ.get("R2_ENDPOINT")
    if not bucket or boto3 is None:
        return
    kwargs = {"endpoint_url": endpoint_url} if endpoint_url else {}
    s3 = boto3.client("s3", **kwargs)
    backend = "r2" if endpoint_url else "s3"
    for fname in ("requests.parquet", "requests.jsonl", "summary.json"):
        p = out_dir / cell_id / fname
        if not p.exists():
            continue
        key = f"phase3/{cell_id}/{fname}"
        try:
            s3.upload_file(str(p), bucket, key)
            print(f"  [upload] {backend}://{bucket}/{key}")
        except Exception as e:
            print(f"  [upload] FAILED {key}: {type(e).__name__}: {e}",
                  file=sys.stderr)


# ===== top-level sweep ======================================================

def sweep(args: argparse.Namespace) -> int:
    try:
        matrix = orch.load_and_validate_matrix(args.matrix)
    except Exception as e:
        print(f"[error] matrix: {e}", file=sys.stderr)
        return 2
    if args.image not in matrix["images"]:
        print(f"[error] image {args.image!r} not in matrix.images "
              f"(have {list(matrix['images'])})", file=sys.stderr)
        return 2
    img = matrix["images"][args.image]

    # N per max_tokens: either one value (broadcast) or per-position.
    if len(args.n_requests) == 1:
        n_per = [args.n_requests[0]] * len(args.max_tokens)
    elif len(args.n_requests) == len(args.max_tokens):
        n_per = list(args.n_requests)
    else:
        print(f"[error] --n-requests must have 1 value or match --max-tokens "
              f"length (got {len(args.n_requests)} vs {len(args.max_tokens)})",
              file=sys.stderr)
        return 2

    sweep_run_id = f"sweep-{args.cell_id_prefix.lower()}-{args.cc_state}-{int(time.time())}"
    print(f"[sweep] run_id={sweep_run_id}")
    print(f"[sweep] cc={args.cc_state}  image={args.image}  (debug-mode only)")
    print(f"[sweep] max_tokens={args.max_tokens}")
    print(f"[sweep] n_per     ={n_per}")
    print(f"[sweep] req_rate  ={args.req_rate}")
    print(f"[sweep] out_dir   ={args.out_dir}")

    # Synthetic deploy cell. The phase3_run_matrix helpers use this dict
    # for URL construction + deploy/teardown invocations. We intentionally
    # do NOT set a `debug` key — the unpatched deploy_target hardcodes
    # --debug, and the local URL helper hardcodes the .debug. subdomain.
    deploy_cell_id = f"{args.cell_id_prefix.lower()}-{args.cc_state}-sweep"
    cell = {
        "cell_id":   deploy_cell_id,
        "condition": "baseline",
        "cc_state":  args.cc_state,
        "image":     args.image,
        "req_rate":  args.req_rate,
        "sweep_run_id": sweep_run_id,
    }

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    deploy_args = Namespace(dry_run=args.dry_run)

    # ---- 1. deploy ----
    print(f"\n[1/5] deploy {deploy_cell_id}-target")
    if not orch.deploy_target(matrix, cell, deploy_args):
        print("[fail] deploy_target returned non-zero", file=sys.stderr)
        return 4

    # ---- 2. status=ready ----
    print(f"\n[2/5] wait for container status=ready")
    status_ok = orch.wait_for_target_status_ready(
        deploy_cell_id, deploy_args,
        timeout=matrix["defaults"]["health_timeout"],
    )
    if not status_ok:
        # Debug-mode only sweep: no non-debug fallback path. If status times
        # out, bail. If you need to recover, SSH in via debug mode and check
        # `docker ps` on the target host CVM.
        print("[fail] status_ready timed out", file=sys.stderr)
        orch.delete_target(deploy_cell_id, deploy_args)
        return 4

    # ---- 3. /health ----
    print(f"\n[3/5] wait for /health")
    if not orch.wait_for_target_health(cell, matrix, deploy_args):
        print("[fail] /health did not become ready", file=sys.stderr)
        orch.delete_target(deploy_cell_id, deploy_args)
        return 4

    # dry-run stops here: we've validated deploy/status/health command shape.
    if args.dry_run:
        print("\n[dry-run] stopping before driver calls and teardown.")
        return 0

    # ---- 4. construct client + warmup ----
    base_url = _target_base_url_debug(
        deploy_cell_id, matrix["org_subdomain"], img["has_v1"],
    )
    api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
    if api_key == "EMPTY":
        print("[warn] VLLM_API_KEY not set; /v1 calls will get 401 from the "
              "Tinfoil shim. Set to your tk_* tenant key before running.",
              file=sys.stderr)
    client = OpenAI(
        base_url=base_url, api_key=api_key,
        max_retries=0, timeout=matrix["defaults"]["timeout"],
    )

    pairs_path = Path(matrix["benchbox"]["pairs_json"])
    try:
        pairs = vllm_drv.load_pairs(pairs_path)
    except Exception as e:
        print(f"[fail] load pairs from {pairs_path}: {e}", file=sys.stderr)
        orch.delete_target(deploy_cell_id, deploy_args)
        return 4
    if len(pairs) < 2:
        print(f"[fail] pairs.json must have >=2 entries for warmup; "
              f"got {len(pairs)}", file=sys.stderr)
        orch.delete_target(deploy_cell_id, deploy_args)
        return 4

    # Warmup with the LAST pair (toxic + benign) so we don't prefix-cache-prime
    # the prompts used in the first measured iteration. With temperature=0 and
    # vLLM's APC, repeat prompts can show artificially low wall — keep warmup
    # disjoint from measurement.
    print(f"\n[4/5] warmup (2 throwaway requests at max_tokens="
          f"{min(args.max_tokens)})")
    last = pairs[-1]
    warm_prompts = [
        (int(last["pair_id"]), "toxic",  last["toxic"]),
        (int(last["pair_id"]), "benign", last["benign"]),
    ]
    _ = vllm_drv.run_cell(
        client, img["model"], warm_prompts,
        xargs={}, max_new_tokens=min(args.max_tokens), req_rate=args.req_rate,
    )

    # ---- 5. iterate max_tokens ----
    iter_summaries: list[dict] = []
    iter_ok: list[bool] = []
    for max_tok, n in zip(args.max_tokens, n_per):
        cell_id = f"{args.cell_id_prefix}-{args.cc_state}-t{max_tok}"
        try:
            n_succ, summary = run_one_iteration(
                client, matrix, cell, max_tok, n, cell_id,
                pairs, out_dir, args.req_rate,
            )
            iter_summaries.append({
                "cell_id":        cell_id,
                "max_new_tokens": max_tok,
                "n_requests":     n,
                "n_success":      n_succ,
                "wall_p50":       summary["overall"]["wall_seconds"].get("p50"),
                "wall_p95":       summary["overall"]["wall_seconds"].get("p95"),
                "tokens_out_p50": summary["overall"]["tokens_out"].get("p50"),
                "tokens_out_p95": summary["overall"]["tokens_out"].get("p95"),
            })
            iter_ok.append(n_succ > 0)
            if not args.no_upload:
                upload_iteration(out_dir, cell_id)
        except Exception as e:
            print(f"[iter-fail] {cell_id}: {type(e).__name__}: {e}",
                  file=sys.stderr)
            iter_summaries.append({
                "cell_id": cell_id, "max_new_tokens": max_tok,
                "error": f"{type(e).__name__}: {e}",
            })
            iter_ok.append(False)
            # Continue — don't lose data from later (or already-completed)
            # iterations on a single transient failure.

    # ---- 6. sweep manifest + teardown ----
    sweep_report = {
        "schema_version": SWEEP_SCHEMA_VERSION,
        "sweep_run_id":   sweep_run_id,
        "cc_state":       args.cc_state,
        "image":          args.image,
        "image_digest":   img["digest"],
        "req_rate":       args.req_rate,
        "cell_id_prefix": args.cell_id_prefix,
        "iterations":     iter_summaries,
    }
    sweep_path = out_dir / f"{sweep_run_id}.json"
    sweep_path.write_text(json.dumps(sweep_report, indent=2))
    print(f"\n[sweep] manifest written: {sweep_path}")

    print(f"\n[5/5] teardown {deploy_cell_id}-target")
    orch.delete_target(deploy_cell_id, deploy_args)

    n_ok    = sum(iter_ok)
    n_total = len(iter_ok)
    print(f"\n[sweep] DONE: {n_ok}/{n_total} iterations with >=1 success")
    if n_ok == 0:
        return 4
    if n_ok < n_total:
        return 1
    return 0


# ===== CLI ==================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--matrix", type=Path, required=True,
                   help="phase3-matrix.yaml (used for org_subdomain, images, "
                        "defaults — image entry selected via --image).")
    p.add_argument("--cc-state", choices=["on", "off"], required=True,
                   help="CC state of the single deploy used for this sweep.")
    p.add_argument("--image", default="vllm",
                   help="Image key in matrix.images. Default 'vllm' (GLM-5.1).")
    p.add_argument("--max-tokens", nargs="+", type=int,
                   default=[32, 128, 512, 1024, 2048],
                   help="max_new_tokens values to sweep. "
                        "Default: 32 128 512 1024 2048.")
    p.add_argument("--n-requests", nargs="+", type=int,
                   default=[100, 100, 50, 50, 50],
                   help="N per max_tokens. Either 1 value (broadcast) or "
                        "one per max_tokens. Default: 100 100 50 50 50.")
    p.add_argument("--req-rate", type=float, default=1.0,
                   help="Sequential send-to-send rate cap (req/s). Default 1.0.")
    p.add_argument("--cell-id-prefix", default="C1",
                   help="Prefix for per-iteration cell_ids "
                        "(e.g. 'C1' → 'C1-off-t128'). Default 'C1'.")
    p.add_argument("--out-dir", type=Path, default=Path("runs/phase3"),
                   help="Output root; per-iter writes go to <out>/<cell-id>/.")
    p.add_argument("--no-upload", action="store_true",
                   help="Skip R2/S3 upload of per-iteration artifacts.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print deploy/status/health command shapes but stop "
                        "before issuing any driver calls. No teardown.")
    return p.parse_args()


def main() -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass
    args = parse_args()
    return sweep(args)


if __name__ == "__main__":
    sys.exit(main())
