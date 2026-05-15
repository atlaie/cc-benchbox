#!/usr/bin/env python3
"""
make_fake_cells.py — synthesize realistic-shaped Phase 3 cell artifacts.

Generates per-cell `requests.parquet` + `summary.json` (and a fake vllm-bench
reference JSON for cells with `vllm_bench_reference: true`) so we can run
`phase3_aggregate.py` end-to-end without real infrastructure.

CC-on cells get ~15% wall overhead and ~5% payload overhead vs their CC-off
counterparts, so the aggregator's delta tables produce meaningful numbers.

Usage:

    python make_fake_cells.py --matrix phase3-matrix.yaml --out-dir runs/phase3/cache
    python phase3_aggregate.py --matrix phase3-matrix.yaml --local-only \\
        --phase2-dir ../cc-deep-eval/runs/phase2_validation

Then inspect runs/phase3/aggregate.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Per-condition baseline profile (CC-off values).
# Tuned against Phase 2 medians: ~9s wall for instrumented, ~12s for gradient.
CONDITION_PROFILE = {
    "baseline":    {"wall_p50": 9.30, "wall_sigma": 0.20, "payload_p50": 3_000,    "payload_sigma": 200,   "tokens_in": 250, "tokens_out": 32},
    "routing":     {"wall_p50": 9.50, "wall_sigma": 0.25, "payload_p50": 8_500,    "payload_sigma": 600,   "tokens_in": 250, "tokens_out": 32},
    "repe_bundle": {"wall_p50": 9.80, "wall_sigma": 0.30, "payload_p50": 12_900,   "payload_sigma": 800,   "tokens_in": 250, "tokens_out": 32},
    "gradient":    {"wall_p50": 12.30, "wall_sigma": 0.40, "payload_p50": 25_000,  "payload_sigma": 1500,  "tokens_in": 250, "tokens_out": 0},
}

CC_ON_WALL_OVERHEAD = 0.15      # +15% wall under CC-on
CC_ON_PAYLOAD_OVERHEAD = 0.05   # +5% payload under CC-on (encryption framing)


def synth_requests(cell: dict, profile: dict, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Build n synthetic per-request rows for one cell."""
    is_on = cell["cc_state"] == "on"
    wall_mu = profile["wall_p50"] * (1 + CC_ON_WALL_OVERHEAD if is_on else 1)
    wall_sigma = profile["wall_sigma"]
    pl_mu = profile["payload_p50"] * (1 + CC_ON_PAYLOAD_OVERHEAD if is_on else 1)
    pl_sigma = profile["payload_sigma"]
    req_rate = cell["req_rate"]
    interval = 1.0 / req_rate

    t0 = time.time()
    walls = rng.normal(wall_mu, wall_sigma, size=n).clip(0.5)
    payloads = rng.normal(pl_mu, pl_sigma, size=n).clip(0).astype(int)
    tokens_in = (rng.normal(profile["tokens_in"], 30, size=n).clip(50)).astype(int)
    tokens_out = np.full(n, profile["tokens_out"], dtype=int)

    rows = []
    for i in range(n):
        t_send = t0 + i * interval
        wall = float(walls[i])
        t_complete = t_send + wall
        cls = "toxic" if i % 2 == 0 else "benign"
        row = {
            "request_id": i,
            "pair_id": i // 2,
            "prompt_class": cls,
            "t_send": t_send,
            "t_first_token": t_send,    # non-streaming, set equal to send
            "t_complete": t_complete,
            "wall_seconds": wall,
            "tokens_in": int(tokens_in[i]),
            "tokens_out": int(tokens_out[i]),
            "payload_bytes": int(payloads[i]),
            "http_status": 200,
            "error": None,
        }
        if cell["condition"] == "gradient":
            # Gradient-specific schema fields
            row["fwd_seconds"] = max(float(rng.normal(wall * 0.83, 0.1)), 0.1)
            row["bwd_seconds"] = max(float(rng.normal(wall * 0.17, 0.05)), 0.05)
            row["server_total_seconds"] = row["fwd_seconds"] + row["bwd_seconds"]
            row["loss"] = float(rng.normal(2.5, 0.3))
            row["target_token_id"] = int(rng.integers(1000, 100000))
            row["target_token"] = f"tok_{row['target_token_id']}"
        else:
            row["completion_text"] = f"<fake completion for request {i}>"
        rows.append(row)

    return pd.DataFrame(rows)


def build_summary(cell: dict, df: pd.DataFrame, image_digest: str) -> dict:
    """Match the shape phase3_run_cell.py produces (driver summary + vllm_bench_aligned)."""
    ok = df[df["error"].isna()]
    n_total = len(df)
    n_success = len(ok)
    walls = ok["wall_seconds"].to_numpy()
    payloads = ok["payload_bytes"].to_numpy()

    t_sends = ok["t_send"].to_numpy()
    t_completes = ok["t_complete"].to_numpy()
    run_wall = float(t_completes.max() - t_sends.min())

    tok_in_total = int(ok["tokens_in"].sum())
    tok_out_total = int(ok["tokens_out"].sum())

    summary = {
        "schema_version": "phase3-driver-v1-fake",
        "cell_id": cell["cell_id"],
        "condition": cell["condition"],
        "cc_state": cell["cc_state"],
        "base_url": f"https://{cell['cell_id'].lower()}-target.debug.fake.containers.tinfoil.dev",
        "image_digest": image_digest,
        "req_rate": cell["req_rate"],
        "n_requests_target": n_total,
        "n_total": n_total,
        "n_success": n_success,
        "success_rate": n_success / n_total if n_total else 0.0,
        "wall_seconds": {
            "p50": float(np.percentile(walls, 50)),
            "p95": float(np.percentile(walls, 95)),
            "max": float(walls.max()),
        },
        "payload_bytes": {
            "p50": float(np.percentile(payloads, 50)),
            "p95": float(np.percentile(payloads, 95)),
        },
        "vllm_bench_aligned": {
            "schema_version": "phase3-run-cell-v1",
            "n_success": n_success,
            "run_wall_seconds": run_wall,
            "mean_e2el_ms": float(walls.mean()) * 1000,
            "median_e2el_ms": float(np.median(walls)) * 1000,
            "p99_e2el_ms": float(np.percentile(walls, 99)) * 1000,
            "max_e2el_ms": float(walls.max()) * 1000,
            "total_input_tokens": tok_in_total,
            "total_generated_tokens": tok_out_total,
            "request_throughput_req_per_s": n_success / run_wall if run_wall > 0 else None,
            "output_token_throughput_tok_per_s": tok_out_total / run_wall if run_wall > 0 else None,
            "total_token_throughput_tok_per_s": (tok_in_total + tok_out_total) / run_wall if run_wall > 0 else None,
            "mean_ttft_ms": None,
            "median_ttft_ms": None,
            "p99_ttft_ms": None,
            "mean_itl_ms": None,
        },
    }
    return summary


def build_vllm_bench_reference(cell: dict, summary: dict) -> dict:
    """Produce a fake vllm-bench-style result JSON for C1 cells.
    Numbers should be within 5% of driver's aligned metrics so the
    cross-check in phase3_aggregate.py looks reasonable.
    """
    aligned = summary["vllm_bench_aligned"]
    # Simulate small driver-vs-bench delta (~2%)
    factor = 1.02
    return {
        "date": "2026-05-15",
        "backend": "openai-chat",
        "model_id": "glm-5-1-fp8",
        "tokenizer_id": "glm-5-1-fp8",
        "num_prompts": summary["n_success"],
        "completed": summary["n_success"],
        "request_throughput": aligned["request_throughput_req_per_s"] / factor,
        "output_throughput": aligned["output_token_throughput_tok_per_s"] / factor,
        "total_token_throughput": aligned["total_token_throughput_tok_per_s"] / factor,
        "mean_ttft_ms": 72.5,
        "median_ttft_ms": 71.8,
        "p99_ttft_ms": 89.2,
        "mean_tpot_ms": 280.4,
        "mean_itl_ms": 280.4,
        "mean_e2el_ms": aligned["mean_e2el_ms"] / factor,
        "median_e2el_ms": aligned["median_e2el_ms"] / factor,
        "p99_e2el_ms": aligned["p99_e2el_ms"] / factor,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("runs/phase3/cache"))
    ap.add_argument("--n-per-cell", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    matrix = yaml.safe_load(args.matrix.read_text())
    cells = matrix["cells"]
    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for cell in cells:
        cell_id = cell["cell_id"]
        cond = cell["condition"]
        profile = CONDITION_PROFILE[cond]
        cell_dir = args.out_dir / cell_id
        cell_dir.mkdir(parents=True, exist_ok=True)

        df = synth_requests(cell, profile, args.n_per_cell, rng)
        df.to_parquet(cell_dir / "requests.parquet", index=False)

        # Determine image_digest from matrix's images map
        img_name = cell["image"]
        digest = matrix["images"][img_name]["digest"]

        summary = build_summary(cell, df, digest)
        (cell_dir / "summary.json").write_text(json.dumps(summary, indent=2))

        if cell.get("vllm_bench_reference"):
            bench_dir = cell_dir / "vllm-bench-reference"
            bench_dir.mkdir(exist_ok=True)
            bench_json = build_vllm_bench_reference(cell, summary)
            # Filename pattern roughly matches what vllm bench produces
            bench_name = f"openai-chat-{cell['req_rate']}qps-{args.n_per_cell}p.json"
            (bench_dir / bench_name).write_text(json.dumps(bench_json, indent=2))

        print(f"[fake] {cell_id}: {args.n_per_cell} rows → {cell_dir}")

    print(f"\nWrote {len(cells)} cell(s) under {args.out_dir}")
    print("\nNext:")
    print(f"  python phase3_aggregate.py --matrix {args.matrix} --local-only \\")
    print(f"      --phase2-dir ../cc-deep-eval/runs/phase2_validation \\")
    print(f"      --cache-dir {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
