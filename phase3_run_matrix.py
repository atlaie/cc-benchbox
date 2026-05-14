#!/usr/bin/env python3
"""
phase3_run_matrix.py — Phase 3 laptop-side matrix orchestrator.

Reads phase3-matrix.yaml. For each cell, in declared order:

    1. tinfoil container create <cell>-target ... [--disable-cc-mode]
    2. Poll target /health via HTTPS (laptop → target endpoint)
    3. SSH into the long-lived benchbox: python phase3_run_cell.py ...
    4. (C1-off / C1-on only) SSH benchbox: vllm bench serve ...
    5. Capture target logs locally, upload to R2 if credentials present
    6. tinfoil container delete <cell>-target

The benchbox is assumed to already be running (long-lived). Phase3 artifacts
(requests.parquet, summary.json, metrics.parquet) are written to R2 by
phase3_run_cell.py running *inside* the benchbox; this orchestrator only
handles deploy/teardown and target-side log capture from the laptop side.

`matrix_report.json` is written incrementally after each cell, so partial
runs (or aborted runs) are inspectable.

Usage:

    # Full matrix
    python phase3_run_matrix.py --matrix phase3-matrix.yaml --out-dir runs/phase3

    # Resume from a specific cell
    python phase3_run_matrix.py --matrix phase3-matrix.yaml --from-cell C2-off

    # Subset
    python phase3_run_matrix.py --matrix phase3-matrix.yaml --only-cells C1-off,C1-on

    # Dry run — print every tinfoil/ssh command, execute none
    python phase3_run_matrix.py --matrix phase3-matrix.yaml --dry-run

Pre-requisites (checked in --preflight unless --skip-preflight):
  - tinfoil CLI installed and authenticated (TINFOIL_API_KEY env)
  - SSH alias in ~/.ssh/config matching `benchbox.ssh_alias` from YAML
  - VLLM_API_KEY env set (or 'EMPTY' default acceptable)
  - For R2 upload of target logs: S3_BUCKET, R2_ENDPOINT_URL, AWS_*  env vars

Exit codes:
  0  all cells succeeded
  1  one or more cells failed
  2  user error (bad YAML, unknown cell, preflight failure, etc.)
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx

try:
    import yaml  # type: ignore
except ImportError:
    print("ERROR: PyYAML required. `pip install pyyaml`", file=sys.stderr)
    sys.exit(2)

try:
    import boto3  # type: ignore
except ImportError:
    boto3 = None


SCHEMA_VERSION = "phase3-run-matrix-v1"
VALID_CONDITIONS = {"baseline", "repe_bundle", "routing", "gradient"}


# ===== YAML load + validate ==================================================

def load_and_validate_matrix(path: Path) -> dict:
    """Parse YAML, validate required structure. Raise ValueError on any issue."""
    if not path.exists():
        raise ValueError(f"matrix file not found: {path}")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level must be a mapping")

    for key in ("org_subdomain", "tinfoil_host", "benchbox", "defaults", "images", "cells"):
        if key not in data:
            raise ValueError(f"{path}: missing top-level key {key!r}")

    bb = data["benchbox"]
    for k in ("name", "ssh_alias", "scripts_dir", "pairs_json", "out_dir"):
        if k not in bb:
            raise ValueError(f"{path}: benchbox.{k} required")

    for img_name, img in data["images"].items():
        for k in ("repo", "tag", "digest"):
            if k not in img:
                raise ValueError(f"{path}: images.{img_name}.{k} required")
        if not img["digest"]:
            raise ValueError(
                f"{path}: images.{img_name}.digest is empty. "
                f"Pin sha256:... before running. "
                f"Use `tinfoil container inspect <prior-deploy>` or read tinfoil-config.yml."
            )

    seen_ids: set[str] = set()
    for cell in data["cells"]:
        cid = cell.get("cell_id")
        if not cid:
            raise ValueError(f"{path}: cell missing cell_id")
        if cid in seen_ids:
            raise ValueError(f"{path}: duplicate cell_id {cid!r}")
        seen_ids.add(cid)
        if cell.get("condition") not in VALID_CONDITIONS:
            raise ValueError(f"{path}: {cid}: invalid condition {cell.get('condition')!r}")
        cc = cell.get("cc_state")
        # YAML 1.1 wart: bare `on`/`off` parse to True/False (PyYAML default).
        # Coerce so unquoted YAML still works; phase3-matrix.yaml header tells
        # readers to quote anyway, but defense in depth is cheap.
        if cc is True:
            cc = "on"
        elif cc is False:
            cc = "off"
        cell["cc_state"] = cc  # write back so downstream sees the string
        if cc not in ("on", "off"):
            raise ValueError(f"{path}: {cid}: cc_state must be 'on'|'off', got {cc!r}")
        if cell.get("image") not in data["images"]:
            raise ValueError(f"{path}: {cid}: image {cell.get('image')!r} not in images map")
        if "req_rate" not in cell:
            raise ValueError(f"{path}: {cid}: req_rate required")

    return data


def filter_cells(cells: list[dict], args: argparse.Namespace) -> list[dict]:
    out = list(cells)
    if args.only_cells:
        wanted = {c.strip() for c in args.only_cells.split(",") if c.strip()}
        out = [c for c in out if c["cell_id"] in wanted]
    if args.skip_cells:
        skip = {c.strip() for c in args.skip_cells.split(",") if c.strip()}
        out = [c for c in out if c["cell_id"] not in skip]
    if args.from_cell:
        start = False
        new_out = []
        for c in out:
            if c["cell_id"] == args.from_cell:
                start = True
            if start:
                new_out.append(c)
        if not new_out:
            raise ValueError(f"--from-cell {args.from_cell!r} not found in cells")
        out = new_out
    return out


# ===== URL construction =====================================================

def _target_endpoint(cell_id: str, org_subdomain: str) -> str:
    """Tinfoil debug-mode endpoint for a target container."""
    return f"https://{cell_id.lower()}-target.debug.{org_subdomain}.containers.tinfoil.dev"


def target_base_url(cell_id: str, org_subdomain: str, has_v1: bool) -> str:
    """Driver-facing base URL. vLLM cells expect /v1 suffix; gradient cell doesn't."""
    base = _target_endpoint(cell_id, org_subdomain)
    return f"{base}/v1" if has_v1 else base


def health_url(cell_id: str, org_subdomain: str) -> str:
    return f"{_target_endpoint(cell_id, org_subdomain)}/health"


def metrics_url(cell_id: str, org_subdomain: str) -> str:
    return f"{_target_endpoint(cell_id, org_subdomain)}/metrics"


# ===== subprocess helpers ===================================================

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _print_cmd(cmd: list[str]) -> None:
    print(f"  $ {' '.join(shlex.quote(c) for c in cmd)}")


def run_cmd(cmd: list[str], dry_run: bool, timeout: Optional[float] = None,
            capture_output: bool = False) -> subprocess.CompletedProcess:
    """Run a command, respecting --dry-run."""
    _print_cmd(cmd)
    if dry_run:
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
    return subprocess.run(cmd, timeout=timeout, capture_output=capture_output, text=True)


# ===== Tinfoil + SSH command builders =======================================

def deploy_target(matrix: dict, cell: dict, args: argparse.Namespace) -> bool:
    img = matrix["images"][cell["image"]]
    target_name = f"{cell['cell_id'].lower()}-target"
    cmd = [
        "tinfoil", "container", "create", target_name,
        "--repo", img["repo"],
        "--tag", img["tag"],
        "--host", matrix["tinfoil_host"],
        "--debug",
        "--yes",
    ]
    if cell["cc_state"] == "off":
        cmd.append("--disable-cc-mode")
    return run_cmd(cmd, dry_run=args.dry_run).returncode == 0


def wait_for_target_health(cell: dict, matrix: dict, args: argparse.Namespace) -> bool:
    url = health_url(cell["cell_id"], matrix["org_subdomain"])
    timeout = matrix["defaults"]["health_timeout"]
    poll = matrix["defaults"]["health_poll_interval"]
    api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
    print(f"  [health] polling {url} (max {timeout:.0f}s, every {poll:.0f}s)")
    if args.dry_run:
        return True
    deadline = time.monotonic() + timeout
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    while time.monotonic() < deadline:
        try:
            r = httpx.get(url, headers=headers, timeout=30.0)
            if r.status_code == 200:
                try:
                    body = r.json()
                    if body.get("status") == "ok":
                        print(f"  [health] ready: {body}")
                        return True
                    print(f"  [health] status={body.get('status')!r}; waiting...")
                except Exception:
                    # vLLM's /health may return empty body with 200 once ready.
                    print(f"  [health] ready (HTTP 200, empty body)")
                    return True
            else:
                print(f"  [health] HTTP {r.status_code}; waiting...")
        except Exception as e:
            print(f"  [health] {type(e).__name__}: {e}")
        time.sleep(poll)
    print(f"  [health] TIMEOUT — /health not ready within {timeout:.0f}s", file=sys.stderr)
    return False


def build_run_cell_remote_cmd(matrix: dict, cell: dict) -> str:
    """Construct the SSH-remote `python phase3_run_cell.py ...` command as a single
    shell-safe string. Caller passes this as the SSH command argument."""
    cell_id = cell["cell_id"]
    img = matrix["images"][cell["image"]]
    has_v1 = cell["image"] == "vllm"
    bb = matrix["benchbox"]
    d = matrix["defaults"]

    parts = [
        "python3", f"{bb['scripts_dir']}/phase3_run_cell.py",
        "--cell-id", cell_id,
        "--condition", cell["condition"],
        "--cc-state", cell["cc_state"],
        "--target-base-url", target_base_url(cell_id, matrix["org_subdomain"], has_v1),
        "--pairs-json", bb["pairs_json"],
        "--out-dir", bb["out_dir"],
        "--image-digest", img["digest"],
        "--n-requests", str(d["n_requests"]),
        "--req-rate", str(cell["req_rate"]),
        "--max-new-tokens", str(d["max_new_tokens"]),
        "--timeout", str(d["timeout"]),
        "--health-timeout", str(d["health_timeout"]),
        "--health-poll-interval", str(d["health_poll_interval"]),
        "--model", d["model"],
        # Orchestrator already health-checked from laptop side; benchbox-side
        # poll would only add latency and noise.
        "--skip-health",
    ]
    if d.get("metrics_interval", 0) > 0:
        parts += [
            "--metrics-url", metrics_url(cell_id, matrix["org_subdomain"]),
            "--metrics-interval", str(d["metrics_interval"]),
        ]
    return " ".join(shlex.quote(p) for p in parts)


def build_vllm_bench_remote_cmd(matrix: dict, cell: dict) -> str:
    """vllm bench serve reference run — C1 cells only. Output in a sibling dir
    next to phase3_run_cell.py's outputs so phase3_aggregate.py can correlate."""
    cell_id = cell["cell_id"]
    bb = matrix["benchbox"]
    d = matrix["defaults"]
    out = f"{bb['out_dir']}/{cell_id}/vllm-bench-reference"
    bench = [
        "vllm", "bench", "serve",
        "--backend", "openai-chat",
        "--model", d["model"],
        "--endpoint", "/v1/chat/completions",
        "--base-url", target_base_url(cell_id, matrix["org_subdomain"], has_v1=False),
        "--dataset-name", "custom",
        "--dataset-path", bb["pairs_jsonl"],
        "--num-prompts", str(d["n_requests"]),
        "--extra-body", '{"chat_template_kwargs":{"enable_thinking":false}}',
        "--save-result",
        "--result-dir", out,
    ]
    bench_str = " ".join(shlex.quote(p) for p in bench)
    return f"mkdir -p {shlex.quote(out)} && {bench_str}"


def ssh_exec(ssh_alias: str, remote_cmd: str, args: argparse.Namespace,
             timeout: Optional[float] = None) -> int:
    cmd = ["ssh", ssh_alias, remote_cmd]
    return run_cmd(cmd, dry_run=args.dry_run, timeout=timeout).returncode


def capture_target_logs(cell_id: str, out_dir: Path, args: argparse.Namespace) -> Optional[Path]:
    target_name = f"{cell_id.lower()}-target"
    log_path = out_dir / cell_id / "target.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["tinfoil", "container", "logs", target_name]
    _print_cmd(cmd)
    print(f"    → {log_path}")
    if args.dry_run:
        return None
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        body = result.stdout or ""
        if result.stderr:
            body += "\n--- STDERR ---\n" + result.stderr
        log_path.write_text(body)
        return log_path
    except Exception as e:
        print(f"  [logs] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def upload_local_to_r2(local_path: Optional[Path], cell_id: str) -> bool:
    if local_path is None or not local_path.exists():
        return False
    bucket = os.environ.get("S3_BUCKET")
    endpoint_url = os.environ.get("R2_ENDPOINT_URL") or os.environ.get("R2_ENDPOINT")
    if not bucket or boto3 is None:
        return False
    kwargs = {"endpoint_url": endpoint_url} if endpoint_url else {}
    try:
        s3 = boto3.client("s3", **kwargs)
        key = f"phase3/{cell_id}/{local_path.name}"
        s3.upload_file(str(local_path), bucket, key)
        backend = "r2" if endpoint_url else "s3"
        print(f"  [upload] {backend}://{bucket}/{key}")
        return True
    except Exception as e:
        print(f"  [upload] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return False


def delete_target(cell_id: str, args: argparse.Namespace) -> bool:
    target_name = f"{cell_id.lower()}-target"
    cmd = ["tinfoil", "container", "delete", target_name, "--yes"]
    return run_cmd(cmd, dry_run=args.dry_run).returncode == 0


# ===== per-cell pipeline ====================================================

def run_one_cell(matrix: dict, cell: dict, args: argparse.Namespace, out_dir: Path) -> dict:
    cell_id = cell["cell_id"]
    report: dict[str, Any] = {
        "cell_id": cell_id,
        "condition": cell["condition"],
        "cc_state": cell["cc_state"],
        "image": cell["image"],
        "req_rate": cell["req_rate"],
        "started": now_iso(),
        "stages": {},
    }

    print(f"\n===== {cell_id} (condition={cell['condition']}, cc={cell['cc_state']}) =====")

    # 1. Deploy target
    print(f"[1/5] deploy target")
    t0 = time.monotonic()
    ok = deploy_target(matrix, cell, args)
    report["stages"]["deploy"] = {"wall_s": time.monotonic() - t0, "ok": ok}
    if not ok:
        report["status"] = "fail"
        report["failed_at"] = "deploy"
        report["ended"] = now_iso()
        return report

    # 2. Wait for /health
    print(f"[2/5] wait for target /health")
    t0 = time.monotonic()
    health_ok = wait_for_target_health(cell, matrix, args)
    report["stages"]["health"] = {"wall_s": time.monotonic() - t0, "ok": health_ok}
    if not health_ok:
        # Try to capture logs + delete even on health failure — don't strand the target.
        log_path = capture_target_logs(cell_id, out_dir, args)
        upload_local_to_r2(log_path, cell_id)
        delete_target(cell_id, args)
        report["status"] = "fail"
        report["failed_at"] = "health"
        report["ended"] = now_iso()
        return report

    # 3. SSH benchbox → phase3_run_cell.py
    print(f"[3/5] run phase3_run_cell.py via SSH")
    t0 = time.monotonic()
    remote_cmd = build_run_cell_remote_cmd(matrix, cell)
    print(f"  remote: {remote_cmd}")
    ssh_alias = matrix["benchbox"]["ssh_alias"]
    rc = ssh_exec(ssh_alias, remote_cmd, args)
    report["stages"]["run_cell"] = {"wall_s": time.monotonic() - t0, "rc": rc}

    # 4. Optional: vllm bench reference for C1-off / C1-on
    if cell.get("vllm_bench_reference"):
        if rc == 0:
            print(f"[4/5] vllm bench reference (C1 only)")
            t0 = time.monotonic()
            bench_cmd = build_vllm_bench_remote_cmd(matrix, cell)
            print(f"  remote: {bench_cmd}")
            bench_rc = ssh_exec(ssh_alias, bench_cmd, args)
            report["stages"]["vllm_bench_reference"] = {
                "wall_s": time.monotonic() - t0, "rc": bench_rc,
            }
        else:
            print(f"[4/5] vllm bench reference: skipped (run_cell failed)")
            report["stages"]["vllm_bench_reference"] = {"skipped": "run_cell failed"}
    else:
        print(f"[4/5] vllm bench reference: not configured for this cell")

    # 5. Logs + delete
    print(f"[5/5] capture target logs + delete")
    t0 = time.monotonic()
    log_path = None
    log_uploaded = False
    if matrix["defaults"].get("capture_target_logs", True):
        log_path = capture_target_logs(cell_id, out_dir, args)
        log_uploaded = upload_local_to_r2(log_path, cell_id)
    delete_ok = delete_target(cell_id, args)
    report["stages"]["teardown"] = {
        "wall_s": time.monotonic() - t0,
        "logs_captured": log_path is not None,
        "logs_uploaded_to_r2": log_uploaded,
        "delete_ok": delete_ok,
    }

    report["status"] = "ok" if rc == 0 else "fail"
    if rc != 0:
        report["failed_at"] = "run_cell"
    report["ended"] = now_iso()
    return report


# ===== preflight ============================================================

def preflight(matrix: dict, args: argparse.Namespace) -> bool:
    """SSH reachability + tinfoil CLI + env vars check."""
    print("[preflight] checking SSH to benchbox")
    ssh_alias = matrix["benchbox"]["ssh_alias"]
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
           ssh_alias, "echo benchbox_reachable && python3 --version"]
    if args.dry_run:
        _print_cmd(cmd)
    else:
        _print_cmd(cmd)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            print(f"[preflight] SSH to {ssh_alias} timed out", file=sys.stderr)
            return False
        if result.returncode != 0:
            print(f"[preflight] SSH to {ssh_alias} failed (rc={result.returncode}):",
                  file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            print(f"  Ensure ~/.ssh/config has a 'Host {ssh_alias}' entry.", file=sys.stderr)
            return False
        last_line = (result.stdout or "").strip().split("\n")[-1]
        print(f"  benchbox: {last_line}")

    print("[preflight] checking tinfoil CLI")
    cmd = ["tinfoil", "--version"]
    if args.dry_run:
        _print_cmd(cmd)
    else:
        _print_cmd(cmd)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        except FileNotFoundError:
            print("[preflight] tinfoil CLI not found on PATH", file=sys.stderr)
            return False
        if result.returncode != 0:
            print("[preflight] tinfoil CLI errored:", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            return False
        print(f"  {result.stdout.strip()}")

    if not args.dry_run:
        if not os.environ.get("TINFOIL_API_KEY"):
            print("[preflight] WARN: TINFOIL_API_KEY not set; CLI may use cached auth")
        if not os.environ.get("VLLM_API_KEY"):
            print("[preflight] WARN: VLLM_API_KEY not set; falling back to 'EMPTY'")
        if not os.environ.get("S3_BUCKET"):
            print("[preflight] WARN: S3_BUCKET not set; target logs won't be uploaded to R2")

    return True


# ===== CLI ==================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--matrix", type=Path, required=True,
                   help="Path to phase3-matrix.yaml.")
    p.add_argument("--out-dir", type=Path, default=Path("runs/phase3"),
                   help="Local directory for matrix_report.json + per-cell target logs.")
    p.add_argument("--from-cell", default=None,
                   help="Resume from this cell_id (skips anything declared before it).")
    p.add_argument("--only-cells", default=None,
                   help="Comma-separated cell_ids to run; others skipped.")
    p.add_argument("--skip-cells", default=None,
                   help="Comma-separated cell_ids to skip.")
    p.add_argument("--on-fail", choices=["abort", "continue"], default="abort",
                   help="On cell failure: abort the whole matrix (default), or continue.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print every tinfoil/ssh command; execute none.")
    p.add_argument("--skip-preflight", action="store_true",
                   help="Skip SSH + tinfoil CLI checks. Use only for offline iteration.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        matrix = load_and_validate_matrix(args.matrix)
    except Exception as e:
        print(f"[error] matrix validation: {e}", file=sys.stderr)
        return 2

    try:
        cells = filter_cells(matrix["cells"], args)
    except ValueError as e:
        print(f"[error] cell filter: {e}", file=sys.stderr)
        return 2

    if not cells:
        print("[error] no cells to run after filtering", file=sys.stderr)
        return 2

    print(f"[matrix] {len(cells)} cell(s): {[c['cell_id'] for c in cells]}")
    print(f"[matrix] out_dir={args.out_dir}")
    if args.dry_run:
        print(f"[matrix] DRY RUN — no commands executed")

    if not args.skip_preflight:
        if not preflight(matrix, args):
            return 2

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "matrix_file": str(args.matrix),
        "dry_run": args.dry_run,
        "started": now_iso(),
        "cells": [],
    }

    aborted = False
    for cell in cells:
        cell_report = run_one_cell(matrix, cell, args, out_dir)
        report["cells"].append(cell_report)
        # Write after every cell so partial runs are inspectable.
        (out_dir / "matrix_report.json").write_text(json.dumps(report, indent=2))
        if cell_report["status"] == "fail" and args.on_fail == "abort":
            print(f"\n[matrix] ABORT — {cell_report['cell_id']} failed at "
                  f"{cell_report.get('failed_at', '?')}")
            aborted = True
            break

    report["ended"] = now_iso()
    report["aborted"] = aborted
    (out_dir / "matrix_report.json").write_text(json.dumps(report, indent=2))

    ok = sum(1 for c in report["cells"] if c["status"] == "ok")
    fail = len(report["cells"]) - ok
    print(f"\n[matrix] DONE: {ok}/{len(report['cells'])} cells ok, {fail} failed")
    print(f"[matrix] report: {out_dir / 'matrix_report.json'}")

    return 0 if (ok == len(cells) and not aborted) else 1


if __name__ == "__main__":
    sys.exit(main())
