#!/usr/bin/env python3
"""
phase3_run_matrix.py — Phase 3 laptop-side matrix orchestrator (v3-grouped).

v3 change vs v2-local: cells are grouped by (image, cc_state, debug) and
each group deploys ONCE. All cells in a group run sequentially against the
shared deploy. Saves model-load time when multiple cells share a deploy
configuration (typical for the GLM-MoE matrix: 6 vllm cells × 2 CC states
becomes 2 deploys instead of 12).

CC-toggle constraint still holds (PHASE2_REFERENCE §9): different CC states
require different deploys. Within one (image, cc_state, debug), instrumentation
choice (vllm_xargs payload) and driver choice (sequential / streaming /
concurrent) are request-level and can vary freely across cells against a
shared target. The `debug` flag is also deploy-time (per docs.tinfoil.sh):
debug-mode and production-mode deploys live at different FQDNs and have
different attestation/SSH semantics, so they cannot share a target.

Driver runs as a local subprocess on the laptop (no SSH/docker exec) because
Tinfoil CC blocks intra-container network egress, preventing the benchbox
from reaching target endpoints. CC overhead is measured as a delta (on vs off)
so laptop→Tinfoil RTT cancels.

PATCHED 2026-05-26:
  - wait_for_group_health now ALSO verifies /v1/models lists the expected
    served-model. Closes the gap where the Tinfoil edge proxy answers
    /health 200 (empty body) before vLLM finishes loading the model
    weights (observed ~25 min lag on GLM-5.1-FP8 / 700 GB).
  - Compat shims added (deploy_target, wait_for_target_status_ready,
    wait_for_target_health, delete_target) so phase3_sweep_max_tokens.py
    (pre-v3, cell-based API) works against this orchestrator.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx

try:
    import yaml  # type: ignore
except ImportError:
    print("ERROR: PyYAML required. `pip install pyyaml`", file=sys.stderr)
    sys.exit(2)


SCHEMA_VERSION = "phase3-run-matrix-v3-grouped"
VALID_CONDITIONS = {"baseline", "repe_bundle", "routing", "gradient", "steer"}


# ===== YAML load + validate =================================================

def load_and_validate_matrix(path: Path) -> dict:
    if not path.exists():
        raise ValueError(f"matrix file not found: {path}")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level must be a mapping")

    for key in ("org_subdomain", "tinfoil_host", "benchbox", "defaults",
                "images", "cells"):
        if key not in data:
            raise ValueError(f"{path}: missing top-level key {key!r}")

    bb = data["benchbox"]
    for k in ("scripts_dir", "pairs_json", "out_dir"):
        if k not in bb:
            raise ValueError(f"{path}: benchbox.{k} required (laptop-side path)")

    for img_name, img in data["images"].items():
        for k in ("repo", "tag", "digest"):
            if k not in img:
                raise ValueError(f"{path}: images.{img_name}.{k} required")
        if not img["digest"]:
            raise ValueError(f"{path}: images.{img_name}.digest empty; pin sha256:...")
        if "has_v1" not in img:
            raise ValueError(
                f"{path}: images.{img_name}.has_v1 required "
                f"(true for OpenAI-compatible vLLM, false for custom sidecars)"
            )
        img.setdefault("model", data["defaults"].get("model"))
        if not img["model"]:
            raise ValueError(
                f"{path}: images.{img_name}.model not set and defaults.model "
                f"missing — cannot infer served-model name"
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
            raise ValueError(
                f"{path}: {cid}: invalid condition {cell.get('condition')!r}"
            )
        cc = cell.get("cc_state")
        if cc is True:
            cc = "on"
        elif cc is False:
            cc = "off"
        cell["cc_state"] = cc
        if cc not in ("on", "off"):
            raise ValueError(f"{path}: {cid}: cc_state must be 'on'|'off', got {cc!r}")
        if cell.get("image") not in data["images"]:
            raise ValueError(
                f"{path}: {cid}: image {cell.get('image')!r} not in images map"
            )
        if "req_rate" not in cell:
            raise ValueError(f"{path}: {cid}: req_rate required")
        cell.setdefault("debug", data["defaults"].get("debug", True))
        if not isinstance(cell["debug"], bool):
            raise ValueError(
                f"{path}: {cid}: debug must be bool, got "
                f"{type(cell['debug']).__name__}"
            )
        driver = cell.get("driver", "sequential")
        if driver not in ("sequential", "stream", "concurrent"):
            raise ValueError(f"{path}: {cid}: invalid driver {driver!r}")
        cell["driver"] = driver
        if driver == "concurrent":
            if ("concurrency" not in cell
                    or not isinstance(cell["concurrency"], int)
                    or cell["concurrency"] < 1):
                raise ValueError(
                    f"{path}: {cid}: concurrent driver needs int concurrency>=1"
                )
        cell["stream"] = bool(cell.get("stream", driver == "stream"))
        cell.setdefault("n_requests", data["defaults"]["n_requests"])
        cell.setdefault("max_new_tokens", data["defaults"]["max_new_tokens"])
        if not isinstance(cell["max_new_tokens"], int) or cell["max_new_tokens"] < 1:
            raise ValueError(
                f"{path}: {cid}: max_new_tokens must be positive int, got "
                f"{cell['max_new_tokens']!r}"
            )
        if "pairs_json" in cell:
            if not isinstance(cell["pairs_json"], str):
                raise ValueError(
                    f"{path}: {cid}: pairs_json must be string path, got "
                    f"{type(cell['pairs_json']).__name__}"
                )
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


# ===== grouping =============================================================

@dataclass
class Group:
    image: str
    cc_state: str
    debug: bool
    cells: list[dict] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, str, bool]:
        return (self.image, self.cc_state, self.debug)

    @property
    def deploy_id(self) -> str:
        debug_suffix = "debug" if self.debug else "prod"
        slug = f"{self.image}-{self.cc_state}-{debug_suffix}"
        return slug.lower().replace("_", "-")

    @property
    def target_name(self) -> str:
        return f"{self.deploy_id}-target"


def group_cells(cells: list[dict]) -> list[Group]:
    by_key: dict[tuple, Group] = {}
    order: list[tuple] = []
    for cell in cells:
        key = (cell["image"], cell["cc_state"], cell.get("debug", True))
        if key not in by_key:
            by_key[key] = Group(image=key[0], cc_state=key[1], debug=key[2])
            order.append(key)
        by_key[key].cells.append(cell)
    return [by_key[k] for k in order]


# ===== URL helpers ==========================================================

def _target_endpoint(target_name: str, org_subdomain: str,
                      debug: bool = True) -> str:
    subdomain = "debug." if debug else ""
    return f"https://{target_name}.{subdomain}{org_subdomain}.containers.tinfoil.dev"


def target_base_url(target_name: str, org_subdomain: str, has_v1: bool,
                     debug: bool = True) -> str:
    base = _target_endpoint(target_name, org_subdomain, debug=debug)
    return f"{base}/v1" if has_v1 else base


def health_url(target_name: str, org_subdomain: str, debug: bool = True) -> str:
    return f"{_target_endpoint(target_name, org_subdomain, debug=debug)}/health"


def metrics_url(target_name: str, org_subdomain: str, debug: bool = True) -> str:
    return f"{_target_endpoint(target_name, org_subdomain, debug=debug)}/metrics"


# ===== subprocess helpers ===================================================

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _print_cmd(cmd: list[str]) -> None:
    print(f"  $ {' '.join(shlex.quote(c) for c in cmd)}")


def run_cmd(cmd: list[str], dry_run: bool, timeout: Optional[float] = None,
             capture_output: bool = False, env: Optional[dict] = None
             ) -> subprocess.CompletedProcess:
    _print_cmd(cmd)
    if dry_run:
        return subprocess.CompletedProcess(args=cmd, returncode=0,
                                            stdout="", stderr="")
    return subprocess.run(cmd, timeout=timeout, capture_output=capture_output,
                           text=True, env=env)


# ===== Tinfoil deploy + status + health (group-level) =======================

def deploy_group(matrix: dict, group: Group, args: argparse.Namespace) -> bool:
    img = matrix["images"][group.image]
    cmd = [
        "tinfoil", "container", "create", group.target_name,
        "--repo", img["repo"],
        "--tag", img["tag"],
        "--host", matrix["tinfoil_host"],
        "--yes",
    ]
    if group.debug:
        cmd.append("--debug")
    if group.cc_state == "off":
        cmd.append("--disable-cc-mode")
    return run_cmd(cmd, dry_run=args.dry_run).returncode == 0


def wait_for_status_ready(target_name: str, args: argparse.Namespace,
                            timeout: float = 3600, poll: float = 30) -> bool:
    print(f"  [status] polling tinfoil for status=ready (max {timeout:.0f}s)")
    if args.dry_run:
        return True
    deadline = time.monotonic() + timeout
    last_status = None
    while time.monotonic() < deadline:
        try:
            r = subprocess.run(
                ["tinfoil", "container", "get", target_name],
                capture_output=True, text=True, timeout=30,
            )
            status = ""
            for line in r.stdout.splitlines():
                if line.strip().startswith("Status:"):
                    status = line.split(":", 1)[1].strip()
                    break
            if status != last_status:
                print(f"  [status] {status}")
                last_status = status
            if status == "ready":
                subprocess.run(["sudo", "-n", "killall", "-HUP", "mDNSResponder"],
                                capture_output=True)
                subprocess.run(["sudo", "-n", "dscacheutil", "-flushcache"],
                                capture_output=True)
                time.sleep(2)
                return True
            if status in ("failed", "deleted"):
                print(f"  [status] aborting — terminal state '{status}'",
                       file=sys.stderr)
                return False
        except Exception as e:
            print(f"  [status] error: {e}")
        time.sleep(poll)
    print(f"  [status] TIMEOUT after {timeout:.0f}s", file=sys.stderr)
    return False


def wait_for_group_health(group, matrix: dict,
                            args: argparse.Namespace) -> bool:
    """Wait for the deploy to be ready to serve REAL inference.

    PATCHED (2026-05-26): two-stage readiness check.

      Stage 1 — /health returns HTTP 200. Necessary but NOT sufficient
      on Tinfoil + vLLM 0.20.0: the edge proxy answers /health with an
      empty-bodied 200 well before vLLM finishes loading the model
      (observed ~25 min lag on GLM-5.1-FP8 / 700 GB weights). The
      original logic treated empty-bodied 200 as ready, causing every
      cell in that window to fail with APIConnectionError.

      Stage 2 — for OpenAI-compatible images (has_v1=True), additionally
      verify /v1/models returns HTTP 200 AND lists the expected served-
      model id. This is the authoritative vLLM-loaded signal — the
      endpoint only responds once the model is in HBM and the engine
      is ready to schedule requests. For non-v1 images (custom sidecars
      like the gradient driver), only /health is checked.

    `group` is duck-typed: needs .target_name, .image, .cc_state, .debug.
    Both the dataclass Group (new orchestrator) and the _SweepGroup shim
    used by phase3_sweep_max_tokens.py satisfy this.
    """
    url = health_url(group.target_name, matrix["org_subdomain"],
                      debug=group.debug)
    img = matrix["images"][group.image]
    expected_model = img["model"]
    v1_models_url: Optional[str] = None
    if img.get("has_v1"):
        v1_base = target_base_url(group.target_name, matrix["org_subdomain"],
                                    has_v1=True, debug=group.debug)
        v1_models_url = f"{v1_base}/models"

    timeout = matrix["defaults"]["health_timeout"]
    poll = matrix["defaults"]["health_poll_interval"]
    api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
    print(f"  [health] polling {url} (max {timeout:.0f}s, every {poll:.0f}s)")
    if v1_models_url:
        print(f"  [health] AND verifying /v1/models lists model="
              f"{expected_model!r}")
    if args.dry_run:
        return True
    deadline = time.monotonic() + timeout
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    while time.monotonic() < deadline:
        # Stage 1 — /health
        health_passed = False
        try:
            r = httpx.get(url, headers=headers, timeout=30.0, verify=False)
            if r.status_code == 200:
                health_passed = True
            else:
                print(f"  [health] /health HTTP {r.status_code}; waiting...")
        except Exception as e:
            print(f"  [health] /health {type(e).__name__}: {e}; waiting...")

        # Stage 2 — /v1/models (only for has_v1=True images)
        if health_passed:
            if not v1_models_url:
                print(f"  [health] ready (HTTP 200; non-v1 image, no /v1 check)")
                return True
            try:
                mr = httpx.get(v1_models_url, headers=headers,
                                timeout=30.0, verify=False)
                if mr.status_code == 200:
                    try:
                        body = mr.json()
                        models = [m.get("id") for m in body.get("data", [])]
                    except Exception:
                        models = []
                    if expected_model in models:
                        print(f"  [health] ready: /health=200 AND "
                              f"/v1/models lists {expected_model!r}")
                        return True
                    print(f"  [health] /v1/models=200 but {expected_model!r} "
                          f"not in {models}; vLLM still warming...")
                else:
                    print(f"  [health] /health=200 but /v1/models HTTP "
                          f"{mr.status_code}; vLLM still loading model...")
            except Exception as e:
                print(f"  [health] /health=200 but /v1/models "
                      f"{type(e).__name__}: {e}; vLLM still loading model...")
        time.sleep(poll)
    print(f"  [health] TIMEOUT — endpoint not ready within {timeout:.0f}s",
           file=sys.stderr)
    return False


def delete_group(group: Group, args: argparse.Namespace) -> bool:
    cmd = ["tinfoil", "container", "delete", group.target_name]
    return run_cmd(cmd, dry_run=args.dry_run).returncode == 0


# ===== compat shims for phase3_sweep_max_tokens.py ==========================
# The sweep script was written against the pre-v3 (cell-based) API and calls
# orch.deploy_target / wait_for_target_status_ready / wait_for_target_health /
# delete_target. These shims wrap the new Group-based API so the sweep works
# without modification, while preserving the sweep's own naming convention
# (target_name = "{cell_id.lower()}-target", e.g. "c1-off-sweep-target").

def deploy_target(matrix: dict, cell: dict, args: argparse.Namespace) -> bool:
    """Compat shim: deploys a Tinfoil container using cell['cell_id'] as the
    target name basis (matching phase3_sweep_max_tokens.py's expectations).
    """
    img = matrix["images"][cell["image"]]
    target_name = f"{cell['cell_id'].lower()}-target"
    cmd = ["tinfoil", "container", "create", target_name,
           "--repo", img["repo"], "--tag", img["tag"],
           "--host", matrix["tinfoil_host"], "--yes"]
    debug = cell.get("debug", matrix["defaults"].get("debug", True))
    if debug:
        cmd.append("--debug")
    if cell["cc_state"] == "off":
        cmd.append("--disable-cc-mode")
    return run_cmd(cmd, dry_run=args.dry_run).returncode == 0


def wait_for_target_status_ready(deploy_cell_id: str,
                                   args: argparse.Namespace,
                                   timeout: float = 3600) -> bool:
    """Compat shim. deploy_cell_id is what the sweep calls its cell-id
    (e.g. "c1-off-sweep"); the real Tinfoil target_name has "-target" appended.
    """
    target_name = f"{deploy_cell_id.lower()}-target"
    return wait_for_status_ready(target_name, args, timeout=timeout)


def wait_for_target_health(cell: dict, matrix: dict,
                             args: argparse.Namespace) -> bool:
    """Compat shim. Builds a synthetic group-shaped object whose target_name
    follows the sweep's naming, then delegates to the PATCHED
    wait_for_group_health (with /v1/models verification)."""
    class _SweepGroup:
        pass
    sg = _SweepGroup()
    sg.image = cell["image"]
    sg.cc_state = cell["cc_state"]
    sg.debug = cell.get("debug", matrix["defaults"].get("debug", True))
    sg.target_name = f"{cell['cell_id'].lower()}-target"
    return wait_for_group_health(sg, matrix, args)


def delete_target(deploy_cell_id: str, args: argparse.Namespace) -> bool:
    """Compat shim. Uses the sweep's deploy naming."""
    target_name = f"{deploy_cell_id.lower()}-target"
    cmd = ["tinfoil", "container", "delete", target_name]
    return run_cmd(cmd, dry_run=args.dry_run).returncode == 0


# ===== driver invocation builders ===========================================

def build_run_cell_local_cmd(matrix: dict, cell: dict, group: Group) -> list[str]:
    cell_id = cell["cell_id"]
    img = matrix["images"][cell["image"]]
    has_v1 = img["has_v1"]
    bb = matrix["benchbox"]
    d = matrix["defaults"]

    venv = os.environ.get("VIRTUAL_ENV")
    python = (f"{venv}/bin/python"
               if venv and Path(f"{venv}/bin/python").exists()
               else sys.executable)

    parts = [
        python, "-u", f"{bb['scripts_dir']}/phase3_run_cell.py",
        "--cell-id", cell_id,
        "--condition", cell["condition"],
        "--cc-state", cell["cc_state"],
        "--target-base-url", target_base_url(
            group.target_name, matrix["org_subdomain"], has_v1, debug=group.debug,
        ),
        "--pairs-json", cell.get("pairs_json", bb["pairs_json"]),
        "--out-dir", bb["out_dir"],
        "--image-digest", img["digest"],
        "--n-requests", str(cell.get("n_requests", d["n_requests"])),
        "--req-rate", str(cell["req_rate"]),
        "--max-new-tokens", str(cell.get("max_new_tokens", d["max_new_tokens"])),
        "--timeout", str(d["timeout"]),
        "--health-timeout", str(d["health_timeout"]),
        "--health-poll-interval", str(d["health_poll_interval"]),
        "--model", img["model"],
        "--skip-health",
        "--no-upload",
        "--driver", cell.get("driver", "sequential"),
    ]
    if cell.get("stream"):
        parts.append("--stream")
    if cell.get("driver") == "concurrent":
        parts += ["--concurrency", str(cell["concurrency"])]
    parts += ["--tinfoil-target-name", group.target_name]
    if cell.get("steer_direction"):
        parts += ["--steer-direction", str(cell["steer_direction"])]
    if cell.get("enable_thinking", False):
        parts.append("--enable-thinking")
    if cell.get("apply_steering_json"):
        parts += ["--apply-steering-json", str(cell["apply_steering_json"])]
    return parts


def build_vllm_bench_local_cmd(matrix: dict, cell: dict,
                                 group: Group) -> list[str]:
    cell_id = cell["cell_id"]
    bb = matrix["benchbox"]
    d = matrix["defaults"]
    out_dir = Path(bb["out_dir"]) / cell_id / "vllm-bench-reference"
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        "vllm", "bench", "serve",
        "--backend", "openai-chat",
        "--model", d["model"],
        "--endpoint", "/v1/chat/completions",
        "--base-url", target_base_url(
            group.target_name, matrix["org_subdomain"], has_v1=False,
            debug=group.debug,
        ),
        "--dataset-name", "custom",
        "--dataset-path", bb.get("pairs_jsonl", "data/pairs.jsonl"),
        "--num-prompts", str(d["n_requests"]),
        "--extra-body", '{"chat_template_kwargs":{"enable_thinking":false}}',
        "--save-result",
        "--result-dir", str(out_dir),
    ]


# ===== group-level pipeline =================================================

def run_one_group(matrix: dict, group: Group, args: argparse.Namespace,
                   out_dir: Path) -> dict:
    report: dict[str, Any] = {
        "deploy_id": group.deploy_id,
        "target_name": group.target_name,
        "image": group.image,
        "cc_state": group.cc_state,
        "debug": group.debug,
        "n_cells": len(group.cells),
        "started": now_iso(),
        "stages": {},
        "cells": [],
    }

    print(f"\n========================================================")
    print(f"  GROUP {group.deploy_id}  ({len(group.cells)} cell(s))")
    print(f"     image={group.image}  cc={group.cc_state}  debug={group.debug}")
    print(f"     cells: {[c['cell_id'] for c in group.cells]}")
    print(f"========================================================")

    def _abort(stage: str, reason: str) -> dict:
        report["status"] = "fail"
        report["failed_at"] = stage
        for c in group.cells:
            report["cells"].append({
                "cell_id": c["cell_id"],
                "condition": c["condition"],
                "cc_state": c["cc_state"],
                "regime": c.get("regime", "sequential"),
                "status": "not-run",
                "reason": reason,
            })
        report["ended"] = now_iso()
        return report

    print(f"\n[group/1] deploy target {group.target_name}")
    t0 = time.monotonic()
    ok = deploy_group(matrix, group, args)
    report["stages"]["deploy"] = {"wall_s": time.monotonic() - t0, "ok": ok}
    if not ok:
        return _abort("deploy", "deploy_group returned non-zero")

    print(f"\n[group/2] wait for container status=ready")
    t0 = time.monotonic()
    status_ok = wait_for_status_ready(
        group.target_name, args,
        timeout=matrix["defaults"]["health_timeout"],
    )
    report["stages"]["status_ready"] = {
        "wall_s": time.monotonic() - t0, "ok": status_ok,
    }
    if not status_ok:
        print(f"  [status] WARN: {'non-debug' if not group.debug else 'debug'} "
            f"group timed out at status=ready; deferring to /health "
            f"as authoritative liveness signal.", file=sys.stderr)
        report["stages"]["status_ready"]["fallback"] = (
            f"{'non-debug' if not group.debug else 'debug'}; deferring to /health"
        )

    print(f"\n[group/3] wait for /health AND /v1/models")
    t0 = time.monotonic()
    health_ok = wait_for_group_health(group, matrix, args)
    report["stages"]["health"] = {"wall_s": time.monotonic() - t0, "ok": health_ok}
    if not health_ok:
        delete_group(group, args)
        return _abort("health", "/health or /v1/models timeout")

    for i, cell in enumerate(group.cells):
        cell_report: dict[str, Any] = {
            "cell_id": cell["cell_id"],
            "condition": cell["condition"],
            "cc_state": cell["cc_state"],
            "regime": cell.get("regime", "sequential"),
            "req_rate": cell.get("req_rate"),
            "max_new_tokens": cell.get("max_new_tokens"),
            "started": now_iso(),
            "stages": {},
        }
        print(f"\n  -------- cell {i + 1}/{len(group.cells)}: {cell['cell_id']} "
              f"(condition={cell['condition']}, "
              f"driver={cell.get('driver','sequential')}) "
              f"--------")

        t0 = time.monotonic()
        local_cmd = build_run_cell_local_cmd(matrix, cell, group)
        _print_cmd(local_cmd)
        rc = 0
        if not args.dry_run:
            rc = subprocess.run(local_cmd, env=os.environ.copy()).returncode
        cell_report["stages"]["run_cell"] = {
            "wall_s": time.monotonic() - t0, "rc": rc,
        }
        cell_report["status"] = "ok" if rc == 0 else "fail"
        if rc != 0:
            cell_report["failed_at"] = "run_cell"

        if cell.get("vllm_bench_reference"):
            if rc != 0:
                print(f"  [bench] skipped (run_cell failed)")
                cell_report["stages"]["vllm_bench_reference"] = {
                    "skipped": "run_cell failed",
                }
            elif shutil.which("vllm") is None:
                print(f"  [bench] vllm CLI not on PATH; skipping reference run")
                cell_report["stages"]["vllm_bench_reference"] = {
                    "skipped": "no local vllm",
                }
            else:
                t0 = time.monotonic()
                bench_cmd = build_vllm_bench_local_cmd(matrix, cell, group)
                _print_cmd(bench_cmd)
                bench_rc = 0
                if not args.dry_run:
                    bench_rc = subprocess.run(
                        bench_cmd, env=os.environ.copy(),
                    ).returncode
                cell_report["stages"]["vllm_bench_reference"] = {
                    "wall_s": time.monotonic() - t0, "rc": bench_rc,
                }

        cell_report["ended"] = now_iso()
        report["cells"].append(cell_report)

    print(f"\n[group/5] teardown {group.target_name}")
    t0 = time.monotonic()
    delete_ok = delete_group(group, args)
    report["stages"]["teardown"] = {
        "wall_s": time.monotonic() - t0, "delete_ok": delete_ok,
    }

    statuses = [c["status"] for c in report["cells"]]
    if all(s == "ok" for s in statuses):
        report["status"] = "ok"
    elif any(s == "ok" for s in statuses):
        report["status"] = "partial"
    else:
        report["status"] = "fail"
    report["ended"] = now_iso()
    return report


# ===== preflight ============================================================

def preflight(matrix: dict, args: argparse.Namespace) -> bool:
    bb = matrix["benchbox"]
    print("[preflight] checking local driver scripts + data")
    required = [
        Path(bb["scripts_dir"]) / "phase3_run_cell.py",
        Path(bb["scripts_dir"]) / "phase3_vllm_driver.py",
        Path(bb["scripts_dir"]) / "phase3_grad_driver.py",
        Path(bb["pairs_json"]),
    ]
    for p in required:
        if not p.exists():
            print(f"[preflight] missing: {p}", file=sys.stderr)
            return False
        print(f"  ok: {p}")

    print("[preflight] checking tinfoil CLI + auth")
    cmd = ["tinfoil", "whoami"]
    if args.dry_run:
        _print_cmd(cmd)
    else:
        _print_cmd(cmd)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        except FileNotFoundError:
            print("[preflight] tinfoil CLI not on PATH", file=sys.stderr)
            return False
        if result.returncode != 0:
            print("[preflight] tinfoil CLI errored:", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            return False
        print(f"  {result.stdout.strip()}")

    if not args.dry_run:
        if not os.environ.get("VLLM_API_KEY"):
            print("[preflight] WARN: VLLM_API_KEY not set; /v1 calls will get 401. "
                  "Set to your tk_* tenant key.", file=sys.stderr)
        if not os.environ.get("S3_BUCKET"):
            print("[preflight] WARN: S3_BUCKET not set; per-cell artifacts won't "
                   "upload to R2")

    return True


# ===== CLI ==================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("runs/phase3"))
    p.add_argument(
        "--from-cell", default=None,
        help="Start from this cell_id (skips earlier cells in YAML order). "
             "Filter is applied BEFORE grouping; the affected cell's group "
             "will only deploy if surviving cells map to it.",
    )
    p.add_argument(
        "--only-cells", default=None,
        help="Comma-separated cell_ids to run. Filter applied before grouping.",
    )
    p.add_argument(
        "--skip-cells", default=None,
        help="Comma-separated cell_ids to skip. Filter applied before grouping.",
    )
    p.add_argument(
        "--on-fail", choices=["abort", "continue"], default="abort",
        help="abort: stop after the first group with status != ok. "
             "continue: complete all groups regardless of failures.",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-preflight", action="store_true")
    return p.parse_args()


def main() -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass
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

    groups = group_cells(cells)

    n_cells = len(cells)
    print(f"[matrix] {n_cells} cell(s) in {len(groups)} group(s):")
    for g in groups:
        cell_summary = ", ".join(c["cell_id"] for c in g.cells)
        plural = "s" if len(g.cells) > 1 else ""
        print(f"  - {g.deploy_id} ({len(g.cells)} cell{plural}): {cell_summary}")
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
        "groups": [],
    }

    aborted = False
    for group in groups:
        group_report = run_one_group(matrix, group, args, out_dir)
        report["groups"].append(group_report)
        (out_dir / "matrix_report.json").write_text(json.dumps(report, indent=2))
        if group_report["status"] != "ok" and args.on_fail == "abort":
            print(f"\n[matrix] ABORT — group {group_report['deploy_id']} "
                  f"status={group_report['status']} "
                  f"(failed_at={group_report.get('failed_at', '?')})")
            aborted = True
            break

    report["ended"] = now_iso()
    report["aborted"] = aborted
    (out_dir / "matrix_report.json").write_text(json.dumps(report, indent=2))

    total_cells = sum(len(g["cells"]) for g in report["groups"])
    cells_ok = sum(
        sum(1 for c in gr["cells"] if c.get("status") == "ok")
        for gr in report["groups"]
    )
    cells_fail = total_cells - cells_ok
    groups_ok = sum(1 for g in report["groups"] if g["status"] == "ok")
    print(f"\n[matrix] DONE: {cells_ok}/{total_cells} cells ok across "
          f"{len(report['groups'])} group(s); "
          f"{groups_ok} group(s) fully ok, {cells_fail} cell failure(s)")
    print(f"[matrix] report: {out_dir / 'matrix_report.json'}")

    return 0 if (cells_ok == total_cells and not aborted) else 1


if __name__ == "__main__":
    sys.exit(main())