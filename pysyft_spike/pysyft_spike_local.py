#!/usr/bin/env python3
"""
pysyft_spike_local.py — local Datasite + Mode B endpoint dry-run.

Step 1 of the §3.2 feasibility spike. Three goals:

  1. Confirm `sy.orchestra.launch(...)` brings up a Datasite on this machine
     (catches Python-version / native-dep issues before we burn a Tinfoil deploy).
  2. Confirm `@sy.api_endpoint(path=...)` + `client.custom_api.add(endpoint=...)`
     works in 0.9.5 — Mode B pre-approved primitive, no `request_code_execution`.
  3. Dump the Datasite's FastAPI route inventory to `openapi_paths.txt`. The
     Tinfoil shim's `paths:` allowlist is explicit; we need this list before
     deploying to Tinfoil in Step 2.

Run from a fresh venv:

    python3.12 -m venv .venv-syft-spike
    source .venv-syft-spike/bin/activate
    pip install 'syft>=0.9.5,<0.9.6'
    python pysyft_spike_local.py

Expected output (in order):
    🚀  launching Datasite "prepilot-spike" on port 8000 ...
    🟢  Datasite ready
    📡  /openapi.json → N paths written to openapi_paths.txt
    🛠   registered endpoint prepilot.hello
    ✅  client.custom_api.prepilot.hello() → "hello from datasite (...)"
    ✅  SPIKE STEP 1 PASS

Exit code 0 on full pass, 1 on any failure. Leaves the Datasite running
on port 8000 unless --shutdown-on-exit is set; useful for interactive
poking from a Jupyter notebook.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# syft is imported lazily so we get a clean error if the venv is wrong.
try:
    import syft as sy
except ImportError as e:
    print(f"[fatal] syft import failed: {e}", file=sys.stderr)
    print("  pip install 'syft>=0.9.5,<0.9.6'", file=sys.stderr)
    sys.exit(1)

import httpx


DATASITE_NAME = "prepilot-spike"
DEFAULT_PORT = 8000
ADMIN_EMAIL = "info@openmined.org"
ADMIN_PASSWORD = "changethis"   # PySyft 0.9.5 default; sufficient for spike.


def dump_openapi_paths(port: int, out_path: Path) -> int:
    """Hit the Datasite's /openapi.json, write a sorted unique-path list.

    Returns the number of paths discovered. The list goes into the Tinfoil
    shim allowlist; PySyft is FastAPI so this is the authoritative source.
    """
    url = f"http://127.0.0.1:{port}/openapi.json"
    r = httpx.get(url, timeout=15.0)
    r.raise_for_status()
    spec = r.json()
    paths = sorted(spec.get("paths", {}).keys())
    out_path.write_text("\n".join(paths) + "\n")
    return len(paths)


def assert_endpoint_works(client: Any) -> str:
    """Define and call a trivial @sy.api_endpoint. Confirms the Mode B
    primitive is wired correctly and the laptop-side client can invoke it."""

    @sy.api_endpoint(
        path="prepilot.hello",
        description="Spike-only echo endpoint. Returns a fixed string.",
        # Settings is the right place for any non-public secret. For the
        # spike we just stash a marker so we can confirm context.settings
        # round-trips at all.
        settings={"marker": "spike-step-1"},
    )
    def hello(context) -> str:
        # context.settings: dict[str, Any]
        return f"hello from datasite (marker={context.settings.get('marker')})"

    res = client.custom_api.add(endpoint=hello)
    # PySyft returns a SyftSuccess/SyftError on add(); we want SyftSuccess.
    if "Error" in type(res).__name__:
        raise RuntimeError(f"custom_api.add failed: {res}")

    # client.custom_api is the admin namespace (add/list/delete endpoints).
    # Calls go through client.api.services.<dotted.path>(...) after a refresh
    # to pick up the new endpoint. Verified on PySyft 0.9.5 against a live
    # Datasite.
    client.refresh()
    result = client.api.services.prepilot.hello()
    return str(result)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--openapi-out", type=Path, default=Path("openapi_paths.txt"))
    ap.add_argument("--shutdown-on-exit", action="store_true",
                     help="Shut the Datasite down on success. Default: leave running "
                          "so you can poke from another shell.")
    args = ap.parse_args()

    # Match the PySyft 0.9.5 README pattern exactly.
    print(f"🚀  launching Datasite '{DATASITE_NAME}' on port {args.port} ...",
          flush=True)
    sy.requires(">=0.9.5,<0.9.6")
    server = sy.orchestra.launch(
        name=DATASITE_NAME,
        port=args.port,
        create_producer=True,
        n_consumers=1,
        dev_mode=False,    # SQLite backing, production-ish; not in-memory.
        reset=True,        # wipe DB so the spike is fresh on every run
    )
    print("🟢  Datasite ready", flush=True)

    try:
        # Login as admin. Default creds per the upstream README.
        client = sy.login(port=args.port, email=ADMIN_EMAIL, password=ADMIN_PASSWORD)

        n_paths = dump_openapi_paths(args.port, args.openapi_out)
        print(f"📡  /openapi.json → {n_paths} paths written to {args.openapi_out}",
              flush=True)

        msg = assert_endpoint_works(client)
        # The endpoint's return value comes back wrapped through the Mode B
        # custom_api machinery; we just check the string is present.
        if "hello from datasite" not in msg:
            print(f"❌  unexpected endpoint return: {msg!r}", file=sys.stderr)
            return 1

        print(f"🛠   registered endpoint prepilot.hello", flush=True)
        print(f"✅  client.custom_api.prepilot.hello() → {msg!r}", flush=True)
        print("✅  SPIKE STEP 1 PASS", flush=True)
        return 0
    except Exception as e:
        print(f"❌  SPIKE STEP 1 FAIL: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    finally:
        if args.shutdown_on_exit:
            try:
                server.land()
            except Exception as e:
                print(f"[warn] server.land() failed: {e}", file=sys.stderr)
        else:
            print(f"\n🔵  Datasite still running on http://127.0.0.1:{args.port}. "
                  f"Ctrl-C to stop, or:\n"
                  f"      python -c 'import syft as sy; "
                  f"sy.orchestra.launch(name=\"{DATASITE_NAME}\").land()'\n",
                  flush=True)
            # Block — the user can ^C when done.
            try:
                while True:
                    time.sleep(60)
            except KeyboardInterrupt:
                print("[info] shutdown requested")
                try:
                    server.land()
                except Exception:
                    pass


if __name__ == "__main__":
    sys.exit(main())
