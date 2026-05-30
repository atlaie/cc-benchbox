#!/usr/bin/env python3
"""
pysyft_spike_call.py — Step 2 acceptance test for the §3.2 spike.

Run from your laptop against the Tinfoil-deployed PySyft Datasite. Confirms:
  (a) /api/v2/metadata is reachable through the Tinfoil shim
  (b) sy.login() succeeds (PySyft session-cookie auth on top of shim bearer)
  (c) custom_api.prepilot.hello() round-trips and returns the expected string

Usage:

    export TINFOIL_API_KEY="<your bearer>"
    export PYSYFT_DATASITE_URL="https://pysyft-spike-<id>.containers.tinfoil.dev"
    python pysyft_spike_call.py --url "$PYSYFT_DATASITE_URL"

Exit codes:
    0  full pass (all three acceptance criteria met)
    1  any failure; stderr carries the diagnosis
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from urllib.parse import urlparse

import httpx

try:
    import syft as sy
except ImportError as e:
    print(f"[fatal] syft import failed: {e}", file=sys.stderr)
    sys.exit(1)


ADMIN_EMAIL = "info@openmined.org"
ADMIN_PASSWORD = "changethis"   # PySyft 0.9.5 default — fine for the spike.


def check_metadata(url: str, bearer: str | None) -> dict:
    """(a) Confirm the shim routes /api/v2/metadata through to PySyft."""
    headers = {"Authorization": f"Bearer {bearer}"} if bearer else {}
    r = httpx.get(f"{url.rstrip('/')}/api/v2/metadata", headers=headers, timeout=30.0)
    r.raise_for_status()
    body = r.json()
    name = body.get("name") or body.get("server_name") or "?"
    print(f"[a] /api/v2/metadata OK  name={name!r}")
    return body


def check_login(url: str) -> "sy.DatasiteClient":
    """(b) Confirm sy.login() handshake works through the shim.

    PySyft's client auto-derives host/port from `url`. We pass `url=` rather
    than `port=` so the laptop client uses HTTPS straight to the Tinfoil shim.
    """
    parsed = urlparse(url)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    print(f"[b] sy.login(url={url}, email={ADMIN_EMAIL}) ...")
    t0 = time.perf_counter()
    client = sy.login(url=url, port=port, email=ADMIN_EMAIL, password=ADMIN_PASSWORD)
    print(f"[b] sy.login OK in {time.perf_counter() - t0:.2f}s; "
          f"client={type(client).__name__}")
    return client


def check_endpoint(client: "sy.DatasiteClient") -> None:
    """(c) Confirm the registered endpoint round-trips.

    The endpoint was registered by `pysyft_datasite_spike_server.py` inside
    the container at startup. We just call it.

    PySyft 0.9.5 call path: `client.refresh()` then
    `client.api.services.<dotted.path>(...)`. `client.custom_api` is the
    admin namespace (add/list/delete endpoints), not the invocation path.
    """
    client.refresh()
    print(f"[c] client.api.services.prepilot.hello() ...")
    t0 = time.perf_counter()
    result = client.api.services.prepilot.hello()
    elapsed = time.perf_counter() - t0
    s = str(result)
    if "hello from datasite" not in s:
        raise RuntimeError(f"unexpected endpoint return value: {s!r}")
    print(f"[c] OK in {elapsed:.2f}s → {s!r}")
    if elapsed > 5.0:
        print(f"[c][warn] round-trip > 5s; Tinfoil shim or PySyft routing may be slow")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("PYSYFT_DATASITE_URL"),
                     help="https://<deploy>.containers.tinfoil.dev "
                          "(or http://127.0.0.1:8000 for the local Step 1 check)")
    ap.add_argument("--api-key", default=os.environ.get("TINFOIL_API_KEY"),
                     help="Tinfoil shim bearer token. Required for shim-auth deploys; "
                          "omit for local Step 1.")
    args = ap.parse_args()

    if not args.url:
        print("[fatal] --url required (or set PYSYFT_DATASITE_URL).", file=sys.stderr)
        return 1

    print(f"[spike] target={args.url}", flush=True)
    try:
        check_metadata(args.url, args.api_key)
        client = check_login(args.url)
        check_endpoint(client)
    except Exception as e:
        print(f"\n❌ SPIKE STEP 2 FAIL: {type(e).__name__}: {e}", file=sys.stderr)
        # Common failure modes worth pre-mapping for the writeup:
        #   httpx ConnectError                → Tinfoil shim doesn't route to PySyft
        #   httpx HTTPStatusError 401         → shim accepts request but PySyft auth misconfigured
        #   sy.login → SyftException timeout  → PySyft up but slow / blocking on something
        #   "endpoint not found"              → custom_api.add failed inside the container
        return 1
    print("\n✅ SPIKE STEP 2 PASS — Phase 1 build cleared")
    return 0


if __name__ == "__main__":
    sys.exit(main())
