#!/usr/bin/env python3
"""
whoami_probe.py — determine which context attribute carries the caller's
identity on THIS PySyft build + Tinfoil debug transport.

`_common.py` resolves auditor_id via `context.user_client.metadata.email`,
which is returning the "unknown" fallback for every caller (admin + auditors
all collapse to one engagement). The official 0.9.5 custom-endpoints docs
specify `context.user.email` instead. This probe registers a PUBLIC endpoint
(must be a real file so PySyft's inspect.getsource succeeds — heredoc funcs
fail) that reports BOTH attributes plus a few other plausible carriers, then
calls it as admin and as auditor_a to see which (if any) carries the email.

Run from cc-benchbox under syft-spike:
    PYTHONPATH=/Users/atlaie/prepilot/cc-deep-eval python whoami_probe.py \
        --datasite-url https://pysyft-m1.debug.pour-demain.containers.tinfoil.dev
"""
import argparse
import os
import sys

import syft as sy


@sy.api_endpoint(path="diag.whoami",
                 description="report caller identity attributes (diagnostic)")
def whoami(context):
    out = {}

    def probe(label, getter):
        try:
            out[label] = getter()
        except Exception as e:  # noqa: BLE001
            out[label] = f"ERR {type(e).__name__}: {e}"

    # the path _common.py currently uses:
    probe("user_client.metadata.email",
          lambda: context.user_client.metadata.email)
    # the path the official 0.9.5 docs specify:
    probe("user.email", lambda: context.user.email)
    # other plausible carriers:
    probe("user.verify_key", lambda: str(context.user.verify_key))
    probe("user_client.email", lambda: context.user_client.email)
    probe("user_client.logged_in_user",
          lambda: context.user_client.logged_in_user)
    probe("type(context.user)", lambda: str(type(context.user)))
    probe("type(context.user_client)",
          lambda: str(type(context.user_client)))
    probe("context.user dir(email-ish)",
          lambda: [a for a in dir(context.user)
                   if any(k in a.lower() for k in ("email", "name", "key", "id"))])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasite-url", default=os.environ.get("DATASITE_URL"))
    ap.add_argument("--port", type=int, default=443)
    args = ap.parse_args()
    if not args.datasite_url:
        print("set --datasite-url or DATASITE_URL", file=sys.stderr)
        return 2

    admin = sy.login(url=args.datasite_url, port=args.port,
                     email="info@openmined.org", password="changethis")
    admin.refresh()

    # (re)register the diagnostic endpoint
    existing = {getattr(v, "path", None)
                for v in admin.custom_api.api_endpoints()}
    if "diag.whoami" in existing:
        for attempt in (
            lambda: admin.api.services.api.delete(endpoint_path="diag.whoami"),
            lambda: admin.custom_api.delete(endpoint_path="diag.whoami"),
            lambda: admin.api.services.api.delete("diag.whoami"),
        ):
            try:
                attempt(); break
            except Exception:  # noqa: BLE001
                continue
        admin.refresh()
    r = admin.custom_api.add(endpoint=whoami)
    print("add:", type(r).__name__)
    admin.refresh()

    callers = {
        "admin": ("info@openmined.org", "changethis"),
        "auditor_a": ("auditor_a@example.com", "auditor_password"),
    }
    for who, (em, pw) in callers.items():
        cl = sy.login(url=args.datasite_url, port=args.port,
                      email=em, password=pw)
        cl.refresh()
        res = cl.api.services.diag.whoami()
        res = res.get() if hasattr(res, "get") else res
        print(f"\n=== called as {who} ({em}) ===")
        if isinstance(res, dict):
            for k, v in res.items():
                print(f"  {k}: {v}")
        else:
            print(f"  (non-dict result: {type(res).__name__}) {res}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
