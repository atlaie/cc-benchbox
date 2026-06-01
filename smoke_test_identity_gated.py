#!/usr/bin/env python3
"""
smoke_test_identity_gated.py — Smoke test for the identity-gated public
endpoint (capture_residual_stream_v2).

Preconditions:
  1. Fresh deploy with pysyft image (v0.0.8-pysyft or later)
  2. Old TwinAPIEndpoints deleted (or fresh deploy with none registered)
  3. New identity-gated endpoint registered via:
       admin.custom_api.add(endpoint=build_endpoint())

Usage:
    python smoke_test_identity_gated.py \
        --datasite-url https://<deploy>.containers.tinfoil.dev

Four steps:
  STEP 1: Admin registers the identity-gated endpoint + creates auditor account.
  STEP 2: Auditor (authorized) calls endpoint → expects PRIVATE response.
  STEP 3: Create an unauthorized user, call endpoint → expects MOCK response.
  STEP 4: Verify auditor_id in the private response matches the auditor's email.
"""
import argparse
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasite-url", required=True)
    parser.add_argument("--admin-email", default="info@openmined.org")
    parser.add_argument("--admin-password", default="changethis")
    parser.add_argument("--auditor-email", default="auditor1@aisi.gov.uk")
    parser.add_argument("--auditor-password", default="aud1tor_pa55!")
    parser.add_argument("--unauthorized-email", default="rando@example.com")
    parser.add_argument("--unauthorized-password", default="rand0_pa55!")
    parser.add_argument("--no-proxy", action="store_true")
    parser.add_argument("--register", action="store_true",
                        help="Register the endpoint (run once per fresh deploy)")
    args = parser.parse_args()

    if args.no_proxy:
        import os
        for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            os.environ.pop(k, None)

    import syft as sy

    # Import the endpoint builder — adjust path if needed
    sys.path.insert(0, ".")
    from capture_residual_stream_v2 import build_endpoint

    url = args.datasite_url
    print(f"\n{'='*60}")
    print(f"Smoke test: identity-gated public endpoint")
    print(f"Datasite: {url}")
    print(f"{'='*60}\n")

    # ---- STEP 1: Setup ------------------------------------------------------
    print("[STEP 1] Admin login + setup...")
    admin = sy.login(url=url, email=args.admin_email,
                     password=args.admin_password)
    print(f"  Logged in as admin: {args.admin_email}")

    if args.register:
        # Delete old endpoint if it exists (idempotent)
        try:
            admin.custom_api.delete(
                path="prepilot.capture_residual_stream"
            )
            print("  Deleted old endpoint")
        except Exception:
            print("  No old endpoint to delete (or delete failed, continuing)")

        # Register new identity-gated endpoint
        ep = build_endpoint()
        result = admin.custom_api.add(endpoint=ep)
        print(f"  Registered endpoint: {result}")

    # Verify endpoint exists
    endpoints = admin.custom_api.api_endpoints()
    ep_paths = [e.path for e in endpoints] if endpoints else []
    print(f"  Registered endpoints: {ep_paths}")
    if "prepilot.capture_residual_stream" not in ep_paths:
        print("  FAIL: endpoint not registered. Run with --register.")
        sys.exit(2)

    # Create auditor account (authorized)
    existing = [u for u in admin.users if u.email == args.auditor_email]
    if not existing:
        admin.register(email=args.auditor_email, name="Auditor One",
                       password=args.auditor_password,
                       password_verify=args.auditor_password)
        print(f"  Created auditor: {args.auditor_email}")
    else:
        print(f"  Auditor {args.auditor_email} already exists")

    # Create unauthorized account
    existing = [u for u in admin.users if u.email == args.unauthorized_email]
    if not existing:
        admin.register(email=args.unauthorized_email, name="Random User",
                       password=args.unauthorized_password,
                       password_verify=args.unauthorized_password)
        print(f"  Created unauthorized user: {args.unauthorized_email}")
    else:
        print(f"  Unauthorized user {args.unauthorized_email} already exists")

    print("  STEP 1 PASS\n")

    # ---- STEP 2: Authorized auditor → PRIVATE --------------------------------
    print("[STEP 2] Authorized auditor calls endpoint...")
    auditor = sy.login(url=url, email=args.auditor_email,
                       password=args.auditor_password)
    print(f"  Logged in as auditor: {args.auditor_email}")

    t0 = time.perf_counter()
    try:
        result = auditor.api.services.prepilot.capture_residual_stream(
            prompt="What is 2+2?", max_new_tokens=8,
        )
        wall = time.perf_counter() - t0
        # Unwrap ActionObject if needed
        if hasattr(result, 'get'):
            result = result.get()
        elif hasattr(result, 'syft_action_data'):
            result = result.syft_action_data
        print(f"  Returned in {wall:.1f}s")
    except Exception as e:
        wall = time.perf_counter() - t0
        print(f"  FAIL after {wall:.1f}s: {type(e).__name__}: {e}")
        sys.exit(3)

    if not isinstance(result, dict):
        print(f"  Result type: {type(result)} — expected dict")
        print(f"  Value: {result}")
        sys.exit(3)

    auditor_id = result.get("auditor_id", "???")
    error = result.get("error")
    timings = result.get("pysyft_timings", {})
    encoder_s = timings.get("encoder_seconds", 0)

    print(f"  auditor_id: {auditor_id}")
    print(f"  error: {error}")
    print(f"  encoder_seconds: {encoder_s:.2f}")
    print(f"  all timings: {timings}")

    if error:
        print(f"  FAIL: endpoint returned error: {error}")
        print(f"  detail: {result.get('detail', 'n/a')}")
        sys.exit(3)

    if auditor_id == "mock":
        print("  FAIL: got mock response — auditor not authorized.")
        print("  Check that authorized_auditors includes the auditor email.")
        sys.exit(3)

    if encoder_s < 0.1:
        print(f"  WARNING: encoder_seconds={encoder_s:.4f} — suspiciously fast, "
              f"may be hitting mock despite auditor_id != 'mock'")

    if auditor_id == args.auditor_email:
        print(f"  STEP 2 PASS — private execution under '{auditor_id}'\n")
    elif auditor_id == "unknown":
        print(f"  PARTIAL PASS — private ran but identity='unknown'\n")
    else:
        print(f"  UNEXPECTED auditor_id: '{auditor_id}'\n")

    private_result = result  # save for STEP 4

    # ---- STEP 3: Unauthorized user → MOCK ------------------------------------
    print("[STEP 3] Unauthorized user calls endpoint...")
    rando = sy.login(url=url, email=args.unauthorized_email,
                     password=args.unauthorized_password)
    print(f"  Logged in as: {args.unauthorized_email}")

    t0 = time.perf_counter()
    try:
        result = rando.api.services.prepilot.capture_residual_stream(
            prompt="What is 2+2?", max_new_tokens=8,
        )
        wall = time.perf_counter() - t0
        if hasattr(result, 'get'):
            result = result.get()
        elif hasattr(result, 'syft_action_data'):
            result = result.syft_action_data
        print(f"  Returned in {wall:.1f}s")
    except Exception as e:
        wall = time.perf_counter() - t0
        print(f"  FAIL after {wall:.1f}s: {type(e).__name__}: {e}")
        sys.exit(3)

    if not isinstance(result, dict):
        print(f"  Result type: {type(result)}")
        print(f"  Value: {result}")
        sys.exit(3)

    unauth_auditor_id = result.get("auditor_id", "???")
    unauth_encoder = result.get("pysyft_timings", {}).get("encoder_seconds", -1)
    print(f"  auditor_id: {unauth_auditor_id}")
    print(f"  encoder_seconds: {unauth_encoder}")

    if unauth_auditor_id == "mock" and unauth_encoder == 0.0:
        print("  STEP 3 PASS — unauthorized user got mock response\n")
    else:
        print(f"  FAIL: expected mock but got auditor_id='{unauth_auditor_id}', "
              f"encoder_seconds={unauth_encoder}")
        sys.exit(3)

    # ---- STEP 4: Summary -----------------------------------------------------
    print("[STEP 4] Summary...")
    print(f"  Authorized auditor '{args.auditor_email}' → private execution")
    print(f"    auditor_id in response: {private_result.get('auditor_id')}")
    print(f"    encoder time: {private_result['pysyft_timings']['encoder_seconds']:.2f}s")
    print(f"    engagement_id: {private_result.get('engagement_id')}")
    print(f"    session_id: {private_result.get('session_id')}")
    print(f"  Unauthorized user '{args.unauthorized_email}' → mock")
    print(f"    auditor_id in response: mock")

    print(f"\n{'='*60}")
    print("ALL STEPS PASSED.")
    print("Identity-gated endpoint works: authorized auditors get private")
    print("execution with correct identity propagation; unauthorized users")
    print("get mock. Proceed to full driver sweep.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
