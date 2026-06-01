"""
capture_residual_stream_v2.py — Identity-gated public endpoint.

Replaces the TwinAPIEndpoint (which routes by PySyft role) with a plain
public endpoint where the function body gates private vs mock execution
based on the caller's email against an admin-controlled whitelist in
`settings`.

Design rationale:
  PySyft 0.9.5 TwinAPIEndpoints route DATA_SCIENTIST callers to the mock
  function. The intended promotion path (syft_function_single_use wrapping
  the endpoint) does not resolve TwinAPIEndpoint references to private
  execution upon approval — it still dispatches mock. This is a PySyft
  limitation (filed upstream).

  The workaround: a plain @sy.api_endpoint (public, pre-approved) with an
  in-body identity gate. The admin controls who gets private execution via
  the `authorized_auditors` settings list. This is arguably more realistic
  for the pilot than PySyft's generic role routing: the facility operator
  controls per-evaluator access to specific capture primitives, not just a
  blanket role-based mock/private split.

  _common.py (baked in the image) is unchanged — call_endpoint already
  resolves context.user.email for auditor identity and the ledger. The gate
  sits in front of it, in the hot-swappable endpoint body.

Registration:
  admin.custom_api.add(endpoint=build_endpoint())

  If the old TwinAPIEndpoint for the same path is already registered,
  delete it first:
      admin.custom_api.delete(path="prepilot.capture_residual_stream")
  then add the new one.
"""

import syft as sy


_ENDPOINT_ID = "prepilot.capture_residual_stream"
_DEFAULT_LAYERS = [12, 23, 39, 51, 62, 70]

# Auditor emails authorized for private execution.
# Passed via settings so the admin can update them without rebuilding.
# The admin's own email is included so existing admin-path runs still work.
_DEFAULT_AUTHORIZED = [
    "info@openmined.org",       # admin (backward compat)
    "auditor1@aisi.gov.uk",     # AISI evaluator
    "auditor2@aisi.gov.uk",     # second evaluator (for isolation demo)
]


@sy.api_endpoint(
    path=_ENDPOINT_ID,
    description=(
        "Capture residual-stream activations at the GLM-5.1 probe layers "
        "([12, 23, 39, 51, 62, 70] by default). Returns the encoder's "
        "Tier-1 bundle (aggregates + plots + signed tar) plus a five-stage "
        "PySyft timing decomposition.\n\n"
        "**Access control:** private execution is gated on the caller's "
        "identity. Authorized auditors get the real capture pipeline; "
        "all other callers receive a zero-filled mock response of the "
        "same shape."
    ),
    settings={
        "authorized_auditors": _DEFAULT_AUTHORIZED,
    },
)
def capture_residual_stream(
    context,
    prompt: str = "",
    layers: list = None,
    max_new_tokens: int = 32,
) -> dict:
    """Identity-gated endpoint body.

    If the caller is in the authorized_auditors whitelist → private path
    (real encoder, real ledger, real bundle).
    Otherwise → zero-filled mock.

    All imports are inline (PySyft serialises the function body).
    """
    # --- Resolve caller identity ---
    try:
        caller_email = context.user.email
    except AttributeError:
        caller_email = "unknown"

    # --- Gate: authorized → private, otherwise → mock ---
    authorized = context.settings.get("authorized_auditors", [])
    if caller_email not in authorized:
        # Return mock — same shape as private, all zeros
        from pysyft_endpoints.endpoints._common import zero_filled_mock
        return zero_filled_mock(
            context,
            endpoint_id="prepilot.capture_residual_stream",
            prompt=prompt,
        )

    # --- Private path: real capture pipeline ---
    from pysyft_endpoints.endpoints import _common
    from pysyft_endpoints.endpoints._common import call_endpoint

    # Hot-override model name (matches tinfoil-config served-model-name)
    _common.DEFAULT_MODEL = "glm-5-1"

    safe_layers = layers or [12, 23, 39, 51, 62, 70]
    safe_layers = [int(L) for L in safe_layers if 0 <= int(L) <= 77]
    if not safe_layers:
        return {
            "error": "invalid_layers",
            "endpoint": "prepilot.capture_residual_stream",
            "detail": "layers must be a non-empty list of ints in [0, 77]",
        }
    xargs = {"output_residual_stream": safe_layers}

    return call_endpoint(
        context,
        endpoint_id="prepilot.capture_residual_stream",
        prompt=prompt,
        xargs=xargs,
        max_new_tokens=max_new_tokens,
    )


def build_endpoint():
    """Returns the endpoint object for admin.custom_api.add().

    This is a plain api_endpoint (already decorated above), not a
    TwinAPIEndpoint. The function is returned directly.
    """
    return capture_residual_stream
