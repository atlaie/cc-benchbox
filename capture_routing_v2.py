"""
capture_routing_v2.py — Identity-gated public endpoint for MoE routing capture.

Same pattern as capture_residual_stream_v2.py: plain @sy.api_endpoint with
in-body identity gate, no TwinAPIEndpoint. See that file for design rationale.

Default layer set: 6-layer probe subset [12, 23, 39, 51, 62, 70] to stay
under the debug shim's ~60s result-return ceiling. The full 75-layer routing
(layers 3..77) exceeds that ceiling; pass layers=[3,4,...,77] only if the
transport supports it (attested build or streaming-aware shim).
"""

import syft as sy


_ENDPOINT_ID = "prepilot.capture_routing"
_DEFAULT_ROUTING_LAYERS = [12, 23, 39, 51, 62, 70]

_DEFAULT_AUTHORIZED = [
    "info@openmined.org",       # admin (backward compat)
    "auditor1@aisi.gov.uk",
    "auditor2@aisi.gov.uk",
]


@sy.api_endpoint(
    path=_ENDPOINT_ID,
    description=(
        "Capture MoE expert routing (top-k indices and weights) at specified "
        "layers. Default: 6-layer probe subset [12, 23, 39, 51, 62, 70]. "
        "Returns the encoder's Tier-1 bundle plus PySyft timing decomposition."
        "\n\n"
        "**Access control:** private execution is gated on the caller's "
        "identity. Authorized auditors get the real capture pipeline; "
        "all other callers receive a zero-filled mock response."
    ),
    settings={
        "authorized_auditors": _DEFAULT_AUTHORIZED,
    },
)
def capture_routing(
    context,
    prompt: str = "",
    layers: list = None,
    max_new_tokens: int = 32,
) -> dict:
    """Identity-gated endpoint body for MoE routing capture."""
    # --- Resolve caller identity ---
    try:
        caller_email = context.user.email
    except AttributeError:
        caller_email = "unknown"

    # --- Gate: authorized → private, otherwise → mock ---
    authorized = context.settings.get("authorized_auditors", [])
    if caller_email not in authorized:
        from pysyft_endpoints.endpoints._common import zero_filled_mock
        return zero_filled_mock(
            context,
            endpoint_id="prepilot.capture_routing",
            prompt=prompt,
        )

    # --- Private path ---
    from pysyft_endpoints.endpoints import _common
    from pysyft_endpoints.endpoints._common import call_endpoint

    _common.DEFAULT_MODEL = "glm-5-1"

    safe_layers = layers or [12, 23, 39, 51, 62, 70]
    # Sanitise: MoE layers are 3..77 on GLM-5.1 (layers 0..2 are dense)
    safe_layers = [int(L) for L in safe_layers if 3 <= int(L) <= 77]
    if not safe_layers:
        return {
            "error": "invalid_layers",
            "endpoint": "prepilot.capture_routing",
            "detail": "layers must be a non-empty list of ints in [3, 77]",
        }
    xargs = {"output_router_logits": safe_layers}

    return call_endpoint(
        context,
        endpoint_id="prepilot.capture_routing",
        prompt=prompt,
        xargs=xargs,
        max_new_tokens=max_new_tokens,
    )


def build_endpoint():
    """Returns the endpoint object for admin.custom_api.add()."""
    return capture_routing
