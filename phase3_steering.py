"""
phase3_steering.py — EII-4 steering condition helper.

Loads a precomputed RepE concept direction (output of Phase 2 D6) and
constructs the vllm_xargs payload for the apply_steering_vectors hook.

The payload is computed once per cell at startup; the same xargs dict is
used for every request in the cell, since steering is a per-cell static
condition (same direction, scale, layer for all prompts).

Working envelope from Phase 2 D6 validation (Cohen's d=2.857):
  - layer = 62
  - scale = 10.0   (scale=20 caused degenerate repetition)
  - sign = -1      (d_signed = toxic_mean - benign_mean; sign=-1 pushes toxic->benign)
  - norm_match = True
  - position_indices = None (apply at all positions)

Wire format references vllm-lens v1.1.0 SteeringVector Pydantic schema:
  - activations: dict (bf16-as-int16 + zstd + b64), shape (1, hidden_size)
  - layer_indices: [62]
  - scale: float
  - norm_match: bool
  - position_indices: optional list[int]

The full list is JSON-stringified per vLLM 0.20.0 vllm_xargs schema
constraints (no nested dicts at the top level).
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Optional

import numpy as np
import zstandard as zstd


HIDDEN_SIZE = 6144
DEFAULT_LAYER = 62
DEFAULT_SCALE = 10.0
DEFAULT_SIGN = -1
DEFAULT_NORM_MATCH = True


def _serialize_tensor_bf16(arr: np.ndarray) -> dict:
    """Encode a float32 numpy array as bf16-stored-as-int16 + zstd + b64.

    Identical to the D6 implementation. Matches the convention vllm-lens uses
    for residual-stream payloads; the server-side deserializer in
    `vllm_lens._helpers._serialize.deserialize_tensor` round-trips this format.
    """
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    f32_bits = arr.view(np.uint32)
    # Round-half-to-even on the discarded low 16 bits.
    rounded = (f32_bits + 0x7FFF + ((f32_bits >> 16) & 1)) >> 16
    bf16_as_uint16 = rounded.astype(np.uint16)
    int16_view = bf16_as_uint16.view(np.int16)
    raw = int16_view.tobytes()
    compressed = zstd.ZstdCompressor(level=1).compress(raw)
    return {
        "data": base64.b64encode(compressed).decode("ascii"),
        "dtype": "int16",
        "original_dtype": "torch.bfloat16",
        "shape": list(arr.shape),
        "compression": "zstd",
    }


def build_steer_xargs(
    direction_path: Path,
    layer: int = DEFAULT_LAYER,
    scale: float = DEFAULT_SCALE,
    sign: int = DEFAULT_SIGN,
    norm_match: bool = DEFAULT_NORM_MATCH,
    position_indices: Optional[list[int]] = None,
) -> dict:
    """Return the static vllm_xargs dict for the steer condition.

    Parameters
    ----------
    direction_path : Path
        Path to a .npy file containing a 1D float32 direction of length
        HIDDEN_SIZE (6144 for GLM-5.1). Output of Phase 2 D6.
    layer : int
        Single layer at which steering is applied. Default 62 (Phase 2 pick).
    scale : float
        Effective coefficient when norm_match=True. Default 10.0 (working envelope).
    sign : int (-1 or +1)
        With d_signed = toxic_mean - benign_mean, sign=-1 pushes toxic->benign.
    norm_match : bool
        If True, vllm-lens rescales the steering vector to match the residual's
        runtime norm at each apply site; `scale` then becomes the effective
        coefficient (typical RepE-norm-matched recipe).
    position_indices : list[int] or None
        Token positions to steer at. None = apply at all positions.

    Returns
    -------
    dict
        `{"apply_steering_vectors": "<json-stringified list of one payload>"}`,
        ready to merge into `vllm_xargs` and pass through `extra_body`.
    """
    d_unit = np.load(direction_path)
    if d_unit.ndim != 1 or d_unit.shape[0] != HIDDEN_SIZE:
        raise ValueError(
            f"direction at {direction_path} must be shape ({HIDDEN_SIZE},); "
            f"got {d_unit.shape}"
        )
    if sign not in (-1, +1):
        raise ValueError(f"sign must be -1 or +1; got {sign}")

    # Apply sign at construction time so the server-side payload is just
    # the effective direction; scale and norm_match handle magnitude.
    direction = (sign * d_unit).reshape(1, HIDDEN_SIZE).astype(np.float32)
    payload = {
        "activations": _serialize_tensor_bf16(direction),
        "layer_indices": [int(layer)],
        "scale": float(scale),
        "norm_match": bool(norm_match),
    }
    if position_indices is not None:
        payload["position_indices"] = [int(p) for p in position_indices]

    return {"apply_steering_vectors": json.dumps([payload])}


def summarize_steer_config(xargs: dict) -> dict:
    """Return a small, log-friendly summary of a steer xargs dict.

    Strips the large base64 `activations.data` blob so the orchestrator and
    summary.json can record the steering configuration without bloating
    artifacts by ~10 KB per cell.
    """
    payload_list = json.loads(xargs["apply_steering_vectors"])
    out = []
    for p in payload_list:
        activations_meta = {
            k: v for k, v in p.get("activations", {}).items() if k != "data"
        }
        out.append({
            "layer_indices": p.get("layer_indices"),
            "scale": p.get("scale"),
            "norm_match": p.get("norm_match"),
            "position_indices": p.get("position_indices"),
            "activations_meta": activations_meta,
        })
    return {"apply_steering_vectors": out}


if __name__ == "__main__":
    # Smoke: build the xargs from the default D6 path and print the config.
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--direction", type=Path,
                   default=Path("runs/phase2_d6_steering/direction_L62.npy"))
    p.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    p.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    p.add_argument("--sign", type=int, choices=[-1, +1], default=DEFAULT_SIGN)
    p.add_argument("--no-norm-match", action="store_true")
    args = p.parse_args()

    xargs = build_steer_xargs(
        args.direction,
        layer=args.layer,
        scale=args.scale,
        sign=args.sign,
        norm_match=not args.no_norm_match,
    )
    print(f"payload_kb: {len(xargs['apply_steering_vectors']) / 1024:.1f}")
    print(json.dumps(summarize_steer_config(xargs), indent=2))
