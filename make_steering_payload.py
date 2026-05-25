#!/usr/bin/env python3
"""
make_steering_payload.py
========================

Builds the `apply_steering_vectors` JSON payload for the Tier 1C
C_H-on-steer cell from Phase 2 D6's saved concept direction at L62.

Why a separate file
-------------------
vllm-lens v1.1.0 expects `apply_steering_vectors` in `vllm_xargs` to
be a JSON-stringified list of SteeringVector dicts (PHASE2_REFERENCE
§5.2 and §11.1 — the natural Python list-of-dicts is rejected at the
FastAPI boundary in vLLM 0.20.0 because `vllm_xargs` is typed as
`Dict[str, Union[str, int, float, List[scalars]]]`).

Pre-building the payload as a file and passing its path to the
driver is cleaner than embedding tensor serialization in the cell's
YAML config or building it inline at request time. Same pattern as
Phase 2's `d6_steering.py` save/load discipline.

What it does
------------
1. Loads `{toxic,benign}_residual.npz` from the D6 outputs
   (`runs/phase2_validation/repe_bundle/`).
2. Computes `d = mean(toxic[L62]) - mean(benign[L62])` and
   unit-normalizes.
3. Multiplies by `sign` (default -1, the known-good sign that
   suppresses refusal; see PHASE2_REFERENCE §8.2).
4. Reshapes to `(1, hidden_size)` per the SteeringVector pydantic
   constraint (`activations.dim() ∈ {2, 3}` AND
   `activations.shape[0] == len(layer_indices)`).
5. Serializes via Phase 1/2's canonical int16/zstd/base64 layout
   so the server-side `_helpers._serialize.deserialize_tensor`
   accepts it.
6. Wraps into a SteeringVector dict and dumps to a JSON file.

The driver loads this JSON, json.dumps the list (re-serializing it),
and sets it as `vllm_xargs["apply_steering_vectors"]` per the
strict-typing workaround.

Default parameters reproduce D6's PASSED smoke (scale=10, sign=-1,
norm_match=True, layer 62) on the held-out toxic prompt — a clean
break of refusal without collapsing into degenerate repetition
(scale=20 collapses; see PHASE2_REFERENCE §8.2 Verdict).

Usage
-----
    python make_steering_payload.py \\
        --residual-dir runs/phase2_validation/repe_bundle \\
        --layer 62 --scale 10.0 --sign -1 --norm-match \\
        --out steering_payload_L62_scale10.json

The output file is then referenced by the driver via the new
`--apply-steering-json` CLI flag (see DRIVER_PATCHES.md).
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np


def _serialize_bf16(arr: np.ndarray) -> dict:
    """
    Serialize a float32/float64 array to Phase 1/2's canonical
    int16/zstd/base64 layout that the server-side
    `_helpers._serialize.deserialize_tensor` accepts.

    Roundtrip: float32 → uint32 view → right-shift 16 → cast to
    int16 (bf16 bit pattern) → bytes → zstd → base64.

    Matches `vllm_lens._helpers._serialize.serialize_tensor` and
    Phase 1 §6.4 (PHASE1_REFERENCE.md).
    """
    try:
        import zstandard as zstd
    except ImportError:
        sys.exit("ERROR: install zstandard:\n    pip install zstandard")

    arr_f32 = np.ascontiguousarray(arr, dtype=np.float32)
    # bf16 truncation: take the top 16 bits of the float32 bit
    # pattern. This is the bit-equivalent of a float32 → bfloat16
    # cast on Intel/NVIDIA hardware (round-to-zero, no rounding).
    bits = arr_f32.view(np.uint32)
    bf16_bits = (bits >> 16).astype(np.uint16)
    int16_view = bf16_bits.view(np.int16)
    raw = int16_view.tobytes()
    compressed = zstd.ZstdCompressor(level=1).compress(raw)
    return {
        "data": base64.b64encode(compressed).decode("ascii"),
        "dtype": "int16",
        "original_dtype": "torch.bfloat16",
        "shape": list(arr.shape),
        "compression": "zstd",
    }


def build_direction(
    residual_dir: Path, layer: int
) -> tuple[np.ndarray, dict]:
    """
    Load Phase 2 residual npz files and compute the unit-normalized
    concept direction at the requested layer.

    Returns (d_unit, diagnostics_dict).
    """
    toxic_path = residual_dir / "toxic_residual.npz"
    benign_path = residual_dir / "benign_residual.npz"
    if not toxic_path.exists() or not benign_path.exists():
        sys.exit(
            f"ERROR: missing residual files under {residual_dir}.\n"
            f"  Expected: toxic_residual.npz, benign_residual.npz\n"
            f"  Generate via Phase 2 `repe_bundle` condition first."
        )

    layer_key = f"layer_{layer:03d}_last_tok"
    with np.load(toxic_path) as tox_npz, np.load(benign_path) as ben_npz:
        if layer_key not in tox_npz.files:
            sys.exit(
                f"ERROR: layer key {layer_key!r} not in {toxic_path}. "
                f"Available: {list(tox_npz.files)}"
            )
        tox = tox_npz[layer_key]  # (n_pairs, hidden_size)
        ben = ben_npz[layer_key]

    if tox.shape != ben.shape:
        sys.exit(f"ERROR: shape mismatch toxic {tox.shape} vs benign {ben.shape}")

    mean_tox = tox.mean(axis=0)
    mean_ben = ben.mean(axis=0)
    d = mean_tox - mean_ben
    norm_d = np.linalg.norm(d)
    if norm_d == 0.0:
        sys.exit("ERROR: difference-of-means direction has zero norm.")
    d_unit = d / norm_d

    diag = {
        "layer": layer,
        "n_pairs": int(tox.shape[0]),
        "hidden_size": int(tox.shape[1]),
        "norm_d_raw": float(norm_d),
        "cos_mean_tox_ben": float(
            np.dot(mean_tox, mean_ben)
            / (np.linalg.norm(mean_tox) * np.linalg.norm(mean_ben))
        ),
    }
    return d_unit.astype(np.float32), diag


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--residual-dir", type=Path,
                    default=Path("runs/phase2_validation/repe_bundle"),
                    help="Directory containing toxic_residual.npz and "
                         "benign_residual.npz (Phase 2 D6 outputs).")
    ap.add_argument("--layer", type=int, default=62,
                    help="Probe layer (default 62, the D6 RepE layer).")
    ap.add_argument("--scale", type=float, default=10.0,
                    help="Steering scale α. D6 verified scale=10 breaks "
                         "refusal cleanly; scale=20 collapses to "
                         "degenerate repetition. Default 10.0.")
    ap.add_argument("--sign", type=int, choices=[-1, 1], default=-1,
                    help="Direction sign. -1 suppresses refusal "
                         "(steer AWAY from toxic-mean direction); "
                         "+1 strengthens it. D6 used -1. Default -1.")
    ap.add_argument("--norm-match", action="store_true", default=True,
                    help="Apply norm_match=True (RepE-canonical; "
                         "scale becomes α relative to runtime residual "
                         "norm). Default True.")
    ap.add_argument("--no-norm-match", dest="norm_match",
                    action="store_false",
                    help="Apply norm_match=False (absolute scale).")
    ap.add_argument("--out", type=Path,
                    default=Path("steering_payload.json"),
                    help="Output JSON file path.")
    args = ap.parse_args()

    d_unit, diag = build_direction(args.residual_dir, args.layer)
    print(f"Direction at L{args.layer}: norm_d_raw={diag['norm_d_raw']:.3f}, "
          f"cos(means)={diag['cos_mean_tox_ben']:.3f}, "
          f"n_pairs={diag['n_pairs']}", file=sys.stderr)

    d_signed = (args.sign * d_unit).reshape(1, -1)
    activations_blob = _serialize_bf16(d_signed)

    steering_vector = {
        "activations": activations_blob,
        "layer_indices": [args.layer],
        "scale": float(args.scale),
        "norm_match": bool(args.norm_match),
        # position_indices omitted → applied at all positions.
    }

    # Wrap in a list (SteeringVector schema is a list, even for a
    # single vector). The driver json.dumps([this_dict]) before
    # placing it in vllm_xargs (PHASE2_REFERENCE §5.2).
    payload = {
        "schema": "vllm_lens.SteeringVector_v1",
        "diagnostics": diag,
        "parameters": {
            "layer": args.layer,
            "scale": args.scale,
            "sign": args.sign,
            "norm_match": args.norm_match,
        },
        "steering_vectors": [steering_vector],
    }

    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"Wrote {args.out} ({args.out.stat().st_size} bytes)",
          file=sys.stderr)
    print(f"\nSummary:", file=sys.stderr)
    print(f"  layer       : {args.layer}", file=sys.stderr)
    print(f"  scale       : {args.scale}", file=sys.stderr)
    print(f"  sign        : {args.sign}", file=sys.stderr)
    print(f"  norm_match  : {args.norm_match}", file=sys.stderr)
    print(f"  activations : shape={d_signed.shape}, dtype=bf16-as-int16",
          file=sys.stderr)


if __name__ == "__main__":
    main()
