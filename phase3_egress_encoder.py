#!/usr/bin/env python3
"""
phase3_egress_encoder.py — Tier-1 evidence encoder for the egress measurement cells.

Consumes ``captures.CaptureResult`` outputs (already-deserialized numpy arrays
from the vllm-lens plugin) and produces a Tier-1-bounded evidence bundle:

    1. ``compute_aggregates``    — small deterministic transforms of the raw
                                    captures.  ~few KB output per prompt.
    2. ``render_plots``          — matplotlib server-side render of canonical
                                    plot templates from aggregates only.
    3. ``build_evidence_bundle`` — tar + ed25519 signature.
    4. ``ExportLedger``          — SQLite per-session token/byte counter.

Each stage is timed independently so phase3_aggregate.py can decompose the
egress overhead into stages.  Designed to plug into a thin driver that wraps
``captures.call_with_capture``; see ``phase3_egress_driver.py`` (separate file).

The measurement decomposition maps to four cells (matching the E0–E3 design
discussed in the white-paper egress-cost section):

    E0  baseline (existing C2-on)                   — no encoder, raw payload
    E1  + PySyft request/release transit only       — stages={}
    E2  + bounded aggregate                         — stages={"aggregate"}
    E3  + plots + bundle + ledger                   — stages={"aggregate", "plot",
                                                              "bundle", "ledger"}

The bounded aggregates are strictly smaller than the raw payload by 2–3 orders
of magnitude for the GLM-5.1-FP8 / probe-layer-set configuration (6 layers ×
6144 hidden × bf16 → ~75 KB/layer raw vs. ~4 KB/layer aggregate). This is what
makes the egress channel Tier-1-compatible per the white paper.

Dependencies beyond captures.py: matplotlib, cryptography, sqlite3 (stdlib).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")  # no display backend; required for headless rendering.
import matplotlib.pyplot as plt
import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

# Single-source-of-truth schema contracts.
from captures import (
    CaptureResult,
    GLM51_N_ROUTED_EXPERTS,
    GLM51_HIDDEN_SIZE,
)


SCHEMA_VERSION = "phase3-egress-encoder-v1"


# ====================================================================
# Bounded aggregates
# ====================================================================

@dataclass
class BoundedAggregates:
    """Compact summaries of a single prompt's captures.

    All numpy arrays are 1-D or 2-D with bounded sizes that do not depend on
    ``seq_len``.  Total size is on the order of:

        activations:   (n_layers,) means/stds + (n_layers, 32) histogram   ≈ 1 KB
        attention:     (n_layers, n_heads) entropy stats × 2               ≈ a few KB
        routing:       (n_layers, n_experts) utilization                    ≈ 75 KB
                       (GLM-5.1: 75 MoE layers × 256 experts × 4B = 76,800 B)

    Total worst case: ~80 KB per prompt — three orders of magnitude smaller
    than the raw 4–25 MB payload, well inside the Tier-1 export budget for a
    50-plot / 20K-token session.
    """

    # Per-layer activation summary.
    activation_layers: Optional[np.ndarray] = None              # (n_layers,) int32
    activation_norm_mean: Optional[np.ndarray] = None           # (n_layers,) f32
    activation_norm_std: Optional[np.ndarray] = None            # (n_layers,) f32
    activation_token_norms_hist: Optional[np.ndarray] = None    # (n_layers, n_bins) i32

    # Per-(layer, head) attention summary.
    attn_layers: Optional[np.ndarray] = None                    # (n_layers,) int32
    attn_entropy_mean: Optional[np.ndarray] = None              # (n_layers, n_heads) f32
    attn_entropy_std: Optional[np.ndarray] = None               # (n_layers, n_heads) f32
    attn_rowmax_mean: Optional[np.ndarray] = None               # (n_layers, n_heads) f32

    # MoE routing summary.
    routing_layers: Optional[np.ndarray] = None                 # (n_layers,) int32
    routing_expert_utilization: Optional[np.ndarray] = None     # (n_layers, n_experts) f32
    routing_entropy_mean: Optional[np.ndarray] = None           # (n_layers,) f32

    # Top-line scalars for ledger accounting.
    seq_len: int = 0
    tokens_in: int = 0
    tokens_out: int = 0

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize to plain JSON. Arrays become nested lists."""
        out: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "seq_len": int(self.seq_len),
            "tokens_in": int(self.tokens_in),
            "tokens_out": int(self.tokens_out),
        }
        for name in (
            "activation_layers", "activation_norm_mean", "activation_norm_std",
            "activation_token_norms_hist",
            "attn_layers", "attn_entropy_mean", "attn_entropy_std", "attn_rowmax_mean",
            "routing_layers", "routing_expert_utilization", "routing_entropy_mean",
        ):
            arr = getattr(self, name)
            out[name] = arr.tolist() if arr is not None else None
        return out

    def total_bytes(self) -> int:
        """Sum of ``.nbytes`` across all populated arrays. Used by the ledger."""
        total = 0
        for name in (
            "activation_layers", "activation_norm_mean", "activation_norm_std",
            "activation_token_norms_hist",
            "attn_layers", "attn_entropy_mean", "attn_entropy_std", "attn_rowmax_mean",
            "routing_layers", "routing_expert_utilization", "routing_entropy_mean",
        ):
            arr = getattr(self, name)
            if arr is not None:
                total += int(arr.nbytes)
        return total


def compute_aggregates(
    result: CaptureResult,
    *,
    n_norm_bins: int = 32,
    norm_range: tuple[float, float] = (0.0, 200.0),
) -> BoundedAggregates:
    """Single-prompt aggregate transform. This is **stage 2** of the egress
    pipeline (after raw-payload deserialization, before plot rendering).

    The transform is deterministic and uses only ``result``'s already-extracted
    numpy arrays (no I/O, no network).  Wall is roughly O(n_layers · seq_len · hidden)
    for the activation path and O(n_layers · seq_len · top_k) for routing —
    both well under 10 ms on commodity CPU for the GLM-5.1 probe-layer set.

    Args:
        result:        Output of ``captures.call_with_capture``.
        n_norm_bins:   Histogram bins for token-norm distribution per layer.
        norm_range:    Histogram clip range for activation norms. GLM-5.1
                       residual stream typically lives in [0, 200]; values
                       outside this range count as out-of-range. Bump for
                       other architectures.
    """
    ba = BoundedAggregates(
        seq_len=0,
        tokens_in=int(result.prompt_tokens or 0),
        # tokens_out is not present on CaptureResult directly; the driver fills
        # it from usage.completion_tokens when constructing metadata. Leave 0
        # here and let the driver override before passing to the bundler.
        tokens_out=0,
    )

    # ---- activations ----
    if result.activations:
        layers = sorted(result.activations.keys())
        ba.activation_layers = np.asarray(layers, dtype=np.int32)
        norm_means: list[float] = []
        norm_stds: list[float] = []
        hists: list[np.ndarray] = []
        seq_lens: list[int] = []
        for L in layers:
            arr = result.activations[L]  # (seq_len, hidden_size), already trimmed by plugin
            if arr.ndim == 3:
                arr = arr[0]
            seq_lens.append(int(arr.shape[0]))
            # L2 norm per token, summarised across the sequence.
            token_norms = np.linalg.norm(arr, axis=-1)
            norm_means.append(float(token_norms.mean()))
            norm_stds.append(float(token_norms.std()))
            h, _ = np.histogram(token_norms, bins=n_norm_bins, range=norm_range)
            hists.append(h.astype(np.int32))
        ba.activation_norm_mean = np.asarray(norm_means, dtype=np.float32)
        ba.activation_norm_std = np.asarray(norm_stds, dtype=np.float32)
        ba.activation_token_norms_hist = np.stack(hists)
        # seq_len should be identical across layers post-trim; take max to be safe.
        ba.seq_len = max(seq_lens) if seq_lens else 0

    # ---- attention stats ----
    if result.attention_stats and "per_head_entropy" in result.attention_stats:
        ent = result.attention_stats["per_head_entropy"]  # (n_layers, n_heads, seq_len)
        if ent.ndim != 3:
            # Schema drift. Don't silently squash to 2D — caller's assert layer
            # in captures.assert_attention_stats_valid should have caught this;
            # surfacing here as well so we fail loud at the encoder boundary.
            raise ValueError(
                f"per_head_entropy expected 3-D (n_layers, n_heads, seq_len); "
                f"got shape={ent.shape}"
            )
        ba.attn_entropy_mean = ent.mean(axis=-1).astype(np.float32)
        ba.attn_entropy_std = ent.std(axis=-1).astype(np.float32)

        rm = result.attention_stats.get("rowmax")
        if rm is not None and rm.ndim == 3:
            ba.attn_rowmax_mean = rm.mean(axis=-1).astype(np.float32)

        # layer_indices is what we want here; fall back to sequential if absent.
        li = result.attention_stats.get("layer_indices")
        if li is not None:
            ba.attn_layers = np.asarray(li, dtype=np.int32)
        else:
            ba.attn_layers = np.arange(ent.shape[0], dtype=np.int32)

    # ---- routing ----
    if result.routing and "topk_ids" in result.routing:
        topk_ids = result.routing["topk_ids"]  # int16, (n_layers, seq_len, top_k)
        n_layers = topk_ids.shape[0]
        # Top-1 expert per (layer, token) → frequency over the prompt's tokens.
        top1 = topk_ids[..., 0]  # (n_layers, seq_len)
        util = np.zeros((n_layers, GLM51_N_ROUTED_EXPERTS), dtype=np.float32)
        for L_idx in range(n_layers):
            row = top1[L_idx].ravel()
            # Clamp any spurious ids; assert_routing_valid should have already
            # rejected out-of-range, but defend against that being skipped.
            row = row[(row >= 0) & (row < GLM51_N_ROUTED_EXPERTS)]
            if row.size:
                counts = np.bincount(row, minlength=GLM51_N_ROUTED_EXPERTS).astype(np.float32)
                util[L_idx] = counts / counts.sum()
        ba.routing_expert_utilization = util

        if "routing_entropy" in result.routing:
            re = result.routing["routing_entropy"]  # (n_layers, seq_len)
            if re.ndim == 2:
                ba.routing_entropy_mean = re.mean(axis=-1).astype(np.float32)

        li = result.routing.get("layer_indices")
        if li is not None:
            ba.routing_layers = np.asarray(li, dtype=np.int32)
        else:
            ba.routing_layers = np.arange(n_layers, dtype=np.int32)

    return ba


# ====================================================================
# Plot renderer (server-side, bounded templates)
# ====================================================================

def render_plots(
    ba: BoundedAggregates,
    out_dir: Path,
    prompt_id: str,
    *,
    dpi: int = 80,
) -> list[Path]:
    """Render the canonical plot set from aggregates only. Stage 3.

    Each plot template takes only the bounded aggregate as input. No raw
    tensors, no auditor-controlled pixel content beyond what the aggregate
    contains. Output is a list of PNG paths.

    Templates rendered when their input is present:
        1. resnorm           — per-layer activation L2 norm (mean ± std)
        2. normhist          — per-layer token-norm distribution heatmap
        3. attn_entropy      — (layer, head) attention entropy heatmap
        4. routing_util      — (layer, expert) top-1 expert utilization heatmap
        5. routing_entropy   — per-layer routing entropy

    Returns:
        List of PNG paths written. Order matches template order above.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    # 1. resnorm
    if ba.activation_norm_mean is not None and ba.activation_layers is not None:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.errorbar(
            ba.activation_layers,
            ba.activation_norm_mean,
            yerr=ba.activation_norm_std,
            marker="o", capsize=3,
        )
        ax.set_xlabel("layer")
        ax.set_ylabel(r"$\|x\|_2$ (mean ± std across tokens)")
        ax.set_title(f"residual stream norm — {prompt_id}")
        ax.grid(alpha=0.3)
        p = out_dir / f"{prompt_id}_resnorm.png"
        fig.tight_layout()
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        paths.append(p)

    # 2. normhist
    if ba.activation_token_norms_hist is not None and ba.activation_layers is not None:
        fig, ax = plt.subplots(figsize=(6, 3))
        im = ax.imshow(
            ba.activation_token_norms_hist,
            aspect="auto", origin="lower", cmap="viridis",
        )
        ax.set_xlabel("norm bin")
        ax.set_ylabel("layer")
        ax.set_yticks(np.arange(len(ba.activation_layers)))
        ax.set_yticklabels(ba.activation_layers)
        ax.set_title(f"per-layer token-norm distribution — {prompt_id}")
        fig.colorbar(im, ax=ax, label="token count")
        p = out_dir / f"{prompt_id}_normhist.png"
        fig.tight_layout()
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        paths.append(p)

    # 3. attn_entropy
    if ba.attn_entropy_mean is not None and ba.attn_layers is not None:
        fig, ax = plt.subplots(figsize=(6, 3))
        im = ax.imshow(
            ba.attn_entropy_mean, aspect="auto", origin="lower", cmap="magma",
        )
        ax.set_xlabel("attention head")
        ax.set_ylabel("layer")
        ax.set_yticks(np.arange(len(ba.attn_layers)))
        ax.set_yticklabels(ba.attn_layers)
        ax.set_title(f"attention entropy (mean over seq) — {prompt_id}")
        fig.colorbar(im, ax=ax, label="nats")
        p = out_dir / f"{prompt_id}_attn_entropy.png"
        fig.tight_layout()
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        paths.append(p)

    # 4. routing_util
    if ba.routing_expert_utilization is not None:
        fig, ax = plt.subplots(figsize=(6, 3))
        im = ax.imshow(
            ba.routing_expert_utilization,
            aspect="auto", origin="lower", cmap="cividis",
        )
        ax.set_xlabel("expert id")
        ax.set_ylabel("MoE layer")
        ax.set_title(f"top-1 expert utilization — {prompt_id}")
        fig.colorbar(im, ax=ax, label="fraction")
        p = out_dir / f"{prompt_id}_routing_util.png"
        fig.tight_layout()
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        paths.append(p)

    # 5. routing_entropy
    if ba.routing_entropy_mean is not None and ba.routing_layers is not None:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(ba.routing_layers, ba.routing_entropy_mean, marker="o")
        ax.set_xlabel("MoE layer")
        ax.set_ylabel("routing entropy (mean over seq)")
        ax.set_title(f"routing entropy per layer — {prompt_id}")
        ax.grid(alpha=0.3)
        p = out_dir / f"{prompt_id}_routing_entropy.png"
        fig.tight_layout()
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        paths.append(p)

    return paths


# ====================================================================
# Evidence bundler
# ====================================================================

def build_evidence_bundle(
    aggregates: BoundedAggregates,
    plot_paths: list[Path],
    metadata: dict[str, Any],
    signing_key: Ed25519PrivateKey,
    out_path: Path,
) -> tuple[Path, int, str, str]:
    """Pack aggregates + plots + metadata into a tar, sign with ed25519.
    Stage 4.

    The signature is over the tar bytes; the tar itself is unencrypted (Tier-1
    artifacts are policy-bounded, not confidentiality-bounded). The IETF RATS
    attestation-bundle binding described in white paper §4.5 sits at a higher
    layer than this function — the signature here is the local artifact
    integrity stamp.

    Returns:
        (bundle_path, bundle_bytes, signature_hex, sha256_hex)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    agg_json = json.dumps(aggregates.to_json_dict(), separators=(",", ":")).encode()
    meta_json = json.dumps(metadata, default=str, separators=(",", ":")).encode()

    with tarfile.open(out_path, "w") as tar:
        _add_bytes(tar, "aggregates.json", agg_json)
        _add_bytes(tar, "metadata.json", meta_json)
        for p in plot_paths:
            tar.add(p, arcname=f"plots/{p.name}")

    bundle_bytes = out_path.read_bytes()
    sha = hashlib.sha256(bundle_bytes).hexdigest()
    signature = signing_key.sign(bundle_bytes)
    return out_path, len(bundle_bytes), signature.hex(), sha


def _add_bytes(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    """tarfile doesn't have a convenience method for in-memory bytes; assemble
    a TarInfo manually."""
    import io
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.mtime = int(time.time())
    tar.addfile(info, io.BytesIO(payload))


# ====================================================================
# Export ledger
# ====================================================================

class ExportLedger:
    """SQLite-backed per-session export accountant. Stage 5.

    Tracks tokens and bytes exported per session against the white-paper
    Tier-1 caps (20,000 generated tokens / 50 plots / 50 exemplars per
    session). The encoder driver records one row per request; the ledger
    enforces caps at query time via ``check_within_caps``.

    The DB lives at ``<out_dir>/export_ledger.sqlite`` so each cell run's
    ledger is self-contained. For multi-session aggregation (engagement-level
    budget per §4.4), use a shared DB path across cells.
    """

    SCHEMA = """
        CREATE TABLE IF NOT EXISTS ledger (
            ts            REAL    NOT NULL,
            session_id    TEXT    NOT NULL,
            request_id    INTEGER NOT NULL,
            tokens_in     INTEGER NOT NULL,
            tokens_out    INTEGER NOT NULL,
            aggregate_bytes INTEGER NOT NULL,
            bundle_bytes  INTEGER NOT NULL,
            bundle_sha256 TEXT    NOT NULL,
            n_plots       INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_session ON ledger(session_id);
    """

    def __init__(self, db_path: Path, session_id: str):
        self.db_path = db_path
        self.session_id = session_id
        self.conn = sqlite3.connect(str(db_path))
        self.conn.executescript(self.SCHEMA)
        self.conn.commit()

    def record(
        self,
        *,
        request_id: int,
        tokens_in: int,
        tokens_out: int,
        aggregate_bytes: int,
        bundle_bytes: int,
        bundle_sha256: str,
        n_plots: int,
    ) -> None:
        self.conn.execute(
            "INSERT INTO ledger VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                time.time(), self.session_id, request_id,
                tokens_in, tokens_out, aggregate_bytes,
                bundle_bytes, bundle_sha256, n_plots,
            ),
        )
        self.conn.commit()

    def session_totals(self) -> dict[str, int]:
        cur = self.conn.execute(
            "SELECT COALESCE(SUM(tokens_in), 0), COALESCE(SUM(tokens_out), 0), "
            "       COALESCE(SUM(aggregate_bytes), 0), COALESCE(SUM(bundle_bytes), 0), "
            "       COALESCE(SUM(n_plots), 0), COUNT(*) "
            "FROM ledger WHERE session_id = ?",
            (self.session_id,),
        )
        ti, to, ab, bb, np_, n = cur.fetchone()
        return {
            "tokens_in": int(ti),
            "tokens_out": int(to),
            "aggregate_bytes": int(ab),
            "bundle_bytes": int(bb),
            "n_plots": int(np_),
            "n_records": int(n),
        }

    def check_within_caps(
        self,
        *,
        max_tokens_out: int = 20_000,
        max_plots: int = 50,
    ) -> dict[str, Any]:
        """White-paper Tier-1 illustrative caps. Returns a status dict the
        driver can log or use to halt."""
        t = self.session_totals()
        return {
            "tokens_out_used": t["tokens_out"],
            "tokens_out_cap": max_tokens_out,
            "tokens_out_within": t["tokens_out"] <= max_tokens_out,
            "n_plots_used": t["n_plots"],
            "n_plots_cap": max_plots,
            "n_plots_within": t["n_plots"] <= max_plots,
        }

    def close(self) -> None:
        self.conn.close()


# ====================================================================
# Pipeline
# ====================================================================

# Stage names. The driver passes a subset of these to ``EgressPipeline``.
STAGE_AGGREGATE = "aggregate"
STAGE_PLOT = "plot"
STAGE_BUNDLE = "bundle"
STAGE_LEDGER = "ledger"
ALL_STAGES = (STAGE_AGGREGATE, STAGE_PLOT, STAGE_BUNDLE, STAGE_LEDGER)


@dataclass
class EgressTimings:
    """Per-request timing breakdown. Written one row per request to
    ``egress_timings.parquet`` for downstream aggregation."""

    request_id: int
    pair_id: int
    aggregate_seconds: float = 0.0
    plot_seconds: float = 0.0
    bundle_seconds: float = 0.0
    ledger_seconds: float = 0.0
    total_seconds: float = 0.0
    # Stage sizing (for cross-checking against bandwidth + ledger).
    aggregate_bytes: int = 0
    bundle_bytes: int = 0
    n_plots: int = 0
    # Stages that actually ran, for the E0/E1/E2/E3 audit trail.
    stages_run: tuple[str, ...] = field(default_factory=tuple)


class EgressPipeline:
    """Configurable pipeline. The set of enabled stages controls which
    E-cell variant runs::

        stages = set()                       → E1 (transit only, no encoder)
        stages = {STAGE_AGGREGATE}           → E2 (aggregate only, no plots)
        stages = {STAGE_AGGREGATE,
                  STAGE_PLOT,
                  STAGE_BUNDLE,
                  STAGE_LEDGER}              → E3 (full pipeline)

    E0 is the existing C2-on cell — does not go through this pipeline at all.
    """

    def __init__(
        self,
        *,
        stages: set[str],
        signing_key: Ed25519PrivateKey,
        out_dir: Path,
        ledger: Optional[ExportLedger] = None,
    ):
        unknown = stages - set(ALL_STAGES)
        if unknown:
            raise ValueError(f"unknown stage(s): {unknown}; valid={ALL_STAGES}")
        if STAGE_PLOT in stages and STAGE_AGGREGATE not in stages:
            raise ValueError("STAGE_PLOT requires STAGE_AGGREGATE")
        if STAGE_BUNDLE in stages and STAGE_AGGREGATE not in stages:
            raise ValueError("STAGE_BUNDLE requires STAGE_AGGREGATE")
        if STAGE_LEDGER in stages and STAGE_BUNDLE not in stages:
            raise ValueError("STAGE_LEDGER requires STAGE_BUNDLE")
        self.stages = stages
        self.signing_key = signing_key
        self.out_dir = out_dir
        self.ledger = ledger
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        result: CaptureResult,
        *,
        request_id: int,
        pair_id: int,
        tokens_in: int,
        tokens_out: int,
        metadata: Optional[dict[str, Any]] = None,
    ) -> EgressTimings:
        """Run the configured stages on one ``CaptureResult``. All stage
        timings are wall-clock perf_counter deltas; the caller's network
        latency to the egress endpoint is not included here (that's the
        PySyft transit slice measured at the driver layer)."""
        t_total_start = time.perf_counter()
        timings = EgressTimings(
            request_id=request_id,
            pair_id=pair_id,
            stages_run=tuple(sorted(self.stages)),
        )

        ba: Optional[BoundedAggregates] = None
        plot_paths: list[Path] = []
        bundle_path: Optional[Path] = None
        bundle_bytes = 0
        bundle_sha = ""

        # ---- stage 2: aggregate ----
        if STAGE_AGGREGATE in self.stages:
            t0 = time.perf_counter()
            ba = compute_aggregates(result)
            ba.tokens_in = tokens_in
            ba.tokens_out = tokens_out
            timings.aggregate_seconds = time.perf_counter() - t0
            timings.aggregate_bytes = ba.total_bytes()

        # ---- stage 3: plots ----
        if STAGE_PLOT in self.stages and ba is not None:
            t0 = time.perf_counter()
            plot_paths = render_plots(
                ba, self.out_dir / "plots", f"r{request_id:04d}"
            )
            timings.plot_seconds = time.perf_counter() - t0
            timings.n_plots = len(plot_paths)

        # ---- stage 4: bundle ----
        if STAGE_BUNDLE in self.stages and ba is not None:
            t0 = time.perf_counter()
            bundle_path = self.out_dir / "bundles" / f"r{request_id:04d}.tar"
            _, bundle_bytes, _sig, bundle_sha = build_evidence_bundle(
                aggregates=ba,
                plot_paths=plot_paths,
                metadata=metadata or {},
                signing_key=self.signing_key,
                out_path=bundle_path,
            )
            timings.bundle_seconds = time.perf_counter() - t0
            timings.bundle_bytes = bundle_bytes

        # ---- stage 5: ledger ----
        if STAGE_LEDGER in self.stages and self.ledger is not None and ba is not None:
            t0 = time.perf_counter()
            self.ledger.record(
                request_id=request_id,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                aggregate_bytes=timings.aggregate_bytes,
                bundle_bytes=bundle_bytes,
                bundle_sha256=bundle_sha or "",
                n_plots=len(plot_paths),
            )
            timings.ledger_seconds = time.perf_counter() - t0

        timings.total_seconds = time.perf_counter() - t_total_start
        return timings


# ====================================================================
# Key handling helpers
# ====================================================================

def load_or_create_signing_key(key_path: Path) -> Ed25519PrivateKey:
    """Load an ed25519 signing key from disk, or generate + persist if absent.

    The key represents the facility-controlled evidence encoder's identity.
    In production this would live in an HSM (white paper §4.5 attestation
    bundles). For the egress measurement, a file-backed key on the local
    machine is sufficient — we're measuring encoder cost, not key-management
    cost.
    """
    from cryptography.hazmat.primitives.serialization import (
        Encoding, PrivateFormat, NoEncryption, load_pem_private_key,
    )

    if key_path.exists():
        return load_pem_private_key(key_path.read_bytes(), password=None)  # type: ignore[return-value]
    key = Ed25519PrivateKey.generate()
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_bytes(
        key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
    )
    return key
