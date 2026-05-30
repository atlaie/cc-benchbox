# PySyft Datasite feasibility spike — `prepilot/cc-benchbox`

> **Goal.** Confirm PySyft 0.9.5 Datasite deploys in a Tinfoil container and
> exposes Mode B `@sy.api_endpoint` calls through the Tinfoil shim. **No
> vLLM, no encoder.** Yes/no in half a day. Pass → commit to Phase 1 build;
> fail → report blocker, pivot to robustness.
>
> Per `handoff_robustness_and_pysyft.md` §3.2.

## Locked design decisions

| Decision | Choice | Rationale |
|---|---|---|
| PySyft version | `syft>=0.9.5,<0.9.6` | Current upstream stable (`sy.requires` in README). |
| Mode B primitive | `@sy.api_endpoint(path=...)` | Pre-approved; no human-in-the-loop approval gate. |
| Datasite launch | `sy.orchestra.launch(port=8000, dev_mode=False, ...)` | In-process; SQLite backing avoids postgres-init blocker. |
| Mock-to-anchor (Phase 1) | `sy.TwinAPIEndpoint(mock_function=, private_function=)` | Anchors to white paper §3.6. Mock = public-data stub; private = real encoder. |
| Engagement ledger backing | Custom SQLite | Measuring write contention is part of the experiment (§3.4 run 3). `context.state` would hide it. |
| Port plan (Phase 1) | 8000 PySyft public; 8001 vLLM loopback; 8002 egress loopback | Matches existing egress image's loopback discipline. |

## Spike steps

### Step 1 — Local API surface discovery (laptop, ~30 min)

PySyft's HTTP route inventory is needed to populate the Tinfoil shim's
`paths:` allowlist. The shim is an explicit allowlist; PySyft is FastAPI
under the hood, so we extract from its OpenAPI dump.

```bash
# Fresh venv to avoid polluting the cc-benchbox env.
python3.12 -m venv .venv-syft-spike
source .venv-syft-spike/bin/activate
pip install 'syft>=0.9.5,<0.9.6'

# Launch local Datasite + dump OpenAPI.
python pysyft_spike_local.py
```

Output: `openapi_paths.txt` — list of every route the Datasite exposes.
Eyeball it, then update the `paths:` list in `tinfoil-config-pysyft-spike.yml`.

**Exit Step 1:** local Datasite reaches `READY`, `/openapi.json` returns,
the local registration script (`@sy.api_endpoint(path="prepilot.hello")`
→ `client.custom_api.prepilot.hello()`) returns `"hello from datasite"`.

### Step 2 — Tinfoil deploy (~2 h end-to-end)

1. Build `prepilot-vllm-lens-pysyft-spike` from `Dockerfile.pysyft-spike`:

   ```bash
   docker build -t ghcr.io/atlaie/prepilot-vllm-lens-pysyft-spike:v0.0.1 \
       -f Dockerfile.pysyft-spike .
   docker push ghcr.io/atlaie/prepilot-vllm-lens-pysyft-spike:v0.0.1
   ```

2. Deploy on a non-CC Tinfoil host with `--debug` for SSH access:

   ```bash
   tinfoil container create pysyft-spike \
       --repo atlaie/cc-deep-eval \
       --tag v0.0.1-pysyft-spike \
       --disable-cc-mode \
       --debug \
       --host control.inf8.tinfoil.sh \
       --yes
   ```

   (Config: `tinfoil-config-pysyft-spike.yml`. No GPU needed.)

3. Watch `docker logs` for `🌍 server is ready` (PySyft's orchestra ready marker).

4. Smoke from laptop:

   ```bash
   export PYSYFT_DATASITE_URL=https://pysyft-spike-<id>.containers.tinfoil.dev
   python pysyft_spike_call.py --url "$PYSYFT_DATASITE_URL"
   ```

**Exit Step 2 (all must pass):**

- `sy.login(url=$PYSYFT_DATASITE_URL, email=..., password=...)` returns a `DatasiteClient`.
- `client.custom_api.prepilot.hello()` returns `"hello from datasite"`.
- Round-trip wall under 5 s (sanity for the Tinfoil shim adding no pathological latency).

## Exit decision

**PASS → Phase 1 build begins immediately.**  Deliverables in priority order:

1. `pysyft_endpoints/ledger/{schema.sql, engagement_ledger.py}` — engagement / session / bundle schema, fail-closed 403 on cap-exceed.
2. `pysyft_endpoints/endpoints/{capture_residual_stream, capture_routing, capture_attention_stats, apply_steering}.py` — four `TwinAPIEndpoint`s, each POSTing to `127.0.0.1:8002/v1/egress_eval` with the right `vllm_xargs` from `captures.py`. Per-stage `perf_counter()` timings returned in response.
3. `phase3_pysyft_driver.py` — laptop-side, mirrors `phase3_egress_driver_v2.EgressRowV2` schema with `pysyft_workflow_seconds`, `pysyft_approval_gate_seconds`, `pysyft_ledger_insert_seconds` added.
4. `pysyft_endpoints/anomaly/plot_request_distribution.py` — KL-divergence anomaly trigger calibrated against a baseline distribution.
5. `analyze_pysyft_overhead.py` — reuses `analyze_cc_deltas.bootstrap_paired_delta`. Output row matches the brief's Table 9 stage decomposition.
6. Three experimental runs (cap-enforcement / anomaly / concurrent), each with its own subdir under `runs/phase3_pysyft/`.
7. `pysyft_writeup.md` — drop-in §4.3 for the brief.

**FAIL → report blocker and pivot to robustness (Exp 7).**

Specific failure modes worth distinguishing in the writeup if we end up there:
- PySyft Datasite container OOMs or hangs on init (postgres / migration issue inside Tinfoil's filesystem).
- Tinfoil shim's WebSocket / HTTP routing breaks PySyft's transport (PySyft uses HTTP only by default in 0.9.5, but worth confirming).
- TLS / auth handshake fails (PySyft assumes its own JWT lifecycle; the Tinfoil shim's bearer-token model may collide).

## Phase 1 architecture (for review before I build it)

```
laptop
  │   phase3_pysyft_driver.py  (httpx + syft client)
  │
  ▼
Tinfoil deploy (CC-on / CC-off, --debug, pinned host)
  │
  ├── port 8000 (public, shim-authenticated)
  │     │ PySyft 0.9.5 Datasite
  │     │   ├── @sy.api_endpoint("prepilot.capture_residual_stream")
  │     │   ├── @sy.api_endpoint("prepilot.capture_routing")
  │     │   ├── @sy.api_endpoint("prepilot.capture_attention_stats")
  │     │   └── @sy.api_endpoint("prepilot.apply_steering")
  │     │
  │     │ Each endpoint body:
  │     │   1. auditor_id = context.user_client.metadata.email
  │     │   2. ledger.start_session_or_403(auditor_id)
  │     │   3. xargs = build_xargs_for_endpoint()    # from captures.CONDITION_PRESETS
  │     │   4. httpx.post("http://127.0.0.1:8002/v1/egress_eval", json=...)
  │     │   5. ledger.record_bundle(session_id, ...)
  │     │   6. return {bundle: ..., timings: {...}}
  │
  ├── port 8001 (loopback)
  │     vLLM 0.20.0 (GLM-5.1-FP8, byte-identical to v0.0.25)
  │
  └── port 8002 (loopback)
        egress_service.py (existing FastAPI, unmodified)
          └── /v1/egress_eval → captures + EgressPipeline + ExportLedger
```

The new `engagement_ledger.py` wraps `phase3_egress_encoder.ExportLedger`
with engagement/session/bundle accounting. Per-session caps (20K tokens,
50 plots, 50 exemplars) are unchanged — the cap added is the
**engagement-level** ceiling (default 100K tokens, 200 plots, 200 exemplars).

The 4 endpoints are deliberately *thin* over the existing encoder: no
changes to `phase3_egress_encoder.py`, no changes to `egress_service.py`.
PySyft is purely a workflow + governance layer wrapping the same encoder
the brief already measured. This is what makes the per-stage decomposition
(workflow / approval / encoder / ledger / bundle-return) directly
comparable to the existing Table 9.
