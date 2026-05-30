# §4.3 — PySyft Mode B integration overhead

Drop-in for the brief. Numbers and statements wrapped in `<<PLACEHOLDER>>`
are filled from `analyze_pysyft_overhead.py`'s output after the experimental
runs complete (M1–M5 in `pysyft_spike/SPIKE_PLAN.md`). The narrative shape
is pre-committed so the section can be polished without writing it from
scratch.

---

## 4.3 PySyft as a workflow + governance layer

The encoder pipeline described in §4 is the smallest unit of confidential
evaluation: a single auditor request crosses the TEE boundary, an
attestation-bound bundle comes back. Production engagements need more — an
engagement-scoped budget, per-auditor scoping, an anomaly trigger on
suspect request distributions, and a mock-to-anchor development workflow
that lets auditors iterate without burning facility time on every keystroke.
The 2024 OpenMined / UK AISI / Anthropic pilot used **PySyft** (`syft 0.9.5`)
to provide exactly these guarantees, and this pre-pilot adopts the same
stack on top of the encoder.<sup>cite:openmined-pysyft, cite:openmined-aisi-pilot</sup>

### 4.3.1 Architecture

PySyft sits between the auditor's laptop and the encoder, **inside the TEE**:

```
auditor  ──verified proxy──▶  PySyft Datasite  ──loopback──▶  egress_service  ──loopback──▶  vLLM
                              (Mode B endpoints)             (encoder + ledger)            (GLM-5.1-FP8)
                                    ↓
                              engagement ledger
                              (SQLite, separate from encoder's
                               per-bundle ledger)
```

Four pre-approved `TwinAPIEndpoint`s expose the encoder's instrumentation
surface:

  - `prepilot.capture_residual_stream` — residual stream at probe layers
  - `prepilot.capture_routing` — MoE top-k routing across all 75 MoE layers
  - `prepilot.capture_attention_stats` — per-head entropy / row-max / top-mass
  - `prepilot.apply_steering` — EII-4 single-layer steering with an
    auditor-supplied direction

Each endpoint is a `TwinAPIEndpoint` with both a `mock_function` (returns a
zero-filled bundle of the right shape, audit-isomorphic for the
development phase of the §3.6 mock-to-anchor workflow) and a
`private_function` (the real path: cap check, loopback POST to the encoder,
engagement-ledger insert, bundle return). Both functions are pre-approved
at endpoint registration; the auditor reaches them as
`client.api.services.prepilot.<endpoint>(...)` without a per-call approval
gate.

### 4.3.2 Engagement-level governance

The encoder's existing per-session ledger (white-paper Tier-1 caps:
20K tokens, 50 plots, 50 exemplars per session) is unchanged. PySyft adds
an **engagement-level** SQLite ledger one level up, with `engagement`,
`session`, and `bundle` tables. Cap-checks (token / plot / exemplar)
happen inside a single `BEGIN IMMEDIATE` transaction with the session
insert, so concurrent workers cannot race past the cap. Cap-exceed returns
a typed `engagement_cap_exceeded` payload rather than raising; the driver
categorises by the `error` field.

A KL-divergence anomaly detector on plot-request distributions fires when
the recent window diverges from a benign baseline by more than a calibrated
threshold (calibration M4, results below).

### 4.3.3 Cost decomposition

The PySyft layer adds five measurable stages to each request:

  1. **workflow dispatch + auditor lookup** — PySyft worker thread receives
     the call and resolves `context.user_client.metadata.email`.
  2. **engagement cap + session start** — `BEGIN IMMEDIATE` txn; SELECT
     cap, SUM(used), INSERT session, COMMIT.
  3. **encoder call (loopback)** — `httpx.post` to `http://127.0.0.1:8002/v1/egress_eval`.
     This is the same call the existing egress driver measured; the only
     new cost is the inter-process hop within the TEE.
  4. **engagement-ledger insert** — one INSERT INTO bundle per call.
  5. **bundle return** — PySyft serialises the response dict back to the
     auditor client via `/api/v2/api_call`.

Table 9 reports the p50 and 95% bootstrap CIs for each stage from
`<<N_REQUESTS>>` paired requests across the four endpoints, both CC-on
and CC-off.

#### Table 9 — Per-stage PySyft overhead

<<PASTE OUTPUT OF analyze_pysyft_overhead.py / pysyft_overhead_table.md>>

#### Figure 9 — Stage decomposition

<<INSERT figures/stage_decomposition.pdf>>

### 4.3.4 PySyft overhead relative to the encoder baseline

A direct comparison: the `phase3_egress_driver_v2` cell at the matched CC
state is the same workload without the PySyft layer (same encoder, same
prompts, same engagement caps not enforced). Paired BCa over matched
`(pair_id, prompt_class)` pairs gives the headline:

> **CC-on:** PySyft adds `<<MEDIAN_ADDED_MS>>` ms (median) per request,
> `<<CI_LOW_MS>>`–`<<CI_HIGH_MS>>` ms 95% CI. Relative to the encoder's
> `<<EGRESS_WALL_P50_MS>>` ms p50, this is `<<RELATIVE_PCT>>%` overhead.
>
> **CC-off:** PySyft adds `<<MEDIAN_ADDED_MS_OFF>>` ms (median),
> `<<CI_LOW_MS_OFF>>`–`<<CI_HIGH_MS_OFF>>` ms 95% CI.
>
> The CC-on / CC-off difference within the PySyft layer itself is
> `<<PYS_CC_DELTA>>` ms — `<<INTERPRETATION>>` consistent with the platform
> CC band measured in §3 (the encoder dominates the CC tax; the PySyft
> layer is unaffected because its loopback hop never crosses the
> attestation boundary).

### 4.3.5 Cap enforcement

The cap-enforcement experiment (M3) provisions an engagement with a
deliberately small token budget and runs the auditor through it. The
engagement-ledger transaction semantics guarantee that the first call
that would exceed the cap returns `engagement_cap_exceeded`, not a partial
bundle. Result: `<<CAP_ENFORCEMENT_OUTCOME>>`.

### 4.3.6 Anomaly trigger

The KL-divergence detector is calibrated against `<<N_BENIGN_REQUESTS>>`
benign requests to set `threshold_nats = <<THRESHOLD>>`, then exercised on
a steganographic test corpus that encodes one bit per plot request by
biasing endpoint selection. The detector fires at request
`<<TRIGGER_INDEX>>` of `<<TOTAL_STEG_REQUESTS>>`, with KL = `<<KL_AT_TRIGGER>>`.
False-positive rate on the benign holdout: `<<FPR>>`.

### 4.3.7 Concurrent writes

The engagement ledger uses SQLite WAL mode and `BEGIN IMMEDIATE` around
every cap-check-then-insert sequence. M5 runs two simulated auditors in
parallel for `<<N_PARALLEL_SESSIONS>>` sessions each; the per-call
`pysyft_ledger_insert_seconds` distribution is `<<SUMMARY>>`. No
cap-violations are observed (the BEGIN IMMEDIATE serialises writes at the
DB boundary), and the median write time `<<grows / does not grow>>`
relative to single-auditor operation.

---

**What this section establishes for the white paper:**

- The mock-to-anchor workflow runs end-to-end against real infrastructure
  in CC mode, satisfying the §3.6 interface-contract requirement that
  development and final execution share the same Mode B surface.
- Engagement-level governance (token / plot / exemplar caps) sits cleanly
  on top of the encoder's existing Tier-1 export ledger, with measurable
  per-stage cost.
- The PySyft layer adds `<<MEDIAN_ADDED_MS>>` ms of overhead on a
  `<<EGRESS_WALL_P50_MS>>` ms encoder baseline — small enough that
  governance + scoping is a near-free addition relative to the cost the
  brief already measured for confidential evaluation.
