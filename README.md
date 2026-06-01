# cc-benchbox

Measurement harness for [*An empirical study of Confidential Compute for frontier AI evaluations*](https://pourdemain.ngo/research) (Pour Demain, May 2026). This repo contains the laptop-side drivers, matrix orchestrator, analysis scripts, and prompt data that produced the results in the technical brief.

The companion repo [`cc-deep-eval`](https://github.com/pourdemain/cc-deep-eval) contains the Tinfoil-deployed vLLM target images, vllm-lens plugin, PySyft Datasite server, and server-side endpoint definitions.

Run data (parquet + JSON) is archived at [Zenodo: TODO-DOI](https://zenodo.org/TODO).

## What this repo does

Orchestrates paired CC-on / CC-off measurement runs against GLM-5.1-FP8 and Llama-3.1-70B-FP8 served by vLLM 0.20.0 inside Tinfoil Containers (Intel TDX, 8×H200 SXM). Dispatches prompts, records wall-time and token counts, computes paired BCa bootstrap CIs, and produces the tables and figures in the brief.

## Structure

```
drivers/dispatch
  phase3_vllm_driver.py          # sequential baseline driver
  phase3_vllm_driver_stream.py   # streaming variant (TTFT capture)
  phase3_vllm_driver_concurrent.py # concurrent dispatch (semaphore-bounded)
  phase3_grad_driver.py          # gradient sidecar driver (EII-2)
  phase3_pysyft_driver.py        # PySyft Mode-B governed-egress driver
  phase3_egress_driver_v2.py     # Tier-1 egress pipeline driver
  streaming_client.py            # SSE parsing for streaming dispatch
  m5_concurrent.py               # multi-evaluator concurrent driver

orchestration
  phase3_run_matrix.py           # v3-grouped matrix orchestrator
  phase3_run_cell.py             # per-cell wrapper
  phase3_aggregate.py            # cross-cell aggregation (parquet → summary)
  phase3_aggregate_egress.py     # egress-specific aggregation
  phase3_sweep_max_tokens.py     # max_tokens sweep driver
  phase3_steering.py             # RepE steering dispatch

analysis
  analyze_cc_deltas.py           # per-cell paired BCa bootstrap CIs
  analyze_combined_phase.py      # two-feature OLS (Tables 4–6, Figure 4)
  analyze_concurrency_sweep.py   # concurrency invariance (Figure 6)
  analyze_harmbench.py           # cross-corpus HarmBench replication
  analyze_max_tokens_sweep.py    # max_tokens sweep analysis
  analyze_phase_decomposition.py # prefill/decode phase split
  analyze_pysyft_overhead.py     # PySyft governance-layer overhead
  analyze_thinking.py            # GSM8K + thinking-mode extrapolation
  analyze_interdeploy.py         # inter-deploy variance (robustness)
  analyze_interdeploy_bootstrap.py
  analyze_invariance_effect_sizes.py
  analyze_mde.py                 # minimum detectable effect
  analyze_ols_sensitivity.py     # leave-one-cell-out OLS sensitivity

figures
  make_headline_two_panel.py     # Figure 1 (left + inset)
  make_headline_three_panel.py   # Figure 1 (three-panel variant)
  make_extrapolation_figure.py   # Figure 5 (GSM8K extrapolation)

data builders
  build_pairs.py                 # 100-pair ToxicChat prompt set
  build_pairs_500.py             # 500-pair extension (concurrency sweep)
  build_ruler_pairs.py           # RULER NIAH pairs (tokens_in sweep)
  prepare_harmbench_pairs.py     # HarmBench cross-corpus pairs
  prepare_gsm8k_pairs.py         # GSM8K + thinking-mode pairs
  make_steering_payload.py       # RepE direction → JSON steering payload

governance / monitoring
  run_m4_anomaly.py              # three-tier adversary ladder (§5.3)
  plot_request_distribution.py   # shape (KL) + regularity (CCE) detector
  run_comprehensive_s5.sh        # §5 comprehensive governance run
  smoke_test_identity_gated.py   # PySyft identity-gate smoke test
  whoami_probe.py                # PySyft identity resolution probe

probes (validation)
  probe_t1_fresh_single.py       # single-request probe
  probe_t2_idle_gap.py           # idle-gap timeout probe
  probe_t3_payload_size.py       # payload-size invariance probe
  probe_t4_token_budget.py       # token-budget cap enforcement probe
  probe_common.py                # shared probe utilities

data/
  pairs.json                     # 100 ToxicChat paired prompts (primary matrix)
  pairs-500.json                 # 500-pair extension
  pairs-660.json                 # 660-pair extension (robustness)
  pairs-harmbench-50pairs.json   # HarmBench cross-corpus set
  pairs-gsm8k.json               # GSM8K thinking-mode set
  pairs_ruler/                   # RULER NIAH sets (tok_in sweep)
  steering/                      # RepE direction + steering payload

configs/
  phase3-matrix.yaml             # primary 18-run matrix
  phase3-matrix-*.yaml           # sweep/augmentation matrices
  tinfoil-config.yml             # Tinfoil deployment reference

infrastructure
  Dockerfile                     # benchbox container image
  release.sh                     # tag → build → measure → Sigstore pipeline
  CLAUDE.md                      # baked into container at /workspace/CLAUDE.md
  .github/workflows/             # CI (tinfoil-build + tinfoil-release)
```

## Reproducing the brief

### Prerequisites

- Python 3.12, conda env with `syft>=0.9.5,<0.9.6`
- `openai`, `httpx`, `numpy`, `pandas`, `scipy`, `matplotlib`, `zstandard`
- Tinfoil CLI and an API key for the `pour-demain` org
- A deployed `cc-deep-eval` target image (see companion repo)

### Running a matrix

```bash
python phase3_run_matrix.py --matrix configs/phase3-matrix.yaml
```

The v3-grouped orchestrator deploys one container per `(image, cc_state, debug)` tuple, runs all cells sharing that tuple against the same deploy, then tears down. Each cell produces `runs/<cell_id>/{requests.parquet, summary.json}`.

### Analysis pipeline

```bash
# Per-cell deltas with paired BCa CIs (Table 3, Figure 2)
python analyze_cc_deltas.py --run-dir runs/phase3/

# Two-feature OLS decomposition (Tables 4–6, Figure 4)
python analyze_combined_phase.py

# Concurrency sweep (Figure 6)
python analyze_concurrency_sweep.py --run-dir runs/phase3/concurrency_sweep_n500/

# GSM8K extrapolation (Figure 5)
python analyze_thinking.py
```

All analysis scripts emit tables to stdout and figures to `runs/phase3/analysis/figures/`.

## Image digests

Per-condition image tags and SHA-256 digests are recorded in each matrix's `matrix_report.json`. The brief's App. A.4 lists the canonical tags:

| Image | Tag | Use |
|-------|-----|-----|
| `prepilot-vllm-lens` | `v0.0.25` | GLM-5.1-FP8 primary matrix |
| `prepilot-vllm-lens-grad` | `v0.1.16-grad` | Gradient sidecar (EII-2) |
| `prepilot-vllm-lens` | `v0.0.27-llama70b-1` | Llama-70B TP=8 |
| `prepilot-vllm-lens` | `v0.0.28-llama70b-tp1` | Llama-70B TP=1 |
| `prepilot-vllm-lens` | `v0.0.32-egress` | Tier-1 egress pipeline |
| `prepilot-vllm-lens` | `v0.0.9-pysyft` | PySyft governed-egress |

## Citation

```
Tlaie Boria, A. (2026). An empirical study of Confidential Compute for
frontier AI evaluations: platform overhead and governed egress on Intel
TDX and H200 SXM. Pour Demain Technical Brief.
```

## License

[TODO: confirm license]
