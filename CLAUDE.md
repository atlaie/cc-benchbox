# Tinfoil benchbox

You are inside a `benchbox` container running in a Tinfoil enclave. The container is built from the `model-benchbox` repo and deployed as a debug-mode Tinfoil Container — attestation is disabled, SSH is enabled, and the enclave is meant for experimentation, not production traffic.

This is a **container**, not the CVM host. SSH lands directly here because debug-mode containers expose sshd. You do not need `nsenter` to reach the host; in fact, you can't — the container is isolated.

## What's pre-installed

| Tool | Purpose |
|------|---------|
| `vllm` + `vllm bench` (latency / serve / throughput) | Built-in vLLM benchmarking |
| `guidellm` | vllm-project's recommended benchmarking framework — live progress, exportable reports, richer load patterns |
| `openai`, `httpx` | Python clients for hitting OpenAI-compatible endpoints |
| `pandas`, `datasets` | Result aggregation, HF dataset loading |
| `git`, `gh`, `vim`, `tmux`, `jq`, `htop`, `curl`, `wget` | Standard dev/diagnostic utilities |

GPU is attached via NVIDIA confidential computing (`runtime: nvidia`, `gpus: all`). Run `nvidia-smi` to confirm.

## Filesystem

- `/workspace` — your working directory. Writable, but **non-persistent**: the enclave filesystem is a ramdisk, so anything written here is lost when the container restarts or redeploys. Save results out (scp, `gh release upload`, push to a git branch).
- `/tinfoil` — host ramdisk mount (if exposed by the runtime). Same volatility caveat.
- `/workspace/CLAUDE.md` — this file.

## Secrets

Set in the Tinfoil dashboard, injected as env vars at boot:
- `GITHUB_TOKEN` — for `gh auth` and private clones
- `HF_TOKEN` — for Hugging Face dataset and model downloads
- `TINFOIL_API_KEY` — for hitting other Tinfoil-deployed inference endpoints

If a secret is missing, the related workflow won't work — set it in **Containers > Secrets** before deploying, and redeploy to pick up changes.

## Benchmarking — the recommended approach

The vllm-project's current guidance: **start with `vllm bench serve` for quick local checks, switch to `guidellm` for anything production-relevant.**

### Three vllm bench subcommands

| Subcommand | Measures | Talks to |
|------------|----------|----------|
| `vllm bench latency` | Single-batch latency | Local engine (in-process, needs GPU) |
| `vllm bench serve` | Online serving (TTFT, TPOT, ITL, e2e) | Running HTTP server (any OpenAI-compatible endpoint) |
| `vllm bench throughput` | Max offline throughput | Local engine (in-process, needs GPU) |

For benching a remote enclave (e.g. a deployed `confidential-*` inference container), `vllm bench serve --base-url https://<name>.<org>.containers.tinfoil.dev` is the right starting point.

### Key metrics to report

Always cite all four — a single number in isolation is meaningless:
- **TTFT** — Time To First Token (perceived responsiveness)
- **TPOT** — Time Per Output Token (steady-state generation speed)
- **ITL** — Inter-Token Latency (jitter / smoothness)
- **e2e** — End-to-End request latency

Report mean, p50, p99 for each. The mean alone hides tail latency.

### Workflow

1. **Sanity check first.** One request, tiny input, single iteration. If something is broken you'll see it instantly. Never start a multi-minute run without a passing few-second check.
2. **State the hypothesis.** Write down what you expect and why before running. Prevents aimless tinkering.
3. **Always have a baseline.** "X is 340 req/s" is meaningless. "X is 340 req/s vs Y at 280 req/s on the same hardware" is a result.
4. **Isolate one variable per run.** Change one thing — TP size, batch size, dataset, concurrency — then measure. Do not change two and attribute results to the change you care about.
5. **Verify the environment.** Before attributing results, confirm GPU count, CUDA version, container image digest, model loaded, no competing workloads (`nvidia-smi`, `docker ps` if accessible, check the metrics endpoint).
6. **Save everything.** Each run gets its own clearly-named file/directory (`results/2026-05-09-vllm-tp1-sharegpt/`). Never overwrite. Save the exact command, config, and environment alongside the numbers.
7. **Record negative results.** A failed experiment that rules out a hypothesis is as valuable as a successful one.
8. **Save out before redeploying.** The ramdisk is gone the moment the container restarts. `scp` results to your laptop or `gh release upload` to a results repo.

### guidellm quickstart

For production-grade benchmarking with full reports:

```bash
guidellm benchmark \
    --target https://<name>.<org>.containers.tinfoil.dev \
    --rate-type concurrent --rate 10 \
    --max-seconds 120 \
    --data "prompt_tokens=512,output_tokens=128"
```

`--rate-type` options: `synchronous`, `concurrent`, `constant`, `poisson`, `throughput`. Pick `concurrent` for closed-loop user simulation, `poisson` for realistic traffic patterns. `throughput` ramps until saturation — use it to find the knee of the latency-throughput curve.

### vllm bench serve quickstart

For a quick check against a running server:

```bash
vllm bench serve \
    --base-url https://<name>.<org>.containers.tinfoil.dev \
    --model <served-model-name> \
    --dataset-name sharegpt \
    --num-prompts 100 \
    --request-rate 5
```

Datasets: `sharegpt`, `burstgpt`, `sonnet`, `random`, `random-mm`, `hf`, `custom`.

`--request-rate inf` measures max throughput; finite values simulate controlled load.

## Git commits

When committing from inside this container, the `gh` and `git` configs are populated from the local `setup.sh` if you ran one, otherwise set manually. Tag enclave-authored commits:

```
Co-authored-by: enclave <enclave@tinfoil.sh>
```

## Saving results out

The ramdisk is volatile. Three ways to persist:

```bash
# Push to a results branch in a results repo
gh repo clone tinfoilsh/bench-results /tmp/results && cd /tmp/results
cp -r /workspace/results/<run-id> .
git checkout -b run/<run-id> && git add . && git commit -m "..." && git push

# scp to your laptop (run from your laptop, not here)
scp -P <ssh-port> -r root@console.tinfoil.sh:/workspace/results/<run-id> ./

# Upload as a GitHub release artifact
gh release create bench-<run-id> /workspace/results/<run-id>/*
```
