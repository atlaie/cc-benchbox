# model-benchbox

A deployable Tinfoil Container with vLLM benchmarking tools baked in. Use this instead of launching `confidential-debug-large` and running `model_devbench/setup.sh` to install everything by hand.

## What's in the image

- `vllm[bench]` — `vllm bench {latency,serve,throughput}` subcommands
- `guidellm` — vllm-project's recommended production benchmarking tool
- `openai`, `httpx`, `pandas`, `datasets` — Python clients and result tooling
- `git`, `gh`, `vim`, `tmux`, `jq`, `htop`, `curl`, `wget`, `openssh-*` — dev / diagnostic utilities

Base image: `vllm/vllm-openai:v0.20.2` (carries CUDA, drivers, vllm). The OpenAI-server entrypoint is overridden — the container runs `sleep infinity` and you SSH into it.

## Release flow

This repo follows the same two-step build-and-release pattern as `confidential-model-router`. Both workflows live in `.github/workflows/`:

- **`tinfoil-build.yml`** — `workflow_dispatch` with a `version` input. Builds the Docker image, pushes it to `ghcr.io/tinfoilsh/model-benchbox`, runs `tinfoilsh/update-container-action` to rewrite `tinfoil-config.yml` with the new digest and create the matching git tag, then triggers `tinfoil-release.yml`.
- **`tinfoil-release.yml`** — `workflow_dispatch`, automatically invoked by `tinfoil-build.yml`. Runs `tinfoilsh/measure-image-action` against the tagged config and registers the release in the Sigstore transparency log.

The flow is single-input: pick a version, run the build workflow, everything else is automatic.

### Cutting a release

1. Open Actions → **Tinfoil Container Build** → **Run workflow**.
2. Enter a version (e.g. `v0.0.1`).
3. The workflow builds the image, updates `tinfoil-config.yml` with the new digest, creates the tag, and chains to `tinfoil-release.yml`.

### Updating the vLLM base image

1. Bump `ARG VLLM_VERSION` in the `Dockerfile`.
2. Commit and push to `main`.
3. Run **Tinfoil Container Build** with a new version (e.g. `v0.0.2`).

## Deploying

Once a version is registered (the build workflow has finished and `tinfoil-release.yml` has succeeded), deploy via the dashboard or CLI:

```bash
tinfoil container create benchbox \
    --repo tinfoilsh/model-benchbox \
    --tag v0.0.1 \
    --debug \
    --ssh-key laptop \
    --secret GITHUB_TOKEN \
    --secret HF_TOKEN \
    --secret TINFOIL_API_KEY

tinfoil container get benchbox    # poll for "ready"
```

The dashboard shows the SSH command on the container's card:

```bash
ssh -p <port> root@console.tinfoil.sh
```

You land directly in `/workspace` inside the benchbox container, with all bench tools on `$PATH`. `/workspace/CLAUDE.md` describes the in-container workflow.

### Tearing down

```bash
tinfoil container delete benchbox
```

## Resource sizing

| Field | Value | Reason |
|-------|-------|--------|
| `cpus` | 16 | Plenty for client-side load gen; bench is rarely CPU-bound |
| `memory` | 65536 (64 GB) | Headroom for HF dataset loading and result aggregation |
| `gpus` | 1 | Required for `vllm bench latency` / `throughput` (in-process). Set to 8 for multi-GPU bench (NVIDIA CC restricts to 1 or 8) |

## Secrets

Set in the dashboard before deploying:

- `GITHUB_TOKEN` — for `gh auth` and private clones
- `HF_TOKEN` — for Hugging Face dataset and model downloads
- `TINFOIL_API_KEY` — for hitting other Tinfoil-deployed inference endpoints

## Relationship to model_devbench

`model_devbench` installs dev tools onto a launched `confidential-debug-large` enclave. It still has uses:

- Setting your local `~/.ssh/config` alias for Cursor Remote-SSH
- Bootstrapping a fresh debug CVM for host-level work (editing `/mnt/ramdisk/config.yml`, restarting `tinfoil-shim`)

For the **benchmarking** path, `model-benchbox` is the lower-friction option: you skip the apt-install loop entirely and get a reproducible, version-pinned environment that boots ready-to-go.

## Limitations

- **No persistent disk.** The enclave is a ramdisk. Save results out (scp, git push, gh release upload) before redeploying.
- **No attestation.** Debug mode disables it by design. Don't use this for production traffic or sensitive data.
