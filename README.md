# model-benchbox

vLLM benchmarking tools in a Tinfoil Container. Deploy in debug mode, SSH in, run benches against any OpenAI-compatible endpoint (or a local in-process engine on the attached GPU).

## First deploy

1. **Cut a release.** `./release.sh v0.0.1` — runs the build → measure-image pipeline end-to-end and watches both workflows (~5 min). Or via the web UI: Actions → **Tinfoil Container Build** → enter `v0.0.1` → run.
2. **Set org secrets** in the [Tinfoil dashboard](https://dash.tinfoil.sh): `GITHUB_TOKEN`, `HF_TOKEN`, `TINFOIL_API_KEY`. Register your laptop SSH key under **SSH Keys**.
3. **Deploy:**

   ```bash
   tinfoil container create benchbox \
       --repo tinfoilsh/model-benchbox --tag v0.0.1 --debug \
       --ssh-key laptop \
       --secret GITHUB_TOKEN --secret HF_TOKEN --secret TINFOIL_API_KEY
   ```

   The SSH command appears on the dashboard card once status is `ready`.

## Run an experiment

SSH lands in `/workspace`. Full workflow guidance is in [`CLAUDE.md`](./CLAUDE.md). Two starting points:

```bash
# Quick check — vllm's built-in
vllm bench serve --base-url https://<name>.<org>.containers.tinfoil.dev \
    --model <served-model-name> --dataset-name sharegpt \
    --num-prompts 100 --request-rate 5

# Production-grade — guidellm (recommended)
guidellm benchmark --target https://<name>.<org>.containers.tinfoil.dev \
    --rate-type concurrent --rate 10 --max-seconds 120 \
    --data "prompt_tokens=512,output_tokens=128"
```

## Save results out

The enclave is a ramdisk — anything in `/workspace` is gone on restart. Before redeploying:

```bash
# Push to a GitHub release
gh release create bench-<run-id> /workspace/results/<run-id>/*

# Or scp from your laptop:
scp -P <ssh-port> -r root@console.tinfoil.sh:/workspace/results/<run-id> ./
```

## Subsequent releases

Edit anything (e.g. bump `ARG VLLM_VERSION` in `Dockerfile`), get it merged to `main`, then run [`./release.sh`](./release.sh). The script auto-bumps the patch version from the highest existing tag, triggers the build → measure pipeline, and watches both runs. Pass an explicit version (`./release.sh v0.1.0`) to override; pass `--force` to skip the prompt when a stale tag exists. Same two-step pattern as `confidential-model-router`.

## File map

- [`Dockerfile`](./Dockerfile) — image contents (`vllm[bench]`, `guidellm`, dev tools)
- [`tinfoil-config.yml`](./tinfoil-config.yml) — enclave resources (128 CPU / 1 TiB / 1 GPU, `cvm-version: 0.8.0`). Bump `gpus` to 8 for multi-GPU bench.
- [`CLAUDE.md`](./CLAUDE.md) — bench workflow + metrics; baked into the image at `/workspace/CLAUDE.md`
- [`release.sh`](./release.sh) — one-shot release script (build + measure + watch)
- [`.github/workflows/`](./.github/workflows) — `tinfoil-build.yml` + `tinfoil-release.yml`

Debug mode disables attestation by design. Don't use this for production traffic.
