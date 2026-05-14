# Tinfoil model-benchbox: a deployable container with vLLM benchmarking tools
# baked in. Build context lives in this repo; tags with prefix `image-v*` push
# to ghcr.io/tinfoilsh/benchbox. The published digest is then pinned in
# tinfoil-config.yml and deployed as a Tinfoil Container.

ARG VLLM_VERSION=v0.20.2
FROM vllm/vllm-openai:${VLLM_VERSION}

ENV DEBIAN_FRONTEND=noninteractive

# Match the dev-tool set that model_devbench installs on a debug enclave.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git vim wget curl ca-certificates \
        openssh-client openssh-sftp-server \
        jq unzip tmux htop \
    && rm -rf /var/lib/apt/lists/*

# GitHub CLI for clone/push from inside the enclave.
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        | tee /usr/share/keyrings/githubcli-archive-keyring.gpg > /dev/null \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update && apt-get install -y --no-install-recommends gh \
    && rm -rf /var/lib/apt/lists/*

# Bench tooling. `vllm[bench]` enables the `vllm bench {latency,serve,throughput}`
# subcommands. guidellm is the vllm-project successor to benchmark_serving.py
# and the recommended tool for production benchmarking — live progress,
# exportable reports, richer workload patterns.
RUN pip install --no-cache-dir \
     "vllm[bench]" \
     guidellm \
     openai \
     httpx \
     pandas \
-    datasets
+    datasets \
+    zstandard \
+    boto3

 WORKDIR /workspace
 COPY CLAUDE.md /workspace/CLAUDE.md
+COPY scripts/ /workspace/scripts/
+COPY data/ /workspace/data/

WORKDIR /workspace

COPY CLAUDE.md /workspace/CLAUDE.md
+COPY scripts/ /workspace/scripts/
+COPY data/ /workspace/data/

# Override the upstream OpenAI-server entrypoint so the container is a
# long-lived bench shell. SSH in via debug-mode and run benches interactively.
ENTRYPOINT []
CMD ["sleep", "infinity"]
