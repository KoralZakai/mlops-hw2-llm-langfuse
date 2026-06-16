#!/usr/bin/env bash
#
# Start vLLM with our Phase 1 starting configuration for Qwen3-30B-A3B
# (MoE, ~3B active params) on 1x H100 80GB.
#
# Workload profile this is tuned for: 1.5-3K-token prompts, short structured
# (SQL) outputs, ~2-3 dependent calls per agent run, target P95 < 5s @ 10 RPS.
# These are a STARTING point - Phase 6 is where you iterate on them while
# watching the Grafana dashboard. Each flag has its justification inline.
#
# Reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html

set -euo pipefail

MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"

exec uv run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 4096 \
    `# prompts <=3K + short output fit in 4K; smaller ctx => more KV blocks => more concurrency` \
    --gpu-memory-utilization 0.92 \
    `# push KV-cache headroom on the 80GB card (leave some slack for activations)` \
    --enable-prefix-caching \
    `# the agent reuses the SAME schema across its 2-3 calls => prefill reuse cuts TTFT a lot` \
    --enable-chunked-prefill \
    `# interleave long prefills with decode => steadier latency for other requests under load` \
    --max-num-seqs 128 \
    `# concurrency ceiling; tune against KV-cache usage vs queue-wait on the dashboard` \
    --kv-cache-dtype fp8
    # ^ fp8 KV cache => ~2x KV headroom for more concurrency. Optional: if quality
    #   regresses in the eval, drop this line first. Single H100 => no tensor-parallel.
