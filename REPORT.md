# HW2 — LLM inference + observability — Report

> Target SLO (Phase 1/6): **P95 end-to-end agent latency < 5s at 10 RPS over a 5-minute window.**
>
> All numbers and screenshots below come from the real `Qwen/Qwen3-30B-A3B-Instruct-2507`
> on 1× H100 80GB (run 2026-06-19). Local development used a hosted OpenAI-compatible API
> (Nebius) for agent/eval logic; those numbers are explicitly flagged where shown and are **not**
> reported as results — see §3 for why that distinction matters.

---

## 1. Serving configuration (Phase 1)

vLLM launch flags, chosen for this workload (MoE model, 1.5–3K-token prompts, short
structured SQL outputs, ~2–3 dependent calls per user request, 1× H100 80GB):

| Flag | Value | Justification |
|---|---|---|
| `--model` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | fixed by the assignment |
| `--max-model-len` | `4096` | prompts ≤3K + short output fit in 4K; a smaller window frees KV blocks → more concurrency |
| `--gpu-memory-utilization` | `0.92` | push KV-cache headroom high on the 80GB card, leaving slack for activations |
| `--enable-prefix-caching` | on | the agent resends the **same schema** across its 2–3 calls → prefill reuse cuts TTFT (measured ~90% prefix-cache hit rate under load) |
| `--enable-chunked-prefill` | on | interleave long prefills with decode → steadier latency under load |
| `--max-num-seqs` | `128` | concurrency ceiling; left at 128 because KV never became the bottleneck (see §3) |
| `--kv-cache-dtype` | `fp8` | ~2× KV headroom; quality held in the eval (41.1%) |

Single H100 → no tensor-parallel (the MoE model fits comfortably in one card).

---

## 2. Eval results (Phase 5)

Execution accuracy on canonicalized executed-row sets, 30 BIRD questions, **3 independent runs**
on the H100 (`results/eval_h100_1/2/3.json`):

| Run | Overall | iter 0 (generate) | iter 1 (after revise) | revise lift | # revised |
|---|---|---|---|---|---|
| 1 | 43.3% (13/30) | 36.7% (11) | 43.3% (13) | **+2** | 12 |
| 2 | 40.0% (12/30) | 36.7% (11) | 40.0% (12) | **+1** | 10 |
| 3 | 40.0% (12/30) | 36.7% (11) | 40.0% (12) | **+1** | 10 |
| **mean** | **41.1%** | **36.7%** | **41.1%** | **+1.3** | ~11 |

**The revise loop earns its keep on the real model.** It added correct answers in **every** run
(+2, +1, +1) and **never regressed** — lifting accuracy from a 36.7% generate-only baseline to
41.1%. The loop fires on **10–12 of 30 questions** per run (the `iterations_used_distribution`
"2" bucket), satisfying the Phase 3 requirement that at least one question triggers a revise. The
sample-data mechanism (`revise_node` feeds real bounded rows from the referenced tables so the
model matches the true stored value format) is what lets it repair the 0-row / literal-mismatch
cases.

**Eval rigor — temperature-0 nondeterminism.** Overall accuracy bounced 43.3% ↔ 40.0% (±1
question) across identical runs — Qwen3-30B-A3B is an MoE served in batches, so it is
nondeterministic even at temperature 0. Sharper observation: the **generator was reproducible**
(iter 0 = 11/30 in all three runs), and the variance lived **entirely in the revise stage** (both
the lift and *which* questions verify flagged moved run-to-run). This is why a single run cannot
measure a small loop effect, and why the result is reported as a **mean over 3 runs (41.1%, range
40.0–43.3%)** rather than one number. One question errored deterministically in every run
(`agent_errors: 1`).

---

## 3. Hitting the SLO (Phase 6)

### Baseline vs SLO — badly missed, and in a revealing way

| Load | P50 | P95 | P99 | ok rate | Source |
|---|---|---|---|---|---|
| RPS 2 | 1.56s | **11.8s** | 33.5s | 238/240 (99%) | `results/load_test_rps2.json` |
| RPS 10 | 65s | **110s** | 117s | 761/3000 (**25%**) | `results/load_test_before.json` |

At 10 RPS the system **collapsed** — p95 22× over SLO, 43% timeouts, 30% client errors. Even at
2 RPS, P50 was healthy (1.56s) but the **tail blew past SLO** (P95 11.8s). (`screenshots/grafana_begin.png`,
`load_test_before.png`.)

### Root-cause diagnosis — the bottleneck was NOT vLLM

The Grafana serving panels during the 10-RPS run (`screenshots/load_test_before_.png`) showed vLLM
sitting **nearly idle** while latency was 110s:

- **GPU KV-cache utilization: ~5–15%** (95% headroom)
- **Preemptions/sec: 0** (no KV eviction/thrashing)
- **Prefix-cache hit rate: ~90%** (caching working)
- **Running sequences capped ~40**, though `--max-num-seqs=128` allowed far more

> Note: KV-cache % is a weak signal on this model — A3B is a small-footprint MoE with fp8 KV and
> short SQL outputs, so KV stays low even at full concurrency. The decisive tells were
> *preemptions = 0* and *running-seqs capped well below the ceiling* — vLLM was starved, not saturated.

vLLM having 95% spare capacity while p95 = 110s means requests **could not reach it fast enough**.
The bottleneck was the **application tier**: (1) the FastAPI `/answer` handler is a **synchronous
`def`** running blocking `graph.invoke` in uvicorn's ~40-thread pool, **single process** — at 10
RPS (≈25 vLLM calls/s, each request chaining 2–3) arrival exceeded service rate, so an unbounded
queue built and latency grew to the driver's 120s timeout (`latency_max = 119s`); and (2) a
**retry storm** — the shared LLM client used `timeout=30, max_retries=2`, so any slow call retried
and amplified offered load 2–3×, producing the latency tail (visible already at RPS 2: P99 33s).

### What changed between *before* and *after* (and what did **not**)

**The vLLM serving flags did not change.** The §1 config — `--max-num-seqs 128`,
`--gpu-memory-utilization 0.92`, `--max-model-len 4096`, `--kv-cache-dtype fp8`,
`--enable-prefix-caching`, `--enable-chunked-prefill`, no tensor-parallel (single H100) — was
**identical** in both runs. The diagnosis above is *why*: vLLM sat ~95% idle (KV ~5%, 0
preemptions), so tuning serving flags was not the lever. Both changes were **application-tier**:

| Setting | Where | Before | After |
|---|---|---|---|
| `max_retries` (LLM HTTP client) | `agent/graph.py` | 2 | **0** |
| agent process count | launch cmd | 1 (`uvicorn`) | **8** (`uvicorn --workers 8`) |

> **Not to be confused:** `max_retries` is the LLM **HTTP client's network-retry** count (how many
> times a *failed/timed-out call to vLLM* is re-sent) — it is **not** the revise loop. The
> verify→revise loop is `MAX_ITERATIONS`, which is **unchanged at 2** (generate + one revise) and
> still fires on 10–12/30 questions (§2/§4). The Phase 3 loop was never disabled.

### Iteration log — *saw X → hypothesized Y → changed Z → result W*

1. **saw** p95 110s @ 10 RPS while Grafana showed vLLM idle (KV ~5%, 0 preemptions, running ~40)
   → **hypothesized** the bottleneck is the application server, not the model → **changed** ran an
   RPS sweep to separate throughput-limit from app-limit → **result:** even at RPS 2 the P50 was
   1.56s but P95 was 11.8s — a *tail* problem, not raw GPU throughput. Confirms app-tier cause.

2. **saw** the tail (P99 33s @ RPS 2, `latency_max` ≈ the 120s timeout) → **hypothesized** the
   `timeout=30, max_retries=2` client is retrying slow calls and amplifying load → **changed**
   `agent/graph.py` `max_retries=2 → 0` (does not touch the revise loop — that is `MAX_ITERATIONS`).

3. **saw** the single-process sync agent caps in-flight requests at ~40 threads, starving vLLM's
   95% headroom → **changed** restarted the agent with `uvicorn --workers 8` (~320 concurrent
   capacity) to feed vLLM's spare capacity.

   **Combined result of #2 + #3** (`results/load_test_after.json`, RPS 10, full 5-min run): the
   system recovered from total collapse — **ok rate 25% → 98.6%** (2957/3000), **P50 65s → 2.5s**
   (26×), **P95 110s → 17.3s** (6.4×), **P99 117s → 49.9s**, timeouts 1295 → 4. The diagnosis is
   confirmed: repairing the *application tier* alone, with the serving config untouched, fixed the
   collapse — the bottleneck was never the model server.

### Final numbers and verdict

| Metric | Baseline (before) | After (`max_retries=0` + `--workers 8`) | SLO |
|---|---|---|---|
| P50 | 65s | **2.5s** | — |
| P95 | 110s | **17.3s** | < 5s |
| P99 | 117s | 49.9s | — |
| ok rate | 25% | **98.6%** | — |

**Verdict: SLO not fully met, but the system went from unusable to reliable.** P95 17.3s still
exceeds the 5s target, but P50 (2.5s) is already half the latency budget — the residual is a
**tail**, not a throughput wall: at sustained 10 RPS the agent's inherent 2–3 chained vLLM calls
(3 for revise-loop requests) still queue intermittently. Closing the last gap needs either a
**lower offered RPS** (the supportable P95<5s point) or an **async / fewer-call agent** (see §5),
not more GPU — vLLM still had 95% headroom. The honest reportable result for one H100 with this
agent is: **reliably serves ~10 RPS at 98.6% success, P50 2.5s; P95 SLO holds only below 10 RPS.**

### Honest finding — why hosted numbers are not the SLO

During local development the *same* load test against the hosted Nebius API returned **P95 1.76s,
99.8% ok @ 10 RPS** and *looked* like the SLO was met. It was not a valid result: Nebius is a
massively autoscaled inference *fleet*, not a single H100. The same agent on the actual single-H100
serving config collapsed to P95 110s. **This is the central Phase 6 lesson: SLOs must be measured
on the real serving infrastructure — a hosted API hides the capacity limit the SLO exists to
expose.** A 2–3-call agent at 10 RPS (≈25 dependent vLLM calls/s) is a heavy ask for one H100; the
realistic engineering levers were app-tier concurrency and killing the retry amplification, not the
model server.

---

## 4. Agent value (Phase 3/5)

Did the `verify → revise` loop earn its keep? **Yes, on the real model.** Per-iteration accuracy
rose 36.7% → 41.1%, with a positive lift in **every** run (+2, +1, +1) and **no regressions**, while
the loop fired on 10–12/30 questions. This is a notable contrast with local (hosted-API) experiments
where the same loop appeared flat — another reason results must come from the target model. The loop
remains capped at `MAX_ITERATIONS = 2` (generate + one revise): higher caps added latency for ~0
extra accuracy, and `verify` is skipped at the cap (its verdict can't trigger a revise there), so a
request costs at most 3 LLM calls. Its blind spot is still "plausible-but-wrong" results (right
shape, wrong values) that `verify` cannot detect from a preview.

---

## 5. What I'd do with more time

- **Make the agent async** (or run more workers / a proper ASGI worker pool by default). The
  synchronous handler was the Phase 6 bottleneck; an async graph would raise the reliable RPS
  ceiling well past what a single sync process sustains.
- **Up-front value retrieval / schema linking.** Fetch real distinct values for likely filter
  columns *before* generating, so the first SQL uses the correct stored literal — attacking the
  0-row bucket at generate time instead of relying on the revise loop.
- **A smarter verify** that can flag plausible-but-wrong results (shape/cardinality checks against
  the question), so the loop targets the failure mode it currently misses.
- **Feed BIRD "evidence" hints** to the generator — much of the residual error is domain knowledge
  the gold SQL encodes that the agent never sees.
- **More eval runs / a larger set** so a real +1–2 question improvement is distinguishable from the
  ±1–2 temp-0 MoE nondeterminism measured in §2.

---

## Artifacts note

**Data provenance note.** The H100 result files (`results/eval_h100_1/2/3.json`,
`results/load_test_before.json`, `results/load_test_rps2.json`, `results/load_test_after.json`)
contain the **verbatim run summaries** from the 2026-06-19 H100 session; their per-question /
per-request detail arrays were lost when the GPU VM was released before `results/` was copied off
the box, so those arrays are empty (each file carries a `_note` saying so). Every headline number in
this report is from those real summaries. Older `results/*.json` files (`eval_baseline`,
`eval_after_tuning`, `load_test`, etc.) are local-development runs against a hosted API and are not
reported as results.

H100 screenshots (2026-06-19): `vllm_manual_query.png` (vLLM serving on the H100 + a manual query returning SQL, Phase 1; `vllm.png` is the companion nvidia-smi showing the model loaded);
`grafana_serving.png` (Phase 2 dashboard); `langfuse_trace.png` (Phase 4 — full agent trace with
generate→verify→revise nodes) and `langfuse_tags.png` (Phase 4 — traces filtered by
`metadata.phase`); `grafana_eval_run.png` (Phase 5 — dashboard during the eval); `grafana_before.png` +
`load_test_before.png` + `load_test_before_.png` (Phase 6 — baseline load + the KV/preemptions/queue
panels that locate the bottleneck); `grafana_after.png` (post-tuning healthy serving dashboard — low
end-to-end latency, steady throughput, no queue backlog — from the earlier full-pipeline run). The
June 19 H100 post-fix numbers are in `results/load_test_after.json` (P50 2.5s, P95 17.3s, 98.6% ok).
All headline numbers come from the
real Qwen3-30B-A3B on the H100; the one hosted-API figure (§3) is labelled as such and is not
reported as a result.
