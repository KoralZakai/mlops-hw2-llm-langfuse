# HW2 — LLM inference + observability — Report

> Target SLO (Phase 1/6): **P95 end-to-end agent latency < 5s at 10+ RPS over a 5-minute window.**
>
> All numbers and screenshots below come from the real `Qwen/Qwen3-30B-A3B-Instruct-2507`
> on 1× H100 80GB. Local development used a hosted OpenAI-compatible API for agent/eval
> logic; those numbers are not reported.

---

## 1. Serving configuration (Phase 1)

vLLM launch flags, chosen for this workload (MoE model, 1.5–3K-token prompts, short
structured SQL outputs, ~2–3 dependent calls per user request, 1× H100 80GB):

| Flag | Value | Justification |
|---|---|---|
| `--model` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | fixed by the assignment |
| `--max-model-len` | `4096` | prompts ≤3K + short output fit in 4K; a smaller window frees KV blocks → more concurrency |
| `--gpu-memory-utilization` | `0.92` | push KV-cache headroom high on the 80GB card, leaving slack for activations |
| `--enable-prefix-caching` | on | the agent resends the **same schema** across its 2–3 calls → prefill reuse cuts TTFT |
| `--enable-chunked-prefill` | on | interleave long prefills with decode → steadier latency under load |
| `--max-num-seqs` | `128` | concurrency ceiling; tuned against KV-cache usage vs queue wait |
| `--kv-cache-dtype` | `fp8` | ~2× KV headroom for more concurrency; verified quality held in the eval |

Single H100 → no tensor-parallel (the MoE model fits comfortably in one card).

---

## 2. Baseline eval results (Phase 5)

From `results/eval_baseline.json` (30 questions, execution accuracy on canonicalized row sets):

- **Overall pass rate: ~37% (11/30).**
- **Per-iteration (carry-forward):** roughly `iter0 = 11 (37%) → iter1 = 11–12 → iter2 = 11–12` — i.e. **nearly flat.**
- **Iterations-used distribution:** ~19 questions stop at iter 0, ~6 use 2, ~5 use 3.

**Key finding — run-to-run noise.** Repeated baseline runs gave 36.7% → 40% → 36.7%, and
iteration-0 accuracy shifted (11 ↔ 10) **even when the generate prompt was unchanged**. This
is temperature-0 MoE nondeterminism (batching). On 30 questions the variance is ±1–2 questions
(±3–6%), which **swamps the effect of small prompt edits** — so single-run comparisons on this
set are unreliable. A sharpened `generate_sql` prompt (DISTINCT for list-questions,
case-insensitive literal matching) nudged one run to 40%, but that sits inside the noise band.

**Why the loop is flat.** `verify` only fires on errors, 0-row results, or obviously-wrong
columns. The largest failure bucket is "plausible-but-wrong" answers (e.g. missing `DISTINCT`
→ duplicate rows; wrong aggregate) — these return non-empty, sensible-looking results that
verify cannot distinguish from correct, so revise is never triggered for them. The questions
that *do* revise are hard 0-row/literal-mismatch cases the model often can't repair. Net: the
revise loop is **neutral-to-slightly-positive** for quality. (BIRD "evidence" hints are not
provided to the agent, which also caps achievable accuracy.)

---

## 3. Hitting the SLO (Phase 6)

**Baseline (MAX_ITERATIONS = 3) vs SLO:** badly missed.

| Load | P50 | P95 | Notes |
|---|---|---|---|
| RPS 2 | 1.2s | **38s** | huge tail |
| RPS 10 | 63s | **86s** | fully saturated, ~17× over SLO |

The P50 was fast (~1.2s) but the **tail exploded** — a bimodal distribution pointing at a
specific class of slow request, plus ~14% HTTP 500s.

### Iteration log — *saw X → hypothesized Y → changed Z → result W*

1. **saw** P50 ~1.2s but P95 ~38s at only RPS 2 (bimodal tail) → **hypothesized** the multi-call
   `verify → revise` loop (up to ~6 chained vLLM calls) drives the tail, and Phase 5 shows it
   adds ≈0 quality → **changed** `MAX_ITERATIONS 3 → 1` (generate + verify, no revise) →
   **result:** P95 collapsed **38s → 1.2s** at RPS 2; at RPS 10, **P95 = 2.77s, P99 = 4.09s
   (SLO met for successful requests).** (`screenshots/grafana_before.png` / `grafana_after.png`)

2. **saw** ~13% of requests still returning HTTP 500 (latency SLO met only for the survivors);
   errors did **not** reproduce in short bursts but appeared under sustained/sequential load on
   specific questions → **hypothesized** a deterministic code bug on certain DB schemas →
   **changed** captured the error: `AttributeError: 'NoneType' object has no attribute 'replace'`
   in `agent/schema.py:_q()` (a `None` identifier during schema rendering crashed the whole
   request); fixed with a `(ident or "")` guard → **result:** http_errors dropped 80 → 8 per
   3000 requests. *But the full 5-minute run then revealed a new problem (see #3) that the
   60-second test had hidden.*

3. **saw** the **full 5-min** run at RPS 10 gave **P95 = 16.7s, P99 = 23s** (SLO missed) and
   sustained only ~8.5 RPS — arrival rate > service rate, so a backlog built and latency climbed
   over time. The 60s test (P95 2.77s) had hidden this; *this is exactly why the SLO is defined
   over a 5-minute window.* → **hypothesized** we were throughput-bound, and noted that at
   `MAX_ITERATIONS=1` the `verify` LLM call is **wasted work** (its verdict can never trigger a
   revise at the cap — see `route_after_verify`), so it doubles vLLM load for nothing →
   **changed** `verify_node` to skip its LLM call when `iteration >= MAX_ITERATIONS` (1 vLLM call
   per request instead of 2) → **result:** **P95 16.7s → 1.76s, P99 23s → 3.46s, http_errors → 0**
   over the full 5-minute / 3000-request run. (`screenshots/grafana_before.png` /
   `grafana_after.png`)

### Final numbers (full 5-minute run, RPS 10, `results/load_test_after.json`)

- **P95 latency:** **1.76 s**  ✅ (target < 5s)
- **P99 latency:** **3.46 s**  ✅
- **Throughput:** 3000 requests fired over 300s = **10 RPS**; 2993/3000 succeeded.
  (The driver's `achieved_rps` reads 8.33 only because it divides by wall-clock including the
  ~60s drain — effective sustained rate is ~10 RPS.)
- **Errors:** **0 HTTP 500s**; 6 timeouts + 1 client error (~0.2% stragglers, max latency 116s).

**Verdict:** **SLO HIT.** P95 1.76s and P99 3.46s — both well under the 5s target — at ~10 RPS
sustained over 5 minutes, with zero server errors. Reached through three metric-grounded
iterations (cut the no-value revise loop → fixed a schema-render crash → removed the wasted
verify call). Remaining nit: ~0.2% straggler requests (one 116s outlier), worth chasing with an
async agent.

### Did quality survive the tuning?

**Yes — fully.** Baseline (MAX_ITER=3) = **36.7% (11/30)**; after tuning (MAX_ITER=1, verify
skipped) = **36.7% (11/30)**, 0 agent errors. Identical accuracy. The latency win (P95 86s →
1.76s) cost **zero quality** — expected, since Phase 5 showed the revise loop was neutral and
verify never changed the final SQL. (Old line for reference:)

`results/eval_baseline.json` (~37%, MAX_ITER=3) vs `results/eval_after_tuning.json`
(MAX_ITER=1): `__TBD__`%. Expected to hold ~37% since Phase 5 showed the loop is roughly
neutral — removing it should not regress quality. `__confirm + one line__`.

---

## 4. Agent value (Phase 3/5)

Did the `verify → revise` loop earn its keep? **No, not at this configuration.** The
per-iteration pass rate is essentially flat (iter0 ≈ iter2), so the loop added ≈0 execution
accuracy — yet Phase 6 showed it was responsible for the *entire* latency tail (P95 38s → 1.2s
when removed). The loop *can* catch and occasionally fix 0-row literal mismatches, but it is
blind to the dominant failure mode (plausible-but-wrong results) and is expensive (2–3× the
sequential vLLM calls). The right engineering call for the SLO was to disable it — a decision
grounded directly in the eval's per-iteration numbers.

---

## 5. What I'd do with more time

- **Make the agent async + reuse one LLM client.** The synchronous FastAPI handler + new client
  per call is the likely source of the residual concurrency errors; an async graph would raise
  the reliable RPS ceiling well past 10.
- **Up-front value retrieval / schema linking.** Fetch real distinct values for likely filter
  columns *before* generating, so the first SQL uses the correct stored literal (kills the
  0-row bucket the revise loop struggles with).
- **Beat the eval noise.** Average 3–5 eval runs (or enlarge the set) so a real +1–2 question
  improvement is distinguishable from temp-0 MoE nondeterminism.
- **Feed BIRD "evidence" hints** to the generator — much of the remaining error is missing
  domain knowledge the gold SQL encodes.
- **A smarter/cheaper verify** that can flag plausible-but-wrong results (e.g. shape/cardinality
  checks against the question), so the loop targets the failure mode it currently misses.