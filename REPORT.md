# HW2 — LLM inference + observability — Report

> Target SLO (Phase 1/6): **P95 end-to-end agent latency < 5s at 10+ RPS over a 5-minute window.**
>
> All numbers, screenshots, and the SLO verdict below come from the real
> `Qwen/Qwen3-30B-A3B-Instruct-2507` on 1× H100. Local development used a hosted
> OpenAI-compatible API for the agent/eval logic; those numbers are not reported.

---

## 1. Serving configuration (Phase 1)

vLLM launch flags and why each was chosen for *this* workload (MoE model,
1.5–3K-token prompts, short structured outputs, ~2–3 dependent calls per user
request, 1× H100 80GB):

| Flag | Value | One-line justification |
|---|---|---|
| `--model` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | fixed by the assignment |
| `--max-model-len` | _TBD_ | cap context to the real prompt envelope (~3K in + short out) so more KV blocks are free for concurrency |
| `--gpu-memory-utilization` | _TBD_ | push KV-cache headroom as high as is stable on 80GB |
| `--enable-prefix-caching` | _TBD_ | agent reuses the same schema across its 2–3 calls → prefill reuse cuts TTFT |
| `--max-num-seqs` | _TBD_ | concurrency ceiling; tune against KV-cache usage vs queue wait |
| `--dtype` / quantization | _TBD_ | MoE fits comfortably; decide FP8/AWQ tradeoff vs latency |
| ... | ... | _fill from the H100 run_ |

_Notes on the MoE / prompt-shape / latency tradeoffs: TBD._

---

## 2. Baseline eval results (Phase 5)

From `results/eval_baseline.json` (30 questions, execution accuracy):

- **Overall pass rate:** _TBD_
- **Per-iteration pass rate (carry-forward):**
  - iter 0 (generate only): _TBD_
  - iter 1 (after 1 revise): _TBD_
  - iter 2: _TBD_
- **Iterations-used distribution:** _TBD_

Commentary: _TBD._

---

## 3. Hitting the SLO (Phase 6)

Baseline vs SLO: _TBD_

Iteration log — *saw X → hypothesized Y → changed Z → result W*:

1. _saw … → hypothesized … → changed … → result …_  (screenshots/grafana_before.png)
2. _…_
3. _…_  (screenshots/grafana_after.png)

Final numbers: _TBD (P95 latency, achieved RPS, from load_test/driver.py)_

Did quality survive the tuning? Compare `results/eval_baseline.json` vs
`results/eval_after_tuning.json`: _TBD._

---

## 4. Agent value (Phase 3/5)

Did the verify→revise loop earn its keep? Cite the per-iteration pass rate — if
iter 2 is meaningfully higher than iter 0, the loop is doing real work; if
they're equal, it isn't.

_One paragraph, grounded in the numbers: TBD._

---

## 5. What I'd do with more time

_Be specific — not "add Kubernetes". E.g. speculative decoding for the short
structured outputs, schema-pruning to shrink prompts and prefill cost, a
cheaper/faster verify model, structured-output decoding to drop the SQL-extraction
regex, batching the agent's independent calls, etc. TBD._
