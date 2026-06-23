"""Eval runner using execution accuracy.

Reads evals/eval_set.jsonl, calls the agent at AGENT_URL on each question,
then compares the agent's SQL output to the gold SQL by *executed rows*
(canonicalized: sorted, stringified, None-coerced to empty).

Helpers (run_sql / canonicalize / matches) are provided. You implement
eval_one() and summarize().

Run:
    uv run python evals/run_eval.py --out results/eval_baseline.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
DEFAULT_OUT_FILE = ROOT / "results" / "eval_baseline.json"
DB_DIR = ROOT / "data" / "bird"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"


# ---------- Helpers (provided) -----------------------------------------

def run_sql(db_id: str, sql: str, timeout: float = 5.0) -> tuple[bool, list[tuple] | None, str | None]:
    """Run sql against db_id in read-only mode. Returns (ok, rows, error)."""
    path = DB_DIR / f"{db_id}.sqlite"
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout) as conn:
            cur = conn.execute(sql)
            rows = cur.fetchall()
            return True, rows, None
    except Exception as e:  # noqa: BLE001
        return False, None, f"{type(e).__name__}: {e}"


def canonicalize(rows: list[tuple] | None) -> list[tuple] | None:
    """Sort rows; coerce cells to str; None -> ''."""
    if rows is None:
        return None
    return sorted(tuple("" if c is None else str(c) for c in row) for row in rows)


def matches(gold_rows: list[tuple] | None, pred_rows: list[tuple] | None) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    return canonicalize(gold_rows) == canonicalize(pred_rows)


# ---------- Implement these (Phase 5) ----------------------------------

def _iteration_sqls(history: list[dict]) -> list[str]:
    """Reconstruct the SQL the agent held after each generate/revise call.

    The agent records one history entry per LLM node. generate_sql and revise
    are the nodes that (re)produce SQL, so the SQL after iteration k is the
    k-th such entry. verify entries carry no SQL and are skipped.
    """
    return [h["sql"] for h in history if h.get("node") in ("generate_sql", "revise") and "sql" in h]


def eval_one(question: dict, agent_url: str) -> dict:
    """Score one question. Return a dict capturing per-iteration correctness."""
    db_id = question["db_id"]
    gold_sql = question["gold_sql"]

    # Gold rows: the ground truth we compare against.
    gold_ok, gold_rows, gold_err = run_sql(db_id, gold_sql)

    record: dict = {
        "db_id": db_id,
        "question": question["question"],
        "gold_sql": gold_sql,
        "gold_ok": gold_ok,
        "gold_error": gold_err,
    }

    # Call the agent over HTTP.
    payload = {"question": question["question"], "db": db_id, "tags": {"phase": "eval"}}
    try:
        resp = httpx.post(agent_url, json=payload, timeout=120.0)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:  # noqa: BLE001
        record.update({
            "agent_ok": False,
            "agent_error": f"{type(e).__name__}: {e}",
            "final_sql": "",
            "iterations": 0,
            "iter_sqls": [],
            "iter_correct": [],
            "correct": False,
        })
        return record

    final_sql = data.get("sql", "")
    history = data.get("history", [])
    iter_sqls = _iteration_sqls(history)
    if not iter_sqls and final_sql:
        # Fall back to the final SQL if history didn't surface per-iteration SQL.
        iter_sqls = [final_sql]

    # Score each iteration's SQL by executed-row match against gold.
    iter_correct: list[bool] = []
    for sql in iter_sqls:
        pred_ok, pred_rows, _ = run_sql(db_id, sql)
        iter_correct.append(pred_ok and matches(gold_rows, pred_rows))

    final_correct = iter_correct[-1] if iter_correct else False

    record.update({
        "agent_ok": bool(data.get("ok", False)),
        "agent_error": data.get("error"),
        "final_sql": final_sql,
        "iterations": data.get("iterations", len(iter_sqls)),
        "iter_sqls": iter_sqls,
        "iter_correct": iter_correct,
        "correct": final_correct,
    })
    return record


def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results.

    Per-iteration carry-forward: if the agent terminated at iteration j < k
    (verify said ok at j, or it hit MAX_ITERATIONS at j < k), treat the
    question's iteration-k result as identical to its iteration-j result.
    The agent stopped emitting; whatever it had at termination is what
    would have been served had we polled at iteration k.
    """
    n = len(results)
    # Deepest the loop went across all questions = number of iteration columns.
    max_iters = max((len(r.get("iter_correct", [])) for r in results), default=0)
    max_iters = max(max_iters, 1)

    def carried(r: dict, k: int) -> bool:
        """Correctness at iteration k with carry-forward past termination."""
        seq = r.get("iter_correct", [])
        if not seq:
            return False
        return seq[k] if k < len(seq) else seq[-1]

    pass_rate_by_iter = []
    for k in range(max_iters):
        n_correct = sum(1 for r in results if carried(r, k))
        pass_rate_by_iter.append({
            "iteration": k,
            "n_correct": n_correct,
            "pass_rate": (n_correct / n) if n else 0.0,
        })

    overall_correct = sum(1 for r in results if r.get("correct"))
    agent_errors = sum(1 for r in results if not r.get("agent_ok"))

    # Distribution of how many iterations each question actually used.
    iter_counts: dict[int, int] = {}
    for r in results:
        used = len(r.get("iter_correct", []))
        iter_counts[used] = iter_counts.get(used, 0) + 1

    return {
        "n_questions": n,
        "overall_pass_rate": (overall_correct / n) if n else 0.0,
        "overall_correct": overall_correct,
        "agent_errors": agent_errors,
        "pass_rate_by_iteration": pass_rate_by_iter,
        "iterations_used_distribution": dict(sorted(iter_counts.items())),
    }


# ---------- Main (provided) --------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.eval_set.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(questions)} eval questions from {args.eval_set}")

    results: list[dict] = []
    t0 = time.monotonic()
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['db_id']}: {q['question'][:60]}...", flush=True)
        results.append(eval_one(q, args.agent_url))
    elapsed = time.monotonic() - t0

    summary = summarize(results)
    out = {
        "summary": summary,
        "wall_clock_seconds": elapsed,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
