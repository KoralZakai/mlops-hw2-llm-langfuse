"""LangGraph agent: text-to-SQL with verify+revise loop.

Graph shape:

    START -> attach_schema -> generate_sql -> execute -> verify
                                                          |
                                              ok=true ----+----> END
                                                          |
                                              ok=false ---+----> revise -> execute -> verify (loop)

Loop is capped at MAX_ITERATIONS total generate/revise calls.

The execute node and the graph wiring are provided. `generate_sql_node` is
filled in as a worked example; you implement `verify`, `revise`, and the
conditional router following the same shape.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from agent import prompts
from agent.execution import ExecutionResult, execute_sql
from agent.schema import db_path, render_schema

# Total generate + revise calls before the loop is forced to stop.
# Tuned in Phase 6: 3 made the verify->revise loop the latency bottleneck (P95 86s)
# while adding ~0 execution accuracy; 1 disabled the loop entirely. 2 keeps the loop
# (one revise can fire, so it earns its Phase-3 keep) while bounding the tail. The
# final verify is skipped at the cap (see verify_node), so a request is at most:
# generate -> verify -> revise -> (verify skipped) = 3 LLM calls worst case.
MAX_ITERATIONS = 2

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
# vLLM ignores the key, but a hosted OpenAI-compatible provider needs a real one.
# Lets you point the agent at e.g. OpenAI while iterating without a running vLLM.
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "not-needed")


@dataclass
class AgentState:
    """State threaded through the graph. Extend with fields you need."""

    question: str
    db_id: str
    schema: str = ""
    sql: str = ""
    execution: ExecutionResult | None = None
    verify_ok: bool = False
    verify_issue: str = ""
    iteration: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

# Single shared chat client, reused across every call and request. Building a new
# ChatOpenAI per call (the old `llm()` factory) churns a fresh HTTP connection pool
# each time, which under concurrent load caused intermittent 500s and capped
# throughput. timeout + max_retries make transient vLLM hiccups non-fatal.
LLM = ChatOpenAI(
    model=VLLM_MODEL,
    base_url=VLLM_BASE_URL,
    api_key=LLM_API_KEY,
    temperature=0.0,
    timeout=30,
    # Phase 6: was 2. Under load, a slow call hit the 30s timeout and retried,
    # amplifying offered load 2-3x and exploding the latency tail (p95 11.8s @
    # 2 RPS, p95 110s @ 10 RPS). vLLM was near-idle (KV ~5%, 0 preemptions), so
    # the bottleneck was the retry storm + single-process agent concurrency, not
    # the model. 0 retries removes the amplification; --workers scales the agent.
    max_retries=0,
)


# ---- Nodes ------------------------------------------------------------

def _attach_schema(state: AgentState) -> dict:
    """Provided. Render the DB schema once at the start of the run."""
    return {"schema": render_schema(state.db_id)}


def _extract_sql(text: str) -> str:
    """Pull a SQL statement out of an LLM reply, stripping markdown fences/prose.

    Intentionally simple: take the first ```sql ... ``` block if there is one,
    otherwise the whole reply. You may need to harden this for your prompts.
    """
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return (fenced.group(1) if fenced else text).strip()


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of an LLM reply, tolerating fences/prose.

    The verifier is asked for a bare {"ok": ..., "issue": ...} object, but
    models sometimes wrap it in ```json fences or add a sentence around it. We
    grab the first balanced-looking {...} and parse it; on any failure we return
    an empty dict so the caller can fall back to a safe default.
    """
    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", candidate, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()
    match = re.search(r"\{.*\}", candidate, re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, ValueError):
        return {}


def generate_sql_node(state: AgentState) -> dict:
    """Worked example - the other LLM nodes follow this same shape.

    Build messages from the prompts, call the shared llm(), extract the SQL,
    and return only the state fields you changed. `iteration` is bumped here
    (and in revise) so route_after_verify can enforce MAX_ITERATIONS.

    This node is wired and ready; fill in GENERATE_SQL_SYSTEM / GENERATE_SQL_USER
    in prompts.py to make it produce real queries.
    """
    response = LLM.invoke([
        ("system", prompts.GENERATE_SQL_SYSTEM),
        ("user", prompts.GENERATE_SQL_USER.format(
            schema=state.schema,
            question=state.question,
        )),
    ])
    sql = _extract_sql(response.content)
    return {
        "sql": sql,
        "iteration": state.iteration + 1,
        "history": state.history + [{"node": "generate_sql", "sql": sql}],
    }


def execute_node(state: AgentState) -> dict:
    """Provided. Runs the SQL and stores the result."""
    return {"execution": execute_sql(state.db_id, state.sql)}


def _sample_rows(
    db_id: str,
    sql: str,
    max_tables: int = 3,
    n_rows: int = 3,
    max_cols: int = 12,
    max_cell: int = 40,
) -> str:
    """Real sample rows from the tables referenced by `sql`.

    When a query returns 0 rows or errors, the usual cause is a literal that
    doesn't match how the value is actually stored. Showing the model a few real
    rows lets it align on the true value format (casing, '-', trailing '.0',
    etc.) instead of guessing the same wrong literal again.

    Bounded on purpose: BIRD has very wide tables (cards has ~60 columns), so we
    cap columns per table and truncate long cells - an unbounded SELECT * blows
    up the prompt and times the revise call out.
    """
    path = db_path(db_id)
    low = sql.lower()
    blocks: list[str] = []

    def trim(c: object) -> str:
        s = "" if c is None else str(c)
        return s if len(s) <= max_cell else s[:max_cell] + "…"

    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
            tables = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
            ]
            used = [t for t in tables if re.search(rf"\b{re.escape(t.lower())}\b", low)]
            for t in used[:max_tables]:
                try:
                    cur = conn.execute(f'SELECT * FROM "{t}" LIMIT {n_rows}')
                    cols = [d[0] for d in cur.description][:max_cols]
                    rows = cur.fetchall()
                except sqlite3.Error:
                    continue
                note = f" (first {max_cols} of {len(cur.description)} columns)" if len(cur.description) > max_cols else ""
                lines = [", ".join(cols)]
                for row in rows:
                    lines.append(" | ".join(trim(c) for c in row[:max_cols]))
                blocks.append(f"Sample rows from {t}{note}:\n" + "\n".join(lines))
    except sqlite3.Error:
        return ""
    if not blocks:
        return ""
    return "\nActual sample data from the relevant tables (use it to match real value formats):\n" + "\n\n".join(blocks) + "\n"


def verify_node(state: AgentState) -> dict:
    """Decide whether state.execution plausibly answers state.question.

    Follow the generate_sql_node pattern: build messages from the VERIFY_*
    prompts, call llm(), parse the reply. Ask the model for a small JSON object
    like {"ok": bool, "issue": str} and parse it defensively - the model may
    wrap it in prose or fences. state.execution.render() gives you a compact
    view of the rows or error to feed into the prompt.

    Return: {"verify_ok": <bool>, "verify_issue": <str>}.
    What counts as "not plausible" is yours to define - see the Phase 3 targets
    in the README.
    """
    # A verify verdict only matters if a revise can follow it. At the iteration
    # cap, revise is unreachable (see route_after_verify), so the verify LLM call
    # is pure wasted latency/throughput - skip it and pass the SQL through.
    if state.iteration >= MAX_ITERATIONS:
        return {
            "verify_ok": True,
            "verify_issue": "",
            "history": state.history + [{"node": "verify", "ok": True, "issue": "skipped (at iteration cap)"}],
        }
    result = state.execution.render() if state.execution is not None else "ERROR: no execution result"
    response = LLM.invoke([
        ("system", prompts.VERIFY_SYSTEM),
        ("user", prompts.VERIFY_USER.format(
            question=state.question,
            sql=state.sql,
            result=result,
        )),
    ])
    verdict = _extract_json(response.content)
    # Default to ok=True when we can't parse a verdict: a malformed verifier
    # reply should not trap us in the revise loop until MAX_ITERATIONS.
    ok = bool(verdict.get("ok", True))
    issue = str(verdict.get("issue", "")) if not ok else ""
    return {
        "verify_ok": ok,
        "verify_issue": issue,
        "history": state.history + [{"node": "verify", "ok": ok, "issue": issue}],
    }


def revise_node(state: AgentState) -> dict:
    """Produce a revised SQL query given state.verify_issue and the prior attempt.

    Same shape as generate_sql_node, but the prompt should include the failing
    SQL, its execution result, and the verifier's complaint so the model can fix
    it. Bump the iteration counter the same way generate_sql_node does so the
    loop terminates.

    Return: {"sql": <str>, "iteration": state.iteration + 1, ...}.
    """
    result = state.execution.render() if state.execution is not None else "ERROR: no execution result"
    # When the prior attempt returned nothing or errored, the fix usually hinges
    # on the real stored value format - feed the model actual sample rows.
    needs_samples = state.execution is None or (not state.execution.ok) or state.execution.row_count == 0
    samples = _sample_rows(state.db_id, state.sql) if needs_samples else ""
    response = LLM.invoke([
        ("system", prompts.REVISE_SYSTEM),
        ("user", prompts.REVISE_USER.format(
            schema=state.schema,
            question=state.question,
            sql=state.sql,
            result=result,
            issue=state.verify_issue,
            samples=samples,
        )),
    ])
    sql = _extract_sql(response.content)
    return {
        "sql": sql,
        "iteration": state.iteration + 1,
        "history": state.history + [{"node": "revise", "sql": sql, "issue": state.verify_issue}],
    }


def route_after_verify(state: AgentState) -> str:
    """Conditional router: return "revise" to loop, "end" to terminate.

    Two reasons to end: the verifier was happy (state.verify_ok), or you've hit
    the iteration cap (state.iteration >= MAX_ITERATIONS). Otherwise, revise.
    """
    if state.verify_ok:
        return "end"
    if state.iteration >= MAX_ITERATIONS:
        return "end"
    return "revise"


# ---- Graph wiring -----------------------------------------------------

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("attach_schema", _attach_schema)
    g.add_node("generate_sql", generate_sql_node)
    g.add_node("execute", execute_node)
    g.add_node("verify", verify_node)
    g.add_node("revise", revise_node)

    g.add_edge(START, "attach_schema")
    g.add_edge("attach_schema", "generate_sql")
    g.add_edge("generate_sql", "execute")
    g.add_edge("execute", "verify")
    g.add_conditional_edges(
        "verify",
        route_after_verify,
        {"revise": "revise", "end": END},
    )
    g.add_edge("revise", "execute")
    return g.compile()


graph = build_graph()
