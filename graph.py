"""LangGraph pipeline.

Batch graph  : fans out one branch per question (Send), results merged with operator.add.
Question graph: analyze -> retrieve -> rerank -> verify -> (infer -> validate | expand_query -> retrieve | abstain)

Tracing: every node is wrapped by @traced, which tags log lines with [run · question · node], times the node, and appends
{node, at_ms, ms, summary, detail} to the question's `trace`. Each batch is also saved to logs/runs/<run_id>.json.
"""
from __future__ import annotations
import asyncio, functools, json, logging, operator, re, time, uuid
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
import llm, store, verification
from logsetup import LOG_DIR, set_context, reset_context
from verification import contains_any

log = logging.getLogger("rag.graph")
RUN_DIR = LOG_DIR / "runs"

# --------------------------------------------------------------------------- state
class QState(TypedDict, total=False):
    # input
    run_id: str
    question_id: str
    question: str
    document_id: str
    t0: float                    # perf_counter at question start (for trace offsets)
    # per-run settings
    k: int                       # initial retrieval candidates
    top_n: int                   # evidence passages kept after rerank
    strategy: str                # hybrid | vector | keyword
    rerank: bool
    judge: bool                  # LLM faithfulness check in the verification score
    prompt: str                  # extra instructions for the answer step
    max_attempts: int            # extra retrieval retries
    # derived
    metrics: list[str]
    period_terms: list[str]
    search_query: str
    candidates: list[dict]
    evidence: list[dict]
    evidence_sufficient: bool
    validation_reasons: list[str]
    route: str
    attempts: int
    # output
    status: str                  # answered | insufficient_evidence | error
    answer: str
    quote: str
    citations: list[dict]
    warnings: list[str]
    verification: dict
    trace: list[dict]

class BatchState(TypedDict):
    questions: list[dict]
    results: Annotated[list[dict], operator.add]

RESULT_KEYS = ("question_id", "question", "status", "answer", "quote", "citations", "attempts", "validation_reasons",
               "metrics", "period_terms", "warnings", "verification", "trace", "evidence")

# --------------------------------------------------------------------------- tracing
def _log_from(fn, level: int, msg: str, *args) -> None:
    """Emit a log record attributed to `fn` (file:line of its definition) rather than to the tracing wrapper."""
    code = fn.__code__
    log.handle(log.makeRecord(log.name, level, code.co_filename, code.co_firstlineno, msg, args, None, func=fn.__name__))

def traced(name: str, summary):
    """Wrap a node: context tags for logs, timing, and a trace entry. A node may return `_detail` (dict) for the trace."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(s: QState) -> dict:
            token = set_context(run=s.get("run_id", "-"), qid=s.get("question_id", "-"), node=name)
            t0 = time.perf_counter(); _log_from(fn, logging.DEBUG, "start | attempt=%s", s.get("attempts"))
            try:
                out = fn(s)
            except Exception:
                log.exception("node failed in %s", fn.__name__); raise
            else:
                ms = (time.perf_counter() - t0) * 1000; detail = out.pop("_detail", {}); text = summary({**s, **out})
                _log_from(fn, logging.INFO, "%s  (%.0f ms)", text, ms)
                if detail: _log_from(fn, logging.DEBUG, "detail: %s", json.dumps(detail, default=str)[:6000])
                entry = {"node": name, "at_ms": round((t0 - s.get("t0", t0)) * 1000), "ms": round(ms), "summary": text, "detail": detail}
                return {**out, "trace": s.get("trace", []) + [entry]}
            finally:
                reset_context(token)
        return wrapper
    return decorator

def _ids(items: list[dict], score_key: str | None = None, n: int = 6) -> str:
    return " ".join(x["passage"]["chunk_id"] + (f"({x[score_key]})" if score_key else "") for x in items[:n]) or "-"

# --------------------------------------------------------------------------- nodes
ANALYZE_PROMPT = ("You prepare retrieval queries over a document (SEC filing, report, contract, ...). From the user question return ONLY JSON: "
                  '{"metrics": [..], "period_terms": [..], "search_query": ".."}. '
                  "metrics = 1-3 line-item labels or key terms exactly as they would literally appear in the document "
                  "(e.g. 'Net income', 'Total assets', 'Cash and cash equivalents'). "
                  "period_terms = the period/date the question asks about, written as literally printed in the document "
                  "(e.g. 'May 28, 2026'); empty list if the question names no period. "
                  "search_query = a concise keyword query to find the right page.")

ANSWER_PROMPT = "Answer ONLY using the cited evidence. If the evidence is incomplete, reply INSUFFICIENT_EVIDENCE."
ANSWER_RULES = ' Copy numbers exactly as printed and state units (e.g. "in millions"). Cite chunk ids in square brackets like [p6].'

@traced("analyze", lambda s: f"metrics={s['metrics']} period={s['period_terms']} query={s['search_query']!r}")
def analyze(s: QState) -> dict:
    out: dict = {"metrics": [], "period_terms": [], "search_query": s["question"], "attempts": 0, "warnings": []}; detail: dict = {}
    try:
        raw, detail["llm"] = llm.call_chat("analyze", ANALYZE_PROMPT, s["question"])
        parsed = llm.parse_json(raw); detail["parsed"] = parsed
        out["metrics"] = [m for m in parsed.get("metrics", []) if isinstance(m, str) and m.strip()]
        out["period_terms"] = [m for m in parsed.get("period_terms", []) if isinstance(m, str) and m.strip()]
        out["search_query"] = parsed.get("search_query") or s["question"]
    except Exception as e:
        log.warning("analyze failed (%s: %s); continuing without metric/period", type(e).__name__, str(e)[:160])
        out["warnings"].append(f"analyze failed: {type(e).__name__}"); detail["error"] = str(e)[:300]
    return {**out, "_detail": detail}

@traced("retrieve", lambda s: f"attempt {s['attempts']} | {len(s['candidates'])} candidates | top: " +
        " ".join(f"{c['passage']['chunk_id']}({c['retrieval_score']} bm25#{c['bm25_rank']} vec#{c['vector_rank']})" for c in s["candidates"][:5]))
def retrieve(s: QState) -> dict:
    hits, timings = store.get_index(s["document_id"]).search(s["search_query"], k=s.get("k", 20), strategy=s.get("strategy", "hybrid"))
    candidates = [{"passage": asdict(h.passage), "retrieval_score": h.score, "bm25": h.bm25, "vector": h.vector,
                   "bm25_rank": h.bm25_rank, "vector_rank": h.vector_rank} for h in hits]
    top = [{"chunk_id": c["passage"]["chunk_id"], "page": c["passage"]["page"], "section": c["passage"]["section"],
            **{k: c[k] for k in ("retrieval_score", "bm25", "vector", "bm25_rank", "vector_rank")}} for c in candidates[:10]]
    return {"candidates": candidates, "attempts": s["attempts"] + 1,
            "_detail": {"query": s["search_query"], "strategy": s.get("strategy", "hybrid"), "k": s.get("k", 20), "timings": timings, "top": top}}

@traced("rerank", lambda s: f"kept {len(s['evidence'])}: {_ids(s['evidence'], 'rerank_score')}")
def rerank(s: QState) -> dict:
    # Illustrative rule-based rerank: boosts for metric / period / section matches. Swap for a cross-encoder in production.
    top_n = s.get("top_n", 5); rows = []
    for item in s["candidates"]:
        p = item["passage"]
        parts = {"retrieval": item["retrieval_score"],
                 "metric": 1.0 if s["metrics"] and contains_any(p["text"], s["metrics"]) else 0.0,
                 "section": 0.5 if s["metrics"] and contains_any(p["section"], s["metrics"]) else 0.0,
                 "period": 1.0 if s["period_terms"] and contains_any(p["text"], s["period_terms"]) else 0.0}
        if s.get("rerank") is False: parts = {"retrieval": item["retrieval_score"]}
        rows.append({**item, "rerank_score": round(sum(parts.values()), 5), "score_parts": parts})
    evidence = rows[:top_n] if s.get("rerank") is False else sorted(rows, key=lambda x: -x["rerank_score"])[:top_n]
    return {"evidence": evidence, "_detail": {"enabled": s.get("rerank") is not False,
            "kept": [{"chunk_id": x["passage"]["chunk_id"], "page": x["passage"]["page"], "score": x["rerank_score"], "parts": x["score_parts"]} for x in evidence]}}

@traced("verify", lambda s: f"sufficient={s['evidence_sufficient']} supporting={_ids(s['evidence'])} -> {s['route']}" +
        (f" | reasons={s['validation_reasons']}" if s["validation_reasons"] else ""))
def verify(s: QState) -> dict:
    """Evidence gate: a page supports the answer only if it contains the metric (and the period, when one was asked)."""
    checks = []
    for x in s["evidence"]:
        c = {"chunk_id": x["passage"]["chunk_id"],
             "metric": contains_any(x["passage"]["text"], s["metrics"]) if s["metrics"] else None,
             "period": contains_any(x["passage"]["text"], s["period_terms"]) if s["period_terms"] else None}
        c["passed"] = c["metric"] is not False and c["period"] is not False
        checks.append(c)
    matching = [x for x, c in zip(s["evidence"], checks) if c["passed"]]
    reasons = []
    if not s["metrics"]: reasons.append("Unsupported or ambiguous metric")
    if not matching: reasons.append("No matching evidence for metric" + (" and period" if s["period_terms"] else ""))
    sufficient = bool(matching) and not reasons
    route = "infer" if sufficient else ("expand_query" if s["attempts"] <= s.get("max_attempts", 1) else "abstain")
    return {"evidence_sufficient": sufficient, "validation_reasons": reasons, "evidence": matching or s["evidence"], "route": route,
            "_detail": {"checks": checks, "attempts": s["attempts"], "retries_allowed": s.get("max_attempts", 1), "route": route}}

@traced("expand_query", lambda s: f"new query={s['search_query']!r}")
def expand_query(s: QState) -> dict:
    return {"search_query": " ".join(filter(None, [*s["metrics"], *s["period_terms"], s["question"]]))}

def make_quote(item: dict, metrics: list[str]) -> str:
    lines = [l for l in item["passage"]["text"].splitlines() if contains_any(l, metrics)] if metrics else []
    return "\n".join(lines[:3]) or item["passage"]["text"][:240]

@traced("infer", lambda s: f"status={s['status']} cites={[c['chunk_id'] for c in s.get('citations', [])]} answer={s.get('answer', '')[:90]!r}")
def infer(s: QState) -> dict:
    evidence = s["evidence"]
    context = "\n\n".join(f"[{x['passage']['chunk_id']}] (page {x['passage']['page']}, {x['passage']['section']})\n{x['passage']['text']}" for x in evidence)
    system = (s.get("prompt") or ANSWER_PROMPT) + ANSWER_RULES
    detail: dict = {"context_chunks": [x["passage"]["chunk_id"] for x in evidence]}
    try:
        answer, detail["llm"] = llm.call_chat("answer", system, f"Question: {s['question']}\nEvidence:\n{context}")
    except Exception as e:
        return {"status": "error", "answer": "", "citations": [], "warnings": s.get("warnings", []) + [f"chat failed: {type(e).__name__}"],
                "_detail": {**detail, "error": str(e)[:300]}}
    answer = answer.strip()
    if "INSUFFICIENT_EVIDENCE" in answer:
        return {"status": "insufficient_evidence", "answer": "", "citations": [], "_detail": detail}
    cited = set(re.findall(r"\bp\d+(?:\.\d+)?\b", answer))
    used = [x for x in evidence if x["passage"]["chunk_id"] in cited] or evidence
    return {"status": "answered", "answer": answer, "quote": make_quote(used[0], s["metrics"]), "_detail": detail,
            "citations": [{"chunk_id": x["passage"]["chunk_id"], "page": x["passage"]["page"], "document_id": x["passage"]["document_id"],
                           "version_id": x["passage"]["version_id"]} for x in used]}

def _validate_summary(s: QState) -> str:
    v = s.get("verification")
    if not v: return f"status={s['status']}"
    parts = " ".join(f"{k}={c['score']}" for k, c in v["components"].items() if c["score"] is not None)
    return f"score={v['score']} ({v['label']}) | {parts}" + (f" | warnings={s['warnings']}" if s.get("warnings") else "")

@traced("validate", _validate_summary)
def validate_answer(s: QState) -> dict:
    if s.get("status") == "error": return {}
    if not s.get("citations") or not s.get("answer"):
        return {"status": "insufficient_evidence", "answer": "", "citations": []}
    cited = {c["chunk_id"] for c in s["citations"]}
    used = [x for x in s["evidence"] if x["passage"]["chunk_id"] in cited]
    v, calls = verification.verify_answer(s, s["answer"], used)
    c, warnings = v["components"], list(s.get("warnings", []))
    if c["numbers"]["score"] is not None and c["numbers"]["score"] < 1: warnings.append(c["numbers"]["detail"])
    if c["judge"]["score"] is not None and c["judge"]["score"] < 1: warnings.append("faithfulness: " + c["judge"]["detail"])
    if v["score"] < verification.LOW_SCORE: warnings.append(f"low verification score ({v['score']})")
    return {"warnings": warnings, "verification": v, "_detail": {"components": c, "llm": calls}}

@traced("abstain", lambda s: f"reasons={s.get('validation_reasons')}")
def abstain(s: QState) -> dict:
    return {"status": "insufficient_evidence", "answer": "", "citations": []}

# --------------------------------------------------------------------------- graphs
def build_question_graph():
    g = StateGraph(QState)
    for fn in (analyze, retrieve, rerank, verify, expand_query, infer, validate_answer, abstain):
        g.add_node(fn.__name__, fn)
    g.add_edge(START, "analyze"); g.add_edge("analyze", "retrieve"); g.add_edge("retrieve", "rerank"); g.add_edge("rerank", "verify")
    g.add_conditional_edges("verify", lambda s: s["route"], {"infer": "infer", "expand_query": "expand_query", "abstain": "abstain"})
    g.add_edge("expand_query", "retrieve"); g.add_edge("infer", "validate_answer"); g.add_edge("validate_answer", END); g.add_edge("abstain", END)
    return g.compile()

QUESTION_GRAPH = build_question_graph()

async def execute_one(s: dict) -> dict:
    token = set_context(run=s["run_id"], qid=s["question_id"], node="-")
    try:
        t0 = time.perf_counter()
        log.info("question: %r | doc=%s k=%s top_n=%s strategy=%s rerank=%s judge=%s retries=%s", s["question"], s["document_id"],
                 s.get("k"), s.get("top_n"), s.get("strategy"), s.get("rerank"), s.get("judge"), s.get("max_attempts"))
        final = await asyncio.to_thread(QUESTION_GRAPH.invoke, {**s, "t0": t0})
        result = {k: final.get(k) for k in RESULT_KEYS}; result["latency_ms"] = round((time.perf_counter() - t0) * 1000)
        v = result["verification"]
        log.info("done: %s in %d ms%s", result["status"], result["latency_ms"], f" | score {v['score']} ({v['label']})" if v else "")
        return result
    finally:
        reset_context(token)

async def run_question(s: BatchState) -> dict:
    return {"results": [await execute_one(s)]}   # Send supplies the question's QState fields

def build_batch_graph():
    g = StateGraph(BatchState)
    g.add_node("run_question", run_question)
    g.add_conditional_edges(START, lambda s: [Send("run_question", {**q, "max_attempts": q.get("max_attempts", 1)}) for q in s["questions"]], ["run_question"])
    g.add_edge("run_question", END)
    return g.compile()

BATCH_GRAPH = build_batch_graph()

def make_batch(texts: list[str], document_id: str, **settings) -> list[dict]:
    store.get_index(document_id)   # fail fast on an unknown document
    return [{"question_id": f"q{i+1}", "question": t, "document_id": document_id, **settings} for i, t in enumerate(texts)]

async def execute_batch(questions: list[dict], max_concurrency: int = 3) -> dict:
    """Run all questions; returns {run_id, results, elapsed_ms}. The full record is saved to logs/runs/<run_id>.json."""
    run_id = uuid.uuid4().hex[:6]; token = set_context(run=run_id, qid="-", node="-"); t0 = time.perf_counter()
    try:
        settings = {k: v for k, v in questions[0].items() if k not in ("question", "question_id")}
        log.info("run start: %d question(s), concurrency=%d, model=%s | settings=%s", len(questions), max_concurrency, llm.chat_deployment(), settings)
        result = await BATCH_GRAPH.ainvoke({"questions": [{**q, "run_id": run_id} for q in questions], "results": []},
                                           config={"max_concurrency": max_concurrency})
        by_id = {x["question_id"]: x for x in result["results"]}
        results = [by_id[q["question_id"]] for q in questions]
        elapsed = round((time.perf_counter() - t0) * 1000)
        log.info("run done in %.1fs: %s", elapsed / 1000, dict(Counter(r["status"] for r in results)))
        record = {"run_id": run_id, "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "elapsed_ms": elapsed,
                  "model": llm.chat_deployment(), "max_concurrency": max_concurrency, "settings": settings, "results": results}
        RUN_DIR.mkdir(parents=True, exist_ok=True); (RUN_DIR / f"{run_id}.json").write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
        log.debug("run record saved: %s", RUN_DIR / f"{run_id}.json")
        return {"run_id": run_id, "results": results, "elapsed_ms": elapsed}
    finally:
        reset_context(token)

def load_run(run_id: str) -> dict | None:
    f = RUN_DIR / f"{''.join(c for c in run_id if c.isalnum())}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None

def list_runs(limit: int = 20) -> list[dict]:
    files = sorted(RUN_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)[:limit] if RUN_DIR.exists() else []
    out = []
    for f in files:
        r = json.loads(f.read_text(encoding="utf-8"))
        out.append({"run_id": r["run_id"], "started_at": r["started_at"], "elapsed_ms": r["elapsed_ms"], "questions": len(r["results"]),
                    "statuses": dict(Counter(x["status"] for x in r["results"]))})
    return out
