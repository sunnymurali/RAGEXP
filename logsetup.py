"""Logging setup.

Every log line carries a trace tag  [run · question · step]  taken from a context variable, so one grep/filter follows a
single question through retrieval, LLM calls and verification (also across threads: asyncio.to_thread copies the context).

  console : INFO  (rich, coloured)          level via LOG_LEVEL
  file    : DEBUG (logs/rag.log, rotating)  full detail: prompts, candidate tables, score breakdowns; every line has file:line + function
  runs    : logs/runs/<run_id>.json         one structured record per batch run (see graph.py)

Follow one question:   grep "a1b2c3 · q2" logs/rag.log
"""
from __future__ import annotations
import contextvars, logging, os, sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path(__file__).parent / "logs"
_ctx: contextvars.ContextVar[dict] = contextvars.ContextVar("trace_ctx", default={})

def set_context(**fields) -> contextvars.Token:
    """Merge fields (run / qid / node) into the current trace context. Pass the returned token to reset_context()."""
    return _ctx.set({**_ctx.get(), **fields})

def reset_context(token: contextvars.Token) -> None:
    _ctx.reset(token)

class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        c = _ctx.get()
        record.run, record.qid, record.node = c.get("run", "-"), c.get("qid", "-"), c.get("node", "-")
        parts = [c[k] for k in ("run", "qid", "node") if c.get(k)]
        record.tag = f"[{' · '.join(parts)}] " if parts else ""
        return True

def setup_logging() -> None:
    root = logging.getLogger("rag")
    if root.handlers: return
    for stream in (sys.stdout, sys.stderr):
        try: stream.reconfigure(encoding="utf-8")
        except Exception: pass
    root.setLevel(logging.DEBUG); root.propagate = False
    ctx_filter = _ContextFilter()

    try:
        from rich.console import Console
        from rich.logging import RichHandler
        console = RichHandler(console=Console(stderr=True), show_path=True, markup=False, rich_tracebacks=True, log_time_format="%H:%M:%S")
        console.setFormatter(logging.Formatter("%(tag)s%(message)s"))
    except ImportError:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(filename)s:%(lineno)d %(tag)s%(message)s", "%H:%M:%S"))
    console.setLevel(os.getenv("LOG_LEVEL", "INFO").upper()); console.addFilter(ctx_filter)
    root.addHandler(console)

    LOG_DIR.mkdir(exist_ok=True)
    file_handler = RotatingFileHandler(LOG_DIR / "rag.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)-7s %(filename)s:%(lineno)d %(funcName)s run=%(run)s q=%(qid)s node=%(node)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
    file_handler.setLevel(logging.DEBUG); file_handler.addFilter(ctx_filter)
    root.addHandler(file_handler)
