"""Command line: ask questions about a PDF.

    python rag.py report.pdf "What was net income?" "What were total assets?"
    python rag.py "What was net income?"            # uses the PDF named by EVIDENCE_PDF in .env

Options: --json (also print full results). Logs: console INFO, full detail in logs/rag.log, run record in logs/runs/<run_id>.json
"""
from __future__ import annotations
import asyncio, json, sys
from pathlib import Path
from logsetup import setup_logging

setup_logging()
import graph, store   # noqa: E402  (imported after logging is configured)

def main(argv: list[str]) -> int:
    flags = [a for a in argv if a.startswith("--")]; args = [a for a in argv if not a.startswith("--")]
    if args and args[0].lower().endswith(".pdf"):
        if not Path(args[0]).exists(): print(f"File not found: {args[0]}"); return 2
        store.ingest_pdf(args[0]); args = args[1:]
    else:
        store.preload_from_env()
    if not store.INDEXES: print("No document: pass a PDF path or set EVIDENCE_PDF in .env"); return 2
    if not args: print(__doc__); return 2

    doc_id = next(iter(store.INDEXES))
    out = asyncio.run(graph.execute_batch(graph.make_batch(args, doc_id)))
    print()
    for r in out["results"]:
        v = r.get("verification"); score = f" score={v['score']} ({v['label']})" if v else ""
        cites = ", ".join(f"{c['chunk_id']} (p.{c['page']})" for c in r["citations"]) or "-"
        print(f"{r['question_id']} [{r['status']}{score}] {r['question']}\n    answer: {r['answer'] or '-'}\n    cites:  {cites}")
        if r["warnings"]: print(f"    warns:  {r['warnings']}")
        if r["validation_reasons"] and r["status"] != "answered": print(f"    why:    {r['validation_reasons']}")
    print(f"\nrun {out['run_id']} · {out['elapsed_ms'] / 1000:.1f}s · trace: logs/runs/{out['run_id']}.json · log: logs/rag.log")
    if "--json" in flags: print(json.dumps(out["results"], indent=2, default=str))
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
