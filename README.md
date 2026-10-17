# Evidence RAG

Ask questions about a PDF and get answers that are **grounded, cited and scored**. LangGraph pipeline, Azure OpenAI
(embeddings + chat via the Responses API), in-memory Qdrant for vectors, local BM25, FastAPI + a small web UI.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Create `.env`:
```
AZURE_OPENAI_ENDPOINT=...                 # embeddings resource
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_API_VERSION=...
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large
AZURE_OPENAI_CHAT_DEPLOYMENT=...          # chat model (Responses API)
AZURE_OPENAI_CHAT_ENDPOINT=...            # optional: if chat lives on a different resource
AZURE_OPENAI_CHAT_API_KEY=...
AZURE_OPENAI_CHAT_API_VERSION=...
EVIDENCE_PDF=C:\path\to\file.pdf          # optional: index this PDF at startup
LOG_LEVEL=INFO                            # console level (the log file is always DEBUG)
TRACE_PAYLOADS=1                          # 0 = keep prompts/responses out of run records
```

## Run
- **UI + API:** `uvicorn server:app --port 8000 --no-access-log` → http://localhost:8000 (upload a PDF, ask questions)
- **CLI:** `python rag.py report.pdf "What was net income?" "What were total assets?"`

## Pipeline
`analyze` (LLM extracts metric / period / search query) → `retrieve` (Qdrant cosine + BM25, reciprocal-rank fusion)
→ `rerank` (rule-based boosts) → `verify` (evidence must contain the metric and period; else retry once, then abstain)
→ `infer` (LLM answers with `[chunk]` citations) → `validate` (0–1 verification score).

Questions run in parallel (one LangGraph `Send` branch each). Uploads are parsed, embedded (batched, with rate-limit
backoff) and indexed in the background; each document gets its own in-memory Qdrant collection (lost on restart; embeddings
are cached in `.cache/`).

**Verification score** = weighted mean of: LLM faithfulness judge (.40), numbers found verbatim (.20), metric present (.15),
period present (.10), valid citation (.10), rank of cited chunk (.05). Inapplicable checks are dropped and weights rescaled.
High ≥ 0.8, low < 0.5. The score is advisory; it never blocks an answer.

## Tracing
Every log line is tagged `[run · question · step]`, so one question can be followed end to end:

| Where | What |
|---|---|
| console | INFO: one line per step with timing, LLM token counts, retrieval ranks (`bm25#` / `vec#`), routing decisions |
| `logs/rag.log` | DEBUG, rotating: prompts and responses, candidate tables, score breakdowns. `grep "a1b2c3 · q2" logs/rag.log` |
| `logs/runs/<run_id>.json` | one structured record per run: settings, results, and per-step `trace` with `detail` |
| `GET /api/runs`, `/api/runs/{run_id}` | list / fetch run records (the UI links the run id) |
| UI | each result card: verification breakdown + step timeline with expandable detail |

HTTP requests are logged with an `X-Request-ID` header value (`http-xxxxxx` in the tag); background uploads use `ingest-xxxxxxxx`.

## Files
`graph.py` pipeline · `store.py` PDF loading, embeddings, Qdrant+BM25 index, document registry · `verification.py` scoring ·
`llm.py` Azure clients and traced calls · `logsetup.py` logging · `server.py` API · `rag.py` CLI · `ui.html` front end.

## Limits / production notes
- Text-based PDFs only (scanned PDFs need OCR first). Max upload 50 MB. One document per question.
- Rerank is a rule-based stand-in; use a cross-encoder or Azure semantic ranker for production.
- Verification is LLM-judged claim support plus literal checks, not formal entailment.
- Swap the in-memory Qdrant for a persistent store (`QdrantClient(path=...)` / server) to survive restarts.
