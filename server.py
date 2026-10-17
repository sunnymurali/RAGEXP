"""FastAPI wrapper + static UI.   Run: uvicorn server:app --port 8000 --no-access-log   then open http://localhost:8000"""
from __future__ import annotations
import logging, threading, time, uuid
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from logsetup import setup_logging, set_context

setup_logging()
import graph, llm, store   # noqa: E402  (imported after logging is configured)

log = logging.getLogger("rag.http")
UPLOAD_DIR = Path(__file__).parent / "uploads"; MAX_UPLOAD_MB = 50

def _reload_uploads() -> None:
    """Re-index earlier uploads after a restart (embeddings come from .cache/, so this is quick). Runs in the background."""
    set_context(run="startup", qid="-", node="-")
    for pdf in sorted(UPLOAD_DIR.glob("*.pdf")):
        try: store.ingest_pdf(str(pdf), pdf.name)
        except Exception: log.exception("could not reload %s", pdf.name)

@asynccontextmanager
async def lifespan(_: FastAPI):
    store.preload_from_env()
    threading.Thread(target=_reload_uploads, daemon=True).start()
    yield

app = FastAPI(title="Evidence RAG", lifespan=lifespan)

@app.middleware("http")
async def access_log(request: Request, call_next):
    rid = uuid.uuid4().hex[:6]; set_context(run=f"http-{rid}", qid="-", node="-"); t0 = time.perf_counter()
    quiet = request.url.path.startswith("/api/upload/") or request.url.path == "/api/info"   # polled by the UI
    try:
        response = await call_next(request)
    except Exception:
        log.exception("%s %s -> unhandled error", request.method, request.url.path); raise
    (log.debug if quiet else log.info)("%s %s -> %d (%.0f ms)", request.method, request.url.path, response.status_code, (time.perf_counter() - t0) * 1000)
    response.headers["X-Request-ID"] = rid
    return response

class RunRequest(BaseModel):
    document_id: str
    questions: list[str] = Field(min_length=1, max_length=20)
    k: int = Field(30, ge=1, le=100)            # initial retrieval candidates
    top_n: int = Field(5, ge=1, le=20)          # evidence passages kept after rerank
    strategy: str = Field("hybrid", pattern="^(hybrid|vector|keyword)$")
    rerank: bool = True
    judge: bool = True                          # LLM faithfulness check inside the verification score
    prompt: str = ""
    max_retries: int = Field(1, ge=0, le=3)
    max_concurrency: int = Field(3, ge=1, le=10)

def doc_info(idx: store.Index) -> dict:
    return {"document_id": idx.doc_id, "version_id": idx.version_id, "chunks": idx.N, "pages": max(p.page for p in idx.passages), "dim": idx.dim}

@app.get("/api/info")
def info():
    return {"chat_model": llm.chat_deployment(), "embedding_model": llm.embedding_deployment(),
            "documents": [doc_info(i) for i in store.INDEXES.values()]}

# --------------------------------------------------------------------------- upload (background ingest)
JOBS: dict[str, dict] = {}

def _ingest_job(job_id: str, path: str, filename: str) -> None:
    set_context(run=f"ingest-{job_id}", qid="-", node="-")
    job = JOBS[job_id]
    def progress(stage: str, done: int, total: int): job.update(stage=stage, done=done, total=total)
    try:
        job.update(stage="done", result=doc_info(store.ingest_pdf(path, filename, progress)))
    except Exception as e:
        log.exception("ingest failed"); job.update(stage="error", error=f"Could not index PDF: {e}")

@app.post("/api/upload")
def upload(file: UploadFile = File(...)):
    """Accepts a text-based PDF and returns a job id; parse -> embed -> Qdrant runs in the background (poll /api/upload/{job_id})."""
    data = file.file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024: raise HTTPException(413, f"File larger than {MAX_UPLOAD_MB} MB")
    if not data.startswith(b"%PDF"): raise HTTPException(400, "Not a PDF file")
    UPLOAD_DIR.mkdir(exist_ok=True); path = (UPLOAD_DIR / store._slug(file.filename or "upload")).with_suffix(".pdf"); path.write_bytes(data)
    job_id = uuid.uuid4().hex[:8]; JOBS[job_id] = {"stage": "queued", "done": 0, "total": 0, "filename": file.filename}
    log.info("upload: %s (%.1f MB) -> background job %s", file.filename, len(data) / 1e6, job_id)
    threading.Thread(target=_ingest_job, args=(job_id, str(path), file.filename or "upload"), daemon=True).start()
    return {"job_id": job_id}

@app.get("/api/upload/{job_id}")
def upload_status(job_id: str):
    if job_id not in JOBS: raise HTTPException(404, "Unknown job")
    return JOBS[job_id]

# --------------------------------------------------------------------------- run + traces
@app.post("/api/run")
async def run(req: RunRequest):
    try:
        batch = graph.make_batch([q.strip() for q in req.questions if q.strip()], req.document_id, k=req.k, top_n=req.top_n, strategy=req.strategy,
                                 rerank=req.rerank, judge=req.judge, prompt=req.prompt.strip(), max_attempts=req.max_retries)
    except KeyError as e:
        raise HTTPException(404, str(e).strip("'\""))
    if not batch: raise HTTPException(400, "No questions")
    return await graph.execute_batch(batch, max_concurrency=req.max_concurrency)

@app.get("/api/runs")
def runs(limit: int = 20):
    return graph.list_runs(limit)

@app.get("/api/runs/{run_id}")
def run_record(run_id: str):
    record = graph.load_run(run_id)
    if not record: raise HTTPException(404, "Unknown run")
    return record

@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "ui.html")
