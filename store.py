"""Documents and retrieval.

PDF -> page chunks -> embeddings (Azure, cached on disk) -> in-memory Qdrant collection.
Search = dense vectors (Qdrant, cosine) + BM25, merged with reciprocal-rank fusion. One Index per document.
"""
from __future__ import annotations
import hashlib, json, logging, math, os, re, time
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable
import llm

log = logging.getLogger("rag.store")
Progress = Callable[[str, int, int], None]   # (stage, done, total)

CACHE_DIR = Path(__file__).parent / ".cache"
EMBED_BATCH = 32          # chunks per embedding request
EMBED_ATTEMPTS = 8        # per batch, on HTTP 429

# --------------------------------------------------------------------------- documents
@dataclass(frozen=True)
class Passage:
    chunk_id: str
    document_id: str
    version_id: str
    page: int
    section: str
    text: str

@dataclass
class Hit:
    passage: Passage
    score: float          # fused (RRF) score
    bm25: float
    vector: float         # cosine similarity
    bm25_rank: int
    vector_rank: int

_BOILERPLATE = re.compile(r"^(table of contents|\(?in millions.*|\(unaudited\)|\d+)$", re.I)

def load_pdf(path: str, doc_id: str = "", max_chars: int = 3500) -> list[Passage]:
    """One passage per page; long pages are split on line boundaries. `page` is the 1-based PDF page index."""
    from pypdf import PdfReader
    t0 = time.perf_counter(); doc_id = doc_id or Path(path).stem
    reader = PdfReader(path); out: list[Passage] = []; empty = 0
    for i, page in enumerate(reader.pages, 1):
        text = (page.extract_text() or "").strip()
        if not text: empty += 1; continue
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        # Section label = first line that is not boilerplate or a company-name banner.
        section = next((l for l in lines if not _BOILERPLATE.match(l) and not l.endswith(("Inc.", "Corp."))), lines[0])[:90]
        parts, cur = [], ""
        for line in lines:
            if cur and len(cur) + len(line) > max_chars: parts.append(cur); cur = ""
            cur += line + "\n"
        parts.append(cur)
        for n, body in enumerate(parts):
            out.append(Passage(f"p{i}" + (f".{n+1}" if len(parts) > 1 else ""), doc_id, "v1", i, section, body.strip()))
    if not out: raise ValueError("No extractable text in this PDF (scanned/image-only PDFs need OCR first)")
    log.info("PDF parsed: %s | %d pages -> %d chunks (%d empty pages) in %.1fs", Path(path).name, len(reader.pages), len(out), empty, time.perf_counter() - t0)
    return out

# --------------------------------------------------------------------------- index
_WORD = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?%?", re.I)
def tokenize(s: str) -> list[str]:
    return _WORD.findall(s.lower())

class Index:
    def __init__(self, passages: list[Passage], progress: Progress | None = None):
        self.passages = passages; self.progress = progress or (lambda *_: None)
        self.doc_id, self.version_id, self.digest = passages[0].document_id, passages[0].version_id, ""
        self.N = len(passages)
        texts = [p.text + " " + p.section for p in passages]
        self._tokens = [tokenize(t) for t in texts]
        self._counts = [Counter(t) for t in self._tokens]
        self.df = Counter(t for toks in self._tokens for t in set(toks))
        self.avg_len = sum(len(t) for t in self._tokens) / max(1, self.N)
        vectors = self._embed(texts)
        self.dim = len(vectors[0])
        self._build_qdrant(vectors)
        log.info("Index ready: %s | %d chunks | dim %d | qdrant:memory + BM25", self.doc_id, self.N, self.dim)

    # -- ingestion
    def _embed(self, texts: list[str]) -> list[list[float]]:
        dep = llm.embedding_deployment()
        key = hashlib.sha1((dep + "\x00" + "\x00".join(texts)).encode()).hexdigest()[:16]
        cache = CACHE_DIR / f"emb_{key}.json"
        if cache.exists():
            log.info("Embeddings: cache hit (%s)", cache.name); return json.loads(cache.read_text())
        t0 = time.perf_counter(); vectors: list[list[float]] = []; embedder = llm.get_embeddings()
        for start in range(0, len(texts), EMBED_BATCH):
            batch = texts[start:start + EMBED_BATCH]; tb = time.perf_counter()
            for attempt in range(1, EMBED_ATTEMPTS + 1):
                try: vectors += embedder.embed_documents(batch); break
                except Exception as e:   # Azure tokens-per-minute limit -> wait and retry
                    rate_limited = "429" in str(e) or "RateLimit" in type(e).__name__
                    if not rate_limited or attempt == EMBED_ATTEMPTS: raise RuntimeError(f"Embedding failed ({type(e).__name__}): {str(e)[:300]}") from e
                    m = re.search(r"retry after (\d+)", str(e), re.I); wait = int(m.group(1)) + 1 if m else min(60, 4 * attempt)
                    log.warning("Embeddings: rate limited on chunks %d-%d of %d; waiting %ds [attempt %d/%d]", start, start + len(batch), len(texts), wait, attempt, EMBED_ATTEMPTS)
                    time.sleep(wait)
            self.progress("embedding", len(vectors), len(texts))
            log.info("Embeddings: %d/%d chunks (batch took %.1fs)", len(vectors), len(texts), time.perf_counter() - tb)
        CACHE_DIR.mkdir(exist_ok=True); cache.write_text(json.dumps(vectors))
        log.info("Embeddings: %d chunks with %s in %.1fs (cached as %s)", len(texts), dep, time.perf_counter() - t0, cache.name)
        return vectors

    def _build_qdrant(self, vectors: list[list[float]]) -> None:
        from qdrant_client import QdrantClient, models
        self.progress("indexing", 0, self.N); t0 = time.perf_counter()
        self.qdrant = QdrantClient(":memory:"); self.collection = "evidence_chunks"
        self.qdrant.create_collection(self.collection, vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.COSINE))
        self.qdrant.upsert(self.collection, points=[models.PointStruct(id=i, vector=v, payload=asdict(p)) for i, (p, v) in enumerate(zip(self.passages, vectors))])
        log.info("Qdrant: upserted %d points into '%s' in %.2fs", self.N, self.collection, time.perf_counter() - t0)

    # -- retrieval
    def _bm25(self, query_tokens: set[str], i: int) -> float:
        counts, length, score = self._counts[i], len(self._tokens[i]), 0.0
        for t in query_tokens:
            if not counts[t]: continue
            idf = math.log(1 + (self.N - self.df[t] + .5) / (self.df[t] + .5))
            score += idf * counts[t] * 2.2 / (counts[t] + 1.2 * (.25 + .75 * length / max(1, self.avg_len)))
        return score

    def search(self, query: str, k: int = 20, strategy: str = "hybrid") -> tuple[list[Hit], dict]:
        """Returns (hits, timings). strategy: hybrid | vector | keyword."""
        t0 = time.perf_counter(); qv = llm.get_embeddings().embed_query(query); t_embed = time.perf_counter()
        points = self.qdrant.query_points(self.collection, query=qv, limit=self.N).points; t_qdrant = time.perf_counter()
        vector = {p.id: p.score for p in points}
        tokens = set(tokenize(query)); bm25 = [self._bm25(tokens, i) for i in range(self.N)]
        bm25_rank = {i: r for r, i in enumerate(sorted(range(self.N), key=lambda i: -bm25[i]), 1)}
        vec_rank = {i: r for r, i in enumerate(sorted(range(self.N), key=lambda i: -vector.get(i, 0.0)), 1)}
        ranks = {"hybrid": (bm25_rank, vec_rank), "keyword": (bm25_rank,), "vector": (vec_rank,)}[strategy]
        fused = {i: sum(1 / (60 + r[i]) for r in ranks) for i in range(self.N)}   # reciprocal-rank fusion
        top = sorted(fused, key=lambda i: -fused[i])[:k]
        timings = {"embed_query_ms": round((t_embed - t0) * 1000), "qdrant_ms": round((t_qdrant - t_embed) * 1000),
                   "total_ms": round((time.perf_counter() - t0) * 1000)}
        log.debug("search %r strategy=%s k=%d -> %s | %s", query, strategy, k, [(self.passages[i].chunk_id, round(fused[i], 5)) for i in top], timings)
        return [Hit(self.passages[i], round(fused[i], 5), round(bm25[i], 3), round(vector.get(i, 0.0), 4), bm25_rank[i], vec_rank[i]) for i in top], timings

# --------------------------------------------------------------------------- registry
INDEXES: dict[str, Index] = {}    # document_id -> Index (each with its own in-memory Qdrant collection)

def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(name).stem).strip("_")[:60] or "document"

def get_index(doc_id: str) -> Index:
    if doc_id not in INDEXES: raise KeyError(f"Unknown document '{doc_id}'")
    return INDEXES[doc_id]

def ingest_pdf(path: str, name: str = "", progress: Progress | None = None) -> Index:
    """Parse, embed and index a PDF. Same name + same content is reused; same name with new content gets a numeric suffix."""
    digest = hashlib.sha1(Path(path).read_bytes()).hexdigest()[:10]; base = _slug(name or path); doc_id = base; n = 1
    while doc_id in INDEXES and INDEXES[doc_id].digest != digest:
        n += 1; doc_id = f"{base}_{n}"
    if doc_id in INDEXES:
        log.info("Ingest: %s already indexed (same content) -> reusing", doc_id); return INDEXES[doc_id]
    t0 = time.perf_counter(); log.info("Ingest: %s -> document_id=%s", Path(path).name, doc_id)
    if progress: progress("parsing", 0, 0)
    idx = Index(load_pdf(path, doc_id), progress); idx.digest = digest
    INDEXES[doc_id] = idx
    log.info("Ingest: %s done in %.1fs", doc_id, time.perf_counter() - t0)
    return idx

def preload_from_env() -> None:
    """Optionally index the PDF named by EVIDENCE_PDF at startup."""
    path = os.getenv("EVIDENCE_PDF", "").strip().strip('"')
    if not path: return
    if Path(path).exists(): ingest_pdf(path)
    else: log.warning("EVIDENCE_PDF=%s not found; skipping preload", path)
