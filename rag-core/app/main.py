"""rag-core: FastAPI + MCP (streamable HTTP) + in-memory ingest queue.

Endpoints:
  GET  /health          - liveness (also used by the container healthcheck)
  GET  /metrics         - prometheus text format (scraped by the main repo's Prometheus)
  POST /ingest          - {"paths": [...]} or {} for a full rescan -> 202 {"job_id"}
  GET  /ingest/{job_id} - job status
  POST /mcp             - MCP streamable-HTTP endpoint for opencode
MCP tools:
  search(query, k, source, path)      - hybrid search (qdrant + Dify datasets
                                         in weaviate), re-ranked by rag-reranker
  list_collections()                  - qdrant collections + dify dataset classes
  get_document(path)                  - full file text from the corpus
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import PlainTextResponse

from chunker import chunk_id, chunk_markdown
from embedder import Embedder, EmbedderError, METRICS, render_metrics
from rerank_client import RerankClient, RerankError
from store import COLLECTION, Store
from weaviate_store import WeaviateError, WeaviateStore

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def _read_key(path: str | None) -> str:
    if path and os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    return os.environ.get("RAG_API_KEY", "")


RAG_API_KEY = _read_key(os.environ.get("RAG_API_KEY_FILE", "/run/secrets/rag_api_key"))
CORPUS_ROOTS = [r for r in os.environ.get("CORPUS_ROOTS", "/corpus/ai-compose").split(",") if r]
STATE_DIR = os.environ.get("STATE_DIR", "/state")
CHUNK_CHARS = int(os.environ.get("CHUNK_CHARS", "2000"))
EMBED_BATCH = int(os.environ.get("EMBED_BATCH", "16"))
# Unified search: how many candidates to pull from each store before the
# reranker picks the final top-k (see unified_search below).
QDRANT_CANDIDATES = int(os.environ.get("QDRANT_CANDIDATES", "20"))
WEAVIATE_PER_CLASS = int(os.environ.get("WEAVIATE_PER_CLASS", "10"))
RERANK_MAX_CANDIDATES = int(os.environ.get("RERANK_MAX_CANDIDATES", "60"))
STATE_FILE = os.path.join(STATE_DIR, "state.json")  # path -> {hash, mtime}

@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Start the ingest worker and the MCP session manager. mcp v2 requires
    the host app's lifespan to enter mcp.session_manager.run() - a mounted
    sub-app's own lifespan never runs, so nothing else starts the manager."""
    global worker_task
    load_state()
    # wait for embedder/qdrant (compose already gates us on health, but be safe)
    for attempt in range(30):
        try:
            dim = len(embedder.embed("dim probe"))
            store.ensure_collection(dim)
            break
        except Exception as e:
            if attempt == 29:
                raise RuntimeError(f"dependencies not ready: {e}") from e
            await asyncio.sleep(2)
    worker_task = asyncio.create_task(worker())
    async with mcp.session_manager.run():
        yield
    if worker_task is not None:
        worker_task.cancel()
    embedder.close()
    wv.close()
    reranker.close()


app = FastAPI(title="rag-core", lifespan=lifespan)

embedder = Embedder()
store = Store()
wv = WeaviateStore()  # Dify datasets; disabled when WEAVIATE_URL is empty
reranker = RerankClient()  # disabled when RERANKER_URL is empty
queue: asyncio.Queue[str] = asyncio.Queue()  # job ids
jobs: dict[str, dict] = {}  # job_id -> status dict
worker_task: asyncio.Task | None = None
state: dict[str, dict] = {}


def load_state() -> None:
    global state
    try:
        with open(STATE_FILE) as f:
            state = json.load(f)
    except OSError:
        state = {}


def save_state() -> None:
    os.makedirs(STATE_DIR, exist_ok=True)
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, STATE_FILE)


# ---------------------------------------------------------------------------
# auth (single key for the whole service; Phase 1: opencode on loopback)
# ---------------------------------------------------------------------------

async def require_key(authorization: str = Header(default="")) -> None:
    if RAG_API_KEY and authorization != f"Bearer {RAG_API_KEY}":
        raise HTTPException(status_code=401, detail="invalid or missing Bearer token")


# ---------------------------------------------------------------------------
# corpus walk + patch detection
# ---------------------------------------------------------------------------

def find_markdown(paths: list[str] | None) -> list[str]:
    """Return corpus .md files. paths=None -> full scan; else relative paths
    (relative to any corpus root) or absolute paths inside a root."""
    found: list[str] = []
    if paths is None:
        for root in CORPUS_ROOTS:
            for dirpath, _dirnames, filenames in os.walk(root):
                for fn in filenames:
                    if fn.lower().endswith((".md", ".markdown")):
                        found.append(os.path.join(dirpath, fn))
        return sorted(found)
    for p in paths:
        candidates = [p] if os.path.isabs(p) else [os.path.join(r, p) for r in CORPUS_ROOTS]
        for c in candidates:
            c = os.path.normpath(c)
            if os.path.isfile(c) and c.lower().endswith((".md", ".markdown")):
                found.append(c)
            elif os.path.isdir(c):
                for dirpath, _d, filenames in os.walk(c):
                    for fn in filenames:
                        if fn.lower().endswith((".md", ".markdown")):
                            found.append(os.path.join(dirpath, fn))
    return sorted(set(found))


def root_name(root: str) -> str:
    return os.path.basename(root.rstrip(os.sep)) or root


def rel_path(abs_path: str) -> str:
    """Path as stored in qdrant: <root-name>/<path-within-root>. The root name
    prefix keeps it unique when two roots hold a file with the same name
    (e.g. README.md in both repos) and stable across hosts."""
    for root in CORPUS_ROOTS:
        if abs_path.startswith(root + os.sep):
            return os.path.join(root_name(root), os.path.relpath(abs_path, root))
    return abs_path


def needs_ingest(abs_path: str) -> bool:
    st = state.get(rel_path(abs_path))
    try:
        mt = os.stat(abs_path)
    except OSError:
        return False
    if st is None:
        return True
    if st.get("mtime") != mt.st_mtime:
        return True
    # mtime unchanged but content may differ (touch, cp -p): check hash lazily
    if st.get("hash"):
        try:
            import hashlib
            h = hashlib.sha256()
            with open(abs_path, "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h.update(block)
            if st["hash"] != h.hexdigest():
                return True
        except OSError:
            return False
    return False


async def ingest_file(abs_path: str) -> tuple[int, int]:
    """Patch one file: delete old points by path, re-chunk, embed, upsert.
    Returns (chunks, 0) on success, raises on failure."""
    with open(abs_path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    if not text.strip():
        return 0, 0
    rel = rel_path(abs_path)
    mtime = os.stat(abs_path).st_mtime
    chunks = chunk_markdown(text, rel, target_chars=CHUNK_CHARS)

    await asyncio.to_thread(store.delete_by_path, rel)
    n = len(chunks)
    for i in range(0, n, EMBED_BATCH):
        batch = chunks[i : i + EMBED_BATCH]
        vecs = await asyncio.to_thread(embedder.embed_batch, [c.text for c in batch])
        points = []
        for c, v in zip(batch, vecs):
            pid = chunk_id(rel, c)
            points.append(
                (pid, v, c.text, {
                    "path": rel,
                    "section": c.section,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "hash": c.file_hash,
                    "mtime": mtime,
                    "model": embedder.model,
                    "dim": embedder.dim,
                })
            )
        # upsert in one call per batch (simpler + faster than per-point)
        from qdrant_client.models import PointStruct

        await asyncio.to_thread(
            store.client.upsert,
            collection_name=COLLECTION,
            points=[PointStruct(id=p[0], vector=p[1], payload={**p[3], "text": p[2]}) for p in points],
        )
    state[rel] = {"hash": chunks[0].file_hash if chunks else None, "mtime": mtime}
    return n, 0


async def worker() -> None:
    while True:
        job_id = await queue.get()
        job = jobs[job_id]
        try:
            paths = job["paths"]
            job["total"] = len(paths)
            for i, p in enumerate(paths, 1):
                job["current"] = p
                try:
                    n, _ = await ingest_file(p)
                    job["done"] += 1
                    job["chunks"] += n
                    METRICS["ingest_files_total"].inc()
                    METRICS["ingest_chunks_total"].add(n)
                except Exception as e:  # keep the job alive, record the error
                    job["errors"] += 1
                    job.setdefault("failed", []).append({"path": p, "error": str(e)})
                    METRICS["ingest_errors_total"].inc()
                job["progress"] = job["done"] / job["total"] if job["total"] else 1.0
                save_state()
            job["status"] = "done"
            job["finished_at"] = time.time()
        except Exception as e:
            job["status"] = "failed"
            job["error"] = str(e)
        finally:
            METRICS["queue_size"].add(-1)
            queue.task_done()


# ---------------------------------------------------------------------------
# unified search: qdrant (infra corpus) + weaviate (Dify datasets) -> rerank
# ---------------------------------------------------------------------------

def _collect_candidates(qvec: list[float], k: int, path: str | None, source: str) -> list[dict]:
    """Pull near-miss candidates from each store. Weaviate results carry no
    score (the fork's near-vector endpoint omits it) but are best-first."""
    cands: list[dict] = []
    if source in ("all", "qdrant"):
        for h in store.search(qvec, max(QDRANT_CANDIDATES, k * 4), 0.25, path):
            cands.append({"source": "qdrant", **h})
    if source in ("all", "dify") and wv.enabled:
        for cls in wv.classes():
            for h in wv.search(cls, qvec, max(WEAVIATE_PER_CLASS, k)):
                cands.append({"source": "dify", **h})
    if len(cands) > RERANK_MAX_CANDIDATES:
        # Cap the rerank batch: keep qdrant whole, then dify in class order.
        q = [c for c in cands if c["source"] == "qdrant"]
        d = [c for c in cands if c["source"] == "dify"]
        cands = q + d[: max(0, RERANK_MAX_CANDIDATES - len(q))]
    return cands


def unified_search(query: str, qvec: list[float], k: int, path: str | None, source: str) -> list[dict]:
    """Search both stores, then re-rank the union with the cross-encoder.
    Falls back to per-store top-k (unranked) when the reranker is down."""
    cands = _collect_candidates(qvec, k, path, source)
    if not cands:
        return []
    if reranker.enabled:
        try:
            ranked = reranker.rerank(query, [c["text"] for c in cands], top_n=k)
            return [
                {**cands[r["index"]], "score": round(r["score"], 4), "ranked_by": "bge-reranker-v2-m3"}
                for r in ranked
            ]
        except RerankError:
            pass
    # Fallback: per-store top-k. qdrant by score; weaviate is already
    # best-first. Scores from different stores are not comparable, so we
    # keep them separate instead of merging.
    hits: list[dict] = []
    for src in ("qdrant", "dify"):
        pool = [c for c in cands if c["source"] == src]
        if src == "qdrant":
            pool = sorted(pool, key=lambda c: c.get("score") or 0.0, reverse=True)
        hits.extend(pool[:k])
    return hits


# ---------------------------------------------------------------------------
# REST API
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "queue": queue.qsize(), "jobs": len(jobs)}


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> str:
    return render_metrics()


@app.post("/search", dependencies=[Depends(require_key)])
async def search_rest(request: Request) -> dict:
    """Convenience search without the MCP handshake. Same path as the MCP
    `search` tool: hybrid (qdrant + dify) with reranking."""
    body = await request.json()
    query = body.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="missing 'query'")
    k = int(body.get("k", 5))
    source = body.get("source", "all")
    if source not in ("all", "qdrant", "dify"):
        raise HTTPException(status_code=400, detail=f"unknown source '{source}'")
    path = body.get("path")
    t0 = time.monotonic()
    try:
        qvec = await asyncio.to_thread(embedder.embed, query)
        hits = await asyncio.to_thread(unified_search, query, qvec, k, path, source)
    except (EmbedderError, WeaviateError) as e:
        raise HTTPException(status_code=502, detail=str(e))
    METRICS["search_total"].inc()
    METRICS["search_seconds_total"].add(time.monotonic() - t0)
    return {"query": query, "hits": hits}


@app.post("/ingest", dependencies=[Depends(require_key)])
async def ingest(request: Request) -> dict:
    body = await request.json() if await request.body() else {}
    paths_arg: list[str] | None = body.get("paths")
    paths = find_markdown(paths_arg)
    if paths_arg is not None and not paths:
        raise HTTPException(status_code=404, detail=f"no markdown found for {paths_arg}")
    if paths_arg is None:
        paths = [p for p in paths if needs_ingest(p)]  # patch mode by default
    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "paths": paths,
        "total": len(paths),
        "done": 0,
        "chunks": 0,
        "errors": 0,
        "progress": 0.0,
        "created_at": time.time(),
    }
    METRICS["queue_size"].inc()
    await queue.put(job_id)
    return {"job_id": job_id, "total": len(paths)}


@app.get("/ingest/{job_id}")
async def ingest_status(job_id: str) -> dict:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="unknown job_id")
    return job


# ---------------------------------------------------------------------------
# MCP server (streamable HTTP) for opencode
# ---------------------------------------------------------------------------

from mcp.server.mcpserver import MCPServer  # noqa: E402  # mcp v2: FastMCP -> MCPServer

mcp = MCPServer("rag", instructions="Search the infrastructure documentation corpus.")


@mcp.tool()
async def search(
    query: str, k: int = 5, source: str = "all", path: str | None = None, collection: str = COLLECTION
) -> list[dict]:
    """Hybrid search: infra corpus (qdrant) + all Dify datasets (weaviate),
    re-ranked by bge-reranker-v2-m3. Returns up to k chunks best-first, each
    with score, source ("qdrant"|"dify"), text, and origin metadata:
    path/section/lines for qdrant, dataset_id/document_id/chunk_index for
    dify. source picks one store ("all" default, "qdrant", "dify"); path
    filters qdrant results to one file. Datasets are uploaded via the Dify
    GUI; the document name is not indexed - identify it from the text."""
    if collection != COLLECTION:
        return [{"error": f"unknown collection '{collection}', available: {COLLECTION}"}]
    if source not in ("all", "qdrant", "dify"):
        return [{"error": f"unknown source '{source}', available: all, qdrant, dify"}]
    t0 = time.monotonic()
    try:
        qvec = await asyncio.to_thread(embedder.embed, query)
        hits = await asyncio.to_thread(unified_search, query, qvec, k, path, source)
    except (EmbedderError, WeaviateError) as e:
        return [{"error": str(e)}]
    METRICS["search_total"].inc()
    METRICS["search_seconds_total"].add(time.monotonic() - t0)
    return hits


@mcp.tool()
def list_collections() -> dict:
    """List qdrant collections (point counts, embedder model) and the Dify
    dataset classes available in weaviate."""
    out: dict = {"collections": store.collections()}
    if wv.enabled:
        try:
            out["dify_datasets"] = wv.classes()
        except WeaviateError as e:
            out["dify_datasets"] = f"weaviate error: {e}"
    else:
        out["dify_datasets"] = "disabled (WEAVIATE_URL not set)"
    return out


@mcp.tool()
def get_document(path: str) -> dict:
    """Return the full text of a corpus file (path relative to a corpus root)."""
    for root in CORPUS_ROOTS:
        rn = root_name(root)
        inner = path[len(rn) + 1 :] if path.startswith(rn + os.sep) else path
        full = os.path.normpath(os.path.join(root, inner))
        if os.path.isfile(full) and full.startswith(os.path.normpath(root)):
            with open(full, encoding="utf-8", errors="replace") as f:
                return {"path": path, "text": f.read()}
    return {"error": f"no such document: {path}"}


# The MCP SDK exposes a Starlette app; mount it and enforce the API key in
# front of it (a dependency on the mount would not fire per-request).
mcp_http = mcp.streamable_http_app()


class _MCPAuth:
    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        # Only the MCP endpoint is protected; other paths fall through to the
        # sub-app (which 404s them) instead of leaking a 401.
        if (
            scope["type"] == "http"
            and RAG_API_KEY
            and scope.get("path", "").startswith("/mcp")
        ):
            headers = dict(scope.get("headers") or [])
            auth = headers.get(b"authorization", b"").decode()
            if auth != f"Bearer {RAG_API_KEY}":
                await send(
                    {
                        "type": "http.response.start",
                        "status": 401,
                        "headers": [(b"content-type", b"application/json")],
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": b'{"detail":"invalid or missing Bearer token"}',
                    }
                )
                return
        await self.app(scope, receive, send)


# Mount at "/": the SDK's default streamable_http_path is "/mcp", so the
# endpoint lands exactly at /mcp. REST routes registered above take
# precedence over this catch-all mount.
app.mount("/", _MCPAuth(mcp_http))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8090)
