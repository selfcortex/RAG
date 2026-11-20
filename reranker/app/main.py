"""rag-reranker: BAAI/bge-reranker-v2-m3 cross-encoder over HTTP.

Serves the exact request/response format expected by Dify's
OpenAI-API-compatible provider plugin (Jina/llama.cpp compatible):

  POST /rerank
    request : {"model": str, "query": str, "documents": [str],
               "top_n": int|None, "return_documents": bool}
    response: {"results": [{"index": int, "relevance_score": float,
                            "document": {"text": str}}]}
  GET /health - liveness (also used by the container healthcheck)

Dify renormalizes the raw scores to 0..1 within each result set
((s - min) / (max - min)), so the cross-encoder's raw scores are
returned as-is.

The model is loaded from MODEL_DIR (a local directory mounted read-only;
`make init` downloads it on the host). No internet access is needed at
runtime - the container sits on the internal rag-net.
"""

from __future__ import annotations

import asyncio
import os
import threading

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

MODEL_DIR = os.environ.get("MODEL_DIR", "/model")


def _read_key(path: str | None) -> str:
    if path and os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    return os.environ.get("RERANK_API_KEY", "")


RERANK_API_KEY = _read_key(
    os.environ.get("RERANK_API_KEY_FILE", "/run/secrets/rerank_api_key")
)

model = None
# One inference at a time: torch already parallelizes inside predict()
# via its intra-op threads; overlapping requests would only contend.
model_lock = threading.Lock()


async def lifespan(_: FastAPI):
    """Load the cross-encoder at startup (blocking -> off the event loop)."""
    global model
    from sentence_transformers import CrossEncoder

    model = await asyncio.to_thread(CrossEncoder, MODEL_DIR, device="cpu")
    yield


app = FastAPI(title="rag-reranker", lifespan=lifespan)


# ---------------------------------------------------------------------------
# auth (single key, same pattern as rag-core)
# ---------------------------------------------------------------------------

async def require_key(authorization: str = Header(default="")) -> None:
    if RERANK_API_KEY and authorization != f"Bearer {RERANK_API_KEY}":
        raise HTTPException(status_code=401, detail="invalid or missing Bearer token")


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------

def _rerank(query: str, documents: list[str], top_n: int | None) -> list[dict]:
    """Score (query, doc) pairs and return them best-first."""
    # Long docs are truncated to the model's max sequence length; our
    # Dify chunks end at ~1024 tokens so this is a safety net, not the norm.
    scores = model.predict(
        [(query, d) for d in documents],
        truncation=True,
        batch_size=len(documents),
    )
    order = sorted(range(len(documents)), key=lambda i: scores[i], reverse=True)
    if top_n and top_n > 0:
        order = order[:top_n]
    return [
        {"index": i, "relevance_score": float(scores[i]), "document": {"text": documents[i]}}
        for i in order
    ]


# ---------------------------------------------------------------------------
# REST API
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/rerank", dependencies=[Depends(require_key)])
async def rerank(request: Request) -> JSONResponse:
    body = await request.json()
    query = body.get("query")
    documents = body.get("documents")
    if not query:
        raise HTTPException(status_code=400, detail="missing 'query'")
    if not isinstance(documents, list) or not documents:
        raise HTTPException(status_code=400, detail="missing 'documents'")
    documents = [str(d) for d in documents]
    top_n = body.get("top_n")
    if top_n is not None:
        try:
            top_n = int(top_n)
        except (TypeError, ValueError):
            top_n = None
    results = await asyncio.to_thread(_rerank, query, documents, top_n)
    return JSONResponse({"results": results})
