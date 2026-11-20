"""Client for the rag-reranker service (BAAI/bge-reranker-v2-m3).

The service speaks the Jina-style contract Dify uses:
POST /rerank {"model","query","documents","top_n","return_documents"} ->
{"results": [{"index","relevance_score","document"}]}.
RERANKER_URL empty -> disabled, rerank() raises RerankError.
"""

from __future__ import annotations

import os

import httpx

RERANKER_URL = os.environ.get("RERANKER_URL", "")


def _read_key(path: str | None) -> str | None:
    if not path:
        return None
    try:
        with open(path) as f:
            key = f.read().strip()
        return key or None
    except OSError:
        return None


RERANK_API_KEY = _read_key(
    os.environ.get("RERANK_API_KEY_FILE", "/run/secrets/rerank_api_key")
)


class RerankError(RuntimeError):
    pass


class RerankClient:
    def __init__(self, url: str = RERANKER_URL, api_key: str | None = RERANK_API_KEY):
        self.url = url.rstrip("/")
        self.enabled = bool(self.url)
        self._client = httpx.Client(
            base_url=self.url,
            timeout=300.0,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        )

    def close(self) -> None:
        self._client.close()

    def health(self) -> bool:
        """True when the service answers /health with a loaded model."""
        if not self.enabled:
            return False
        try:
            r = self._client.get("/health", timeout=10.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[dict]:
        """Score (query, doc) pairs, best-first. Returns
        [{"index": int, "score": float}, ...] with at most top_n entries."""
        if not self.enabled:
            raise RerankError("reranker not configured (RERANKER_URL empty)")
        try:
            r = self._client.post(
                "/rerank",
                json={
                    "model": "bge-reranker-v2-m3",
                    "query": query,
                    "documents": documents,
                    "top_n": top_n,
                    "return_documents": False,
                },
            )
        except httpx.HTTPError as e:
            raise RerankError(f"reranker unreachable: {e}") from e
        if r.status_code != 200:
            raise RerankError(f"reranker HTTP {r.status_code}: {r.text[:200]}")
        results = r.json().get("results", [])
        return [
            {"index": int(x["index"]), "score": float(x["relevance_score"])}
            for x in results
        ]
