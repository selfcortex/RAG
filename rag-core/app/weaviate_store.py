"""Client for the Dify Weaviate fork: vector search over Dify datasets.

Dify stores every dataset as its own class named
Vector_index_<dataset-uuid>_Node. The fork's old-style REST endpoint
POST /v1/search/{class}/near-vector takes {"vector": [...], "limit": N}
and returns {"results": [{"id": ..., "properties": {...}}]} (no distance
field; results are best-first). The GraphQL endpoint of this fork does
not accept standard GraphQL bodies, so the REST path is the only one.

The document NAME is not in Weaviate (it lives in Dify's Postgres);
results carry dataset_id / document_id / chunk_index instead.
WEAVIATE_URL empty -> the store is disabled and search() returns [].
"""

from __future__ import annotations

import os

import httpx

WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "")
WEAVIATE_CLASS_PREFIX = "Vector_index_"


def _read_key(path: str | None) -> str | None:
    if not path:
        return None
    try:
        with open(path) as f:
            key = f.read().strip()
        return key or None
    except OSError:
        return None


WEAVIATE_API_KEY = _read_key(
    os.environ.get("WEAVIATE_API_KEY_FILE", "/run/secrets/weaviate_api_key")
)


class WeaviateError(RuntimeError):
    pass


class WeaviateStore:
    def __init__(self, url: str = WEAVIATE_URL, api_key: str | None = WEAVIATE_API_KEY):
        self.url = url.rstrip("/")
        self.enabled = bool(self.url)
        self._client = httpx.Client(
            base_url=self.url,
            timeout=120.0,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        )

    def close(self) -> None:
        self._client.close()

    def _get(self, path: str) -> dict:
        try:
            r = self._client.get(path)
        except httpx.HTTPError as e:
            raise WeaviateError(f"weaviate unreachable: {e}") from e
        if r.status_code != 200:
            raise WeaviateError(f"weaviate HTTP {r.status_code}: {r.text[:200]}")
        return r.json()

    def _post(self, path: str, body: dict) -> dict:
        try:
            r = self._client.post(path, json=body)
        except httpx.HTTPError as e:
            raise WeaviateError(f"weaviate unreachable: {e}") from e
        if r.status_code != 200:
            raise WeaviateError(f"weaviate HTTP {r.status_code}: {r.text[:200]}")
        return r.json()

    def classes(self) -> list[str]:
        """All Dify dataset classes, best-effort order (schema order)."""
        d = self._get("/v1/schema")
        return [
            c["class"]
            for c in d.get("classes", [])
            if str(c.get("class", "")).startswith(WEAVIATE_CLASS_PREFIX)
        ]

    def search(self, cls: str, vector: list[float], limit: int = 10) -> list[dict]:
        """Near-vector search in one dataset class, best-first (no score)."""
        if not self.enabled:
            return []
        d = self._post(f"/v1/search/{cls}/near-vector", {"vector": vector, "limit": limit})
        out = []
        for r in d.get("results", []):
            p = r.get("properties") or {}
            out.append(
                {
                    "text": p.get("text"),
                    "dataset_id": p.get("dataset_id"),
                    "document_id": p.get("document_id"),
                    "doc_id": p.get("doc_id"),
                    "chunk_index": p.get("chunk_index"),
                    "class": cls,
                }
            )
        return out
