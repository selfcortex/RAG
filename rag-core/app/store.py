"""Qdrant client wrapper: collection management, patch upsert, delete-by-path."""

from __future__ import annotations

import os
import time

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

def _read_key(path: str | None) -> str | None:
    if not path:
        return None
    try:
        with open(path) as f:
            key = f.read().strip()
        return key or None
    except OSError:
        return None


QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY = _read_key(os.environ.get("QDRANT_API_KEY_FILE", "/run/secrets/qdrant_api_key"))
COLLECTION = os.environ.get("COLLECTION", "docs")
MODEL_NAME = os.environ.get("EMBEDDER_MODEL", "bge-m3")


class Store:
    def __init__(self, url: str = QDRANT_URL, api_key: str | None = QDRANT_API_KEY):
        self.client = QdrantClient(url=url, api_key=api_key, timeout=60)

    def ensure_collection(self, dim: int) -> None:
        if not self.client.collection_exists(COLLECTION):
            self.client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                metadata={"embedder_model": MODEL_NAME, "created": time.time()},
            )
            return
        existing = self.client.get_collection(COLLECTION)
        # qdrant-client 1.20: metadata and vector config live under .config
        # (CollectionInfo no longer exposes .vectors / .metadata directly).
        params = existing.config.params
        # Refuse to mix models: a different embedder model means
        # vectors in incompatible spaces.
        meta_model = (existing.config.metadata or {}).get("embedder_model")
        if meta_model and meta_model != MODEL_NAME:
            raise RuntimeError(
                f"collection '{COLLECTION}' was indexed with embedder model "
                f"'{meta_model}', current model is '{MODEL_NAME}'. "
                f"Drop the collection or change EMBEDDER_MODEL."
            )
        vec_cfg = params.vectors
        size = vec_cfg.size if hasattr(vec_cfg, "size") else list(vec_cfg.values())[0].size
        if size != dim:
            raise RuntimeError(
                f"collection '{COLLECTION}' has dim {size}, "
                f"embedder produced {dim}. Delete the collection to reindex."
            )

    def upsert_chunk(self, point_id: str, vector: list[float], text: str, payload: dict) -> None:
        self.client.upsert(
            collection_name=COLLECTION,
            points=[PointStruct(id=point_id, vector=vector, payload={**payload, "text": text})],
        )

    def delete_by_path(self, path: str) -> int:
        """Delete all points of one file. Returns how many were removed (best effort)."""
        res = self.client.delete(
            collection_name=COLLECTION,
            points_selector=Filter(
                must=[FieldCondition(key="path", match=MatchValue(value=path))]
            ),
        )
        return int(getattr(res, "deleted", 0) or 0)

    def search(self, vector: list[float], k: int = 5, min_score: float = 0.25, path: str | None = None):
        must = [FieldCondition(key="path", match=MatchValue(value=path))] if path else None
        hits = self.client.query_points(
            collection_name=COLLECTION,
            query=vector,
            limit=k,
            score_threshold=min_score,
            query_filter=Filter(must=must) if must else None,
            with_payload=True,
        )
        return [
            {
                "score": h.score,
                "path": h.payload.get("path"),
                "section": h.payload.get("section"),
                "text": h.payload.get("text"),
                "start_line": h.payload.get("start_line"),
                "end_line": h.payload.get("end_line"),
            }
            for h in hits.points
        ]

    def collections(self) -> list[dict]:
        out = []
        for c in self.client.get_collections().collections:
            info = self.client.get_collection(c.name)
            vec_cfg = info.config.params.vectors
            size = vec_cfg.size if hasattr(vec_cfg, "size") else list(vec_cfg.values())[0].size
            out.append(
                {
                    "name": c.name,
                    "points": info.points_count,
                    "dim": size,
                    "embedder_model": (info.config.metadata or {}).get("embedder_model"),
                }
            )
        return out
