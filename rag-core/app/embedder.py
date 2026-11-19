"""Client for the CPU embedder (llama.cpp --embeddings on port 8081)."""

from __future__ import annotations

import os
import time

import httpx

EMBEDDER_URL = os.environ.get("EMBEDDER_URL", "http://embedder:8081")
EMBEDDER_MODEL = os.environ.get("EMBEDDER_MODEL", "bge-m3")


class EmbedderError(RuntimeError):
    pass


class Embedder:
    def __init__(self, url: str = EMBEDDER_URL, model: str = EMBEDDER_MODEL):
        self._client = httpx.Client(base_url=url, timeout=300.0)
        self.model = model
        self.dim: int | None = None  # learned from the first response

    def close(self) -> None:
        self._client.close()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """POST /v1/embeddings with a batch of texts. Returns vectors in order."""
        t0 = time.monotonic()
        try:
            r = self._client.post(
                "/v1/embeddings",
                json={"model": self.model, "input": texts},
            )
        except httpx.HTTPError as e:
            raise EmbedderError(f"embedder unreachable: {e}") from e
        if r.status_code != 200:
            raise EmbedderError(f"embedder HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()["data"]
        data.sort(key=lambda d: d["index"])  # keep request order
        vecs = [d["embedding"] for d in data]
        if self.dim is None:
            self.dim = len(vecs[0])
        METRICS["embed_seconds_total"].add(time.monotonic() - t0)
        METRICS["embed_batches_total"].inc()
        return vecs

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]


# --- tiny metrics registry (prometheus text format on /metrics) -------------

class _Counter:
    def __init__(self) -> None:
        self.value = 0.0

    def inc(self, n: float = 1) -> None:
        self.value += n

    def add(self, n: float) -> None:
        self.value += n


METRICS = {
    "embed_seconds_total": _Counter(),
    "embed_batches_total": _Counter(),
    "search_seconds_total": _Counter(),
    "search_total": _Counter(),
    "ingest_files_total": _Counter(),
    "ingest_chunks_total": _Counter(),
    "ingest_errors_total": _Counter(),
    "queue_size": _Counter(),
}


def render_metrics() -> str:
    lines = []
    for name, c in METRICS.items():
        kind = "counter" if "total" in name else "gauge"
        lines.append(f"# HELP {name} rag-core {name}")
        lines.append(f"# TYPE {name} {kind}")
        lines.append(f"{name} {c.value:.6f}")
    return "\n".join(lines) + "\n"
