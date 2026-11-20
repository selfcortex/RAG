# rag-reranker: BAAI/bge-reranker-v2-m3 cross-encoder (CPU) served in the
# /rerank format expected by Dify's OpenAI-API-compatible provider plugin.
# Base image pinned by digest (python:3.12-slim, verified 2026-11-19).
FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

# Pinned versions (verified 2026-11-20): keep the stack reproducible.
# CPU-only torch from the PyTorch CPU index (no CUDA wheels).
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
      torch==2.15.0+cpu \
 && pip install --no-cache-dir \
      fastapi==0.165.0 \
      uvicorn==0.54.0 \
      sentence-transformers==6.3.0

WORKDIR /app
COPY app/ /app/

RUN useradd -m rag \
 && mkdir -p /model \
 && chown -R rag:rag /app /model
USER rag

EXPOSE 8091
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8091"]
