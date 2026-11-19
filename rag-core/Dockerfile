# rag-core: FastAPI + MCP server + qdrant client.
# Base image pinned by digest (python:3.12-slim, verified 2026-11-19).
FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

# Pinned versions (verified 2026-11-19): keep the stack reproducible.
RUN pip install --no-cache-dir \
      fastapi==0.165.0 \
      uvicorn==0.54.0 \
      mcp==2.3.0 \
      qdrant-client==1.20.0 \
      httpx==0.28.1

WORKDIR /app
COPY app/ /app/

RUN useradd -m rag \
 && mkdir -p /state \
 && chown -R rag:rag /app /state
USER rag

EXPOSE 8090
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8090"]
