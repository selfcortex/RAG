#!/bin/bash
# Applies the weaviate uuid->str patch (idempotent), then starts the API.
# Never blocks startup: if the patch fails, the original entrypoint still runs.
/app/api/.venv/bin/python /app/patches/patch-weaviate-uuid.py \
    || echo "[dify-api-patch] WARNING: weaviate uuid patch failed, continuing without it"
exec /bin/bash /entrypoint.sh "$@"
