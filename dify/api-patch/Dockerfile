# Patched Dify API image: stringifies uuid-typed Weaviate properties on read.
#
# Why: Dify 1.20.0 + the Weaviate fork's auto-schema created uuid-typed
# collection properties (doc_id, document_id). The weaviate client parses
# them back into uuid.UUID objects, but Dify's retrieval code expects
# strings (documents.id comes back as str; document_segments.index_node_id
# is a varchar column). Result: every retrieved document was silently
# dropped in RetrievalService.format_retrieval_documents -> search
# (GUI hit-testing and /retrieve) returned [].
#
# The patch is applied automatically on every container start by the
# entrypoint wrapper and is idempotent. It therefore survives
# `docker compose down && up` without any manual step.
#
# See ../README.md for the full story and canary check.
FROM langgenius/dify-api:1.20.0

USER root
COPY patch-weaviate-uuid.py /app/patches/patch-weaviate-uuid.py
COPY entrypoint-patched.sh /entrypoint-patched.sh
RUN chmod +x /entrypoint-patched.sh
# The base image runs as the dify user; the entrypoint patch step relies on
# dify being able to write the site-packages file (verified).
USER dify

ENTRYPOINT ["/bin/bash", "/entrypoint-patched.sh"]
