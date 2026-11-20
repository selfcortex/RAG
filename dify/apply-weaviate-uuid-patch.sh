#!/usr/bin/env bash
# EMERGENCY FALLBACK ONLY. With the dify-api-patched image the patch is applied
# automatically on every container start (see api-patch/ and README.md).
# Use this script only if a container runs from the unpatched image.
#
# Re-applies the weaviate UUID->str metadata patch to dify-api-1.
#
# Why: Dify 1.20.0 + this Weaviate fork's auto-schema created uuid-typed
# collection properties (doc_id, document_id, ...). The weaviate client
# parses them back into uuid.UUID objects, but Dify's retrieval code expects
# strings (documents.id comes back as str; document_segments.index_node_id is
# a varchar column). Result: every retrieved document was silently dropped in
# RetrievalService.format_retrieval_documents -> GUI/API search returned [].
#
# The patch stringifies uuid.UUID values in the two search methods of
# dify_vdb_weaviate/weaviate_vector.py (site-packages inside the container).
#
# It survives `docker restart dify-api-1` but is LOST when the container is
# recreated (`docker compose down && up`). Re-run this script after any
# container recreation. Idempotent: safe to run repeatedly.

set -euo pipefail

CONTAINER="dify-api-1"
TARGET="/app/api/.venv/lib/python3.13/site-packages/dify_vdb_weaviate/weaviate_vector.py"

docker exec "$CONTAINER" python3 - "$TARGET" <<'PYEOF'
import sys

path = sys.argv[1]
src = open(path).read()

marker = "properties = {k: (str(v) if isinstance(v, _uuid.UUID) else v) for k, v in properties.items()}"
if src.count(marker) == 2:
    print("patch already applied, nothing to do")
    sys.exit(0)
if marker in src:
    print("ERROR: partial patch found, fix manually")
    sys.exit(1)

old = "            properties = dict(obj.properties or {})"
new = (
    "            properties = dict(obj.properties or {})\n"
    "            # Dify treats all metadata ids as strings; the weaviate client\n"
    "            # parses uuid-typed properties into uuid.UUID objects, which breaks\n"
    "            # dict membership checks and varchar IN-queries downstream.\n"
    "            properties = {k: (str(v) if isinstance(v, _uuid.UUID) else v) for k, v in properties.items()}"
)
n = src.count(old)
if n != 2:
    print("ERROR: expected 2 occurrences of the target line, found %d" % n)
    sys.exit(1)

src = src.replace(old, new)
open(path, "w").write(src)

import ast
ast.parse(src)
print("patch applied (2 sites), syntax OK")
PYEOF

echo "restarting $CONTAINER ..."
docker restart "$CONTAINER"
echo "done. verify: docker exec $CONTAINER sh -c 'grep -c \"_uuid.UUID) else v\" $TARGET'  -> expect 2"
