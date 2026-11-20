#!/usr/bin/env python3
"""Idempotently patch dify_vdb_weaviate to stringify uuid-typed properties.

The Weaviate fork used by Dify auto-creates collection properties with
uuid data types when the values look like UUIDs. The weaviate client then
returns uuid.UUID objects instead of strings, which breaks Dify's
retrieval pipeline (dict membership checks and varchar IN-queries).
This script stringifies the affected values in the two search methods
of dify_vdb_weaviate/weaviate_vector.py.

Safe to run repeatedly: it detects an already-patched file and exits 0.
"""
import ast
import sys

TARGETS = [
    "/app/api/.venv/lib/python3.13/site-packages/dify_vdb_weaviate/weaviate_vector.py",
]

MARKER = (
    "properties = {k: (str(v) if isinstance(v, _uuid.UUID) else v)"
    " for k, v in properties.items()}"
)
OLD = "            properties = dict(obj.properties or {})"
NEW = (
    "            properties = dict(obj.properties or {})\n"
    "            # Dify treats all metadata ids as strings; the weaviate client\n"
    "            # parses uuid-typed properties into uuid.UUID objects, which breaks\n"
    "            # dict membership checks and varchar IN-queries downstream.\n"
    "            " + MARKER
)


def main() -> int:
    for path in TARGETS:
        try:
            src = open(path).read()
        except OSError as e:
            print(f"[dify-api-patch] cannot read {path}: {e}")
            return 1
        if src.count(MARKER) == 2:
            print(f"[dify-api-patch] {path}: already patched")
            continue
        if MARKER in src:
            print(f"[dify-api-patch] ERROR: partial patch in {path}, fix manually")
            return 1
        n = src.count(OLD)
        if n != 2:
            print(f"[dify-api-patch] ERROR: expected 2 occurrences of the target line, found {n} in {path}")
            return 1
        src = src.replace(OLD, NEW)
        ast.parse(src)  # syntax check before writing
        open(path, "w").write(src)
        print(f"[dify-api-patch] {path}: patched (2 sites)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
