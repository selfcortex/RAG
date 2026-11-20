# Dify 1.20.0 — Weaviate UUID patch

Dify deployment lives at `/mnt/GameLinux/dify` (not versioned). This directory
holds the one fix that must survive container recreation, plus documentation.

## The bug

Dify's Weaviate fork auto-creates collection properties on first insert. Because
the values look like UUIDs, `doc_id` / `document_id` were created with **uuid**
data types. The weaviate client parses uuid-typed properties into `uuid.UUID`
objects, but Dify's retrieval code expects **strings** (`documents.id` comes
back as `str`; `document_segments.index_node_id` is a `varchar` column).

Consequences in `RetrievalService.format_retrieval_documents`:

1. `document_id not in dataset_documents` was always true (UUID != str) — every
   retrieved document was **silently dropped**, no error logged, search
   (GUI hit-testing and `POST /v1/datasets/<id>/retrieve`) returned `records: []`.
2. Latent landmine: even past the dict check, `index_node_id IN (uuid, ...)`
   against a varchar column would raise
   `operator does not exist: character varying = uuid`.

The bug was latent since the very first dataset upload (the schema was
auto-created then); it is not related to the READONLY/disk incident.

## The fix

`api-patch/` builds `dify-api-patched:1.20.0` from the official
`langgenius/dify-api:1.20.0`:

- `patch-weaviate-uuid.py` — stringifies `uuid.UUID` property values in the two
  search methods of `dify_vdb_weaviate/weaviate_vector.py`. Idempotent.
- `entrypoint-patched.sh` — runs the patch on **every container start**, then
  execs the original entrypoint. A patch failure never blocks startup.

All services that use the `dify-api` image (`api`, `api_websocket`, `worker`,
`worker_knowledge_fs_upgrade`, `worker_beat`) point at `dify-api-patched:1.20.0`;
only `api` carries the `build:` definition.

## Operations

Rebuild + redeploy (e.g. after upgrading Dify version in the compose file):

```bash
cd /mnt/GameLinux/dify
docker compose build api
docker compose up -d
```

No manual patching needed — the entrypoint applies it. The old host-side script
`apply-weaviate-uuid-patch.sh` remains as an emergency fallback (e.g. if the
container was started from the unpatched image for some reason).

## Canary check (run after any Dify redeploy)

```bash
docker exec dify-api-1 sh -c "curl -s -o /tmp/canary.json -w '%{http_code}\n' \
  -X POST 'http://127.0.0.1:5001/v1/datasets/<dataset-uuid>/retrieve' \
  -H 'Authorization: Bearer <dataset-api-token>' \
  -H 'Content-Type: application/json' \
  -d '{\"query\":\"<known phrase from test_file>\"}'"
docker exec dify-api-1 sh -c "python3 -c \"import json; print('records:', len(json.load(open('/tmp/canary.json')).get('records', [])))\""
```

Expected: HTTP 200 and `records:` > 0. If it is `0`, the patch is missing —
check `docker logs dify-api-1 | grep dify-api-patch`.
