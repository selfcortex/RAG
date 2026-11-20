COMPOSE ?= docker compose
# Host directory with the bge-reranker-v2-m3 weights (downloaded by init,
# mounted read-only into rag-reranker). Keep in sync with .env RERANK_MODEL_DIR.
RERANK_MODEL_DIR ?= /mnt/AI/models/bge-reranker-v2-m3

.PHONY: help init up down restart logs ps check ingest rescan search rerank

init: ## One-time setup: .env from example + secrets + ai-net
	@test -f .env || cp .env.example .env
	@mkdir -p secrets
	@test -f secrets/rag_api_key || head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n' > secrets/rag_api_key
	@test -f secrets/qdrant_api_key || head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n' > secrets/qdrant_api_key
	@test -f secrets/rerank_api_key || head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n' > secrets/rerank_api_key
	@chmod 644 secrets/rag_api_key secrets/qdrant_api_key secrets/rerank_api_key
	@docker network inspect ai-net >/dev/null 2>&1 || docker network create ai-net
	@$(MAKE) model
	@echo ".env, secrets/, ai-net and the reranker model are ready (all gitignored)"

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*## "}{printf "  \033[36m%-8s\033[0m %s\n", $$1, $$2}'

up: ## Build (if needed) and start the RAG stack
	$(COMPOSE) up -d --wait

down: ## Stop and remove the RAG stack (volumes kept)
	$(COMPOSE) down

restart: ## Restart the RAG stack
	$(COMPOSE) restart

logs: ## Tail RAG logs
	$(COMPOSE) logs -f --tail=100

ps: ## Show RAG status
	$(COMPOSE) ps

check: ## Validate compose config
	$(COMPOSE) config -q

model: ## Download bge-reranker-v2-m3 (only missing files) into $(RERANK_MODEL_DIR)
	@mkdir -p $(RERANK_MODEL_DIR)
	@for f in config.json model.safetensors sentencepiece.bpe.model \
		special_tokens_map.json tokenizer.json tokenizer_config.json; do \
		if [ ! -f "$(RERANK_MODEL_DIR)/$$f" ]; then \
			echo "downloading bge-reranker-v2-m3/$$f ..."; \
			curl -fL --retry 3 -o "$(RERANK_MODEL_DIR)/$$f" \
				"https://huggingface.co/BAAI/bge-reranker-v2-m3/resolve/main/$$f" || exit 1; \
		else \
			echo "bge-reranker-v2-m3/$$f already present"; \
		fi; \
	done
	@echo "reranker model ready in $(RERANK_MODEL_DIR)"

ingest: ## Full rescan of the corpus (POST /ingest)
	@KEY=$$(cat secrets/rag_api_key); curl -fsS -X POST http://127.0.0.1:8090/ingest \
		-H "Authorization: Bearer $$KEY" -H "Content-Type: application/json" -d '{}'

rescan: ## Alias for ingest
	$(MAKE) ingest

search: ## Quick CLI search: make search Q="..."
	@test -n "$(Q)" || { echo 'usage: make search Q="your question"'; exit 1; }
	@KEY=$$(cat secrets/rag_api_key); curl -fsS http://127.0.0.1:8090/search \
		-H "Authorization: Bearer $$KEY" \
		-H "Content-Type: application/json" \
		-d '{"query":"$(Q)","k":3}'

rerank: ## Quick CLI rerank test: make rerank Q="..." D="text A" D2="text B"
	@test -n "$(Q)" || { echo 'usage: make rerank Q="query" D="doc A" [D2="doc B" ...]'; exit 1; }
	@KEY=$$(cat secrets/rerank_api_key); \
	DOCS=""; \
	for v in $(D) $(D2) $(D3) $(D4); do [ -n "$$v" ] && DOCS="$$DOCS,\"$$v\""; done; \
	curl -fsS http://127.0.0.1:8091/rerank \
		-H "Authorization: Bearer $$KEY" \
		-H "Content-Type: application/json" \
		-d "{\"model\":\"bge-reranker-v2-m3\",\"query\":\"$(Q)\",\"documents\":[${DOCS#,}]}"
