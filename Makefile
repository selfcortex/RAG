COMPOSE ?= docker compose

.PHONY: help init up down restart logs ps check ingest rescan search

init: ## One-time setup: .env from example + secrets + ai-net
	@test -f .env || cp .env.example .env
	@mkdir -p secrets
	@test -f secrets/rag_api_key || head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n' > secrets/rag_api_key
	@test -f secrets/qdrant_api_key || head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n' > secrets/qdrant_api_key
	@chmod 644 secrets/rag_api_key secrets/qdrant_api_key
	@docker network inspect ai-net >/dev/null 2>&1 || docker network create ai-net
	@echo ".env, secrets/ and ai-net are ready (all gitignored)"

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
