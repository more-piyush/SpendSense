# ============================================================================
# Firefly III MLOps — data role Makefile (unified-image edition)
# ----------------------------------------------------------------------------
# All data-role scripts live inside a single image: firefly-data:latest
# Built once from pipelines/Dockerfile, re-used by every container.
#
# Common workflow:
#   make build            build the unified image
#   make up               bring up MinIO + features + feedback
#   make ingest           BLS ingestion
#   make all              full pipeline build (ingest → generate → pipeline → quality → split)
#   make emulate          production-traffic emulator (Ctrl-C to stop)
#   make retrain-dataset  build retraining dataset from feedback + CE Survey
#   make drift            PSI drift monitor
#   make down / reset     stop / wipe
# ============================================================================

COMPOSE ?= docker compose
SHELL   := /bin/bash

.DEFAULT_GOAL := help

help:
	@awk 'BEGIN{FS=":.*##"; printf "\nTargets:\n"} \
	/^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ---- image & infra --------------------------------------------------------
build: ## Build the unified firefly-data image
	$(COMPOSE) build

up: ## Bring up MinIO + features + feedback services
	$(COMPOSE) up -d minio minio-init features feedback
	@echo ""
	@echo "MinIO console: http://localhost:9001  (firefly-access-key / firefly-secret-key)"
	@echo "Features API:  http://localhost:8080/health"
	@echo "Feedback API:  http://localhost:8100/health"

down: ## Stop containers (keeps MinIO data volume)
	$(COMPOSE) down

reset: ## Destroy everything including MinIO volume (fresh start)
	$(COMPOSE) down -v
	@echo "MinIO volume destroyed. Image kept — run 'make build' if you need to rebuild."

nuke: ## Destroy everything AND remove the image
	$(COMPOSE) down -v --rmi local
	@echo "Image + volume + containers all gone."

ps: ## Show running containers
	$(COMPOSE) ps

logs: ## Tail logs for always-on services
	$(COMPOSE) logs -f features feedback

# ---- one-shot data jobs ---------------------------------------------------
# `ingest` is the generic runner service. We override its command: per target.
ingest: ## Ingest BLS CE Survey into raw/processed buckets
	$(COMPOSE) --profile jobs run --rm ingest python3 ingest_bls_data.py

generate: ## CE Survey → synthetic merchant-level transactions
	$(COMPOSE) --profile jobs run --rm ingest python3 generate_synthetic_data.py

pipeline: ## Run full data_pipeline.py (external training data)
	$(COMPOSE) --profile jobs run --rm ingest python3 data_pipeline.py config.yaml

quality: ## Run Soda data quality checks on training data
	$(COMPOSE) --profile jobs run --rm ingest python3 soda_quality_checks.py

split: ## Chronological train/val/test split
	$(COMPOSE) --profile jobs run --rm ingest python3 batch_training_pipeline.py

# ---- closed-loop components -----------------------------------------------
emulate: ## Run production traffic emulator (hits serving + feedback)
	$(COMPOSE) --profile demo up emulator

retrain-dataset: ## Build retraining dataset from feedback + CE Survey
	$(COMPOSE) --profile jobs run --rm retrain-job

drift: ## Run PSI drift monitor
	$(COMPOSE) --profile jobs run --rm drift-job

# ---- aggregate ------------------------------------------------------------
all: build up ingest generate pipeline quality split ## Full from-scratch build
	@echo ""
	@echo "=========================================================="
	@echo "  DATA PIPELINE COMPLETE"
	@echo "  MinIO console: http://localhost:9001"
	@echo "=========================================================="

.PHONY: help build up down reset nuke ps logs ingest generate pipeline \
        quality split emulate retrain-dataset drift all
