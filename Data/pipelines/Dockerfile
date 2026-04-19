# ============================================================================
# Firefly III MLOps — Data Role: unified Dockerfile
# ----------------------------------------------------------------------------
# One image. Many containers. Every data-role script runs from this single
# image; docker-compose varies the `command:` per service.
#
# Replaces:
#   Dockerfile.ingest       Dockerfile.features     Dockerfile.feedback
#   Dockerfile.emulator     Dockerfile.retraining   Dockerfile.drift
#   Dockerfile.pipeline     Dockerfile.batch        Dockerfile.generator
#
# Build once:
#   docker build -t firefly-data:latest -f pipelines/Dockerfile pipelines/
#
# Run as any role (the `command` overrides the default CMD):
#   docker run firefly-data                                   # prints help
#   docker run firefly-data python3 ingest_bls_data.py        # ingestion
#   docker run -p 8080:8080 firefly-data python3 online_feature_service.py
#   docker run -p 8100:8100 firefly-data python3 feedback_service.py
#   docker run firefly-data python3 data_generator.py         # emulator
#   docker run firefly-data python3 build_retraining_dataset.py
#   docker run firefly-data python3 drift_monitor.py
#
# ----------------------------------------------------------------------------
# Design choices:
#   * python:3.11-slim           — pinned. Same interpreter for every role.
#   * --no-cache-dir             — smaller image, faster CI.
#   * Single pip install layer   — one cache-friendly layer, superset of every
#                                  role's requirements. Marginal cost for
#                                  services that don't need pandas is a few MB.
#   * curl added                 — docker-compose healthchecks use it.
#   * Non-root user              — containers run as `firefly` (UID 1000), so
#                                  a compromise of one container doesn't give
#                                  root on the shared network.
# ============================================================================
FROM python:3.11-slim

# ---- OS-level deps ---------------------------------------------------------
# curl   — healthcheck endpoints (GET /health)
# gcc    — some wheel builds (scipy/numpy fallback on odd arches)
# tini   — tiny init process so Ctrl-C and docker stop work cleanly (otherwise
#          Python processes leak when compose sends SIGTERM)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        tini \
    && rm -rf /var/lib/apt/lists/*

# ---- Python deps (superset of every data-role script) ---------------------
# feedback_service.py, online_feature_service.py → stdlib + minio + pyyaml
# data_generator.py                              → + requests
# build_retraining_dataset.py, drift_monitor.py  → + pandas + numpy + pyarrow
# ingest_bls_data.py, data_pipeline.py,          → + boto3
# generate_synthetic_data.py, batch_training     → (same)
# soda_quality_checks.py                         → (same)
#
# Pin versions so the image is reproducible across builds. Bump deliberately.
RUN pip install --no-cache-dir \
        pandas==2.2.3 \
        numpy==1.26.4 \
        pyarrow==15.0.2 \
        minio==7.2.5 \
        boto3==1.34.101 \
        requests==2.32.3 \
        pyyaml==6.0.1

# ---- Non-root user ---------------------------------------------------------
RUN useradd --create-home --uid 1000 firefly
WORKDIR /app

# ---- Application code ------------------------------------------------------
# Copy every pipeline script + the unified config. Images stay small because
# the scripts themselves are tiny (< 150 KB total).
COPY --chown=firefly:firefly *.py        /app/
COPY --chown=firefly:firefly config.yaml /app/

USER firefly

# ---- Runtime defaults ------------------------------------------------------
ENV CONFIG_PATH=/app/config.yaml \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# tini makes SIGTERM propagation work. Without it, `docker compose down`
# waits the full 10s timeout then force-kills — ugly in demos.
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default: print available roles. Compose always overrides this.
CMD ["python3", "-c", "print('''\\n\
Firefly III Data Role — unified image\\n\
\\n\
Override CMD with one of:\\n\
  python3 ingest_bls_data.py           BLS CE Survey ingestion\\n\
  python3 data_pipeline.py config.yaml CE Survey ETL\\n\
  python3 generate_synthetic_data.py   Synthetic transaction generator\\n\
  python3 batch_training_pipeline.py   Chronological split\\n\
  python3 soda_quality_checks.py       Data quality gate\\n\
  python3 online_feature_service.py    Feature service (:8080)\\n\
  python3 feedback_service.py          Feedback service (:8100)\\n\
  python3 data_generator.py            Production traffic emulator\\n\
  python3 build_retraining_dataset.py  Retraining dataset builder\\n\
  python3 drift_monitor.py             PSI drift monitor\\n\
''')"]
