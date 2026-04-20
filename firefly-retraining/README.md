# Firefly III — Retraining Data Pipeline

The data-person deliverable for the Firefly III MLOps capstone. This pipeline
reads production logs + external training data, merges them into versioned
retraining datasets, and writes them to the `retraining-data` bucket for the
training team to consume.

## What this pipeline does

Two scheduled jobs:

| Job | File | Cadence | Source | Output |
|---|---|---|---|---|
| Categorization | `build_categorization_dataset.py` | Weekly (Sun 3 AM) | `production-logs/categorization/` + `training-data/ce_survey/categorization/` | `retraining-data/categorization/v=YYYY-MM-DD/` |
| Trend detection | `build_trend_detection_dataset.py` | Monthly (1st, 3 AM) | `production-logs/anomaly_feedback/` + `training-data/ce_survey/anomaly/` | `retraining-data/anomaly/v=YYYY-MM-DD/` |

Both follow the same 6-step pipeline:

```
  ┌─────────┐   ┌──────────┐   ┌───────────┐   ┌────────┐   ┌───────────────┐   ┌──────────────┐
  │ Ingest  │ → │ Validate │ → │ Transform │ → │ Weight │ → │ Merge + split │ → │ Version+write│
  └─────────┘   └──────────┘   └───────────┘   └────────┘   └───────────────┘   └──────────────┘
```

## Output layout (for training team)

Every successful run writes a versioned folder:

```
retraining-data/
├── categorization/
│   ├── v=2026-04-20/
│   │   ├── train.parquet        # 70%, chronologically earliest
│   │   ├── val.parquet          # 15%
│   │   ├── test.parquet         # 15%, chronologically latest
│   │   └── manifest.json        # metadata — read this first
│   ├── v=2026-04-27/
│   │   └── ...
│   └── latest.json              # pointer to newest version
└── anomaly/
    └── (same structure)
```

### `manifest.json` schema

```json
{
  "version": "v=2026-04-20",
  "built_at": "2026-04-20T03:15:42Z",
  "model_target": "categorization",
  "lookback_days": 7,
  "rows": {"train": 4200, "val": 900, "test": 900},
  "category_distribution": {"Groceries": 1820, "Dining": 740, "...": "..."},
  "source_mix": {"production": 2100, "external": 2100},
  "training_data_hash": "sha256:9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
  "feature_version": "categorization.v1",
  "schema_version": "1.0"
}
```

Training team uses `training_data_hash` as the `training_data_hash` field in the
model registry (§9 of the project doc) — it guarantees reproducibility.

### How to find the latest dataset

Read `retraining-data/<task>/latest.json`:

```json
{
  "version": "v=2026-04-20",
  "path": "retraining-data/categorization/v=2026-04-20/",
  "built_at": "2026-04-20T03:15:42Z"
}
```

Old versions are NEVER deleted — you can always roll back to a previous dataset.

---

## Running locally

### Prerequisites

- Python 3.11+
- Access to the MinIO instance (credentials in `.env`)

### Setup

```bash
cd firefly-retraining-pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then fill in real credentials
```

### Run one-off

```bash
./scripts/run_local.sh categorization     # weekly job
./scripts/run_local.sh anomaly            # monthly job
```

### Run with Docker

```bash
docker build -t firefly-retraining .
docker run --env-file .env firefly-retraining python build_categorization_dataset.py
```

### Test the pipeline before serving is live

Serving won't produce real logs until integration is done. To test the
pipeline with fake data:

```bash
# 1. Generate 5000 fake categorization events
python generate_test_data.py --type categorization --count 5000 \
    --output samples/fake_events.jsonl

# 2. Upload them to production-logs bucket
python generate_test_data.py --type categorization --count 5000 --upload

# 3. Run the pipeline
python build_categorization_dataset.py
```

### Validate a sample file against the schema

Useful before committing — or for the serving team to self-test:

```bash
python validate_schema.py --file samples/fake_events.jsonl --type categorization
```

---

## Scheduled runs (Chameleon VM)

Install the cron entries:

```bash
sudo cp cron/firefly-retraining.cron /etc/cron.d/firefly-retraining
sudo chmod 644 /etc/cron.d/firefly-retraining
sudo systemctl restart cron
```

Logs go to `/var/log/firefly-retraining.log`.

---

## Quality gates

A run **fails loudly** (non-zero exit, visible in cron logs) if:

| Check | Threshold |
|---|---|
| Total rows after validation | ≥ 500 |
| Schema reject rate | < 5% |
| Unique categories | ≥ 10 |
| Production/external ratio | 0.4 – 0.6 |
| Per-category minimum examples | ≥ 20 |
| Future-dated timestamps | 0 |

Failing a gate means **don't promote** — the training team won't pick up a
broken dataset.

---

## Project layout

```
firefly-retraining-pipeline/
├── README.md                          # this file
├── SCHEMA_CONTRACT.md                 # hand this to serving team
├── requirements.txt
├── .env.example
├── config.py                          # all constants in one place
├── schemas.py                         # Pydantic validators
├── utils.py                           # MinIO + DuckDB helpers
├── build_categorization_dataset.py    # weekly job
├── build_trend_detection_dataset.py   # monthly job
├── validate_schema.py                 # CLI validator
├── generate_test_data.py              # fake-data generator for testing
├── Dockerfile
├── cron/
│   └── firefly-retraining.cron
├── scripts/
│   └── run_local.sh
└── samples/                           # local test fixtures
```

---

## References to project doc

| Section | What it defines |
|---|---|
| §4.1.4 | Categorization feedback mechanism (accepted / overridden) |
| §4.1.3 | Confidence abstention threshold (0.7) |
| §4.2.2 | 20-feature vector for anomaly detection |
| §4.2.4 | Helpful / Not Useful / Expected feedback |
| §5.1.2 | Phase 2 fine-tuning — 50/50 mix + sample weights |
| §7.1.3 | Training data pipeline steps |
| §7.2 | Candidate selection rules |
| §7.3 | Leakage prevention (temporal split, shared features) |
| §8.1 | Retraining cadence (weekly / monthly) |
| §9 | Model registry fields requiring `training_data_hash` |
