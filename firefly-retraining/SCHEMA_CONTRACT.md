# Production Log Schema Contract

**Owner:** Data team (Prachiti)
**Consumer:** Serving team
**Version:** 2.0
**Last updated:** 2026-04-21

This document defines the exact format the serving service writes to the
`production-logs` MinIO bucket. The data team's retraining pipeline reads these
logs to build training datasets. If the format drifts, retraining breaks.

---

## 1. Two event streams

Serving writes two parallel streams:

| Stream | Event type | Purpose | Used for retraining? |
|---|---|---|---|
| Inference | `interactions/categorization`, `interactions/trend` | Raw model prediction, no user label | No — drift monitoring only |
| Feedback | `feedback/categorization`, `feedback/trend` | User-confirmed outcome (the label) | **Yes** |

The retraining pipeline in this repo reads **`feedback/*`** events. The
`interactions/*` events are ignored here — they're a separate monitoring concern.

---

## 2. Storage conventions (both streams)

- **Bucket:** `production-logs`
- **Format:** one JSON object per file (single-event files)
- **File naming:** `<YYYYMMDDTHHMMSSmilli>Z_<hex12>.json`
- **Partitioning:** Hive-style `year=/month=/day=/` recommended
- **Compression:** none
- **Timestamps:** ISO 8601 UTC with microseconds and offset
  (`YYYY-MM-DDTHH:MM:SS.ffffff+00:00`)

---

## 3. Categorization feedback event (CONFIRMED)

### 3.1 Path

```
s3://production-logs/feedback/categorization/year=YYYY/month=MM/day=DD/<stamp>_<id>.json
```

### 3.2 Envelope

```json
{
  "event_id": "fb_<hex16>",
  "recorded_at": "2026-04-21T01:14:08.590447+00:00",
  "event_type": "feedback/categorization",
  "feedback": { /* see 3.3 */ }
}
```

### 3.3 `feedback` body

| Field | Type | Notes |
|---|---|---|
| `task` | `"categorization"` | const |
| `transaction_id` | string | references the inference event |
| `user_id` | string | no hashing required in current logs |
| `model_family` | string | e.g. `"distilbert"` |
| `model_version` | string | semver `^\d+\.\d+\.\d+$` |
| `action` | enum | `accepted` \| `overridden` \| `abstained` \| `ignored` |
| `predicted_value` | object \| null | `{category, confidence?}` |
| `final_value` | object \| null | `{category}` — **null iff action=ignored** |
| `metadata.description` | string | transaction description |
| `metadata.amount` | string | **logged as string** — pipeline coerces to float |
| `metadata.currency` | string | 3-letter ISO code |
| `metadata.source` | string (opt) | e.g. `"firefly-ui"` |
| `metadata.feedback_origin` | string (opt) | e.g. `"create-transaction"` |
| `timestamp` | ISO 8601 datetime | when the user took the action |

### 3.4 `action` enum

| Value | When to log | `final_value` |
|---|---|---|
| `accepted` | User kept top suggestion (max_confidence ≥ 0.7) | top predicted category |
| `overridden` | User changed the prediction | user's chosen category |
| `abstained` | max_confidence < 0.7 → user was prompted | user's manual pick |
| `ignored` | User closed page without engaging | **must be null** |

### 3.5 Example (verbatim from production)

```json
{
  "event_id": "fb_fce2a1bc151743a8",
  "recorded_at": "2026-04-21T01:14:08.590447+00:00",
  "feedback": {
    "task": "categorization",
    "transaction_id": "6d6a31a2-bfe5-46ce-aefd-fb8fca00f61f",
    "user_id": "1",
    "model_family": "distilbert",
    "model_version": "1.0.0",
    "action": "accepted",
    "predicted_value": {"category": "Shopping", "confidence": 0.9604},
    "final_value": {"category": "Shopping"},
    "metadata": {
      "source": "firefly-ui",
      "description": "Walmart groceries and household supplies",
      "amount": "43.54",
      "currency": "USD",
      "feedback_origin": "create-transaction"
    },
    "timestamp": "2026-04-21T01:14:08.484Z"
  },
  "event_type": "feedback/categorization"
}
```

---

## 4. Trend feedback event (PROVISIONAL)

> **Status:** no real sample received yet. Shape below mirrors the
> categorization envelope. Confirm and tighten once one lands.

### 4.1 Path

```
s3://production-logs/feedback/trend/year=YYYY/month=MM/day=DD/<stamp>_<id>.json
```

### 4.2 Envelope

```json
{
  "event_id": "fb_<hex16>",
  "recorded_at": "2026-04-21T...+00:00",
  "event_type": "feedback/trend",
  "feedback": { /* see 4.3 */ }
}
```

### 4.3 `feedback` body

| Field | Type | Notes |
|---|---|---|
| `task` | `"trend_detection"` | const |
| `user_id` | string | — |
| `category` | string | spending category the alert was about |
| `period` | string | `YYYY-MM` |
| `model_family` | string | e.g. `"xgboost_optuna"` |
| `model_version` | string | semver |
| `action` | enum | `helpful` \| `not_useful` \| `expected` \| `ignored` |
| `features` | object | **all 20 keys** — see 4.4 |
| `predicted_value` | object | `{predicted_next_month_spend, anomaly_detection, trend_analysis}` |
| `final_value` | object \| null | `{user_feedback, actual_next_month_spend?}` — null iff action=ignored |
| `metadata` | object | free-form |
| `timestamp` | ISO 8601 datetime | — |

### 4.4 `features` — exact 20 keys (§4.2.2)

```
current_spend, rolling_mean_1m, rolling_mean_3m, rolling_mean_6m,
rolling_std_3m, rolling_std_6m, deviation_ratio, share_of_wallet,
hist_share_of_wallet, txn_count, hist_txn_count_mean, avg_txn_size,
hist_avg_txn_size, days_since_last_txn, month_of_year,
spending_velocity, weekend_txn_ratio, total_monthly_spend,
elevated_cat_count, budget_utilization
```

### 4.5 `action` → training weight

| Value | Meaning | Weight |
|---|---|---|
| `helpful` | Alert was correct | 1.0 |
| `expected` | Spending is normal — shift target to predicted | 0.5 |
| `not_useful` | Alert was spurious | 1.0 (negative example) |
| `ignored` | No feedback | excluded from training |

---

## 5. Self-test before deploying

```bash
python validate_schema.py --file /path/to/sample.json --type categorization_feedback
python validate_schema.py --file /path/to/sample.json --type trend_feedback
```

The validator exits non-zero on any validation error.

---

## 6. Change process

1. Bump `model_version` if output semantics change
2. Ping the data team (Prachiti) before any field add/rename/remove
3. Update this doc's version + "Last updated"

Adding optional fields is safe. Renaming or removing fields is **breaking** —
coordinate first.
