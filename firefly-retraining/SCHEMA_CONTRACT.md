# Production Log Schema Contract

**Owner:** Data team (Prachiti)
**Consumer:** Serving team
**Version:** 1.0
**Last updated:** 2026-04-20

This document defines the exact format the serving service must write to the
`production-logs` MinIO bucket. The data team's retraining pipeline reads these
logs to build training datasets. If the format drifts, retraining breaks.

---

## 1. Storage conventions (both event types)

- **Bucket:** `production-logs`
- **Format:** JSONL — one JSON object per line, UTF-8 encoded, newline-terminated
- **Compression:** none (append performance > storage cost at this scale)
- **Partitioning:** Hive-style `year=/month=/day=/hour=/` so queries can scan
  only relevant time windows
- **Write semantics:** append-only; one file per writer per hour is ideal
  (e.g. `events_<uuid>.jsonl` where the UUID is the writer/process ID)
- **Timestamps:** always ISO 8601 UTC (`YYYY-MM-DDTHH:MM:SSZ`)
- **User IDs:** always `sha256:<hex>` — never raw user identifiers

---

## 2. Categorization events

Written every time the DistilBERT model makes a prediction on a new transaction.

### 2.1 Path

```
s3://production-logs/categorization/year=YYYY/month=MM/day=DD/hour=HH/events_<uuid>.jsonl
```

Example:
```
s3://production-logs/categorization/year=2026/month=04/day=20/hour=14/events_a3f4.jsonl
```

### 2.2 JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "CategorizationEvent",
  "type": "object",
  "required": [
    "event_id", "timestamp", "user_id", "transaction_description",
    "amount", "currency", "country", "predicted_categories",
    "prediction_probabilities", "model_version", "user_action", "final_category"
  ],
  "properties": {
    "event_id":                 { "type": "string", "format": "uuid" },
    "timestamp":                { "type": "string", "format": "date-time" },
    "user_id":                  { "type": "string", "pattern": "^sha256:[a-f0-9]{64}$" },
    "transaction_description":  { "type": "string", "minLength": 1 },
    "amount":                   { "type": "number", "minimum": 0 },
    "currency":                 { "type": "string", "pattern": "^[A-Z]{3}$" },
    "country":                  { "type": "string", "pattern": "^[A-Z]{2}$" },
    "predicted_categories":     { "type": "array", "items": { "type": "string" }, "minItems": 1 },
    "prediction_probabilities": { "type": "array", "items": { "type": "number", "minimum": 0.0, "maximum": 1.0 } },
    "model_version":            { "type": "string", "pattern": "^\\d+\\.\\d+\\.\\d+$" },
    "user_action":              { "type": "string", "enum": ["accepted", "overridden", "abstained", "ignored"] },
    "final_category":           { "type": ["string", "null"] }
  },
  "allOf": [
    {
      "if":   { "properties": { "user_action": { "const": "ignored" } } },
      "then": { "properties": { "final_category": { "type": "null" } } },
      "else": { "properties": { "final_category": { "type": "string", "minLength": 1 } } }
    }
  ]
}
```

### 2.3 `user_action` enum — all 4 cases

| Value | When to log this | `final_category` |
|---|---|---|
| `accepted` | User kept the top suggestion (max prob ≥ 0.7) | Top predicted category |
| `overridden` | User changed the prediction | The user's chosen category |
| `abstained` | Max prob < 0.7 → user was prompted to pick manually | The user's manual pick |
| `ignored` | User closed page without engaging | `null` |

> The data pipeline computes `feedback_weight` from `user_action` during the
> Transform step. Do NOT log `feedback_weight` from serving — it is a data-team
> policy, not a serving concern.

### 2.4 Example (one line per event)

```json
{"event_id":"a3f4b1e2-9c5d-4a7b-8e3f-1d2c3b4a5e6f","timestamp":"2026-04-20T14:03:11Z","user_id":"sha256:abc123...","transaction_description":"KROGER #1247 SPRINGFIELD MO","amount":82.40,"currency":"USD","country":"US","predicted_categories":["Groceries","Food"],"prediction_probabilities":[0.91,0.78],"model_version":"2.3.1","user_action":"accepted","final_category":"Groceries"}
{"event_id":"b7c2d3e4-1a2b-3c4d-5e6f-7a8b9c0d1e2f","timestamp":"2026-04-20T14:15:22Z","user_id":"sha256:abc123...","transaction_description":"AMZN MKTP US*2K4MR1LO3","amount":42.30,"currency":"USD","country":"US","predicted_categories":["Groceries","Food"],"prediction_probabilities":[0.82,0.54],"model_version":"2.3.1","user_action":"overridden","final_category":"Dining"}
{"event_id":"c9d8e7f6-5a4b-3c2d-1e0f-9a8b7c6d5e4f","timestamp":"2026-04-20T14:28:09Z","user_id":"sha256:abc123...","transaction_description":"SQ *RANDOM VENDOR LLC 4521","amount":15.75,"currency":"USD","country":"US","predicted_categories":["Shopping","Other"],"prediction_probabilities":[0.42,0.38],"model_version":"2.3.1","user_action":"abstained","final_category":"Coffee"}
{"event_id":"d1e2f3a4-5b6c-7d8e-9f0a-1b2c3d4e5f6a","timestamp":"2026-04-20T14:45:33Z","user_id":"sha256:def456...","transaction_description":"NETFLIX.COM 888-6384","amount":15.99,"currency":"USD","country":"US","predicted_categories":["Subscriptions","Entertainment"],"prediction_probabilities":[0.88,0.71],"model_version":"2.3.1","user_action":"ignored","final_category":null}
```

---

## 3. Anomaly / trend-detection feedback events

Written every time the XGBoost + Isolation Forest pipeline emits an alert **and**
optionally updated when the user clicks Helpful / Not Useful / Expected.

### 3.1 Path

```
s3://production-logs/anomaly_feedback/year=YYYY/month=MM/day=DD/feedback_<uuid>.jsonl
```

(No hour partition — alerts are batch-generated nightly per §4.2 of the project
doc; daily resolution is sufficient.)

### 3.2 JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "AnomalyFeedbackEvent",
  "type": "object",
  "required": [
    "event_id", "timestamp", "user_id", "category", "month",
    "feature_vector", "anomaly_score", "predicted_spend", "actual_spend",
    "direction", "magnitude_pct", "alert_severity", "model_version"
  ],
  "properties": {
    "event_id":        { "type": "string", "format": "uuid" },
    "timestamp":       { "type": "string", "format": "date-time" },
    "user_id":         { "type": "string", "pattern": "^sha256:[a-f0-9]{64}$" },
    "category":        { "type": "string", "minLength": 1 },
    "month":           { "type": "string", "pattern": "^\\d{4}-\\d{2}$" },
    "feature_vector": {
      "type": "object",
      "required": [
        "current_spend", "rolling_mean_1m", "rolling_mean_3m", "rolling_mean_6m",
        "rolling_std_3m", "rolling_std_6m", "deviation_ratio", "share_of_wallet",
        "hist_share_of_wallet", "txn_count", "hist_txn_count_mean", "avg_txn_size",
        "hist_avg_txn_size", "days_since_last_txn", "month_of_year",
        "spending_velocity", "weekend_txn_ratio", "total_monthly_spend",
        "elevated_cat_count", "budget_utilization"
      ]
    },
    "anomaly_score":   { "type": "number", "minimum": 0.0, "maximum": 1.0 },
    "predicted_spend": { "type": "number" },
    "actual_spend":    { "type": "number" },
    "direction":       { "type": "integer", "enum": [-1, 1] },
    "magnitude_pct":   { "type": "number" },
    "alert_severity":  { "type": "string", "enum": ["low", "medium", "high"] },
    "top_factors":     { "type": "array", "items": { "type": "string" }, "maxItems": 3 },
    "user_feedback":   { "type": ["string", "null"], "enum": ["helpful", "not_useful", "expected", null] },
    "model_version":   { "type": "string", "pattern": "^\\d+\\.\\d+\\.\\d+$" }
  }
}
```

### 3.3 `feature_vector` — exact 20 keys (§4.2.2)

Names must match **exactly** — do not abbreviate or rename.

```
current_spend, rolling_mean_1m, rolling_mean_3m, rolling_mean_6m,
rolling_std_3m, rolling_std_6m, deviation_ratio, share_of_wallet,
hist_share_of_wallet, txn_count, hist_txn_count_mean, avg_txn_size,
hist_avg_txn_size, days_since_last_txn, month_of_year,
spending_velocity, weekend_txn_ratio, total_monthly_spend,
elevated_cat_count, budget_utilization
```

### 3.4 `user_feedback` semantics

| Value | Meaning | Training weight |
|---|---|---|
| `helpful` | Alert was correct | 1.0 (confirm) |
| `expected` | Spending is normal for this user | 0.5 (adjust target down) |
| `not_useful` | Alert was spurious | 1.0 (negative example) |
| `null` | User has not given feedback yet | excluded from training |

### 3.5 Example

```json
{"event_id":"b7c2d3e4-1a2b-3c4d-5e6f-7a8b9c0d1e2f","timestamp":"2026-04-20T02:15:00Z","user_id":"sha256:abc123...","category":"Dining","month":"2026-04","feature_vector":{"current_spend":412.50,"rolling_mean_1m":280.00,"rolling_mean_3m":265.40,"rolling_mean_6m":258.10,"rolling_std_3m":42.30,"rolling_std_6m":38.90,"deviation_ratio":1.55,"share_of_wallet":0.18,"hist_share_of_wallet":0.12,"txn_count":11,"hist_txn_count_mean":8.3,"avg_txn_size":37.50,"hist_avg_txn_size":31.90,"days_since_last_txn":1,"month_of_year":4,"spending_velocity":1.40,"weekend_txn_ratio":0.45,"total_monthly_spend":2290.00,"elevated_cat_count":3,"budget_utilization":0.82},"anomaly_score":0.74,"predicted_spend":275.00,"actual_spend":412.50,"direction":1,"magnitude_pct":50.0,"alert_severity":"medium","top_factors":["deviation_ratio","spending_velocity","share_of_wallet"],"user_feedback":"expected","model_version":"1.4.2"}
```

---

## 4. How to self-test before deploying

Run the validator against a sample file of your log output:

```bash
python validate_schema.py --file /path/to/sample.jsonl --type categorization
python validate_schema.py --file /path/to/sample.jsonl --type anomaly_feedback
```

The validator prints per-line errors and exits non-zero if anything is invalid.
Use this in your CI pipeline before shipping a serving change.

---

## 5. Change process

Any schema change requires:

1. Bump `model_version` if output semantics change
2. Ping the data team (Prachiti) — new fields need to be wired into the
   retraining pipeline before serving starts writing them
3. Update this doc's version number and "Last updated"

**Breaking changes are not allowed without coordination.** Adding optional
fields is safe; renaming/removing fields is not.
