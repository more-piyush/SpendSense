"""Generate fake feedback-event JSON files so you can test the pipeline
before real feedback lands.

Matches the production-log envelope observed in the wild:
    {event_id, recorded_at, event_type, feedback: {...}}

Usage:
    # Categorization feedback — write locally
    python generate_test_data.py --type categorization --count 5000 \\
        --output samples/fake_categorization/

    # Upload directly into production-logs
    python generate_test_data.py --type categorization --count 5000 --upload
    python generate_test_data.py --type trend --count 500 --upload
"""
import argparse
import json
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import config
import utils


CATEGORIES = [
    "Groceries", "Dining", "Transport", "Utilities", "Subscriptions",
    "Entertainment", "Shopping", "Healthcare", "Insurance", "Housing",
    "Coffee", "Fuel", "Fitness", "Travel", "Education",
]

MERCHANTS = {
    "Groceries":     ["KROGER #1247", "TRADER JOES #482", "WHOLE FOODS 0123"],
    "Dining":        ["CHIPOTLE 2048", "PANERA #0912", "SWEETGREEN NYC 04"],
    "Transport":     ["UBER *TRIP", "LYFT *RIDE", "NYC MTA *METRO"],
    "Utilities":     ["AMEREN MISSOURI", "CONED ENERGY", "NATIONAL GRID"],
    "Subscriptions": ["NETFLIX.COM", "SPOTIFY USA", "ADOBE *CC"],
    "Entertainment": ["AMC THEATRES 24", "STEAM GAMES", "TICKETMASTER"],
    "Shopping":      ["AMZN MKTP US*2K4MR", "TARGET #0892", "BEST BUY #321"],
    "Healthcare":    ["CVS/PHARMACY 07821", "WALGREENS 3298"],
    "Insurance":     ["BLUE CROSS BLUE SHIELD", "GEICO AUTO INSURANCE"],
    "Housing":       ["RENT PMT COMMON", "HOME DEPOT #0412"],
    "Coffee":        ["STARBUCKS 02219", "BLUE BOTTLE COFFEE", "DUNKIN #332"],
    "Fuel":          ["SHELL OIL 5521", "EXXONMOBIL 4481", "BP #8820"],
    "Fitness":       ["EQUINOX NYC", "PELOTON MEMBERSHIP"],
    "Travel":        ["DELTA AIR 0061", "MARRIOTT HOTELS", "AIRBNB * HMRQ"],
    "Education":     ["COURSERA *SPEC", "NYU BURSAR", "UDEMY *COURSE"],
}


def _iso(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ts.microsecond:06d}+00:00"


# ─────────────────────────────────────────────────────────────────────────────
# Categorization feedback event
# ─────────────────────────────────────────────────────────────────────────────
def fake_categorization_event(rng: random.Random, ts: datetime) -> dict:
    true_cat = rng.choice(CATEGORIES)
    merchant = rng.choice(MERCHANTS[true_cat])
    roll = rng.random()

    if roll < 0.60:
        action, final_cat = "accepted", true_cat
        pred_cat, pred_conf = true_cat, round(rng.uniform(0.75, 0.98), 4)
    elif roll < 0.85:
        action, final_cat = "overridden", true_cat
        pred_cat = rng.choice([c for c in CATEGORIES if c != true_cat])
        pred_conf = round(rng.uniform(0.7, 0.9), 4)
    elif roll < 0.95:
        action, final_cat = "abstained", true_cat
        pred_cat = rng.choice(CATEGORIES)
        pred_conf = round(rng.uniform(0.35, 0.65), 4)
    else:
        action, final_cat = "ignored", None
        pred_cat = rng.choice(CATEGORIES)
        pred_conf = round(rng.uniform(0.7, 0.9), 4)

    feedback = {
        "task": "categorization",
        "transaction_id": str(uuid.uuid4()),
        "user_id": str(rng.randint(1, 50)),
        "model_family": "distilbert",
        "model_version": "1.0.0",
        "action": action,
        "predicted_value": {"category": pred_cat, "confidence": pred_conf},
        "final_value": {"category": final_cat} if final_cat is not None else None,
        "metadata": {
            "source": "firefly-ui",
            "description": merchant,
            "amount": f"{round(rng.uniform(3.0, 450.0), 2)}",
            "currency": "USD",
            "feedback_origin": "create-transaction",
        },
        "timestamp": _iso(ts),
    }
    return {
        "event_id": "fb_" + uuid.uuid4().hex[:16],
        "recorded_at": _iso(ts),
        "feedback": feedback,
        "event_type": "feedback/categorization",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Trend feedback event (PROVISIONAL — mirrors categorization envelope)
# ─────────────────────────────────────────────────────────────────────────────
def fake_trend_event(rng: random.Random, ts: datetime) -> dict:
    mean = rng.uniform(50.0, 800.0)
    current = mean * rng.uniform(0.4, 2.5)
    predicted = mean * rng.uniform(0.8, 1.2)
    action = rng.choice(["helpful", "not_useful", "expected", "ignored"])

    features = {
        "current_spend": round(current, 2),
        "rolling_mean_1m": round(mean * rng.uniform(0.9, 1.1), 2),
        "rolling_mean_3m": round(mean, 2),
        "rolling_mean_6m": round(mean * rng.uniform(0.95, 1.05), 2),
        "rolling_std_3m": round(mean * 0.15, 2),
        "rolling_std_6m": round(mean * 0.13, 2),
        "deviation_ratio": round(current / mean, 3),
        "share_of_wallet": round(rng.uniform(0.05, 0.3), 3),
        "hist_share_of_wallet": round(rng.uniform(0.05, 0.25), 3),
        "txn_count": rng.randint(1, 20),
        "hist_txn_count_mean": round(rng.uniform(3, 12), 1),
        "avg_txn_size": round(current / max(rng.randint(1, 20), 1), 2),
        "hist_avg_txn_size": round(mean / 8, 2),
        "days_since_last_txn": rng.randint(0, 30),
        "month_of_year": ts.month,
        "spending_velocity": round(rng.uniform(0.5, 1.6), 2),
        "weekend_txn_ratio": round(rng.uniform(0.2, 0.6), 2),
        "total_monthly_spend": round(current * rng.uniform(5, 12), 2),
        "elevated_cat_count": rng.randint(0, 5),
        "budget_utilization": round(rng.uniform(0.3, 1.2), 2),
    }

    predicted_value = {
        "predicted_next_month_spend": round(predicted, 2),
        "anomaly_detection": {
            "ensemble_score": round(rng.uniform(0.3, 0.95), 4),
            "xgb_residual_score": round(rng.uniform(0.3, 0.95), 4),
            "isolation_forest_score": round(rng.uniform(0.0, 0.3), 4),
            "is_anomaly": rng.random() < 0.2,
            "anomaly_threshold": 0.7,
        },
        "trend_analysis": {
            "predicted_change_pct": round(rng.uniform(-40, 40), 2),
            "trend_direction": rng.choice(["increasing", "decreasing", "stable"]),
            "spending_vs_history": rng.choice(["above_average", "below_average", "in_line"]),
            "deviation_from_3m_mean": round(current - mean, 2),
        },
    }

    final_value = None
    if action != "ignored":
        final_value = {
            "user_feedback": action,
            "actual_next_month_spend": round(current, 2),
        }

    feedback = {
        "task": "trend_detection",
        "user_id": str(rng.randint(1, 50)),
        "category": rng.choice(CATEGORIES),
        "period": ts.strftime("%Y-%m"),
        "model_family": "xgboost_optuna",
        "model_version": "1.1.0",
        "action": action,
        "features": features,
        "predicted_value": predicted_value,
        "final_value": final_value,
        "metadata": {"source": "firefly-ui"},
        "timestamp": _iso(ts),
    }
    return {
        "event_id": "fb_" + uuid.uuid4().hex[:16],
        "recorded_at": _iso(ts),
        "feedback": feedback,
        "event_type": "feedback/trend",
    }


# ─────────────────────────────────────────────────────────────────────────────
def write_local(events: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for e in events:
        ts_raw = e["recorded_at"]
        stamp = ts_raw.replace(":", "").replace("-", "").replace(".", "")[:20]
        p = out_dir / f"{stamp}Z_{uuid.uuid4().hex[:12]}.json"
        with open(p, "w") as f:
            json.dump(e, f)


def upload_partitioned(events: list[dict], event_type: str) -> None:
    """Upload one JSON file per event into Hive-partitioned paths."""
    client = utils.get_minio_client()
    utils.ensure_bucket(client, config.BUCKET_PRODUCTION_LOGS)

    prefix = (
        config.PREFIX_CATEGORIZATION_FEEDBACK
        if event_type == "categorization"
        else config.PREFIX_TREND_FEEDBACK
    )

    uploaded = 0
    for e in events:
        ts = datetime.fromisoformat(e["recorded_at"])
        key_prefix = (
            f"{prefix}/year={ts.year}/month={ts.month:02d}/day={ts.day:02d}"
        )
        stamp = ts.strftime("%Y%m%dT%H%M%S%f")[:20]
        key = f"{key_prefix}/{stamp}Z_{uuid.uuid4().hex[:12]}.json"
        local = config.TMP_DIR / f"fake_{uuid.uuid4().hex[:8]}.json"
        with open(local, "w") as f:
            json.dump(e, f)
        utils.upload_file(client, config.BUCKET_PRODUCTION_LOGS, key, local)
        uploaded += 1
        if uploaded % 500 == 0:
            print(f"  uploaded {uploaded}/{len(events)}")
    print(f"  uploaded {uploaded} events → "
          f"s3://{config.BUCKET_PRODUCTION_LOGS}/{prefix}/...")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", required=True, choices=["categorization", "trend"])
    ap.add_argument("--count", type=int, default=1000)
    ap.add_argument("--output", type=Path, default=None,
                    help="Local directory for JSON files")
    ap.add_argument("--upload", action="store_true",
                    help="Upload to production-logs bucket (partitioned)")
    ap.add_argument("--spread-days", type=int, default=7,
                    help="Spread event timestamps over the last N days")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    now = datetime.now(timezone.utc)
    gen = (fake_categorization_event if args.type == "categorization"
           else fake_trend_event)

    events = []
    for _ in range(args.count):
        ts = now - timedelta(seconds=rng.randint(0, args.spread_days * 86400))
        events.append(gen(rng, ts))

    print(f"Generated {len(events)} fake {args.type} feedback events")

    if args.output:
        write_local(events, args.output)
        print(f"Wrote {args.output}/*.json")

    if args.upload:
        upload_partitioned(events, args.type)
        print("Upload complete.")

    if not args.output and not args.upload:
        for e in events[:3]:
            print(json.dumps(e, indent=2))


if __name__ == "__main__":
    main()
