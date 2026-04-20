"""Generate fake production-log JSONL so you can test the pipeline
before serving is wired up.

Usage:
    # Write fake events to a local file
    python generate_test_data.py --type categorization --count 5000 \\
        --output samples/fake_categorization.jsonl

    # Upload fake events directly into the MinIO production-logs bucket
    python generate_test_data.py --type categorization --count 5000 --upload

    python generate_test_data.py --type anomaly_feedback --count 500 --upload
"""
import argparse
import hashlib
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
    "Groceries":     ["KROGER #1247", "TRADER JOES #482", "WHOLE FOODS 0123", "SAFEWAY #0018"],
    "Dining":        ["CHIPOTLE 2048", "PANERA #0912", "SWEETGREEN NYC 04", "MCDONALD'S 7742"],
    "Transport":     ["UBER *TRIP", "LYFT *RIDE", "NYC MTA *METRO", "SHELL OIL 5521"],
    "Utilities":     ["AMEREN MISSOURI", "CONED ENERGY", "NATIONAL GRID", "SPECTRUM INTERNET"],
    "Subscriptions": ["NETFLIX.COM", "SPOTIFY USA", "ADOBE *CC", "HBO MAX"],
    "Entertainment": ["AMC THEATRES 24", "STEAM GAMES", "TICKETMASTER", "PLAYSTATION NET"],
    "Shopping":      ["AMZN MKTP US*2K4MR", "TARGET #0892", "BEST BUY #321", "IKEA CONSH"],
    "Healthcare":    ["CVS/PHARMACY 07821", "WALGREENS 3298", "QUEST DIAGNOSTICS"],
    "Insurance":     ["BLUE CROSS BLUE SHIELD", "GEICO AUTO INSURANCE", "STATE FARM RE"],
    "Housing":       ["RENT PMT COMMON", "HOA FEES", "HOME DEPOT #0412"],
    "Coffee":        ["STARBUCKS 02219", "BLUE BOTTLE COFFEE", "DUNKIN #332"],
    "Fuel":          ["SHELL OIL 5521", "EXXONMOBIL 4481", "BP #8820"],
    "Fitness":       ["EQUINOX NYC", "PELOTON MEMBERSHIP", "PLANET FITNESS 0021"],
    "Travel":        ["DELTA AIR 0061", "MARRIOTT HOTELS", "AIRBNB * HMRQ"],
    "Education":     ["COURSERA *SPEC", "NYU BURSAR", "UDEMY *COURSE"],
}


def hashed_user(i: int) -> str:
    h = hashlib.sha256(f"user-{i}".encode()).hexdigest()
    return f"sha256:{h}"


def fake_categorization_event(rng: random.Random, ts: datetime) -> dict:
    true_cat = rng.choice(CATEGORIES)
    merchant = rng.choice(MERCHANTS[true_cat])
    # Top prediction might be right, wrong, or uncertain
    roll = rng.random()
    if roll < 0.60:
        # accepted — model got it right with high confidence
        pred = [true_cat, rng.choice([c for c in CATEGORIES if c != true_cat])]
        probs = [round(rng.uniform(0.75, 0.98), 3), round(rng.uniform(0.2, 0.5), 3)]
        action, final = "accepted", true_cat
    elif roll < 0.85:
        # overridden — model wrong, user corrected
        wrong = rng.choice([c for c in CATEGORIES if c != true_cat])
        pred = [wrong, true_cat]
        probs = [round(rng.uniform(0.7, 0.9), 3), round(rng.uniform(0.3, 0.6), 3)]
        action, final = "overridden", true_cat
    elif roll < 0.95:
        # abstained — confidence low, user picked
        pred = rng.sample(CATEGORIES, 2)
        probs = [round(rng.uniform(0.35, 0.65), 3), round(rng.uniform(0.2, 0.5), 3)]
        action, final = "abstained", true_cat
    else:
        # ignored — no label
        pred = rng.sample(CATEGORIES, 2)
        probs = [round(rng.uniform(0.7, 0.9), 3), round(rng.uniform(0.3, 0.6), 3)]
        action, final = "ignored", None

    return {
        "event_id": str(uuid.uuid4()),
        "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "user_id": hashed_user(rng.randint(1, 50)),
        "transaction_description": merchant,
        "amount": round(rng.uniform(3.0, 450.0), 2),
        "currency": "USD",
        "country": "US",
        "predicted_categories": pred,
        "prediction_probabilities": probs,
        "model_version": "2.3.1",
        "user_action": action,
        "final_category": final,
    }


def fake_anomaly_event(rng: random.Random, ts: datetime) -> dict:
    mean = rng.uniform(50.0, 800.0)
    current = mean * rng.uniform(0.4, 2.5)
    predicted = mean * rng.uniform(0.8, 1.2)
    feedback = rng.choice(["helpful", "not_useful", "expected", None])
    return {
        "event_id": str(uuid.uuid4()),
        "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "user_id": hashed_user(rng.randint(1, 50)),
        "category": rng.choice(CATEGORIES),
        "month": ts.strftime("%Y-%m"),
        "feature_vector": {
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
        },
        "anomaly_score": round(rng.uniform(0.3, 0.95), 3),
        "predicted_spend": round(predicted, 2),
        "actual_spend": round(current, 2),
        "direction": 1 if current > predicted else -1,
        "magnitude_pct": round(abs(current - predicted) / predicted * 100, 1),
        "alert_severity": rng.choice(["low", "medium", "high"]),
        "top_factors": rng.sample(
            ["deviation_ratio", "spending_velocity", "share_of_wallet",
             "txn_count", "budget_utilization"], 3),
        "user_feedback": feedback,
        "model_version": "1.4.2",
    }


def write_local(events: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def upload_partitioned(events: list[dict], event_type: str) -> None:
    """Upload events into Hive-partitioned paths by timestamp."""
    client = utils.get_minio_client()
    utils.ensure_bucket(client, config.BUCKET_PRODUCTION_LOGS)

    prefix = (
        config.PREFIX_CATEGORIZATION_LOGS
        if event_type == "categorization"
        else config.PREFIX_ANOMALY_FEEDBACK
    )
    buckets: dict[str, list[dict]] = {}
    for e in events:
        ts = datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00"))
        if event_type == "categorization":
            key_prefix = (
                f"{prefix}/year={ts.year}/month={ts.month:02d}/"
                f"day={ts.day:02d}/hour={ts.hour:02d}"
            )
        else:
            key_prefix = (
                f"{prefix}/year={ts.year}/month={ts.month:02d}/day={ts.day:02d}"
            )
        buckets.setdefault(key_prefix, []).append(e)

    for key_prefix, items in buckets.items():
        local = config.TMP_DIR / f"fake_{uuid.uuid4().hex[:8]}.jsonl"
        with open(local, "w") as f:
            for e in items:
                f.write(json.dumps(e) + "\n")
        key = f"{key_prefix}/events_{uuid.uuid4().hex[:8]}.jsonl"
        utils.upload_file(client, config.BUCKET_PRODUCTION_LOGS, key, local)
        print(f"  uploaded {len(items):>5} events → s3://{config.BUCKET_PRODUCTION_LOGS}/{key}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", required=True, choices=["categorization", "anomaly_feedback"])
    ap.add_argument("--count", type=int, default=1000)
    ap.add_argument("--output", type=Path, default=None, help="Local JSONL file path")
    ap.add_argument("--upload", action="store_true",
                    help="Upload to production-logs bucket (partitioned)")
    ap.add_argument("--spread-days", type=int, default=7,
                    help="Spread event timestamps over the last N days")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    now = datetime.now(timezone.utc)
    gen = fake_categorization_event if args.type == "categorization" else fake_anomaly_event

    events = []
    for _ in range(args.count):
        ts = now - timedelta(seconds=rng.randint(0, args.spread_days * 86400))
        events.append(gen(rng, ts))

    print(f"Generated {len(events)} fake {args.type} events")

    if args.output:
        write_local(events, args.output)
        print(f"Wrote {args.output}")

    if args.upload:
        upload_partitioned(events, args.type)
        print("Upload complete.")

    if not args.output and not args.upload:
        # Default: print first 3 so the user can eyeball
        for e in events[:3]:
            print(json.dumps(e, indent=2))


if __name__ == "__main__":
    main()
