#!/usr/bin/env python3
"""
Data generator: sends synthetic transactions to hypothetical
Firefly III ML service endpoints and logs to MinIO.
Run for 2-3 minutes for demo video.
"""
import io, json, random, time, uuid
from datetime import datetime
from minio import Minio

client = Minio("localhost:9000", access_key="firefly-access-key",
               secret_key="firefly-secret-key", secure=False)

TRANSACTIONS = [
    {"description": "KROGER #1247 SPRINGFIELD IL", "amount": 67.42, "category": "Food at home"},
    {"description": "STARBUCKS #892 CHICAGO IL", "amount": 5.75, "category": "Food away"},
    {"description": "DUKE ENERGY BILL PAYMENT", "amount": 142.00, "category": "Utilities"},
    {"description": "AMAZON MARKETPLACE PURCHASE", "amount": 29.99, "category": "Miscellaneous"},
    {"description": "WALGREENS PHARMACY #3301", "amount": 18.45, "category": "Healthcare"},
    {"description": "SHELL OIL 5420384710 ANN ARBOR", "amount": 52.80, "category": "Transportation"},
    {"description": "NETFLIX.COM MONTHLY", "amount": 15.49, "category": "Entertainment"},
    {"description": "BLUE CROSS HEALTH INS", "amount": 320.00, "category": "Healthcare"},
    {"description": "CHIPOTLE 2203 COLUMBUS OH", "amount": 13.25, "category": "Food away"},
    {"description": "HOME DEPOT #4219 PORTLAND OR", "amount": 89.74, "category": "Household"},
    {"description": "UBER TRIP HELP.UBER.COM", "amount": 14.33, "category": "Transportation"},
    {"description": "WHOLE FOODS MARKET #10298", "amount": 103.61, "category": "Food at home"},
    {"description": "TARGET 00025073 DENVER CO", "amount": 44.19, "category": "Household"},
    {"description": "SPOTIFY USA MUSIC", "amount": 9.99, "category": "Entertainment"},
    {"description": "PROGRESSIVE INSURANCE PMT", "amount": 180.00, "category": "Insurance"},
    {"description": "MACYS #4821 CHARLOTTE NC", "amount": 67.50, "category": "Clothing"},
    {"description": "COMCAST CABLE BILL", "amount": 89.99, "category": "Utilities"},
    {"description": "CVS PHARMACY #8821", "amount": 22.40, "category": "Healthcare"},
    {"description": "DOORDASH ORDER 994821", "amount": 31.75, "category": "Food away"},
    {"description": "STATE FARM AUTO INS", "amount": 210.00, "category": "Insurance"},
]

def predict_category(description):
    """Simulate ML model prediction."""
    desc = description.lower()
    if any(w in desc for w in ["kroger","walmart","whole foods","trader","aldi","safeway"]):
        return "Food at home", round(random.uniform(0.85, 0.97), 3)
    if any(w in desc for w in ["starbucks","mcdonalds","chipotle","doordash","subway"]):
        return "Food away", round(random.uniform(0.82, 0.95), 3)
    if any(w in desc for w in ["energy","electric","gas","comcast","verizon","at&t"]):
        return "Utilities", round(random.uniform(0.88, 0.96), 3)
    if any(w in desc for w in ["walgreens","cvs","pharmacy","health","medical","blue cross"]):
        return "Healthcare", round(random.uniform(0.80, 0.93), 3)
    if any(w in desc for w in ["shell","bp","chevron","uber","lyft","progressive","geico"]):
        return "Transportation", round(random.uniform(0.79, 0.92), 3)
    if any(w in desc for w in ["netflix","spotify","hulu","amazon prime","amc"]):
        return "Entertainment", round(random.uniform(0.90, 0.98), 3)
    if any(w in desc for w in ["insurance","state farm","allstate"]):
        return "Insurance", round(random.uniform(0.85, 0.95), 3)
    return "Miscellaneous", round(random.uniform(0.45, 0.65), 3)

def log_to_minio(txn, predicted, confidence):
    record = {**txn, "predicted_category": predicted,
              "confidence": confidence, "feedback": None,
              "logged_at": datetime.now().isoformat()}
    key = f"requests/{datetime.now().strftime('%Y/%m/%d')}/{txn['transaction_id']}.json"
    data = json.dumps(record).encode()
    client.put_object("production-logs", key, io.BytesIO(data), len(data))

print("="*55)
print("FIREFLY III DATA GENERATOR")
print("Simulating user transactions hitting ML service")
print("Press Ctrl+C to stop")
print("="*55)

count = 0
correct = 0
while True:
    base = random.choice(TRANSACTIONS)
    txn = {
        "transaction_id": str(uuid.uuid4()),
        "description": base["description"],
        "amount": round(base["amount"] * random.uniform(0.85, 1.15), 2),
        "currency": "USD",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "true_category": base["category"],
    }

    predicted, confidence = predict_category(txn["description"])
    log_to_minio(txn, predicted, confidence)

    match = predicted == txn["true_category"]
    if match: correct += 1
    count += 1
    accuracy = 100 * correct // count

    icon = "✓" if match else "✗"
    print(f"[{count:04d}] {icon} ${txn['amount']:7.2f} | "
          f"{txn['description'][:38]:<38} | "
          f"pred={predicted:<15} conf={confidence:.2f} | acc={accuracy}%")

    time.sleep(2)
