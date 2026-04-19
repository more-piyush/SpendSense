#!/usr/bin/env python3
"""
Online feature computation service for real-time inference (Q2.5).
POST /compute-features → returns feature vector for DistilBERT inference.
"""
import json, re, math
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler

USER_HISTORY = {}

def preprocess_description(description):
    text = description.lower()
    text = re.sub(r'\b\d{4,}\b', '[NUM]', text)
    text = re.sub(r'[*#@]', ' ', text)
    return ' '.join(text.split())

def compute_features(transaction):
    user_id = transaction.get("user_id", "default_user")
    amount  = float(transaction.get("amount", 0))
    desc    = transaction.get("description", "")
    date_str= transaction.get("date", datetime.now().strftime("%Y-%m-%d"))

    # Text features
    processed_text = preprocess_description(desc)
    log_amount     = math.log1p(amount)

    # Date features
    try:
        date = datetime.strptime(date_str, "%Y-%m-%d")
    except:
        date = datetime.now()
    month      = date.month
    dow        = date.weekday()
    is_weekend = 1 if dow >= 5 else 0

    # User history
    history = USER_HISTORY.get(user_id, {"transactions": [], "n": 0})
    n_past  = history["n"]
    history["transactions"].append({"amount": amount, "date": date_str})
    history["transactions"] = history["transactions"][-100:]
    history["n"] = n_past + 1
    USER_HISTORY[user_id] = history

    # Rolling amount features
    past_amounts = [t["amount"] for t in history["transactions"][:-1]]
    avg_past = sum(past_amounts) / len(past_amounts) if past_amounts else amount
    deviation = amount / max(avg_past, 1e-8)

    return {
        "processed_description": processed_text,
        "original_description":  desc,
        "amount":                amount,
        "log_amount":            round(log_amount, 4),
        "month_of_year":         month,
        "day_of_week":           dow,
        "is_weekend":            is_weekend,
        "n_past_transactions":   n_past,
        "avg_past_amount":       round(avg_past, 2),
        "deviation_from_avg":    round(deviation, 4),
        "user_id":               user_id,
        "feature_version":       "1.0.0",
        "computed_at":           datetime.now().isoformat(),
    }

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/compute-features":
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            try:
                txn      = json.loads(body)
                features = compute_features(txn)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(features, indent=2).encode())
            except Exception as e:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "service": "online-feature-service", "version": "1.0.0"}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {format % args}")

if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 8080), Handler)
    print("="*50)
    print("ONLINE FEATURE SERVICE running on port 8080")
    print("Endpoints:")
    print("  GET  /health")
    print("  POST /compute-features")
    print("="*50)
    server.serve_forever()
