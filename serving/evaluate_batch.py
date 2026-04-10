from locust import HttpUser, task, between
import json

BATCH = {
    "transactions": [
        {"transaction_id": f"txn_{i}", "description": "WHOLE FOODS MARKET #10847 AUSTIN TX",
         "amount": 87.43, "currency": "USD", "country": "US", "timestamp": "2026-03-15T14:32:18Z"}
        for i in range(8)
    ]
}

class BatchUser(HttpUser):
    wait_time = between(0.1, 0.3)
    @task
    def predict_batch(self):
        self.client.post('/predict_batch', json=BATCH,
                         headers={'Content-Type': 'application/json'})
