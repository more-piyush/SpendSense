from locust import HttpUser, task, between
import json
SAMPLE = {
    'transaction_id': 'txn_20260315_0042',
    'description': 'WHOLE FOODS MARKET #10847 AUSTIN TX',
    'amount': 87.43,
    'currency': 'USD',
    'country': 'US',
    'timestamp': '2026-03-15T14:32:18Z'
}
class ServingUser(HttpUser):
    wait_time = between(0.1, 0.3)
    @task
    def predict(self):
        self.client.post('/predict', json=SAMPLE,
                         headers={'Content-Type': 'application/json'})
