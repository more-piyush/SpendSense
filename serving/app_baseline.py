from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from datetime import datetime, timezone
import time, os
app = FastAPI()
LABELS = ['Groceries','Food & Dining','Transport','Shopping','Entertainment','Utilities','Healthcare','Travel']
classifier = pipeline('text-classification', model='distilbert-base-uncased',
    return_all_scores=True, device=-1)
class TxInput(BaseModel):
    transaction_id: str
    description: str
    amount: float = 0.0
    currency: str = 'USD'
    country: str = 'US'
    timestamp: str = ''
@app.get('/health')
def health(): return {'status': 'ok'}
@app.post('/predict')
def predict(tx: TxInput):
    t0 = time.time()
    raw = classifier(tx.description)[0]
    latency_ms = round((time.time() - t0) * 1000, 2)
    scored = sorted(zip(LABELS, [r['score'] for r in raw]), key=lambda x: -x[1])
    results = [{'category': c, 'confidence': round(s, 4)} for c, s in scored if s > 0.5]
    confidence = scored[0][1] if scored else 0.0
    return {
        'transaction_id': tx.transaction_id,
        'model': 'distilbert-categorization',
        'model_version': '1.2.0',
        'predicted_categories': results,
        'abstained': confidence < 0.7,
        'abstention_threshold': 0.7,
        'max_confidence': round(confidence, 4),
        'inference_time_ms': latency_ms,
        'timestamp': datetime.now(timezone.utc).isoformat()
    }
