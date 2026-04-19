from fastapi import FastAPI
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast
import onnxruntime as ort
import numpy as np, time, os
from datetime import datetime, timezone
from typing import List
app = FastAPI()
LABELS = ['Groceries','Food & Dining','Transport','Shopping','Entertainment','Utilities','Healthcare','Travel']
MODEL_PATH = os.getenv('ONNX_MODEL_PATH', 'onnx_model/model.onnx')
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
sess = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
class TxInput(BaseModel):
    transaction_id: str
    description: str
    amount: float = 0.0
    currency: str = 'USD'
    country: str = 'US'
    timestamp: str = ''
class BatchInput(BaseModel):
    transactions: List[TxInput]
def run_inference(texts):
    enc = tokenizer(texts, return_tensors='np', padding=True, truncation=True, max_length=48)
    out = sess.run(None, {'input_ids': enc['input_ids'], 'attention_mask': enc['attention_mask']})
    scores = 1 / (1 + np.exp(-out[0]))  # sigmoid
    return scores
@app.get('/health')
def health(): return {'status': 'ok'}
@app.post('/predict')
def predict(tx: TxInput):
    t0 = time.time()
    scores = run_inference([tx.description])[0]
    latency_ms = round((time.time() - t0) * 1000, 2)
    results = [{'category': LABELS[i], 'confidence': round(float(scores[i]), 4)}
               for i in np.argsort(-scores) if scores[i] > 0.5]
    confidence = float(np.max(scores))
    return {
        'transaction_id': tx.transaction_id,
        'model': 'distilbert-categorization',
        'model_version': '1.2.0-onnx',
        'predicted_categories': results,
        'abstained': confidence < 0.7,
        'abstention_threshold': 0.7,
        'max_confidence': round(confidence, 4),
        'inference_time_ms': latency_ms,
        'timestamp': datetime.now(timezone.utc).isoformat()
    }
@app.post('/predict_batch')
def predict_batch(batch: BatchInput):
    t0 = time.time()
    texts = [tx.description for tx in batch.transactions]
    all_scores = run_inference(texts)
    latency_ms = round((time.time() - t0) * 1000, 2)
    results = []
    for tx, scores in zip(batch.transactions, all_scores):
        cats = [{'category': LABELS[i], 'confidence': round(float(scores[i]), 4)}
                for i in np.argsort(-scores) if scores[i] > 0.5]
        results.append({'transaction_id': tx.transaction_id, 'predicted_categories': cats,
                         'max_confidence': round(float(np.max(scores)), 4)})
    return {'results': results, 'batch_size': len(texts), 'inference_time_ms': latency_ms}
