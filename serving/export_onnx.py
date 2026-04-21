from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import DistilBertTokenizerFast
import os
print('Exporting DistilBERT to ONNX...')
model = ORTModelForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', export=True)
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
os.makedirs('onnx_model', exist_ok=True)
model.save_pretrained('onnx_model')
tokenizer.save_pretrained('onnx_model')
print('Done. ONNX model saved to onnx_model/')
