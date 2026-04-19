# training/Dockerfile
# Training container for Firefly III ML features
# Supports both DistilBERT (PyTorch) and XGBoost (scikit-learn) training
# All training runs are logged to MLflow

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies
RUN pip install --no-cache-dir \
    transformers==4.36.0 \
    datasets==2.16.0 \
    xgboost==2.0.3 \
    scikit-learn==1.3.2 \
    mlflow==3.9.0 \
    optuna==3.4.0 \
    pandas==2.1.4 \
    numpy==1.26.2 \
    pyarrow==14.0.2 \
    shap==0.44.0 \
    pyyaml==6.0.1 \
    psutil==5.9.7 \
    matplotlib==3.8.2 \
    psycopg2-binary==2.9.9 \
    requests==2.31.0 \
    boto3==1.28.64 \
    s3fs==2023.10.0 \
    fsspec==2023.10.0

# Copy training scripts and configs
COPY train_categorization.py .
COPY train_trend_detection.py .
COPY utils.py .
COPY data_pipeline.py .
COPY compute_features.py .
COPY promote_model.py .
COPY retrain.py .
COPY explain.py .
COPY generate_training_data.py .
COPY configs/ configs/

ENTRYPOINT ["python"]
