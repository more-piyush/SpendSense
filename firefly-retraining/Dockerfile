FROM python:3.11-slim

WORKDIR /app

# Install Python deps first so they cache separately from source
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the pipeline source
COPY . .

ENV PYTHONUNBUFFERED=1 \
    FIREFLY_TMP=/tmp/firefly-retraining

# Default command just prints help — the cron entries invoke specific scripts
CMD ["python", "-c", "print('Use: python build_categorization_dataset.py  or  python build_trend_detection_dataset.py')"]
