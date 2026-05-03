FROM python:3.12-slim

WORKDIR /app

# system deps for sklearn / numpy wheels (slim base lacks libgomp etc.)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# only ship what the app needs at runtime — the recommender uses the tabular
# XGBoost model (compact) and the trends lookup. The 1.5GB sparse matrices
# stay out of the image; they're only needed for retraining.
COPY src/ ./src/
COPY app/ ./app/
COPY data/X_train_clean.parquet data/X_test_clean.parquet ./data/
COPY data/y_train.parquet data/y_test.parquet ./data/
COPY data/kickstarter_clean.parquet ./data/
COPY data/trend_lookup.pkl ./data/
COPY models/xgb_tabular.json ./models/

EXPOSE 8501

# health endpoint for orchestrators
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health').read()"

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
