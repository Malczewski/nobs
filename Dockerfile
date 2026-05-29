FROM python:3.12-slim

# Keep Python lean and unbuffered for proper container logging.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first to leverage Docker layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Persisted SQLite dedup DB lives here (mounted as a volume in compose).
RUN mkdir -p /data
ENV DB_PATH=/data/nobs.db

CMD ["python", "-m", "app.main"]
