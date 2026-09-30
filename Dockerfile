# Use lightweight Python 3.10 base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CHROMA_PERSIST_DIRECTORY=/app/chroma_db

# Set working directory
WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and scripts
COPY app/ /app/app/
COPY scripts/ /app/scripts/
COPY data/ /app/data/
COPY chroma_db/ /app/chroma_db/
COPY .env.example /app/.env

# Pre-download and run ingestion during build (optional/persistable)
RUN python scripts/ingest.py

# Expose port
EXPOSE 8000

# Run FastAPI app with Uvicorn server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
