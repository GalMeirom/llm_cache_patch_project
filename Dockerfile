# syntax=docker/dockerfile:1
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# Install deps first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Project files
COPY . .

# Entrypoint checks a sentinel and runs apply_overlay only once
COPY docker/entrypoint.py /entrypoint.py
ENTRYPOINT ["python", "/entrypoint.py"]

# Default command (override in `docker run`)
CMD ["pytest", "-q"]
