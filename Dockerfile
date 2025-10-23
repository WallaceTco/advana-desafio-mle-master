# Dockerfile
FROM python:3.10-slim

# Avoid creating .pyc files and use unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Working directory inside the container
WORKDIR /app

# Minimal system dependencies (for building/scikit-learn/numpy/pandas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (to leverage Docker layer caching)
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy the rest of the project (includes challenge/, tests/, data/, Makefile, etc.)
COPY . .

# Cloud Run exposes the port in the $PORT env var; default to 8080
ENV PORT=8080

# Startup command: run FastAPI with Uvicorn
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]