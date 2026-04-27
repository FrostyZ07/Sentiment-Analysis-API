# ════════════════════════════════════════════════════════
# Stage 1: Builder — install dependencies
# ════════════════════════════════════════════════════════
FROM python:3.11-slim AS builder

WORKDIR /install

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# ════════════════════════════════════════════════════════
# Stage 2: Runtime — minimal final image
# ════════════════════════════════════════════════════════
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install/lib /usr/local/lib
COPY --from=builder /install/bin /usr/local/bin

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/models && \
    chown -R appuser:appuser /app

# Copy application source
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY models/ ./models/
COPY static/ ./static/


USER appuser

EXPOSE 8000

ENV PORT=8000 \
    MODEL_PATH=/app/models/distilbert-sentiment \
    LOG_LEVEL=INFO

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
