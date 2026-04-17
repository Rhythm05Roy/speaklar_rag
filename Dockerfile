# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for faiss-cpu and sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Pre-download embedding model into the image (avoids cold-download at runtime)
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('intfloat/multilingual-e5-small')"


# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local
# Copy pre-downloaded model cache
COPY --from=builder /root/.cache /root/.cache

# Non-root user for security
RUN groupadd -r speaklar && useradd -r -g speaklar speaklar \
    && chown -R speaklar:speaklar /app

# Copy source code
COPY --chown=speaklar:speaklar . .

# Data and index directories
RUN mkdir -p /app/data/indexes && chown -R speaklar:speaklar /app/data

USER speaklar

EXPOSE 8000

# Health-check using the /readiness endpoint
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/readiness || exit 1

ENTRYPOINT ["uvicorn", "api.main:app", \
    "--host", "0.0.0.0", \
    "--port", "8000", \
    "--workers", "1", \
    "--loop", "uvloop", \
    "--log-level", "info"]
