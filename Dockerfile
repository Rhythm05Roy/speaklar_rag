# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build
ENV PYTHONPATH=/install/lib/python3.11/site-packages

# System deps for faiss-cpu, sentence-transformers, and ONNX runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Pre-download + export embedding model to ONNX (baked into image — no cold start)
RUN python -c "\
from optimum.onnxruntime import ORTModelForFeatureExtraction; \
from transformers import AutoTokenizer; \
model = ORTModelForFeatureExtraction.from_pretrained( \
    'intfloat/multilingual-e5-small', export=True); \
model.save_pretrained('/build/onnx_model'); \
AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small') \
    .save_pretrained('/build/onnx_model'); \
print('ONNX model exported')"


# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system deps (curl for healthcheck only)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages
COPY --from=builder /install /usr/local

# Copy pre-built ONNX model (eliminates 20s cold-export on startup)
COPY --from=builder /build/onnx_model /app/onnx_model

# Also copy HuggingFace cache (tokenizer files)
COPY --from=builder /root/.cache /root/.cache

# Non-root user for security
RUN groupadd -r speaklar && useradd -r -g speaklar speaklar \
    && chown -R speaklar:speaklar /app

# Copy application source
COPY --chown=speaklar:speaklar . .

# Data and index directories (will be mounted as volumes in production)
RUN mkdir -p /app/data/indexes && chown -R speaklar:speaklar /app/data

USER speaklar

EXPOSE 8000

# Health-check via /readiness endpoint
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:8000/readiness || exit 1

# uvloop for maximum async throughput
ENTRYPOINT ["uvicorn", "api.main:app", \
    "--host", "0.0.0.0", \
    "--port", "8000", \
    "--workers", "1", \
    "--loop", "uvloop", \
    "--log-level", "info", \
    "--no-access-log"]
