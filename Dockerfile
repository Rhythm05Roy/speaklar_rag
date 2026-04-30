# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Create venv so runtime stage can copy it cleanly
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-bake ONNX model using the EXACT same code path as embedder.py
# This populates ~/.cache/huggingface with the exported ONNX artifacts.
# Without this, every container startup exports from scratch (~70s).
RUN python -c "\
import os; \
os.environ['TRANSFORMERS_CACHE'] = '/opt/hf-cache'; \
os.environ['HF_HOME'] = '/opt/hf-cache'; \
from sentence_transformers import SentenceTransformer; \
model = SentenceTransformer('intfloat/multilingual-e5-small', backend='onnx', \
    model_kwargs={'cache_folder': '/opt/hf-cache'}); \
model.max_seq_length = 64; \
_ = model.encode('warmup', convert_to_numpy=True); \
print('ONNX model pre-baked successfully')"


# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy venv and pre-baked model cache from builder
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/hf-cache /opt/hf-cache

# Point HuggingFace to the baked-in cache
ENV PATH="/opt/venv/bin:$PATH" \
    HF_HOME="/opt/hf-cache" \
    TRANSFORMERS_CACHE="/opt/hf-cache" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Non-root user
RUN groupadd -r speaklar && useradd -r -g speaklar speaklar \
    && chown -R speaklar:speaklar /app /opt/hf-cache

# Copy source code
COPY --chown=speaklar:speaklar . .

# Ensure data/indexes dir exists (will be overridden by volume mount)
RUN mkdir -p /app/data/indexes && chown -R speaklar:speaklar /app/data

USER speaklar

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:8000/readiness || exit 1

ENTRYPOINT ["uvicorn", "api.main:app", \
    "--host", "0.0.0.0", \
    "--port", "8000", \
    "--workers", "1", \
    "--loop", "uvloop", \
    "--log-level", "info", \
    "--no-access-log"]
