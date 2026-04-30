# Speaklar Bangla RAG

Production-oriented Bangla retrieval-augmented assistant for product search and multi-turn catalog QA. The system is built around a low-latency deterministic hot path for factual product questions, with hybrid retrieval and LLM generation reserved for queries that cannot be answered safely from structured catalog signals.

## What This System Does

- Serves Bangla product queries over a FastAPI API.
- Resolves short follow-up turns such as `দাম কত টাকা?` using prior conversational context.
- Answers exact product and product-group questions deterministically when possible.
- Falls back to hybrid retrieval plus LLM generation for open-ended or unsupported queries.
- Exposes health, readiness, metrics, session management, and WebSocket interfaces.

This codebase is optimized for the assessment-style requirement:

1. Q1: `আপনাদের কোম্পানি কি নুডুলস বিক্রি করে?`
2. Q2: `দাম কত টাকা?`

The intended behavior is:

- Q1 stores `নুডুলস` as structured conversational context.
- Q2 resolves against that context.
- Q2 is answered from an in-memory grouped catalog lookup without calling the embedder or the LLM.

## Architecture

The runtime is intentionally split into two answer paths.

### 1. Deterministic Hot Path

Used for factual queries that can be answered directly from catalog metadata:

- exact product price
- exact product brand
- exact product size / pack
- exact product category
- product-group existence queries
- product-group price range queries
- context-aware follow-up questions that resolve to one of the above

Examples:

- `মিল্কভিটা নুডুলস ৫০০ গ্রাম দাম কত?`
- `মিল্কভিটা নুডুলস ৫০০ গ্রামের ব্র্যান্ড কী?`
- `আপনাদের কোম্পানি কি নুডুলস বিক্রি করে?`
- `দাম কত টাকা?` after a prior noodle turn

This path runs before embedding, FAISS, BM25, semantic cache, and LLM generation.

### 2. Retrieval + LLM Fallback Path

Used only when the system cannot safely answer deterministically:

- open-ended product questions
- unsupported attributes
- broad conversational requests
- weak or ambiguous matches that require synthesis

Fallback flow:

1. query embedding
2. semantic cache lookup
3. FAISS vector search + BM25 lexical search in parallel
4. reciprocal rank fusion
5. conservative deterministic rendering from retrieved docs when possible
6. prompt construction
7. primary LLM with fallback provider chain

## High-Level Request Flow

### Deterministic exact/group flow

```text
request
  -> context resolver
  -> structured session context lookup
  -> deterministic responder
  -> template/group response
```

### Fallback RAG flow

```text
request
  -> context resolver
  -> structured session context lookup
  -> deterministic responder miss
  -> embed query
  -> semantic cache
  -> FAISS + BM25
  -> RRF fusion
  -> deterministic render from retrieved docs OR LLM generation
```

## Core Components

### API layer

- [api/main.py](api/main.py)
  FastAPI app, startup lifecycle, REST endpoints, WebSocket endpoint, readiness and metrics.
- [api/pipeline.py](api/pipeline.py)
  End-to-end orchestrator with deterministic pre-embedding routing.
- [api/middleware.py](api/middleware.py)
  Request IDs, latency headers, and Redis-backed rate limiting.

### Context and conversation state

- [context/resolver.py](context/resolver.py)
  Bangla coreference resolution and short-follow-up rewriting.
- [context/rewriter.py](context/rewriter.py)
  Query rewriting utilities.
- [session/store.py](session/store.py)
  Redis-backed session history, last entities, and structured deterministic target context.

### Deterministic answer engine

- [generation/deterministic.py](generation/deterministic.py)
  Intent detection, exact-name matching, grouped product-type lookups, range answers, and follow-up resolution from stored target context.
- [utils/product_metadata.py](utils/product_metadata.py)
  Offline and runtime metadata enrichment for brand, product type, and pack size extraction.

### Retrieval

- [retrieval/embedder.py](retrieval/embedder.py)
  Singleton multilingual E5 embedder with ONNX-first loading.
- [retrieval/faiss_store.py](retrieval/faiss_store.py)
  FAISS cosine-similarity search with persistence.
- [retrieval/bm25_store.py](retrieval/bm25_store.py)
  Bangla BM25 lexical search with persistence.
- [retrieval/cache.py](retrieval/cache.py)
  Two-tier semantic cache: in-process cosine similarity plus Redis exact-match.
- [retrieval/fusion.py](retrieval/fusion.py)
  Reciprocal rank fusion over FAISS and BM25 results.

### Generation and observability

- [generation/generator.py](generation/generator.py)
  Prompt builder and LLM provider facade for Groq, Gemini, and OpenAI.
- [utils/metrics.py](utils/metrics.py)
  In-memory metrics and Prometheus export.
- [utils/logger.py](utils/logger.py)
  Structured application logging.

### Offline data and indexing

- [indexer/pipeline.py](indexer/pipeline.py)
  Batch normalization, enrichment, embedding, and index persistence.
- [data/products.csv](data/products.csv)
  Product dataset.
- [data/generate_mock_data.py](data/generate_mock_data.py)
  Mock data generation utilities.

## Product Resolution Policy

The code intentionally distinguishes between exact product queries and broader product-group queries.

- Exact query -> exact answer
- Partial grouped query -> narrowed grouped answer
- Broad product-group query -> group-level answer, typically a price range
- Unsupported attribute -> deterministic unavailable response or fallback
- Open-ended conversational query -> LLM path

Examples:

- `মিল্কভিটা নুডুলস ৫০০ গ্রাম দাম কত?`
  -> `মিল্কভিটা নুডুলস ৫০০ গ্রামের দাম ৫৬ টাকা।`
- `নুডুলসের দাম কত?`
  -> `নুডুলসের দাম ৫৬ টাকা থেকে ৪১৬ টাকা পর্যন্ত।`
- `আপনাদের কোম্পানি কি নুডুলস বিক্রি করে?`
  -> `হ্যাঁ, নুডুলস বিক্রি করা হয়।`

This avoids the common failure mode where a broad query randomly returns the price of a single SKU.

## Setup

### Prerequisites

- Python 3.10+
- Redis
- One configured LLM provider for fallback generation:
  - `GROQ_API_KEY`, or
  - `OPENAI_API_KEY`, or
  - `GEMINI_API_KEY`

### Local environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file or update the existing one. At minimum, configure:

```bash
GROQ_API_KEY=...
OPENAI_API_KEY=...
GEMINI_API_KEY=...
LLM_PRIMARY=groq
LLM_FALLBACK=openai
REDIS_URL=redis://localhost:6379/0
DATA_DIR=./data
INDEX_DIR=./data/indexes
```

### Start infrastructure

```bash
docker-compose up -d redis
```

Or start the full stack:

```bash
docker-compose up -d
```

This compose file includes:

- Redis
- API service
- Prometheus
- Grafana

### Build indexes

If indexes are not already present:

```bash
python indexer/pipeline.py --data-path ./data/products.csv --index-dir ./data/indexes
```

### Run the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Or:

```bash
make run
```

## API

### `POST /query`

Request:

```json
{
  "query": "নুডুলসের দাম কত?",
  "session_id": "demo-session"
}
```

Response shape:

```json
{
  "session_id": "demo-session",
  "request_id": "4aa74369",
  "query": "নুডুলসের দাম কত?",
  "response": "নুডুলসের দাম ৫৬ টাকা থেকে ৪১৬ টাকা পর্যন্ত।",
  "coref_resolved": false,
  "cache_hit": false,
  "retrieved_docs": [
    {
      "name": "রাধুনী নুডুলস ১ কেজি",
      "price": 110,
      "score": 0.0323,
      "sources": ["faiss", "bm25"]
    }
  ],
  "latencies": {
    "context_ms": 2.1,
    "exact_lookup_ms": 1.4,
    "embed_ms": 0.0,
    "cache_ms": 0.0,
    "retrieval_ms": 0.0,
    "fusion_ms": 0.0,
    "template_ms": 1.4,
    "llm_ms": 0.0,
    "total_ms": 4.3
  }
}
```

### `GET /health`

Basic liveness plus Redis and cache stats.

### `GET /readiness`

Readiness check that confirms the API pipeline and search indexes are loaded.

### `GET /metrics`

Prometheus-format metrics endpoint.

### `GET /metrics/json`

JSON-formatted in-memory metrics snapshot.

### `DELETE /session/{session_id}`

Deletes session history, last entities, and structured target context.

### `WebSocket /ws`

Bidirectional endpoint for interactive use. The current implementation streams the already-built final response chunked by word; it is not yet true token streaming from the underlying LLM provider.

## Example Conversations

### Requirement-style context-aware flow

```text
Q1: আপনাদের কোম্পানি কি নুডুলস বিক্রি করে?
A1: হ্যাঁ, নুডুলস বিক্রি করা হয়।

Q2: দাম কত টাকা?
A2: নুডুলসের দাম ৫৬ টাকা থেকে ৪১৬ টাকা পর্যন্ত।
```

### Exact product flow

```text
Q: মিল্কভিটা নুডুলস ৫০০ গ্রাম দাম কত?
A: মিল্কভিটা নুডুলস ৫০০ গ্রামের দাম ৫৬ টাকা।
```

### Fallback LLM flow

```text
Q: ভালো নুডুলস কোনটা?
A: retrieval + LLM path
```

## Performance Model

This repository has two different latency profiles.

### Deterministic path

Designed for:

- exact factual product lookups
- grouped product-type existence and price questions
- context-aware follow-up lookups

Expected behavior:

- no embedder call
- no FAISS/BM25 call
- no LLM call

This is the path intended to satisfy the sub-100ms requirement.

### Retrieval / LLM path

Used for:

- open-ended generation
- unsupported fields
- conversational questions
- ambiguous cases that cannot be safely answered deterministically

This path is more expensive and should not be treated as a hard sub-100ms SLA path in production.

## Configuration

Settings are defined in [config.py](config.py). Important variables include:

```bash
GROQ_API_KEY=
GROQ_MODEL=llama-3.1-8b-instant
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.0-flash
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
LLM_PRIMARY=groq
LLM_FALLBACK=openai

REDIS_URL=redis://localhost:6379/0
REDIS_CACHE_TTL=3600
REDIS_SESSION_TTL=3600

CACHE_SIMILARITY_THRESHOLD=0.98
SEMANTIC_CACHE_MAX_ENTRIES=2000

API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false
LOG_LEVEL=INFO

DATA_DIR=./data
INDEX_DIR=./data/indexes
```

## Development

### Common commands

```bash
make install
make docker-up
make run
make test
make lint
make format
```

### Tests

The repository includes unit and pipeline coverage for:

- context resolution
- deterministic answer routing
- retrieval logic
- latency-oriented orchestration
- LLM fallback behavior

Run:

```bash
pytest tests/ -v --cov=. --cov-report=term-missing
```

For environments where the local `.env` contains non-boolean debug values, override `DEBUG` when running tests:

```bash
DEBUG=false pytest tests/ -v
```

### Validation and benchmarking

- [bootstrap.py](bootstrap.py)
  environment checks and quickstart validation
- [benchmark.py](benchmark.py)
  assessment-oriented benchmark script
- [validate_fixes.py](validate_fixes.py)
  static validation helper
- [validate_static.py](validate_static.py)
  additional validation helper

## Deployment Notes

- The Docker image pre-bakes the ONNX embedding model to avoid export-on-startup delays.
- The compose stack mounts `./data` into the API container so indexes can survive restarts.
- Readiness depends on both FAISS and BM25 indexes being available.
- Redis failures degrade gracefully to in-memory behavior for sessions and semantic caching.
- One worker is used in the container entrypoint to keep model memory predictable.

## Current Limitations

- Broad product-group answers are intentionally conservative and mainly implemented for sell/existence and price queries.
- The WebSocket endpoint is response-chunk streaming, not provider-native token streaming.
- The deterministic path depends on metadata quality extracted from semi-structured product names and descriptions.
- LLM fallback quality and latency depend on provider configuration and network conditions.

## License

This repository is distributed under the MIT License per [pyproject.toml](pyproject.toml).
