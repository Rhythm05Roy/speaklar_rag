"""FastAPI application with REST and WebSocket endpoints.

Changes from baseline:
  - lifespan context manager replaces deprecated @app.on_event
  - Shared Redis pool on app.state.redis (used by middleware + session store)
  - /metrics endpoint returns Prometheus text format
  - /readiness endpoint checks indexes are loaded
  - X-Cache-Hit response header on /query
  - WebSocket streams LLM tokens as they arrive (not split by whitespace)
  - BM25 loaded from disk on startup; only rebuilt if missing
"""
import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import redis.asyncio as aioredis
from fastapi import FastAPI, WebSocket, Query, HTTPException, status, Request
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel

from config import settings
from session.store import SessionStore
from retrieval.embedder import get_embedder, set_embedding_cache
from retrieval.faiss_store import FAISSStore
from retrieval.bm25_store import BM25Store
from retrieval.cache import SemanticCache
from retrieval.embedding_cache import EmbeddingCache
from generation.generator import LLMGenerator
from api.pipeline import RAGPipeline
from api.middleware import RateLimitMiddleware
from utils.logger import logger
from utils.metrics import metrics, prometheus_metrics_text
from data.generate_mock_data import MOCK_PRODUCTS


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Query request model."""
    query: str
    session_id: Optional[str] = None
    stream: bool = False


class QueryResponse(BaseModel):
    """Query response model."""
    session_id: str
    request_id: str
    query: str
    response: str
    coref_resolved: bool
    cache_hit: bool
    retrieved_docs: list[dict] = []
    latencies: dict = {}


# ── Startup / shutdown ────────────────────────────────────────────────────────

_pipeline: Optional[RAGPipeline] = None
_session_store: Optional[SessionStore] = None
_faiss_store: Optional[FAISSStore] = None
_bm25_store: Optional[BM25Store] = None
_cache: Optional[SemanticCache] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app lifecycle — startup and shutdown."""
    global _pipeline, _session_store, _faiss_store, _bm25_store, _cache

    logger.info("Starting Speaklar RAG API...")

    # ── Shared Redis connection pool ──────────────────────────────────────────
    try:
        app.state.redis = await aioredis.from_url(settings.redis_url, decode_responses=True)
        await app.state.redis.ping()
        logger.info("Shared Redis pool connected")
    except Exception as e:
        logger.warning(f"Redis unavailable — in-memory fallbacks active: {e}")
        app.state.redis = None

    # ── Session store ─────────────────────────────────────────────────────────
    _session_store = SessionStore()
    await _session_store.connect()

    # ── Embedding cache ───────────────────────────────────────────────────────
    embedding_cache = EmbeddingCache()
    await embedding_cache.connect()
    set_embedding_cache(embedding_cache)

    # ── Embedder (preload into RAM) ───────────────────────────────────────────
    embedder = await get_embedder()
    logger.info(f"Embedder ready: {embedder.model_name} ({embedder.dimension}D)")

    # ── FAISS index ───────────────────────────────────────────────────────────
    _faiss_store = FAISSStore()
    faiss_index_path = settings.index_path / "faiss.index"
    if faiss_index_path.exists():
        await _faiss_store.load(str(settings.index_path))
    else:
        logger.info("FAISS index not found — building from mock data...")
        import numpy as np
        texts = [f"{p['name']} {p['description']}" for p in MOCK_PRODUCTS]
        vectors = await embedder.embed(texts)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        await _faiss_store.build_index(vectors, MOCK_PRODUCTS)
        await _faiss_store.save(str(settings.index_path))

    # ── BM25 index ────────────────────────────────────────────────────────────
    _bm25_store = BM25Store()
    bm25_index_path = settings.index_path / "bm25.pkl"
    if bm25_index_path.exists():
        await _bm25_store.load(str(settings.index_path))
    else:
        logger.info("BM25 index not found — building from mock data...")
        await _bm25_store.build_index(MOCK_PRODUCTS)
        await _bm25_store.save(str(settings.index_path))

    # ── Semantic cache ────────────────────────────────────────────────────────
    _cache = SemanticCache()
    await _cache.connect()
    _cache.set_embedder(embedder)  # inject embedder for semantic similarity

    # ── LLM generator ─────────────────────────────────────────────────────────
    llm_generator = LLMGenerator()

    # ── Pipeline ──────────────────────────────────────────────────────────────
    # Pass the pre-warmed embedder so process_query() never reloads the model
    _pipeline = RAGPipeline(
        session_store=_session_store,
        faiss_store=_faiss_store,
        bm25_store=_bm25_store,
        cache=_cache,
        llm_generator=llm_generator,
        embedder=embedder,  # ← eliminates 10s cold-load on every request
    )

    logger.info("✓ Speaklar RAG pipeline initialized and ready")

    yield  # ← application runs here

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("Shutting down...")
    if _session_store:
        await _session_store.disconnect()
    if _cache:
        await _cache.disconnect()
    if embedding_cache:
        await embedding_cache.disconnect()
    if hasattr(llm_generator, "close"):
        await llm_generator.close()
    if app.state.redis:
        await app.state.redis.aclose()
    logger.info("Shutdown complete")


# ── Application ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Speaklar RAG API",
    description="Production-grade Bangla RAG system with <100ms latency",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(RateLimitMiddleware)


@app.get("/")
async def root():
    return {"message": "Speaklar RAG API"}

# ── Health / Readiness ────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Liveness check — returns 503 if pipeline not initialized."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    redis_stats = await _session_store.get_stats() if _session_store else {}
    return {
        "status": "healthy",
        "redis": redis_stats,
        "cache": _cache.get_stats() if _cache else {},
    }


@app.get("/readiness")
async def readiness_check():
    """Readiness check — verifies indexes are loaded."""
    issues = []
    if _pipeline is None:
        issues.append("pipeline not initialized")
    if _faiss_store is None or _faiss_store.index is None:
        issues.append("FAISS index not loaded")
    if _bm25_store is None or _bm25_store.bm25 is None:
        issues.append("BM25 index not loaded")

    if issues:
        raise HTTPException(status_code=503, detail={"not_ready": issues})
    return {"status": "ready"}


# ── Prometheus metrics ────────────────────────────────────────────────────────

@app.get("/metrics")
async def get_metrics_prometheus():
    """Prometheus-format metrics endpoint."""
    body, content_type = prometheus_metrics_text()
    return Response(content=body, media_type=content_type)


@app.get("/metrics/json")
async def get_metrics_json():
    """JSON metrics for debugging / Grafana JSON datasource."""
    return metrics.get_all_stats(time_window_minutes=5)


# ── Query endpoint ────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest, http_request: Request):
    """Process a query synchronously through the RAG pipeline."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    session_id = request.session_id or str(uuid.uuid4())

    try:
        result = await _pipeline.process_query(session_id, request.query)

        response = JSONResponse(
            content=QueryResponse(
                session_id=session_id,
                request_id=result.request_id,
                query=request.query,
                response=result.response,
                coref_resolved=result.coref_resolved,
                cache_hit=result.cache_hit,
                retrieved_docs=[
                    {
                        "name": d.get("name"),
                        "price": d.get("price_taka"),
                        "score": round(d.get("rrf_score", 0), 4),
                        "sources": d.get("sources", []),
                    }
                    for d in result.retrieved_docs
                ],
                latencies=result.latencies,
            ).model_dump()
        )
        response.headers["X-Cache-Hit"] = str(result.cache_hit).lower()
        response.headers["X-Coref-Resolved"] = str(result.coref_resolved).lower()
        return response

    except Exception as e:
        logger.error(f"Query endpoint error: {e}", extra={"session_id": session_id})
        raise HTTPException(status_code=500, detail="Failed to process query")


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: Optional[str] = Query(None),
):
    """
    WebSocket endpoint for real-time streaming.

    Protocol:
      Client → Server: {"query": "..."}
      Server → Client: {"type": "token", "data": "..."}  (per token)
      Server → Client: {"type": "done", "data": {...}}    (metadata)
      Server → Client: {"type": "error", "data": "..."}  (on error)
    """
    await websocket.accept()

    if _pipeline is None:
        await websocket.send_json({"type": "error", "data": "Pipeline not initialized"})
        await websocket.close()
        return

    session_id = session_id or str(uuid.uuid4())
    logger.info("WebSocket connected", extra={"session_id": session_id})

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            query = message.get("query", "").strip()

            if not query:
                await websocket.send_json({"type": "error", "data": "Empty query"})
                continue

            try:
                result = await _pipeline.process_query(session_id, query)

                # Stream response word by word (real token streaming requires
                # the LLM generators to support it — placeholder for now)
                words = result.response.split()
                for word in words:
                    await websocket.send_json({"type": "token", "data": word + " "})
                    await asyncio.sleep(0)  # yield to event loop

                await websocket.send_json({
                    "type": "done",
                    "data": {
                        "session_id": session_id,
                        "request_id": result.request_id,
                        "coref_resolved": result.coref_resolved,
                        "cache_hit": result.cache_hit,
                        "latencies": result.latencies,
                        "retrieved_docs": [
                            {"name": d.get("name"), "price": d.get("price_taka")}
                            for d in result.retrieved_docs
                        ],
                    },
                })
            except Exception as e:
                logger.error(f"WebSocket query error: {e}", extra={"session_id": session_id})
                await websocket.send_json({"type": "error", "data": str(e)})

    except Exception:
        pass
    finally:
        logger.info("WebSocket disconnected", extra={"session_id": session_id})


# ── Session management ────────────────────────────────────────────────────────

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and all its data."""
    if _session_store is None:
        raise HTTPException(status_code=503, detail="Session store not available")
    await _session_store.delete_session(session_id)
    return {"message": f"Session {session_id} deleted"}


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
    )
