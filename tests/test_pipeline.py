"""Tests for end-to-end pipeline orchestration.

Uses shared conftest fixtures for all dummy components. Tests verify:
  - Full pipeline returns valid response structure
  - Semantic cache is populated on first call (LLM runs once)
  - Second identical query is served from cache (LLM NOT called again)
  - Coreference resolved query is correctly passed to retrieval
"""
import asyncio
import numpy as np
import pytest

from api.pipeline import RAGPipeline
from tests.conftest import (
    DummySessionStore, DummyFAISSStore, DummyBM25Store,
    DummyCache, DummyLLM, DummyEmbedder,
)


async def _build_pipeline(monkeypatch) -> tuple[RAGPipeline, DummyCache, DummyLLM]:
    """Helper: build a pipeline with all external calls mocked."""
    async def fake_get_embedder():
        return DummyEmbedder()
    monkeypatch.setattr("api.pipeline.get_embedder", fake_get_embedder)

    cache = DummyCache()
    llm = DummyLLM()
    pipeline = RAGPipeline(
        session_store=DummySessionStore(),
        faiss_store=DummyFAISSStore(),
        bm25_store=DummyBM25Store(),
        cache=cache,
        llm_generator=llm,
    )
    return pipeline, cache, llm


@pytest.mark.asyncio
async def test_pipeline_returns_valid_response(monkeypatch):
    """Pipeline must return a non-empty response with valid fields."""
    pipeline, _, _ = await _build_pipeline(monkeypatch)
    result = await pipeline.process_query("session-1", "চালের দাম কত?")

    assert result.response == "চালের দাম ৭০ টাকা।"
    assert result.session_id == "session-1"
    assert result.query == "চালের দাম কত?"
    assert isinstance(result.latencies, dict)
    assert "total_ms" in result.latencies


@pytest.mark.asyncio
async def test_pipeline_caches_full_response(monkeypatch):
    """The pipeline caches the LLM response and reuses it on the second call."""
    pipeline, cache, llm = await _build_pipeline(monkeypatch)

    first = await pipeline.process_query("session-1", "চালের দাম কত?")
    # Allow fire-and-forget cache.set() task to complete
    await asyncio.sleep(0.02)
    second = await pipeline.process_query("session-1", "চালের দাম কত?")

    assert first.response == "চালের দাম ৭০ টাকা।"
    assert second.response == "চালের দাম ৭০ টাকা।"
    # LLM must only be called once — second call served from cache
    assert llm.calls == 1

    # Verify something was stored in the cache (key is normalized query)
    assert len(cache.payloads) > 0
    stored_payload = next(iter(cache.payloads.values()))
    assert stored_payload["response"] == "চালের দাম ৭০ টাকা।"


@pytest.mark.asyncio
async def test_pipeline_coref_resolved_flag(monkeypatch):
    """Elliptic Q2 after Q1 with an entity should set coref_resolved=True."""
    async def fake_get_embedder():
        return DummyEmbedder()
    monkeypatch.setattr("api.pipeline.get_embedder", fake_get_embedder)

    session = DummySessionStore()
    pipeline = RAGPipeline(
        session_store=session,
        faiss_store=DummyFAISSStore(),
        bm25_store=DummyBM25Store(),
        cache=DummyCache(),
        llm_generator=DummyLLM(),
    )

    # Q1 — establishes entity নুডুলস in session
    await pipeline.process_query("coref-sess", "নুডুলসের দাম কত?")
    await asyncio.sleep(0.01)  # let history write settle

    # Q2 — elliptic, should resolve to "নুডুলসের দাম কত?"
    result = await pipeline.process_query("coref-sess", "দাম কত?")
    assert result.coref_resolved is True


@pytest.mark.asyncio
async def test_pipeline_cache_miss_calls_llm(monkeypatch):
    """On a cache miss, LLM must be called exactly once."""
    pipeline, cache, llm = await _build_pipeline(monkeypatch)

    result = await pipeline.process_query("miss-sess", "তেলের দাম কত?")

    assert result.response == "চালের দাম ৭০ টাকা।"  # DummyLLM always returns this
    assert llm.calls == 1
    assert result.cache_hit is False
