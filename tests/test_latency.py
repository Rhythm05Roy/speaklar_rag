"""Latency budget tests for the RAG pipeline.

Tests that the non-LLM path (context + embed + retrieval + fusion + prompt)
stays well under 30ms, leaving the full budget for the LLM call.

All LLM calls are mocked — these tests are pure latency regression guards.
"""
import time
import pytest
import numpy as np

from api.pipeline import RAGPipeline
from tests.conftest import (
    DummySessionStore, DummyFAISSStore, DummyBM25Store,
    DummyCache, DummyLLM, DummyEmbedder,
)


async def _make_pipeline(monkeypatch) -> RAGPipeline:
    """Build a pipeline with all external calls mocked."""
    async def fake_get_embedder():
        return DummyEmbedder()

    monkeypatch.setattr("api.pipeline.get_embedder", fake_get_embedder)

    return RAGPipeline(
        session_store=DummySessionStore(),
        faiss_store=DummyFAISSStore(),
        bm25_store=DummyBM25Store(),
        cache=DummyCache(),
        llm_generator=DummyLLM(response="চালের দাম ৭০ টাকা।"),
    )


@pytest.mark.asyncio
async def test_non_llm_path_under_30ms(monkeypatch):
    """
    The entire pipeline EXCLUDING LLM generation should stay under 30ms.

    Even with a ~3ms embed, ~8ms retrieval, the non-LLM work budget is <30ms.
    This leaves ~70ms for the LLM call within the 100ms SLA.
    """
    pipeline = await _make_pipeline(monkeypatch)

    t0 = time.perf_counter()
    result = await pipeline.process_query("latency-test", "চালের দাম কত?")
    total_ms = (time.perf_counter() - t0) * 1000

    non_llm_ms = total_ms - result.latencies.get("llm_ms", 0)
    assert non_llm_ms < 30, (
        f"Non-LLM pipeline took {non_llm_ms:.1f}ms — exceeds 30ms budget. "
        f"Breakdown: {result.latencies}"
    )


@pytest.mark.asyncio
async def test_context_resolution_under_5ms(monkeypatch):
    """Context resolution stage must complete in under 5ms."""
    pipeline = await _make_pipeline(monkeypatch)
    result = await pipeline.process_query("ctx-latency", "দাম কত?")

    context_ms = result.latencies.get("context_ms", 999)
    assert context_ms < 5.0, (
        f"Context resolution took {context_ms:.1f}ms — exceeds 5ms budget"
    )


@pytest.mark.asyncio
async def test_retrieval_under_15ms(monkeypatch):
    """Parallel FAISS + BM25 retrieval must complete in under 15ms."""
    pipeline = await _make_pipeline(monkeypatch)
    result = await pipeline.process_query("ret-latency", "চালের দাম কত?")

    retrieval_ms = result.latencies.get("retrieval_ms", 999)
    assert retrieval_ms < 15.0, (
        f"Retrieval took {retrieval_ms:.1f}ms — exceeds 15ms budget"
    )


@pytest.mark.asyncio
async def test_cache_hit_under_5ms(monkeypatch):
    """A semantic cache hit must return in under 5ms (skips retrieval + LLM)."""
    pipeline = await _make_pipeline(monkeypatch)

    # First call to populate cache
    await pipeline.process_query("cache-test", "চালের দাম কত?")

    # Wait briefly for fire-and-forget cache write (create_task) to complete
    import asyncio
    await asyncio.sleep(0.01)

    # Second call — should be a cache hit
    t0 = time.perf_counter()
    result = await pipeline.process_query("cache-test", "চালের দাম কত?")
    hit_ms = (time.perf_counter() - t0) * 1000

    if result.cache_hit:
        assert hit_ms < 5.0, f"Cache HIT took {hit_ms:.1f}ms — expected under 5ms"


@pytest.mark.asyncio
async def test_pipeline_returns_valid_response_structure(monkeypatch):
    """Pipeline must always return a structurally valid RAGResponse."""
    pipeline = await _make_pipeline(monkeypatch)
    result = await pipeline.process_query("struct-test", "নুডুলসের দাম কত?")

    assert result.query == "নুডুলসের দাম কত?"
    assert isinstance(result.response, str) and len(result.response) > 0
    assert isinstance(result.retrieved_docs, list)
    assert isinstance(result.latencies, dict)
    assert "total_ms" in result.latencies
    assert isinstance(result.coref_resolved, bool)
    assert isinstance(result.cache_hit, bool)
    assert isinstance(result.request_id, str)


@pytest.mark.asyncio
async def test_pipeline_caches_response_and_skips_llm(monkeypatch):
    """On second identical query, LLM must NOT be called again."""
    import asyncio

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

    first = await pipeline.process_query("sess-1", "চালের দাম কত?")
    await asyncio.sleep(0.01)  # allow fire-and-forget tasks to settle
    second = await pipeline.process_query("sess-1", "চালের দাম কত?")

    assert first.response == "চালের দাম ৭০ টাকা।"
    assert second.response == "চালের দাম ৭০ টাকা।"
    assert llm.calls == 1  # second call served from cache
