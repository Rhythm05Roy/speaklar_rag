"""Tests for end-to-end pipeline orchestration."""
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
    """The pipeline caches deterministic responses and reuses them."""
    pipeline, cache, llm = await _build_pipeline(monkeypatch)

    first = await pipeline.process_query("session-1", "চালের দাম কত?")
    # Allow fire-and-forget cache.set() task to complete
    await asyncio.sleep(0.02)
    second = await pipeline.process_query("session-1", "চালের দাম কত?")

    assert first.response == "চালের দাম ৭০ টাকা।"
    assert second.response == "চালের দাম ৭০ টাকা।"
    # Deterministic price lookup should skip the LLM entirely
    assert llm.calls == 0

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
    """Unsupported intents should fall back to the LLM on cache miss."""
    pipeline, cache, llm = await _build_pipeline(monkeypatch)

    result = await pipeline.process_query("miss-sess", "চাল সম্পর্কে বলো")

    assert result.response == "চালের দাম ৭০ টাকা।"  # DummyLLM always returns this
    assert llm.calls == 1
    assert result.cache_hit is False


@pytest.mark.asyncio
async def test_pipeline_skips_embedding_for_exact_product_query(monkeypatch):
    """Exact deterministic queries should return before the embedder runs."""
    async def exploding_get_embedder():
        class ExplodingEmbedder:
            async def embed(self, texts):
                raise AssertionError("Embedder should not run on exact deterministic query")

        return ExplodingEmbedder()

    monkeypatch.setattr("api.pipeline.get_embedder", exploding_get_embedder)

    pipeline = RAGPipeline(
        session_store=DummySessionStore(),
        faiss_store=DummyFAISSStore(),
        bm25_store=DummyBM25Store(),
        cache=DummyCache(),
        llm_generator=DummyLLM(),
    )

    result = await pipeline.process_query("exact-sess", "মিল্কভিটা কালিজিরা চাল ৫ কেজি দাম কত")

    assert result.response == "মিল্কভিটা কালিজিরা চাল ৫ কেজির দাম ৬৭০ টাকা।"
    assert result.latencies["embed_ms"] == 0.0
    assert result.latencies["llm_ms"] == 0.0


@pytest.mark.asyncio
async def test_pipeline_sell_query_uses_exact_fast_path(monkeypatch):
    """Catalog existence queries should use the exact deterministic fast path."""
    pipeline, _, llm = await _build_pipeline(monkeypatch)

    result = await pipeline.process_query("sell-sess", "আপনাদের কোম্পানি কি নুডুলস বিক্রি করে?")

    assert result.response == "হ্যাঁ, নুডুলস বিক্রি করা হয়।"
    assert result.latencies["embed_ms"] == 0.0
    assert llm.calls == 0


@pytest.mark.asyncio
async def test_pipeline_q2_follow_up_price_uses_group_fast_path(monkeypatch):
    """Requirement path: Q2 should resolve from context and avoid embedding/LLM."""
    async def exploding_get_embedder():
        class ExplodingEmbedder:
            async def embed(self, texts):
                raise AssertionError("Embedder should not run on deterministic Q2 follow-up")

        return ExplodingEmbedder()

    monkeypatch.setattr("api.pipeline.get_embedder", exploding_get_embedder)

    session = DummySessionStore()
    llm = DummyLLM()
    pipeline = RAGPipeline(
        session_store=session,
        faiss_store=DummyFAISSStore(),
        bm25_store=DummyBM25Store(),
        cache=DummyCache(),
        llm_generator=llm,
    )

    q1 = await pipeline.process_query("req-sess", "আপনাদের কোম্পানি কি নুডুলস বিক্রি করে?")
    q2 = await pipeline.process_query("req-sess", "দাম কত টাকা?")
    await asyncio.sleep(0.01)

    assert q1.response == "হ্যাঁ, নুডুলস বিক্রি করা হয়।"
    assert q2.coref_resolved is True
    assert q2.response == "নুডুলসের দাম ৫৬ টাকা থেকে ৪১৬ টাকা পর্যন্ত।"
    assert q2.latencies["embed_ms"] == 0.0
    assert q2.latencies["llm_ms"] == 0.0
    assert llm.calls == 0
    assert session.context["req-sess"]["product_type"] == "নুডুলস"
