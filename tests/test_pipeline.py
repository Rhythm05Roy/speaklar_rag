"""Tests for the end-to-end pipeline orchestration."""
import numpy as np
import pytest

from api.pipeline import RAGPipeline


class DummySessionStore:
    """Minimal async session store for pipeline tests."""

    def __init__(self):
        self.history = {}
        self.entities = {}

    async def append_to_history(self, session_id, turn):
        self.history.setdefault(session_id, []).append(turn)

    async def get_last_entities(self, session_id):
        return self.entities.get(session_id)

    async def set_last_entities(self, session_id, entities):
        self.entities[session_id] = entities


class DummyFAISSStore:
    """Static FAISS stub."""

    async def search(self, query_vector, top_k=5):
        return [
            {"id": "p1", "name": "চাল", "price_taka": 70, "category": "খাদ্য", "description": "মিনিকেট চাল", "score": 0.9}
        ][:top_k]


class DummyBM25Store:
    """Static BM25 stub."""

    async def search(self, query, top_k=5):
        return [
            {"id": "p1", "name": "চাল", "price_taka": 70, "category": "খাদ্য", "description": "মিনিকেট চাল", "score": 0.8}
        ][:top_k]


class DummyCache:
    """In-memory cache stub with the production interface."""

    def __init__(self):
        self.payloads = {}

    async def get(self, query):
        return self.payloads.get(query)

    async def set(self, query, payload):
        self.payloads[query] = payload


class DummyLLM:
    """Static LLM stub with call counting."""

    def __init__(self):
        self.calls = 0

    async def generate(self, system_prompt, user_prompt, stream=False):
        self.calls += 1
        return "চালের দাম ৭০ টাকা।"


class DummyEmbedder:
    """Static embedder stub."""

    async def embed(self, texts):
        return np.ones(384, dtype=np.float32)


@pytest.mark.asyncio
async def test_pipeline_caches_full_response(monkeypatch):
    """The pipeline should cache and reuse the generated response."""
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

    first = await pipeline.process_query("session-1", "চালের দাম কত?")
    second = await pipeline.process_query("session-1", "চালের দাম কত?")

    assert first.response == "চালের দাম ৭০ টাকা।"
    assert second.response == "চালের দাম ৭০ টাকা।"
    assert llm.calls == 1
    assert cache.payloads["চালের দাম কত?"]["response"] == "চালের দাম ৭০ টাকা।"
