"""Test configuration and fixtures."""
import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from session.store import SessionStore
from retrieval.faiss_store import FAISSStore
from retrieval.bm25_store import BM25Store
from retrieval.cache import SemanticCache


# ── Event loop ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def event_loop():
    """Session-scoped event loop to avoid re-creating per test."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ── Session store ─────────────────────────────────────────────────────────────

@pytest.fixture
async def session_store():
    """In-memory session store (no Redis required for tests)."""
    store = SessionStore(redis_url="redis://localhost:16379/0")  # intentionally wrong port
    await store.connect()  # will fall back to in-memory
    yield store
    await store.disconnect()


# ── FAISS store ───────────────────────────────────────────────────────────────

@pytest.fixture
async def faiss_store():
    """FAISS store seeded with 100 mock product vectors."""
    store = FAISSStore()
    vectors = np.random.rand(100, 384).astype(np.float32)
    products = [
        {
            "id": f"prod_{i:03d}",
            "name": f"পণ্য {i}",
            "category": "টেস্ট",
            "description": f"বিবরণ {i}",
            "price_taka": 50 + i,
        }
        for i in range(100)
    ]
    await store.build_index(vectors, products)
    yield store


# ── BM25 store ────────────────────────────────────────────────────────────────

@pytest.fixture
async def bm25_store():
    """BM25 store seeded with Bangla product names."""
    store = BM25Store()
    products = [
        {
            "id": f"prod_{i:03d}",
            "name": f"নুডুলস {i}",
            "category": "খাদ্য",
            "description": f"ডিম নুডুলস বিস্কুট {i}",
            "price_taka": 60 + i,
        }
        for i in range(50)
    ]
    await store.build_index(products)
    yield store


# ── Semantic cache ────────────────────────────────────────────────────────────

@pytest.fixture
async def semantic_cache():
    """Semantic cache with in-memory fallback."""
    cache = SemanticCache(redis_url="redis://localhost:16379/0")  # intentionally wrong port
    await cache.connect()
    yield cache
    await cache.clear()
    await cache.disconnect()


# ── Dummy components for pipeline tests ──────────────────────────────────────

class DummySessionStore:
    def __init__(self):
        self.history: dict = {}
        self.entities: dict = {}
        self.context: dict = {}
        self._use_in_memory = True

    async def append_to_history(self, sid, turn):
        self.history.setdefault(sid, []).append(turn)

    async def get_history(self, sid):
        return self.history.get(sid, [])

    async def get_last_entities(self, sid):
        return self.entities.get(sid)

    async def set_last_entities(self, sid, entities):
        self.entities[sid] = entities

    async def get_session_context(self, sid):
        return self.history.get(sid, []), self.entities.get(sid)

    async def set_last_context(self, sid, context):
        self.context[sid] = context

    async def get_last_context(self, sid):
        return self.context.get(sid)


class DummyFAISSStore:
    def __init__(self):
        self.products = {
            "p1": {
                "id": "p1",
                "name": "চাল",
                "price_taka": 70,
                "category": "খাদ্য",
                "description": "মিনিকেট চাল",
            },
            "prod_exact": {
                "id": "prod_exact",
                "name": "মিল্কভিটা কালিজিরা চাল ৫ কেজি",
                "price_taka": 670,
                "category": "খাদ্যশস্য",
                "description": "সুগন্ধী কালিজিরা চাল, ৫ কেজি, মিল্কভিটা ব্র্যান্ড",
            },
            "prod_noodles": {
                "id": "prod_noodles",
                "name": "রাধুনী নুডুলস ১ কেজি",
                "price_taka": 110,
                "category": "তাৎক্ষণিক খাদ্য",
                "description": "ডিম নুডুলস, ১ কেজি, রাধুনী ব্র্যান্ড",
            },
            "prod_noodles_2": {
                "id": "prod_noodles_2",
                "name": "মিল্কভিটা নুডুলস ৫০০ গ্রাম",
                "price_taka": 56,
                "category": "তাৎক্ষণিক খাদ্য",
                "description": "ডিম নুডুলস, ৫০০ গ্রাম, মিল্কভিটা ব্র্যান্ড",
            },
            "prod_noodles_3": {
                "id": "prod_noodles_3",
                "name": "সুরভি নুডুলস ৫ কেজি",
                "price_taka": 416,
                "category": "তাৎক্ষণিক খাদ্য",
                "description": "ডিম নুডুলস, ৫ কেজি, সুরভি ব্র্যান্ড",
            },
        }

    async def search(self, query_vector, top_k=5):
        return [
            {"id": "p1", "name": "চাল", "price_taka": 70, "category": "খাদ্য",
             "description": "মিনিকেট চাল", "score": 0.9}
        ]


class DummyBM25Store:
    async def search(self, query, top_k=5):
        return [
            {"id": "p1", "name": "চাল", "price_taka": 70, "category": "খাদ্য",
             "description": "মিনিকেট চাল", "score": 0.8}
        ]


class DummyCache:
    def __init__(self):
        self.payloads: dict = {}
        self.call_count = 0

    async def get(self, query, query_embedding=None):
        self.call_count += 1
        return self.payloads.get(query)

    async def set(self, query, payload, query_embedding=None):
        self.payloads[query] = payload


class DummyLLM:
    def __init__(self, response: str = "চালের দাম ৭০ টাকা।"):
        self.calls = 0
        self._response = response

    async def generate(self, system_prompt, user_prompt, stream=False):
        self.calls += 1
        return self._response


class DummyTimeoutLLM:
    """LLM that always raises TimeoutError."""
    async def generate(self, system_prompt, user_prompt, stream=False):
        raise asyncio.TimeoutError("Simulated timeout")


class DummyEmbedder:
    async def embed(self, texts):
        if isinstance(texts, list):
            return np.ones((len(texts), 384), dtype=np.float32)
        return np.ones(384, dtype=np.float32)


@pytest.fixture
def dummy_session_store():
    return DummySessionStore()

@pytest.fixture
def dummy_cache():
    return DummyCache()

@pytest.fixture
def dummy_llm():
    return DummyLLM()

@pytest.fixture
def dummy_embedder():
    return DummyEmbedder()
