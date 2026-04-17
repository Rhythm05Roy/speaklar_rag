"""Tests for retrieval pipeline (FAISS, BM25, Fusion)."""
import pytest
import numpy as np
from retrieval.fusion import reciprocal_rank_fusion


class TestRecipocalRankFusion:
    """Test RRF algorithm."""

    def test_empty_inputs(self):
        """Test with empty inputs."""
        result = reciprocal_rank_fusion([], [])
        assert result == []

    def test_single_source(self):
        """Test fusion with single source."""
        faiss_results = [
            {"id": "1", "name": "Product 1", "score": 0.9},
            {"id": "2", "name": "Product 2", "score": 0.8},
        ]
        result = reciprocal_rank_fusion(faiss_results, [])
        assert len(result) == 2
        assert result[0]["id"] == "1"  # Highest RRF score

    def test_fusion_merges_sources(self):
        """Test that RRF correctly merges both sources."""
        faiss_results = [
            {"id": "1", "name": "Product 1", "score": 0.9},
            {"id": "2", "name": "Product 2", "score": 0.7},
        ]
        bm25_results = [
            {"id": "2", "name": "Product 2", "score": 0.8},
            {"id": "3", "name": "Product 3", "score": 0.6},
        ]

        result = reciprocal_rank_fusion(faiss_results, bm25_results, k=60)
        
        # Product 2 should rank high because it appears in both
        product_ids = [r["id"] for r in result]
        assert "2" in product_ids

    def test_rrf_score_normalization(self):
        """Test that RRF scores are in reasonable range."""
        results = [{"id": str(i), "name": f"P{i}", "score": 0.5} for i in range(10)]
        fused = reciprocal_rank_fusion(results, [], k=60)
        
        assert all(0 <= r.get("rrf_score", 0) <= 1.0 for r in fused)

    def test_source_tracking(self):
        """Test that source attribution is tracked."""
        faiss_results = [{"id": "1", "name": "P1", "score": 0.9}]
        bm25_results = [{"id": "2", "name": "P2", "score": 0.8}]

        fused = reciprocal_rank_fusion(faiss_results, bm25_results)
        
        # Check source tracking
        for result in fused:
            assert "sources" in result
            assert isinstance(result["sources"], list)


class TestFAISSSearch:
    """Test FAISS search."""

    @pytest.mark.asyncio
    async def test_search_returns_valid_results(self, faiss_store):
        """Test FAISS search returns products."""
        query_vector = np.random.rand(384).astype(np.float32)
        results = await faiss_store.search(query_vector, top_k=3)
        
        assert len(results) <= 3
        assert all("id" in r and "name" in r for r in results)

    @pytest.mark.asyncio
    async def test_search_latency(self, faiss_store):
        """Test FAISS search is fast."""
        import time
        query_vector = np.random.rand(384).astype(np.float32)
        
        t0 = time.time()
        results = await faiss_store.search(query_vector)
        latency_ms = (time.time() - t0) * 1000
        
        # Should be under 5ms for 5k vectors
        assert latency_ms < 10  # Allow 10ms for test overhead


class TestBM25Search:
    """Test BM25 search."""

    @pytest.mark.asyncio
    async def test_search_returns_results(self, bm25_store):
        """Test BM25 search."""
        results = await bm25_store.search("নুডুলস", top_k=3)
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_search_exact_term(self, bm25_store):
        """Test BM25 finds exact terms."""
        results = await bm25_store.search("নুডুলস", top_k=5)
        # Should find results with নুডুলস
        assert len(results) > 0


class TestSemanticCache:
    """Test semantic cache."""

    @pytest.mark.asyncio
    async def test_cache_set_get(self, semantic_cache):
        """Test cache set/get (in-process LRU tier with embedding)."""
        import numpy as np
        query = "নুডুলসের দাম কত?"
        emb = np.ones(384, dtype=np.float32)
        payload = [{"id": "1", "name": "Product 1"}]

        await semantic_cache.set(query, payload, query_embedding=emb)
        cached = await semantic_cache.get(query, query_embedding=emb)

        assert cached is not None
        assert len(cached) == 1

    @pytest.mark.asyncio
    async def test_cache_miss(self, semantic_cache):
        """Test cache miss."""
        import numpy as np
        emb = np.zeros(384, dtype=np.float32)  # zero vector — should never hit
        result = await semantic_cache.get("unique_query_xyz_12345", query_embedding=emb)
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_clear(self, semantic_cache):
        """Test cache clear."""
        import numpy as np
        emb = np.ones(384, dtype=np.float32)
        await semantic_cache.set("test_query", [{"id": "1"}], query_embedding=emb)
        await semantic_cache.clear()

        result = await semantic_cache.get("test_query", query_embedding=emb)
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_dict_payload(self, semantic_cache):
        """Test cache can store structured payloads used by the pipeline."""
        import numpy as np
        emb = np.ones(384, dtype=np.float32) * 0.5
        payload = {
            "response": "নুডুলসের দাম ৬০ টাকা।",
            "retrieved_docs": [{"id": "1", "name": "নুডুলস"}],
        }

        await semantic_cache.set("structured_query", payload, query_embedding=emb)
        cached = await semantic_cache.get("structured_query", query_embedding=emb)

        assert cached == payload
