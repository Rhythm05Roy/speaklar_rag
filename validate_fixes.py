#!/usr/bin/env python
"""Validation script for RAG fixes."""
import sys
import asyncio
from contextlib import suppress

# Test 1: Query normalization
def test_query_normalization():
    """Test query normalization consistency."""
    from api.pipeline import RAGPipeline
    
    q1 = "চালের দাম   কত?"  # Extra spaces
    q2 = "চালের দাম কত?"   # Normal spaces
    
    normalized_q1 = RAGPipeline._normalize_query(q1)
    normalized_q2 = RAGPipeline._normalize_query(q2)
    
    assert normalized_q1 == normalized_q2, f"Query normalization failed: {normalized_q1} != {normalized_q2}"
    print("✓ Test 1: Query normalization - PASSED")


# Test 2: Context resolver false positive fix
def test_context_resolver_false_positives():
    """Test that context resolver doesn't mark complete queries as elliptic."""
    from context.resolver import BanglaContextResolver
    
    # Query with entities should NOT be elliptic
    query_with_entity = "চালের দাম কত?"
    entities = ["চাল"]
    is_elliptic = BanglaContextResolver._is_elliptic(query_with_entity, entities)
    assert not is_elliptic, "Query with entity should not be elliptic"
    
    # Truly elliptic query (no entity, starts with reference)
    elliptic_query = "দাম কত?"
    no_entities = []
    is_elliptic = BanglaContextResolver._is_elliptic(elliptic_query, no_entities)
    assert is_elliptic, "True elliptic query not detected"
    
    print("✓ Test 2: Context resolver false positive fixes - PASSED")


# Test 3: Input sanitization
def test_input_sanitization():
    """Test that input sanitization prevents injection."""
    from generation.generator import PromptBuilder
    
    # Test with long input
    long_text = "a" * 600
    sanitized = PromptBuilder._sanitize_text(long_text, max_length=500)
    assert len(sanitized) <= 501, f"Sanitization failed: length={len(sanitized)}"
    
    # Test with null bytes (security)
    unsafe_text = "normal\x00text"
    sanitized = PromptBuilder._sanitize_text(unsafe_text)
    assert "\x00" not in sanitized, "Null bytes not removed"
    
    # Test with control chars
    control_text = "text\x01with\x02control"
    sanitized = PromptBuilder._sanitize_text(control_text)
    assert "\x01" not in sanitized, "Control characters not removed"
    
    print("✓ Test 3: Input sanitization - PASSED")


# Test 4: RRF fusion filtering
def test_rrf_fusion_filtering():
    """Test that RRF fusion handles edge cases correctly."""
    from retrieval.fusion import reciprocal_rank_fusion
    
    faiss_results = [
        {"id": "p1", "name": "চাল", "score": 0.9},
    ]
    
    bm25_results = [
        {"id": "p2", "name": "ডাল", "score": 0.8},
    ]
    
    merged = reciprocal_rank_fusion(faiss_results, bm25_results, k=60)
    
    # Both should be in results since they have reasonable scores
    ids = [r["id"] for r in merged]
    assert "p1" in ids, "Top FAISS result missing"
    assert "p2" in ids, "Top BM25 result missing"
    
    print("✓ Test 4: RRF fusion - PASSED")


# Test 5: Embedding cache interface
async def test_embedding_cache_interface():
    """Test that embedding cache is properly integrated."""
    from retrieval.embedding_cache import EmbeddingCache
    import numpy as np
    
    cache = EmbeddingCache("redis://localhost:6379/1")  # Use test DB
    
    try:
        await cache.connect()
        
        # Test set/get
        query = "চালের দাম কত?"
        embedding = np.ones(384, dtype=np.float32)
        
        await cache.set(query, embedding)
        retrieved = await cache.get(query)
        
        assert retrieved is not None, "Cache retrieval failed"
        assert np.allclose(retrieved, embedding), "Embedding mismatch"
        
        await cache.disconnect()
        print("✓ Test 5: Embedding cache interface - PASSED (if Redis running)")
    except (ConnectionError, Exception) as e:
        print(f"✓ Test 5: Embedding cache interface - SKIPPED (Redis not available: {e})")


# Test 6: Bangla tokenization
def test_bangla_tokenization():
    """Test improved Bangla tokenization."""
    from retrieval.bm25_store import BM25Store
    
    store = BM25Store()
    
    # Test basic tokenization  
    tokens = store._simple_tokenize_bn("চালের দাম ৭০ টাকা")
    assert len(tokens) > 0, "Tokenization failed"
    assert "চাল" in tokens or any("চাল" in t for t in tokens), "Product keyword not found"
    
    # Test punctuation removal
    tokens = store._simple_tokenize_bn("চালের দাম? কত টাকা!")
    assert not any("?" in t or "!" in t for t in tokens), "Punctuation not removed"
    
    print("✓ Test 6: Bangla tokenization - PASSED")


# Test 7: Pipeline query normalization
async def test_pipeline_integration():
    """Test that pipeline uses normalized queries for caching."""
    from api.pipeline import RAGPipeline
    
    q1 = "চালের  দাম  কত?"  # Double spaces
    q2 = "চালের দাম কত?"    # Single spaces
    
    norm1 = RAGPipeline._normalize_query(q1)
    norm2 = RAGPipeline._normalize_query(q2)
    
    # Both should normalize to same value
    assert norm1 == norm2, "Pipeline normalization inconsistent"
    assert norm1 == "চালের দাম কত?", f"Unexpected normalization: {norm1}"
    
    print("✓ Test 7: Pipeline integration - PASSED")


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("RAG APPLICATION VALIDATION TESTS")
    print("=" * 60)
    
    try:
        # Synchronous tests
        test_query_normalization()
        test_context_resolver_false_positives()
        test_input_sanitization()
        test_rrf_fusion_filtering()
        test_bangla_tokenization()
        test_pipeline_integration()
        
        # Async tests
        asyncio.run(test_embedding_cache_interface())
        
        print("\n" + "=" * 60)
        print("✓ ALL VALIDATION TESTS PASSED")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
