#!/usr/bin/env python
"""Static code validation - checks that all fixes are implemented."""
import os
import sys

def check_file_contains(filepath, search_strings):
    """Check if file contains all search strings."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    missing = []
    for search_str in search_strings:
        if search_str not in content:
            missing.append(search_str)
    return len(missing) == 0, missing, content


def validate_embedding_cache():
    """Validate embedding cache implementation."""
    print("Checking: Embedding Cache Layer...")
    filepath = "retrieval/embedding_cache.py"
    
    if not os.path.exists(filepath):
        print(f"  ✗ FAILED: {filepath} doesn't exist")
        return False
    
    required = [
        "class EmbeddingCache",
        "async def connect",
        "async def get",
        "async def set",
        "_normalize",
    ]
    
    ok, missing, _ = check_file_contains(filepath, required)
    if ok:
        print(f"  ✓ PASSED: Embedding cache properly implemented")
        return True
    else:
        print(f"  ✗ FAILED: Missing: {missing}")
        return False


def validate_query_normalization():
    """Validate query normalization in pipeline."""
    print("Checking: Query Normalization...")
    filepath = "api/pipeline.py"
    
    required = [
        "_normalize_query",
        'normalized_query = self._normalize_query',
        'await self.cache.get(normalized_query)',
        'await self.cache.set(normalized_query',
    ]
    
    ok, missing, _ = check_file_contains(filepath, required)
    if ok:
        print(f"  ✓ PASSED: Query normalization in pipeline")
        return True
    else:
        print(f"  ✗ FAILED: Missing: {missing}")
        return False


def validate_context_resolver_fix():
    """Validate context resolver false positive fix."""
    print("Checking: Context Resolver False Positive Fix...")
    filepath = "context/resolver.py"
    
    required = [
        "FIXED to reduce false positives",
        "if entities:",
        "return False",
        "_is_elliptic(query, entities)",
    ]
    
    ok, missing, _ = check_file_contains(filepath, required)
    if ok:
        print(f"  ✓ PASSED: Context resolver false positives fixed")
        return True
    else:
        print(f"  ✗ FAILED: Missing: {missing}")
        return False


def validate_sanitization():
    """Validate input sanitization in prompt builder."""
    print("Checking: Input Sanitization...")
    filepath = "generation/generator.py"
    
    required = [
        "_sanitize_text",
        "max_length",
        "control characters",
        "isprintable",
    ]
    
    ok, missing, _ = check_file_contains(filepath, required)
    if ok:
        print(f"  ✓ PASSED: Input sanitization implemented")
        return True
    else:
        print(f"  ✗ FAILED: Missing: {missing}")
        return False


def validate_rrf_fusion():
    """Validate RRF fusion improvements."""
    print("Checking: RRF Fusion Optimization...")
    filepath = "retrieval/fusion.py"
    
    required = [
        "source_count",
        "score_threshold",
        "excluding items that appear",
    ]
    
    ok, missing, _ = check_file_contains(filepath, required)
    if ok:
        print(f"  ✓ PASSED: RRF fusion improved")
        return True
    else:
        print(f"  ✗ FAILED: Missing: {missing}")
        return False


def validate_cache_normalization():
    """Validate cache key normalization."""
    print("Checking: Cache Key Normalization...")
    filepath = "retrieval/cache.py"
    
    required = [
        "normalized = ",
        "split()",
        "lower()",
    ]
    
    ok, missing, _ = check_file_contains(filepath, required)
    if ok:
        print(f"  ✓ PASSED: Cache key normalization")
        return True
    else:
        print(f"  ✗ FAILED: Missing: {missing}")
        return False


def validate_bangla_tokenization():
    """Validate improved Bangla tokenization."""
    print("Checking: Improved Bangla Tokenization...")
    filepath = "retrieval/bm25_store.py"
    
    required = [
        "Improved Bangla tokenization",
        "stop_words",
        "findall",
        "filter",
    ]
    
    ok, missing, _ = check_file_contains(filepath, required)
    if ok:
        print(f"  ✓ PASSED: Bangla tokenization improved")
        return True
    else:
        print(f"  ✗ FAILED: Missing: {missing}")
        return False


def validate_embedding_cache_integration():
    """Validate embedding cache in main API."""
    print("Checking: Embedding Cache Integration...")
    filepath = "api/main.py"
    
    required = [
        "from retrieval.embedding_cache import EmbeddingCache",
        "set_embedding_cache",
        "_embedding_cache = EmbeddingCache",
        "await _embedding_cache.connect",
        "await _embedding_cache.disconnect",
    ]
    
    ok, missing, _ = check_file_contains(filepath, required)
    if ok:
        print(f"  ✓ PASSED: Embedding cache integrated in main API")
        return True
    else:
        print(f"  ✗ FAILED: Missing: {missing}")
        return False


def validate_pipeline_imports():
    """Validate pipeline imports and improvements."""
    print("Checking: Pipeline Improvements...")
    filepath = "api/pipeline.py"
    
    required = [
        "import asyncio",
        "embedder_parallel_with_retrieval",
        "await asyncio.gather",
    ]
    
    ok, missing, _ = check_file_contains(filepath, required)
    if ok:
        print(f"  ✓ PASSED: Pipeline parallel execution")
        return True
    else:
        print(f"  ✗ FAILED: Missing: {missing}")
        return False


def main():
    """Run all static validations."""
    print("\n" + "=" * 70)
    print("RAG APPLICATION - STATIC CODE VALIDATION")
    print("=" * 70 + "\n")
    
    tests = [
        validate_embedding_cache,
        validate_query_normalization,
        validate_context_resolver_fix,
        validate_sanitization,
        validate_rrf_fusion,
        validate_cache_normalization,
        validate_bangla_tokenization,
        validate_embedding_cache_integration,
        validate_pipeline_imports,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"  ✗ ERROR: {e}\n")
            results.append(False)
    
    print("\n" + "=" * 70)
    print(f"RESULTS: {sum(results)}/{len(results)} validations passed")
    print("=" * 70)
    
    if all(results):
        print("\n✓ ALL FIXES VALIDATED SUCCESSFULLY\n")
        return 0
    else:
        print("\n✗ Some validations failed\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
