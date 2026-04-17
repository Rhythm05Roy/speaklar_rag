"""Reciprocal rank fusion for hybrid search."""
from typing import List, Dict, Any


def reciprocal_rank_fusion(faiss_results: List[Dict[str, Any]], 
                           bm25_results: List[Dict[str, Any]], 
                           k: int = 60) -> List[Dict[str, Any]]:
    """
    Merge and rank results from FAISS and BM25 using Reciprocal Rank Fusion.
    
    Formula: RRF(d) = Σ (1 / (k + rank(d)))
    
    where rank(d) is the position of document d in the ranking (1-indexed)
    and k is the parameter (typically 60).
    
    Args:
        faiss_results: List of results from FAISS vector search
        bm25_results: List of results from BM25 lexical search  
        k: Fusion parameter (higher k gives more uniform weighting)
        
    Returns:
        Merged and ranked results sorted by RRF score (descending),
        excluding items that appear in only one source with zero score
    """
    # Dictionary to accumulate RRF scores: product_id -> score
    rrf_scores: Dict[str, float] = {}
    product_data: Dict[str, Dict[str, Any]] = {}
    source_count: Dict[str, int] = {}  # Track how many sources contributed
    
    # Process FAISS results (rank 1, 2, 3, ...)
    for rank, result in enumerate(faiss_results, start=1):
        product_id = result.get("id")
        if product_id:
            rrf_score = 1.0 / (k + rank)
            rrf_scores[product_id] = rrf_scores.get(product_id, 0) + rrf_score
            product_data[product_id] = result
            source_count[product_id] = source_count.get(product_id, 0) + 1
    
    # Process BM25 results
    for rank, result in enumerate(bm25_results, start=1):
        product_id = result.get("id")
        if product_id:
            rrf_score = 1.0 / (k + rank)
            rrf_scores[product_id] = rrf_scores.get(product_id, 0) + rrf_score
            # Update with BM25 data if not already present
            if product_id not in product_data:
                product_data[product_id] = result
            source_count[product_id] = source_count.get(product_id, 0) + 1
    
    # Sort by RRF score (descending) and create output
    merged_results = []
    for product_id in sorted(rrf_scores.keys(), key=lambda pid: rrf_scores[pid], reverse=True):
        product = product_data[product_id].copy()
        product["rrf_score"] = rrf_scores[product_id]
        product["sources"] = []
        
        # Track which sources contributed
        if any(r.get("id") == product_id for r in faiss_results):
            product["sources"].append("faiss")
        if any(r.get("id") == product_id for r in bm25_results):
            product["sources"].append("bm25")
        
        # Only include if: (a) appears in both sources, OR (b) strong score from one source
        # This prevents including borderline results ranked by only one retriever
        score_threshold = 1.0 / (k + 10)  # Roughly top-10 candidate
        if len(product["sources"]) >= 2 or rrf_scores[product_id] >= score_threshold:
            merged_results.append(product)
    
    return merged_results
