"""FAISS vector store for cosine similarity search.

Key change from baseline:
  - Switched from IndexFlatL2 → IndexFlatIP quantizer (inner product).
  - multilingual-e5-small produces L2-normalized vectors; cosine similarity
    equals inner product on unit vectors — using L2 distance was incorrect.
  - Vectors are normalized before add() and search() via faiss.normalize_L2().
"""
import asyncio
import pickle
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
import faiss
from utils.logger import logger
from utils.metrics import metrics


class FAISSStore:
    """FAISS IVF-Flat index for fast cosine similarity search."""

    def __init__(self, dimension: int = 384, nlist: int = 32):
        """Initialize FAISS store."""
        self.dimension = dimension
        self.nlist = nlist
        self.index: Optional[faiss.Index] = None
        self.id_map: Dict[int, str] = {}       # row index → product id
        self.products: Dict[str, Dict[str, Any]] = {}  # product id → metadata

    # ── Index building ────────────────────────────────────────────────────────

    async def build_index(self, vectors: np.ndarray, products: List[Dict[str, Any]]) -> None:
        """
        Build FAISS IVF-Flat index from normalized vectors.

        Args:
            vectors:  numpy array of shape (n_products, 384), float32
            products: list of product dicts with 'id', 'name', etc.
        """
        assert vectors.shape[1] == self.dimension, (
            f"Vector dimension mismatch: got {vectors.shape[1]}, expected {self.dimension}"
        )

        start = time.perf_counter()
        loop = asyncio.get_event_loop()

        try:
            def build() -> faiss.Index:
                vecs = vectors.astype(np.float32)
                # Normalize to unit sphere — cosine sim = inner product on L2-normalized vecs
                faiss.normalize_L2(vecs)

                # Inner-product quantizer + IVF-Flat index
                quantizer = faiss.IndexFlatIP(self.dimension)
                idx = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist, faiss.METRIC_INNER_PRODUCT)

                if vecs.shape[0] >= self.nlist:
                    idx.train(vecs)

                idx.add(vecs)
                idx.nprobe = 8  # probe 8/32 cells — good recall for 5k vectors
                return idx

            self.index = await loop.run_in_executor(None, build)

            for i, prod in enumerate(products):
                self.id_map[i] = prod["id"]
                self.products[prod["id"]] = prod

            latency_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "Built FAISS index (IVF-Flat, InnerProduct)",
                extra={
                    "num_vectors": len(vectors),
                    "dimension": self.dimension,
                    "nlist": self.nlist,
                    "latency_ms": round(latency_ms, 1),
                },
            )
        except Exception as e:
            logger.error(f"Failed to build FAISS index: {e}", extra={"error": str(e)})
            raise

    # ── Search ────────────────────────────────────────────────────────────────

    async def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for top-k most similar products using cosine similarity.

        Args:
            query_vector: Query embedding, shape (384,) or (1, 384)
            top_k:        Number of results to return

        Returns:
            List of product dicts enriched with 'score' and 'source' fields
        """
        if self.index is None:
            logger.warning("FAISS index not initialized")
            return []

        assert query_vector.shape[-1] == self.dimension, (
            f"Query vector dimension mismatch: got {query_vector.shape[-1]}, expected {self.dimension}"
        )

        start = time.perf_counter()
        try:
            loop = asyncio.get_event_loop()

            def search_fn() -> tuple:
                q = query_vector.reshape(1, -1).astype(np.float32)
                faiss.normalize_L2(q)  # normalize query to unit sphere
                scores, indices = self.index.search(q, top_k)
                return scores[0], indices[0]

            scores, indices = await loop.run_in_executor(None, search_fn)

            results: List[Dict[str, Any]] = []
            for idx, score in zip(indices, scores):
                if idx >= 0:  # -1 means no result
                    product_id = self.id_map.get(int(idx))
                    if product_id and product_id in self.products:
                        product = self.products[product_id].copy()
                        # Inner product on normalized vecs = cosine similarity (range −1 to 1)
                        product["score"] = float(score)
                        product["source"] = "faiss"
                        results.append(product)

            latency_ms = (time.perf_counter() - start) * 1000
            metrics.record("faiss_search", latency_ms, success=True)
            return results

        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            metrics.record("faiss_search", latency_ms, success=False, error_type="FAISSSearchError")
            logger.error(f"FAISS search failed: {e}", extra={"error": str(e)})
            return []

    # ── Persistence ───────────────────────────────────────────────────────────

    async def save(self, path: str) -> None:
        """Save index and metadata to disk."""
        try:
            path_obj = Path(path)
            path_obj.mkdir(parents=True, exist_ok=True)

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: faiss.write_index(self.index, str(path_obj / "faiss.index")),
            )

            with open(path_obj / "faiss_metadata.pkl", "wb") as f:
                pickle.dump({"id_map": self.id_map, "products": self.products}, f, protocol=5)

            logger.info(f"Saved FAISS index to {path}")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}", extra={"error": str(e)})

    async def load(self, path: str) -> None:
        """Load index and metadata from disk."""
        try:
            path_obj = Path(path)
            loop = asyncio.get_event_loop()

            self.index = await loop.run_in_executor(
                None,
                lambda: faiss.read_index(str(path_obj / "faiss.index")),
            )
            # Restore nprobe after loading (not serialized by faiss.write_index)
            if hasattr(self.index, "nprobe"):
                self.index.nprobe = 8

            with open(path_obj / "faiss_metadata.pkl", "rb") as f:
                data = pickle.load(f)
                self.id_map = data["id_map"]
                self.products = data["products"]

            logger.info(f"Loaded FAISS index from {path} ({len(self.products)} products)")
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}", extra={"error": str(e)})
            raise
