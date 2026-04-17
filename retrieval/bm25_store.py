"""BM25 lexical search with Bangla tokenization and disk persistence.

Key improvements over baseline:
  - NFC normalization before tokenization
  - save() / load() methods — index is now persisted to disk, not rebuilt
    from scratch on every pod startup (saves 2–5s cold-start time)
  - Expanded stop-word list
"""
import asyncio
import pickle
import re
import time
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from utils.logger import logger
from utils.metrics import metrics


# Bangla stop words — do not contribute to BM25 scoring
_STOP_WORDS: frozenset[str] = frozenset({
    "এর", "এ", "এই", "এত", "যে", "যা", "এবং", "অথবা", "তো", "না",
    "হয়", "হয়েছে", "হবে", "হচ্ছে", "আছে", "আছো", "আছি",
    "কি", "কী", "কীভাবে", "কোথায়", "কে", "কাকে", "তার", "তারা",
    "আমার", "আমাদের", "আপনার", "আপনাদের", "এখানে", "সেখানে",
    "বলুন", "জানান", "দিন", "পাই", "চাই", "করেন", "করি",
})


class BM25Store:
    """BM25 keyword search index for Bangla products."""

    def __init__(self) -> None:
        """Initialize BM25 store."""
        self.bm25: Optional[BM25Okapi] = None
        self.products: Dict[str, Dict[str, Any]] = {}
        self.products_list: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []

    # ── Tokenization ──────────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize Bangla text for BM25 indexing.

        Steps:
          1. NFC normalization
          2. Lowercase
          3. Extract Bangla and alphanumeric tokens
          4. Remove stop words
          5. Drop single-character tokens
        """
        text = unicodedata.normalize("NFC", text).lower().strip()
        tokens = re.findall(r"[\u0980-\u09FF]+|[a-z0-9]+", text)
        return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]

    # ── Index building ────────────────────────────────────────────────────────

    async def build_index(self, products: List[Dict[str, Any]]) -> None:
        """
        Build BM25 index from products.

        Args:
            products: list of product dicts with 'name', 'category', 'description'
        """
        start = time.perf_counter()
        loop = asyncio.get_event_loop()

        try:
            def build() -> tuple:
                corpus: List[List[str]] = []
                for prod in products:
                    text = (
                        f"{prod.get('name', '')} "
                        f"{prod.get('category', '')} "
                        f"{prod.get('description', '')}"
                    )
                    corpus.append(self._tokenize(text))
                return BM25Okapi(corpus), corpus

            self.bm25, self.tokenized_corpus = await loop.run_in_executor(None, build)
            self.products_list = products
            for prod in products:
                self.products[prod["id"]] = prod

            latency_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "Built BM25 index",
                extra={"num_products": len(products), "latency_ms": round(latency_ms, 1)},
            )
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}", extra={"error": str(e)})
            raise

    # ── Search ────────────────────────────────────────────────────────────────

    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for products using BM25 ranking.

        Args:
            query: Search query string (Bangla or mixed)
            top_k: Number of results to return

        Returns:
            List of product dicts enriched with 'score' and 'source' fields
        """
        if self.bm25 is None:
            logger.warning("BM25 index not initialized")
            return []

        start = time.perf_counter()
        try:
            loop = asyncio.get_event_loop()

            def search_fn() -> List[tuple]:
                tokens = self._tokenize(query)
                if not tokens:
                    return []
                scores = self.bm25.get_scores(tokens)
                top_indices = sorted(
                    range(len(scores)), key=lambda i: scores[i], reverse=True
                )[:top_k]
                return [(i, scores[i]) for i in top_indices if scores[i] > 0]

            ranked = await loop.run_in_executor(None, search_fn)

            results: List[Dict[str, Any]] = []
            for idx, score in ranked:
                if idx < len(self.products_list):
                    product = self.products_list[idx].copy()
                    product["score"] = min(score / 30.0, 1.0)  # normalize to [0, 1]
                    product["source"] = "bm25"
                    results.append(product)

            latency_ms = (time.perf_counter() - start) * 1000
            metrics.record("bm25_search", latency_ms, success=True)
            return results

        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            metrics.record("bm25_search", latency_ms, success=False, error_type="BM25SearchError")
            logger.error(f"BM25 search failed: {e}", extra={"error": str(e)})
            return []

    # ── Persistence ───────────────────────────────────────────────────────────

    async def save(self, path: str) -> None:
        """Pickle BM25 index and metadata to disk.

        Saves: bm25 model, tokenized corpus, and products list.
        Enables fast load on startup instead of rebuilding from scratch.
        """
        try:
            path_obj = Path(path)
            path_obj.mkdir(parents=True, exist_ok=True)
            bm25_path = path_obj / "bm25.pkl"

            loop = asyncio.get_event_loop()

            def _write() -> None:
                with open(bm25_path, "wb") as f:
                    pickle.dump(
                        {
                            "bm25": self.bm25,
                            "tokenized_corpus": self.tokenized_corpus,
                            "products_list": self.products_list,
                            "products": self.products,
                        },
                        f,
                        protocol=5,
                    )

            await loop.run_in_executor(None, _write)
            logger.info(f"Saved BM25 index to {bm25_path}")
        except Exception as e:
            logger.error(f"Failed to save BM25 index: {e}", extra={"error": str(e)})

    async def load(self, path: str) -> None:
        """Load BM25 index and metadata from disk.

        Args:
            path: Directory containing 'bm25.pkl'
        """
        try:
            bm25_path = Path(path) / "bm25.pkl"
            if not bm25_path.exists():
                raise FileNotFoundError(f"BM25 index not found at {bm25_path}")

            loop = asyncio.get_event_loop()

            def _read() -> dict:
                with open(bm25_path, "rb") as f:
                    return pickle.load(f)

            data = await loop.run_in_executor(None, _read)
            self.bm25 = data["bm25"]
            self.tokenized_corpus = data["tokenized_corpus"]
            self.products_list = data["products_list"]
            self.products = data["products"]

            logger.info(f"Loaded BM25 index from {bm25_path} ({len(self.products_list)} products)")
        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}", extra={"error": str(e)})
            raise
