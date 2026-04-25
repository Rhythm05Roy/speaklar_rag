"""Semantic cache with true vector similarity lookup.

Upgrade from baseline (hash-only) to a two-tier cache:

Tier 1 — In-process LRU (OrderedDict):
  Stores (embedding, payload) pairs. On lookup, computes cosine similarity
  between incoming query embedding and all cached embeddings. Cache HIT when
  max cosine sim >= threshold (default 0.92). Lookup time: O(n) where n =
  cache size (≤ 2000 entries × ~1.5 KB = ~3 MB RAM).
  Target latency: ~2ms for 2000 entries.

Tier 2 — Redis hash-based (exact match fallback):
  Same SHA-256 key as before. Used as a cross-process / cross-pod exact hit,
  and to persist responses across restarts.

This gives ~10–15ms total on repeated/similar queries (cache hit),
vs ~75–90ms for a full retrieval + LLM call (cache miss).
"""
import hashlib
import json
import time
from collections import OrderedDict
from typing import Optional, Any, TYPE_CHECKING
import numpy as np
import redis.asyncio as aioredis
from config import settings
from utils.logger import logger
from utils.metrics import metrics

if TYPE_CHECKING:
    from retrieval.embedder import EmbedderSingleton


class SemanticCache:
    """Two-tier semantic cache: in-process cosine-sim + Redis exact-match."""

    def __init__(
        self,
        redis_url: Optional[str] = None,
        ttl: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        max_entries: Optional[int] = None,
    ) -> None:
        self.redis_url = redis_url or settings.redis_url
        self.redis: Optional[aioredis.Redis] = None
        self.ttl = ttl if ttl is not None else settings.redis_cache_ttl
        self.threshold = similarity_threshold if similarity_threshold is not None \
            else settings.cache_similarity_threshold
        self.max_entries = max_entries if max_entries is not None \
            else settings.semantic_cache_max_entries
        self.query_prefix = "cache:query:"

        # Tier 1: in-process semantic cache {normalized_query: (embedding, payload)}
        self._lru: OrderedDict[str, tuple[np.ndarray, Any]] = OrderedDict()

        # Reference to embedder (set via set_embedder() after initialization)
        self._embedder: Optional["EmbedderSingleton"] = None

    def set_embedder(self, embedder: "EmbedderSingleton") -> None:
        """Inject embedder reference (avoids circular import at module load time)."""
        self._embedder = embedder

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Connect to Redis (falls back to in-memory if unavailable)."""
        try:
            self.redis = await aioredis.from_url(self.redis_url, decode_responses=True)
            await self.redis.ping()
            logger.info("Connected to Redis (SemanticCache)", extra={"service": "SemanticCache"})
        except Exception as e:
            logger.warning(
                f"Redis unavailable, using in-process semantic cache only: {e}",
                extra={"service": "SemanticCache"},
            )
            self.redis = None

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self.redis:
            if hasattr(self.redis, "aclose"):
                await self.redis.aclose()
            else:
                await self.redis.close()

    # ── Cache key ─────────────────────────────────────────────────────────────

    def _make_key(self, query: str) -> str:
        """SHA-256 based key for exact Redis lookup."""
        normalized = " ".join(query.strip().split()).lower()
        key_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return f"{self.query_prefix}{key_hash}"

    # ── Lookup ────────────────────────────────────────────────────────────────

    async def get(self, query: str, query_embedding: Optional[np.ndarray] = None) -> Optional[Any]:
        """
        Look up cache for a query.

        Two-stage lookup:
        1. Tier 1: cosine-sim scan of in-process LRU (if embedder available)
        2. Tier 2: Redis exact-match by SHA-256 key

        Args:
            query:           Normalized query string
            query_embedding: Pre-computed embedding (avoids double embed on miss path)

        Returns:
            Cached payload dict or None
        """
        start = time.perf_counter()
        norm_query = " ".join(query.strip().split()).lower()

        # ── Tier 1: semantic similarity ───────────────────────────────────────
        if query_embedding is not None and len(self._lru) > 0:
            best_sim, best_key = self._find_most_similar(query_embedding)
            if best_sim >= self.threshold:
                _, payload = self._lru[best_key]
                # Move to end (LRU update)
                self._lru.move_to_end(best_key)
                latency_ms = (time.perf_counter() - start) * 1000
                metrics.record("cache_hit", latency_ms, success=True)
                logger.debug(
                    f"Semantic cache HIT (sim={best_sim:.3f})",
                    extra={"query": query[:50], "similarity": round(best_sim, 3)},
                )
                return payload

        # ── Tier 2: Redis exact match ─────────────────────────────────────────
        if self.redis:
            key = self._make_key(norm_query)
            try:
                cached = await self.redis.get(key)
                if cached:
                    payload = json.loads(cached)
                    latency_ms = (time.perf_counter() - start) * 1000
                    metrics.record("cache_hit", latency_ms, success=True)
                    return payload
            except Exception as e:
                logger.warning(f"Redis cache lookup failed: {e}")

        latency_ms = (time.perf_counter() - start) * 1000
        metrics.record("cache_miss", latency_ms, success=True)
        return None

    def _find_most_similar(self, query_embedding: np.ndarray) -> tuple[float, str]:
        """Find the most similar cached embedding via cosine similarity."""
        q = query_embedding.astype(np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return -1.0, ""

        best_sim = -1.0
        best_key = ""
        for key, (emb, _) in self._lru.items():
            sim = float(np.dot(q, emb) / (q_norm * np.linalg.norm(emb) + 1e-9))
            if sim > best_sim:
                best_sim = sim
                best_key = key
        return best_sim, best_key

    # ── Store ─────────────────────────────────────────────────────────────────

    async def set(
        self,
        query: str,
        payload: Any,
        query_embedding: Optional[np.ndarray] = None,
    ) -> None:
        """
        Cache payload for a query.

        Writes to both Tier 1 (in-process) and Tier 2 (Redis).

        Args:
            query:           Normalized query string
            payload:         JSON-serializable dict to cache
            query_embedding: Pre-computed embedding (for Tier 1 semantic lookup)
        """
        norm_query = " ".join(query.strip().split()).lower()

        # ── Tier 1: in-process LRU ────────────────────────────────────────────
        if query_embedding is not None:
            emb = query_embedding.astype(np.float32)
            if norm_query in self._lru:
                self._lru.move_to_end(norm_query)
            self._lru[norm_query] = (emb, payload)
            # Evict oldest if over capacity
            while len(self._lru) > self.max_entries:
                self._lru.popitem(last=False)

        # ── Tier 2: Redis exact match ─────────────────────────────────────────
        if self.redis:
            key = self._make_key(norm_query)
            try:
                await self.redis.setex(key, self.ttl, json.dumps(payload))
            except Exception as e:
                logger.warning(f"Failed to write to Redis cache: {e}")

    # ── Maintenance ───────────────────────────────────────────────────────────

    async def clear(self) -> None:
        """Clear all cache entries (both tiers)."""
        self._lru.clear()
        if self.redis:
            try:
                keys = await self.redis.keys(f"{self.query_prefix}*")
                if keys:
                    await self.redis.delete(*keys)
                logger.info(f"Cleared {len(keys)} Redis cache entries")
            except Exception as e:
                logger.warning(f"Failed to clear Redis cache: {e}")

    def get_stats(self) -> dict:
        """Return cache statistics."""
        return {
            "lru_entries": len(self._lru),
            "lru_max": self.max_entries,
            "threshold": self.threshold,
            "redis_connected": self.redis is not None,
        }
