"""Cache layer for query embeddings to avoid recomputation."""
import hashlib
import time
from typing import Optional
import numpy as np
import redis.asyncio as aioredis
from config import settings
from utils.logger import logger
from utils.metrics import metrics


class EmbeddingCache:
    """Redis-backed cache for embeddings with fast lookups."""

    def __init__(self, redis_url: Optional[str] = None, ttl: Optional[int] = None):
        """Initialize embedding cache."""
        self.redis_url = redis_url or settings.redis_url
        self.redis = None
        self.ttl = ttl if ttl is not None else settings.redis_cache_ttl * 2  # 2x session TTL
        self.prefix = "embed:query:"
        self._in_memory_cache = {}  # Fallback when Redis unavailable

    async def connect(self) -> None:
        """Connect to Redis (falls back to in-memory if unavailable)."""
        try:
            self.redis = await aioredis.from_url(self.redis_url, decode_responses=False)
            await self.redis.ping()
            logger.info("Connected to Redis (embedding cache)", extra={"service": "EmbeddingCache"})
        except Exception as e:
            logger.warning(f"Redis unavailable, using in-memory embedding cache: {e}", extra={"service": "EmbeddingCache"})
            # In-memory fallback
            self._in_memory_cache = {}
            self.redis = None

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self.redis:
            if hasattr(self.redis, "aclose"):
                await self.redis.aclose()
            else:
                await self.redis.close()

    def _make_key(self, query: str) -> str:
        """Generate cache key for query embedding."""
        # Normalize query: strip whitespace, lowercase
        normalized = query.strip().lower()
        key_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return f"{self.prefix}{key_hash}"

    async def get(self, query: str) -> Optional[np.ndarray]:
        """
        Get cached embedding for a query.
        
        Args:
            query: Query string
            
        Returns:
            Embedding as numpy array if found, None otherwise
        """
        start = time.perf_counter()
        key = self._make_key(query)
        
        try:
            if self.redis:
                cached_bytes = await self.redis.get(key)
                if cached_bytes:
                    # Deserialize numpy array from bytes
                    embedding = np.frombuffer(cached_bytes, dtype=np.float32)
                    latency_ms = (time.perf_counter() - start) * 1000
                    metrics.record("embedding_cache_hit", latency_ms, success=True)
                    logger.debug(f"Embedding cache hit for query: {query[:50]}")
                    return embedding
            else:
                # In-memory fallback
                if key in self._in_memory_cache:
                    embedding = self._in_memory_cache[key]
                    latency_ms = (time.perf_counter() - start) * 1000
                    metrics.record("embedding_cache_hit", latency_ms, success=True)
                    logger.debug(f"Embedding cache hit (in-memory) for query: {query[:50]}")
                    return embedding
            
            latency_ms = (time.perf_counter() - start) * 1000
            metrics.record("embedding_cache_miss", latency_ms, success=True)
            return None
            
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            metrics.record("embedding_cache_error", latency_ms, success=False)
            logger.warning(f"Embedding cache lookup failed: {e}")
            return None

    async def set(self, query: str, embedding: np.ndarray) -> None:
        """
        Cache an embedding for a query.
        
        Args:
            query: Query string
            embedding: Embedding vector as numpy array
        """
        key = self._make_key(query)
        
        try:
            if self.redis:
                # Serialize numpy array to bytes
                embedding_bytes = embedding.astype(np.float32).tobytes()
                await self.redis.setex(key, self.ttl, embedding_bytes)
            else:
                # In-memory fallback
                self._in_memory_cache[key] = embedding.astype(np.float32).copy()
            
            logger.debug(f"Cached embedding for query: {query[:50]}")
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")

    async def clear(self) -> None:
        """Clear all embedding cache entries."""
        if not self.redis:
            return
        
        try:
            keys = await self.redis.keys(f"{self.prefix}*")
            if keys:
                await self.redis.delete(*keys)
            logger.info(f"Cleared {len(keys)} embedding cache entries")
        except Exception as e:
            logger.warning(f"Failed to clear embedding cache: {e}")
