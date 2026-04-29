"""Singleton embedding model wrapper using multilingual-e5-small."""
import time
import asyncio
from typing import Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from utils.logger import logger
from utils.metrics import metrics


# Cache will be injected at runtime
_embedding_cache = None


class EmbedderSingleton:
    """Lazy-loaded singleton for the multilingual-e5-small embedding model."""

    _instance: Optional["EmbedderSingleton"] = None
    _model: Optional[SentenceTransformer] = None
    _loading = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    async def get_instance(cls) -> "EmbedderSingleton":
        """Get or create the singleton instance with async loading."""
        instance = cls()
        if cls._model is None and not cls._loading:
            cls._loading = True
            await instance._load_model()
            cls._loading = False
        return instance

    async def _load_model(self) -> None:
        """Load the embedding model with ONNX Runtime backend and run warmup."""
        loop = asyncio.get_event_loop()
        start = time.perf_counter()

        def _load_and_optimize() -> SentenceTransformer:
            # ONNX Runtime provides ~2x speedup over PyTorch on CPU (14ms vs 25ms)
            # First load exports to ONNX (one-time ~70s), cached to disk after that
            try:
                model = SentenceTransformer(
                    "intfloat/multilingual-e5-small",
                    backend="onnx",
                )
                logger.info("Loaded embedding model with ONNX backend")
            except Exception as onnx_err:
                logger.warning(f"ONNX backend failed ({onnx_err}), falling back to PyTorch")
                model = SentenceTransformer("intfloat/multilingual-e5-small")

            # Limit sequence length — product queries are short (≤30 tokens)
            model.max_seq_length = 64

            # Warmup — forces compilation/graph optimization on the cold path
            _ = model.encode("warmup", convert_to_numpy=True)
            return model

        try:
            self._model = await loop.run_in_executor(None, _load_and_optimize)
            latency_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "Loaded embedding model",
                extra={
                    "model": "multilingual-e5-small",
                    "dimensions": 384,
                    "max_seq_length": 64,
                    "backend": "onnx",
                    "latency_ms": round(latency_ms, 1),
                },
            )
        except Exception as e:
            logger.error(
                f"Failed to load embedding model: {e}",
                extra={"model": "multilingual-e5-small", "error": str(e)},
            )
            raise

    async def embed(self, texts: str | list[str]) -> np.ndarray:
        """
        Embed one or more texts with caching.
        
        Args:
            texts: Single string or list of strings to embed
            
        Returns:
            numpy array of shape (n, 384) or (384,) for single text
        """
        if self._model is None:
            await self._load_model()

        # Handle single text with cache
        if isinstance(texts, str):
            if _embedding_cache:
                cached = await _embedding_cache.get(texts)
                if cached is not None:
                    return cached
            
            start = time.perf_counter()
            try:
                loop = asyncio.get_event_loop()
                embedding = await loop.run_in_executor(
                    None,
                    lambda: self._model.encode(texts, convert_to_numpy=True)
                )
                
                latency_ms = (time.perf_counter() - start) * 1000
                metrics.record("embed", latency_ms, success=True)
                
                # Cache single embedding
                if _embedding_cache:
                    await _embedding_cache.set(texts, embedding)
                
                return embedding
            except Exception as e:
                latency_ms = (time.perf_counter() - start) * 1000
                metrics.record("embed", latency_ms, success=False, error_type="EmbeddingError")
                logger.error(
                    f"Embedding failed: {e}",
                    extra={"error": str(e), "text_type": type(texts).__name__},
                )
                raise

        # Handle list of texts
        start = time.perf_counter()
        try:
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: self._model.encode(texts, convert_to_numpy=True)
            )
            
            latency_ms = (time.perf_counter() - start) * 1000
            metrics.record("embed", latency_ms, success=True)
            
            return embeddings
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            metrics.record("embed", latency_ms, success=False, error_type="EmbeddingError")
            logger.error(
                f"Embedding failed: {e}",
                extra={"error": str(e), "text_type": type(texts).__name__},
            )
            raise

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return 384

    @property
    def model_name(self) -> str:
        """Get model name."""
        return "multilingual-e5-small"


# Global accessor
async def get_embedder() -> EmbedderSingleton:
    """Get the global embedder instance."""
    return await EmbedderSingleton.get_instance()


def set_embedding_cache(cache) -> None:
    """Set the embedding cache instance (called during initialization)."""
    global _embedding_cache
    _embedding_cache = cache

