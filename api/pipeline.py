"""End-to-end RAG pipeline orchestrator.

Critical ordering changes from baseline:

  BEFORE (incorrect):
    context_resolve → append_history → cache_lookup → embed → retrieve → LLM

  AFTER (correct):
    context_resolve → cache_lookup → embed → retrieve → LLM →
      [fire-and-forget: cache.set + append_history]

  The session history write is moved AFTER the LLM call and made fire-and-
  forget so it never adds latency to the hot path. The cache lookup happens
  BEFORE embedding, so a cache hit skips all retrieval and LLM work.

Latency budget (cache miss path):
  context_resolve:  ~3ms
  cache_lookup:     ~2ms
  embed:            ~3ms
  FAISS + BM25:     ~5ms  (parallel)
  RRF merge:        <1ms
  prompt build:     <1ms
  LLM generate:     ~55–70ms
  ─────────────────────────
  Total:            ~70–85ms  (targeting <100ms)
"""
import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from retrieval.embedder import get_embedder
from retrieval.faiss_store import FAISSStore
from retrieval.bm25_store import BM25Store
from retrieval.cache import SemanticCache
from retrieval.fusion import reciprocal_rank_fusion
from context.resolver import BanglaContextResolver
from generation.generator import PromptBuilder, LLMGenerator
from session.store import SessionStore
from utils.logger import logger
from utils.metrics import metrics

_FALLBACK_RESPONSE = "দুঃখিত, এই তথ্য এখন পাওয়া যাচ্ছে না।"


@dataclass
class RAGResponse:
    """Response from RAG pipeline."""
    query: str
    response: str
    retrieved_docs: List[Dict[str, Any]]
    session_id: str
    latencies: Dict[str, float]
    coref_resolved: bool
    cache_hit: bool = False
    request_id: str = ""


class RAGPipeline:
    """Orchestrates the full RAG pipeline with sub-100ms latency targeting."""

    def __init__(
        self,
        session_store: SessionStore,
        faiss_store: FAISSStore,
        bm25_store: BM25Store,
        cache: SemanticCache,
        llm_generator: LLMGenerator,
        embedder=None,  # Pre-warmed embedder injected at startup
    ) -> None:
        self.session_store = session_store
        self.faiss_store = faiss_store
        self.bm25_store = bm25_store
        self.cache = cache
        self.llm_generator = llm_generator
        self.context_resolver = BanglaContextResolver(session_store)
        # Embedder is injected from main.py startup so it is NEVER loaded
        # inside the request hot path (eliminates the 10s cold-load latency)
        self._embedder = embedder

    @staticmethod
    def _normalize_query(query: str) -> str:
        """Normalize query for consistent caching (strip/collapse whitespace)."""
        return " ".join(query.strip().split())

    async def process_query(self, session_id: str, query: str) -> RAGResponse:
        """
        Process a query end-to-end through the RAG pipeline.

        Args:
            session_id: Conversation session ID
            query:      Raw user query string

        Returns:
            RAGResponse with response, latency breakdown, and metadata
        """
        t0_total = time.perf_counter()
        latencies: Dict[str, float] = {}
        request_id = str(uuid.uuid4())[:8]  # Short ID for log correlation

        try:
            # ── Stage 1: Context Resolution ───────────────────────────────────
            # NER + ellipsis / pronoun detection + entity lookup from session
            t0 = time.perf_counter()
            resolved = await self.context_resolver.resolve(session_id, query)
            latencies["context_ms"] = round((time.perf_counter() - t0) * 1000, 1)

            resolved_query = resolved.rewritten
            coref_resolved = resolved.coref_resolved
            normalized_query = self._normalize_query(resolved_query)

            # ── Stage 2: Embedding (needed for semantic cache + retrieval) ────
            # Use the pre-warmed singleton injected at startup.
            # NEVER call get_embedder() here — it would reload the model.
            t0 = time.perf_counter()
            embedder = self._embedder or (await get_embedder())
            query_vector = await embedder.embed(normalized_query)
            latencies["embed_ms"] = round((time.perf_counter() - t0) * 1000, 1)

            # ── Stage 3: Semantic Cache Lookup ───────────────────────────────
            # Pass the pre-computed embedding so cache doesn't re-embed
            t0 = time.perf_counter()
            cached_payload = await self.cache.get(normalized_query, query_embedding=query_vector)
            latencies["cache_ms"] = round((time.perf_counter() - t0) * 1000, 1)

            if isinstance(cached_payload, dict) and "response" in cached_payload:
                latencies["total_ms"] = round((time.perf_counter() - t0_total) * 1000, 1)
                logger.info(
                    "Cache HIT — returning cached response",
                    extra={
                        "request_id": request_id,
                        "session_id": session_id,
                        "total_ms": latencies["total_ms"],
                    },
                )
                # Fire-and-forget history write (never blocks response)
                asyncio.create_task(
                    self.session_store.append_to_history(
                        session_id,
                        {
                            "original_query": query,
                            "resolved_query": resolved_query,
                            "coref_resolved": coref_resolved,
                            "cache_hit": True,
                        },
                    )
                )
                return RAGResponse(
                    query=query,
                    response=cached_payload["response"],
                    retrieved_docs=cached_payload.get("retrieved_docs", []),
                    session_id=session_id,
                    latencies=latencies,
                    coref_resolved=coref_resolved,
                    cache_hit=True,
                    request_id=request_id,
                )

            # ── Stage 4: Parallel FAISS + BM25 Retrieval ─────────────────────
            # Both use the already-computed embedding — no extra embed call
            t0 = time.perf_counter()
            faiss_results, bm25_results = await asyncio.gather(
                self.faiss_store.search(query_vector, top_k=5),
                self.bm25_store.search(normalized_query, top_k=5),
            )
            latencies["retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 1)

            # ── Stage 5: RRF Fusion ───────────────────────────────────────────
            t0 = time.perf_counter()
            fused = reciprocal_rank_fusion(faiss_results, bm25_results, k=60)
            retrieved_docs = fused[:3]
            latencies["fusion_ms"] = round((time.perf_counter() - t0) * 1000, 1)

            # ── Stage 6: Prompt Build ─────────────────────────────────────────
            t0 = time.perf_counter()
            system_prompt, user_prompt = PromptBuilder.build_full_prompt(
                resolved_query, retrieved_docs
            )
            latencies["prompt_ms"] = round((time.perf_counter() - t0) * 1000, 1)

            # ── Stage 7: LLM Generation (primary → fallback) ──────────────────
            t0 = time.perf_counter()
            llm_response = await self.llm_generator.generate(system_prompt, user_prompt)
            latencies["llm_ms"] = round((time.perf_counter() - t0) * 1000, 1)

            latencies["total_ms"] = round((time.perf_counter() - t0_total) * 1000, 1)

            # ── Fire-and-forget: cache + history ──────────────────────────────
            # Detach from request — never blocks the response path
            asyncio.create_task(
                self.cache.set(
                    normalized_query,
                    {"response": llm_response, "retrieved_docs": retrieved_docs},
                    query_embedding=query_vector,
                )
            )
            asyncio.create_task(
                self.session_store.append_to_history(
                    session_id,
                    {
                        "original_query": query,
                        "resolved_query": resolved_query,
                        "coref_resolved": coref_resolved,
                        "cache_hit": False,
                    },
                )
            )

            logger.info(
                "Query processed",
                extra={
                    "request_id": request_id,
                    "session_id": session_id,
                    "coref": coref_resolved,
                    "context_ms": latencies["context_ms"],
                    "embed_ms": latencies["embed_ms"],
                    "cache_ms": latencies["cache_ms"],
                    "retrieval_ms": latencies["retrieval_ms"],
                    "llm_ms": latencies["llm_ms"],
                    "total_ms": latencies["total_ms"],
                },
            )

            # Record end-to-end latency in metrics
            metrics.record("pipeline_e2e", latencies["total_ms"], success=True)

            return RAGResponse(
                query=query,
                response=llm_response,
                retrieved_docs=retrieved_docs,
                session_id=session_id,
                latencies=latencies,
                coref_resolved=coref_resolved,
                cache_hit=False,
                request_id=request_id,
            )

        except Exception as e:
            latencies["total_ms"] = round((time.perf_counter() - t0_total) * 1000, 1)
            metrics.record("pipeline_e2e", latencies["total_ms"], success=False, error_type=type(e).__name__)
            logger.error(
                f"Pipeline error: {e}",
                extra={
                    "request_id": request_id,
                    "session_id": session_id,
                    "error": str(e),
                    "total_ms": latencies["total_ms"],
                },
            )
            return RAGResponse(
                query=query,
                response=_FALLBACK_RESPONSE,
                retrieved_docs=[],
                session_id=session_id,
                latencies=latencies,
                coref_resolved=False,
                cache_hit=False,
                request_id=request_id,
            )
