"""Metrics collection: in-memory stats + Prometheus export.

Adds prometheus_client Histograms and Counters alongside the existing
in-memory MetricsCollector. Prometheus metrics are exposed at /metrics
in the standard text format.
"""
import time
from collections import defaultdict
from threading import Lock
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta

try:
    from prometheus_client import (
        Histogram, Counter, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False


# ── Prometheus metrics (no-op if library not installed) ───────────────────────

if _PROMETHEUS_AVAILABLE:
    _registry = CollectorRegistry()

    _REQUEST_LATENCY = Histogram(
        "rag_request_latency_ms",
        "End-to-end RAG request latency in ms",
        buckets=[10, 25, 50, 75, 100, 150, 200, 500, 1000],
        registry=_registry,
    )
    _LLM_LATENCY = Histogram(
        "rag_llm_latency_ms",
        "LLM generation latency in ms",
        ["provider"],
        buckets=[10, 30, 50, 75, 100, 150, 200],
        registry=_registry,
    )
    _RETRIEVAL_LATENCY = Histogram(
        "rag_retrieval_latency_ms",
        "Retrieval latency in ms (FAISS + BM25 parallel)",
        buckets=[1, 3, 5, 10, 20, 50],
        registry=_registry,
    )
    _CACHE_HITS = Counter(
        "rag_cache_hits_total",
        "Number of semantic cache hits",
        registry=_registry,
    )
    _CACHE_MISSES = Counter(
        "rag_cache_misses_total",
        "Number of semantic cache misses",
        registry=_registry,
    )
    _COREF_RESOLVED = Counter(
        "rag_coref_resolved_total",
        "Number of queries resolved via coreference",
        registry=_registry,
    )
    _LLM_FALLBACK = Counter(
        "rag_llm_fallback_total",
        "Number of times the LLM fallback provider was used",
        registry=_registry,
    )
    _LLM_DOUBLE_FAILURE = Counter(
        "rag_llm_double_failure_total",
        "Number of times both LLM providers failed",
        registry=_registry,
    )


def prometheus_metrics_text() -> tuple[bytes, str]:
    """Return Prometheus metrics in text format."""
    if _PROMETHEUS_AVAILABLE:
        return generate_latest(_registry), CONTENT_TYPE_LATEST
    return b"# prometheus_client not installed\n", "text/plain"


# ── In-memory snapshot store ──────────────────────────────────────────────────

@dataclass
class MetricSnapshot:
    """Point-in-time metric reading."""
    timestamp: datetime
    operation: str
    latency_ms: float
    success: bool
    error_type: Optional[str] = None


class MetricsCollector:
    """In-memory metrics + Prometheus bridge."""

    def __init__(self, window_size: int = 1000) -> None:
        self.window_size = window_size
        self.metrics: Dict[str, List[MetricSnapshot]] = defaultdict(list)
        self.lock = Lock()

    def record(
        self,
        operation: str,
        latency_ms: float,
        success: bool = True,
        error_type: Optional[str] = None,
    ) -> None:
        """Record a metric snapshot and update Prometheus counters."""
        snapshot = MetricSnapshot(
            timestamp=datetime.now(),
            operation=operation,
            latency_ms=latency_ms,
            success=success,
            error_type=error_type,
        )
        with self.lock:
            self.metrics[operation].append(snapshot)
            if len(self.metrics[operation]) > self.window_size:
                self.metrics[operation] = self.metrics[operation][-self.window_size:]

        # ── Update Prometheus metrics ──────────────────────────────────────
        if _PROMETHEUS_AVAILABLE:
            if operation == "pipeline_e2e" and success:
                _REQUEST_LATENCY.observe(latency_ms)
            elif operation in ("gemini_generate", "openai_generate") and success:
                provider = "gemini" if "gemini" in operation else "openai"
                _LLM_LATENCY.labels(provider=provider).observe(latency_ms)
            elif operation == "faiss_search" and success:
                _RETRIEVAL_LATENCY.observe(latency_ms)
            elif operation == "cache_hit":
                _CACHE_HITS.inc()
            elif operation == "cache_miss":
                _CACHE_MISSES.inc()
            elif operation == "context_resolve" and success:
                # Coreference counter bumped in pipeline, not here
                pass
            elif operation == "llm_fallback_used":
                _LLM_FALLBACK.inc()
            elif operation == "llm_double_failure":
                _LLM_DOUBLE_FAILURE.inc()

    def get_stats(self, operation: str, time_window_minutes: int = 5) -> Dict[str, float]:
        """Get aggregated stats for an operation within a time window."""
        cutoff = datetime.now() - timedelta(minutes=time_window_minutes)
        with self.lock:
            snapshots = [m for m in self.metrics.get(operation, []) if m.timestamp > cutoff]

        if not snapshots:
            return {
                "operation": operation, "count": 0,
                "p50_ms": 0, "p95_ms": 0, "p99_ms": 0,
                "mean_ms": 0, "max_ms": 0, "error_rate": 0,
            }

        latencies = sorted(m.latency_ms for m in snapshots if m.success)
        errors = sum(1 for m in snapshots if not m.success)
        n = len(snapshots)

        return {
            "operation": operation,
            "count": n,
            "p50_ms": round(latencies[len(latencies) // 2] if latencies else 0, 2),
            "p95_ms": round(latencies[int(len(latencies) * 0.95)] if latencies else 0, 2),
            "p99_ms": round(latencies[int(len(latencies) * 0.99)] if latencies else 0, 2),
            "mean_ms": round(sum(latencies) / len(latencies) if latencies else 0, 2),
            "max_ms": round(max(latencies) if latencies else 0, 2),
            "error_rate": round((errors / n) * 100 if n else 0, 2),
        }

    def get_all_stats(self, time_window_minutes: int = 5) -> Dict[str, Dict[str, float]]:
        """Get aggregated stats for all recorded operations."""
        with self.lock:
            operations = list(self.metrics.keys())
        return {op: self.get_stats(op, time_window_minutes) for op in operations}

    def clear(self) -> None:
        """Clear all in-memory metrics."""
        with self.lock:
            self.metrics.clear()


# Global singleton
metrics = MetricsCollector()
