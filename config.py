"""Application configuration and settings."""
import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ── OpenAI Configuration ─────────────────────────────────────────────────
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    # Realistic timeout for LLM APIs — TCP+TLS alone can take 200ms+ on first call.
    # The <100ms SLA targets the retrieval path (context+embed+faiss+bm25 = ~30ms).
    openai_timeout_ms: int = 5000

    # ── Google Gemini Configuration ───────────────────────────────────────────
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    gemini_timeout_ms: int = 5000

    # ── Groq Configuration (LPU — ultra-low latency) ─────────────────────────
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    groq_timeout_ms: int = 5000

    # ── LLM Routing ───────────────────────────────────────────────────────────
    # "groq" | "gemini" | "openai"  — primary provider; the others are fallbacks
    llm_primary: str = "groq"
    llm_fallback: str = "openai"

    # ── Redis Configuration ───────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    redis_cache_ttl: int = 3600
    redis_session_ttl: int = 3600

    # ── Semantic Cache ────────────────────────────────────────────────────────
    # Cosine similarity threshold for semantic cache hit (0–1)
    cache_similarity_threshold: float = 0.92
    # Maximum in-process LRU cache entries (each ~1.5 KB for 384-dim float32)
    semantic_cache_max_entries: int = 2000

    # ── Application Configuration ─────────────────────────────────────────────
    debug: bool = False
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── Feature Flags ─────────────────────────────────────────────────────────
    enable_cache: bool = True
    enable_monitoring: bool = True

    # ── Paths ─────────────────────────────────────────────────────────────────
    data_dir: str = "./data"
    index_dir: str = "./data/indexes"

    class Config:
        env_file = ".env"
        case_sensitive = False

    @property
    def data_path(self) -> Path:
        """Get data directory path."""
        path = Path(self.data_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def index_path(self) -> Path:
        """Get index directory path."""
        path = Path(self.index_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def validate_startup(self) -> list[str]:
        """Validate configuration on startup. Returns list of validation errors."""
        errors = []

        # At least one LLM must be configured
        if not self.openai_api_key and not self.gemini_api_key:
            errors.append(
                "No LLM API key configured. Set OPENAI_API_KEY and/or GEMINI_API_KEY."
            )

        if not self.redis_url:
            errors.append("REDIS_URL is required but not set")

        if self.llm_primary not in ("groq", "gemini", "openai"):
            errors.append("LLM_PRIMARY must be 'groq', 'gemini', or 'openai'")

        return errors


# Global settings instance
settings = Settings()
