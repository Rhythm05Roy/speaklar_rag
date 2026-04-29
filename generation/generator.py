"""LLM integration: Groq LPU (primary) + Gemini/OpenAI (fallback).

Latency budget: <90ms total for LLM generation.
Design:
  - GroqGenerator: primary, targets 40–80ms TTFT on LPU hardware
  - GeminiGenerator: fallback, targets 60–100ms TTFT
  - OpenAIGenerator: fallback, targets 200–500ms TTFT
  - LLMGenerator facade: tries primary, falls back on timeout/error
  - Deterministic Bangla fallback: returned if all providers fail
  - temperature=0.1 for factual product answers
  - Prompt token budget enforced: ~100 system + ~200 context + ~20 query
"""
import asyncio
import time
from typing import List, Dict, Any, Optional, AsyncIterator

import httpx
from openai import AsyncOpenAI, APITimeoutError

from config import settings
from utils.logger import logger
from utils.metrics import metrics

# Deterministic fallback when both LLM providers fail
_FALLBACK_RESPONSE = "দুঃখিত, এই তথ্য এখন পাওয়া যাচ্ছে না।"


# ── Prompt Builder ────────────────────────────────────────────────────────────

class PromptBuilder:
    """Builds minimal prompts within the <320 token budget."""

    # System prompt optimized for Llama 3.1 — use English instructions
    # (Llama follows English instructions more reliably) but Bangla output
    SYSTEM_PROMPT_BN = (
        "You are a Bangla product information assistant. "
        "Answer ONLY in Bangla using the product data provided below. "
        "The context section contains the relevant products with their prices. "
        "Give a short, direct answer based on the data. "
        "If the product is NOT in the context, reply: 'দুঃখিত, এই তথ্য উপলব্ধ নয়'।"
    )

    @staticmethod
    def _sanitize(text: str, max_chars: int = 400) -> str:
        """Strip control chars and truncate."""
        text = "".join(c for c in text if c.isprintable() or c.isspace())
        if len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0] + "…"
        return text.strip()

    @staticmethod
    def build_context_prompt(retrieved_docs: List[Dict[str, Any]]) -> str:
        """Build context section from top-3 retrieved docs (~200 tokens max)."""
        if not retrieved_docs:
            return ""
        lines = ["প্রাসঙ্গিক তথ্য:"]
        for i, doc in enumerate(retrieved_docs[:3], 1):
            name = PromptBuilder._sanitize(doc.get("name", "অজানা"), 60)
            cat = PromptBuilder._sanitize(doc.get("category", ""), 30)
            price = PromptBuilder._sanitize(str(doc.get("price_taka", "")), 15)
            # Trim description to 80 chars to stay within token budget
            desc = PromptBuilder._sanitize(doc.get("description", ""), 80)

            line = f"{i}. {name}"
            if cat:
                line += f" ({cat})"
            if price:
                line += f" — দাম: {price} টাকা"
            if desc:
                line += f" — {desc}"
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def build_user_prompt(query: str, context: str) -> str:
        """Combine context and query into user message (~20 query tokens)."""
        query = PromptBuilder._sanitize(query, 200)
        if context:
            return f"{context}\n\nপ্রশ্ন: {query}"
        return query

    @staticmethod
    def build_full_prompt(query: str, retrieved_docs: List[Dict[str, Any]]) -> tuple[str, str]:
        """Return (system_prompt, user_prompt)."""
        system = PromptBuilder.SYSTEM_PROMPT_BN
        context = PromptBuilder.build_context_prompt(retrieved_docs)
        user = PromptBuilder.build_user_prompt(query, context)
        return system, user


# ── Gemini Generator (primary) ────────────────────────────────────────────────

class GeminiGenerator:
    """Google Gemini Flash generator — primary LLM provider.

    Uses the new `google-genai` SDK (google.genai).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        # Product answers are 1 sentence — 60 tokens is plenty. Each extra token = ~15ms
        max_tokens: int = 60,
    ) -> None:
        self.api_key = api_key or settings.gemini_api_key
        self.model = model or "gemini-2.0-flash"
        self.timeout_ms = timeout_ms if timeout_ms is not None else settings.gemini_timeout_ms
        self.timeout_s = self.timeout_ms / 1000.0
        self.max_tokens = max_tokens

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set")

        import google.genai as genai
        import google.genai.types as genai_types
        # google-genai uses httpx internally with keep-alive by default
        self._client = genai.Client(api_key=self.api_key)
        self._genai_types = genai_types

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response using the google-genai SDK."""
        start = time.perf_counter()
        combined = f"{system_prompt}\n\n{user_prompt}"
        try:
            loop = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self._client.models.generate_content(
                        model=self.model,
                        contents=combined,
                        config=self._genai_types.GenerateContentConfig(
                            temperature=0.1,
                            max_output_tokens=self.max_tokens,
                        ),
                    )
                ),
                timeout=self.timeout_s,
            )
            text = response.text or ""
            latency_ms = (time.perf_counter() - start) * 1000
            metrics.record("gemini_generate", latency_ms, success=True)
            logger.info(f"Gemini generated in {latency_ms:.0f}ms")
            return text.strip()
        except asyncio.TimeoutError:
            latency_ms = (time.perf_counter() - start) * 1000
            metrics.record("gemini_timeout", latency_ms, success=False, error_type="TimeoutError")
            logger.warning(f"Gemini timed out after {self.timeout_ms}ms")
            raise
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            metrics.record("gemini_error", latency_ms, success=False, error_type=type(e).__name__)
            logger.error(f"Gemini generation failed: {e}")
            raise


# ── OpenAI Generator (fallback) ───────────────────────────────────────────────

class OpenAIGenerator:
    """OpenAI gpt-4o-mini generator — fallback LLM provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        # Product answers are 1 sentence — 60 tokens is plenty. Each extra token = ~15ms
        max_tokens: int = 60,
    ) -> None:
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.openai_model
        self.timeout_ms = timeout_ms if timeout_ms is not None else settings.openai_timeout_ms
        self.timeout_s = self.timeout_ms / 1000.0
        self.max_tokens = max_tokens

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")

        # Keep-alive pool: reuse TCP connections so subsequent requests skip
        # the ~200ms TCP+TLS handshake overhead
        limits = httpx.Limits(max_keepalive_connections=5, max_connections=10, keepalive_expiry=30)
        self.http_client = httpx.AsyncClient(timeout=self.timeout_s, limits=limits)
        self.client = AsyncOpenAI(api_key=self.api_key, http_client=self.http_client)

    async def close(self) -> None:
        await self.http_client.aclose()

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response with 90ms timeout."""
        start = time.perf_counter()
        try:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=self.max_tokens,
                ),
                timeout=self.timeout_s + 0.5,
            )
            text = response.choices[0].message.content or "" if response.choices else ""
            latency_ms = (time.perf_counter() - start) * 1000
            metrics.record("openai_generate", latency_ms, success=True)
            return text.strip()
        except (asyncio.TimeoutError, APITimeoutError):
            latency_ms = (time.perf_counter() - start) * 1000
            metrics.record("openai_timeout", latency_ms, success=False, error_type="TimeoutError")
            logger.warning(f"OpenAI timed out after {self.timeout_ms}ms")
            raise
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            metrics.record("openai_error", latency_ms, success=False, error_type=type(e).__name__)
            logger.error(f"OpenAI generation failed: {e}")
            raise


# ── Groq Generator (primary — LPU ultra-low latency) ────────────────────────

class GroqGenerator:
    """Groq LPU generator — fastest available inference (~60ms TTFT).

    Groq uses OpenAI-compatible API with custom base_url.
    LPU (Language Processing Unit) hardware delivers ~750-1200 tok/s
    on Llama 3.3 8B, with deterministic latency and minimal variance.
    """

    GROQ_BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        max_tokens: int = 60,
    ) -> None:
        self.api_key = api_key or settings.groq_api_key
        self.model = model or settings.groq_model
        self.timeout_ms = timeout_ms if timeout_ms is not None else settings.groq_timeout_ms
        self.timeout_s = self.timeout_ms / 1000.0
        self.max_tokens = max_tokens

        if not self.api_key:
            raise ValueError("GROQ_API_KEY not set")

        # Keep-alive pool for TCP connection reuse
        limits = httpx.Limits(max_keepalive_connections=5, max_connections=10, keepalive_expiry=30)
        self.http_client = httpx.AsyncClient(timeout=self.timeout_s, limits=limits)
        # Groq is OpenAI-compatible — just set base_url
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.GROQ_BASE_URL,
            http_client=self.http_client,
        )

    async def close(self) -> None:
        await self.http_client.aclose()

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response via Groq LPU."""
        start = time.perf_counter()
        try:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=self.max_tokens,
                ),
                timeout=self.timeout_s,
            )
            text = response.choices[0].message.content or "" if response.choices else ""
            latency_ms = (time.perf_counter() - start) * 1000
            metrics.record("groq_generate", latency_ms, success=True)
            logger.info(f"Groq generated in {latency_ms:.0f}ms ({self.model})")
            return text.strip()
        except (asyncio.TimeoutError, APITimeoutError):
            latency_ms = (time.perf_counter() - start) * 1000
            metrics.record("groq_timeout", latency_ms, success=False, error_type="TimeoutError")
            logger.warning(f"Groq timed out after {self.timeout_ms}ms")
            raise
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            metrics.record("groq_error", latency_ms, success=False, error_type=type(e).__name__)
            logger.error(f"Groq generation failed: {e}")
            raise


# ── LLM Generator Facade ──────────────────────────────────────────────────────

class LLMGenerator:
    """
    Facade that routes to primary LLM, falls back to secondary on failure.

    Provider order controlled by settings.llm_primary:
      "groq"    → primary=Groq (fastest), fallback=OpenAI/Gemini
      "gemini"  → primary=Gemini, fallback=OpenAI
      "openai"  → primary=OpenAI, fallback=Gemini

    On total failure, returns the deterministic Bangla fallback string.
    """

    def __init__(self) -> None:
        self._primary = None
        self._fallback = None
        self._primary_name = settings.llm_primary
        self._fallback_name = settings.llm_fallback
        self.model = f"{self._primary_name} (primary)"

        self._init_providers()

    def _init_providers(self) -> None:
        """Initialize primary and fallback providers based on config."""
        providers: dict = {}

        # Groq (LPU — fastest)
        if settings.groq_api_key:
            try:
                providers["groq"] = GroqGenerator()
            except Exception as e:
                logger.warning(f"Groq init failed: {e}")

        # Gemini
        if settings.gemini_api_key:
            try:
                providers["gemini"] = GeminiGenerator()
            except Exception as e:
                logger.warning(f"Gemini init failed: {e}")

        # OpenAI
        if settings.openai_api_key:
            try:
                providers["openai"] = OpenAIGenerator()
            except Exception as e:
                logger.warning(f"OpenAI init failed: {e}")

        if not providers:
            raise RuntimeError(
                "No LLM provider initialized. Set GROQ_API_KEY, GEMINI_API_KEY, and/or OPENAI_API_KEY."
            )

        # Assign primary / fallback based on config
        self._primary = providers.get(self._primary_name) or next(iter(providers.values()))
        fallback_candidates = [p for k, p in providers.items() if k != self._primary_name]
        self._fallback = fallback_candidates[0] if fallback_candidates else None

        provider_names = list(providers.keys())
        logger.info(
            f"LLM initialized — primary: {self._primary_name}, "
            f"fallback: {self._fallback_name if self._fallback else 'none'}",
            extra={"providers": provider_names},
        )

    async def close(self) -> None:
        """Close HTTP clients."""
        for provider in [self._primary, self._fallback]:
            if isinstance(provider, (OpenAIGenerator, GroqGenerator)):
                await provider.close()

    async def generate(self, system_prompt: str, user_prompt: str, stream: bool = False) -> str:
        """
        Generate response with primary → fallback → deterministic fallback chain.

        Args:
            system_prompt: System message
            user_prompt:   User message (includes context)
            stream:        Ignored — streaming handled at WebSocket layer

        Returns:
            Generated response string
        """
        # Try primary
        if self._primary:
            try:
                return await self._primary.generate(system_prompt, user_prompt)
            except Exception as e:
                logger.warning(
                    f"Primary LLM ({self._primary_name}) failed, trying fallback: {e}"
                )
                metrics.record("llm_fallback_triggered", 0, success=True)

        # Try fallback
        if self._fallback:
            try:
                result = await self._fallback.generate(system_prompt, user_prompt)
                metrics.record("llm_fallback_used", 0, success=True)
                return result
            except Exception as e:
                logger.error(f"Fallback LLM ({self._fallback_name}) also failed: {e}")

        # Both failed — return deterministic Bangla fallback
        metrics.record("llm_double_failure", 0, success=False, error_type="DoubleLLMFailure")
        return _FALLBACK_RESPONSE
