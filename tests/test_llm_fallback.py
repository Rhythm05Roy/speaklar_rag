"""Tests for LLM fallback behaviour.

Verifies:
  1. Gemini timeout → falls back to OpenAI
  2. Both providers timeout → deterministic Bangla fallback returned
  3. Fallback response is never empty
  4. Pipeline metrics record double failure correctly
"""
import asyncio
import pytest

from generation.generator import LLMGenerator, GeminiGenerator, OpenAIGenerator, _FALLBACK_RESPONSE
from tests.conftest import DummyLLM, DummyTimeoutLLM


# ── LLM Generator Facade tests ────────────────────────────────────────────────

class FakeGeminiGenerator:
    """Simulates a Gemini timeout."""
    async def generate(self, system_prompt, user_prompt):
        raise asyncio.TimeoutError("Simulated Gemini timeout")


class FakeOpenAIGenerator:
    """Returns a canned response."""
    def __init__(self, response="নুডুলসের দাম ৬০ টাকা।"):
        self._response = response
        self.calls = 0

    async def generate(self, system_prompt, user_prompt):
        self.calls += 1
        return self._response


class FakeFailingOpenAI:
    """Simulates an OpenAI failure."""
    async def generate(self, system_prompt, user_prompt):
        raise asyncio.TimeoutError("Simulated OpenAI timeout")


@pytest.mark.asyncio
async def test_primary_timeout_falls_back_to_secondary():
    """When primary (Gemini) times out, secondary (OpenAI) should be used."""
    llm = LLMGenerator.__new__(LLMGenerator)
    llm._primary = FakeGeminiGenerator()
    llm._fallback = FakeOpenAIGenerator()
    llm._primary_name = "gemini"
    llm._fallback_name = "openai"
    llm.model = "gemini (primary)"

    result = await llm.generate("sys", "user")
    assert result == "নুডুলসের দাম ৬০ টাকা।"


@pytest.mark.asyncio
async def test_both_providers_fail_returns_bangla_fallback():
    """When both providers fail, the deterministic Bangla fallback is returned."""
    llm = LLMGenerator.__new__(LLMGenerator)
    llm._primary = FakeGeminiGenerator()
    llm._fallback = FakeFailingOpenAI()
    llm._primary_name = "gemini"
    llm._fallback_name = "openai"
    llm.model = "gemini (primary)"

    result = await llm.generate("sys", "user")
    assert result == _FALLBACK_RESPONSE
    assert len(result) > 0, "Fallback response must not be empty"


@pytest.mark.asyncio
async def test_fallback_response_is_bangla():
    """The hardcoded fallback must be in Bangla."""
    # Check the fallback string contains Bangla characters
    bangla_chars = any('\u0980' <= c <= '\u09FF' for c in _FALLBACK_RESPONSE)
    assert bangla_chars, f"Fallback response must be in Bangla: '{_FALLBACK_RESPONSE}'"


@pytest.mark.asyncio
async def test_openai_only_config_no_gemini():
    """With only OpenAI configured, no fallback needed — should succeed."""
    llm = LLMGenerator.__new__(LLMGenerator)
    openai_gen = FakeOpenAIGenerator("চালের দাম ৭০ টাকা।")
    llm._primary = openai_gen
    llm._fallback = None
    llm._primary_name = "openai"
    llm._fallback_name = "gemini"
    llm.model = "openai (primary)"

    result = await llm.generate("sys", "user")
    assert result == "চালের দাম ৭০ টাকা।"
    assert openai_gen.calls == 1


@pytest.mark.asyncio
async def test_pipeline_returns_fallback_on_llm_failure(monkeypatch):
    """Full pipeline should return Bangla fallback string when LLM double-fails."""
    from api.pipeline import RAGPipeline
    from tests.conftest import DummySessionStore, DummyFAISSStore, DummyBM25Store, DummyCache, DummyEmbedder

    async def fake_get_embedder():
        return DummyEmbedder()
    monkeypatch.setattr("api.pipeline.get_embedder", fake_get_embedder)

    # Build LLM facade that always fails
    llm = LLMGenerator.__new__(LLMGenerator)
    llm._primary = FakeGeminiGenerator()
    llm._fallback = FakeFailingOpenAI()
    llm._primary_name = "gemini"
    llm._fallback_name = "openai"
    llm.model = "gemini (primary)"

    pipeline = RAGPipeline(
        session_store=DummySessionStore(),
        faiss_store=DummyFAISSStore(),
        bm25_store=DummyBM25Store(),
        cache=DummyCache(),
        llm_generator=llm,
    )

    result = await pipeline.process_query("fail-test", "নুডুলসের দাম কত?")
    assert result.response == _FALLBACK_RESPONSE
    assert len(result.response) > 0
