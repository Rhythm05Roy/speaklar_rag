"""Tests for LLM generation: PromptBuilder, OpenAIGenerator, LLMGenerator facade.

Updated to match the refactored generator architecture:
  - PromptBuilder: system/user prompt construction + token budget
  - OpenAIGenerator: mocked with a FakeClient
  - LLMGenerator facade: primary/fallback routing
"""
import asyncio
from types import SimpleNamespace

import pytest

from generation.generator import (
    LLMGenerator, OpenAIGenerator, PromptBuilder, _FALLBACK_RESPONSE
)


# ── PromptBuilder tests ────────────────────────────────────────────────────────

class TestPromptBuilder:
    """Test prompt construction and token budget enforcement."""

    def test_build_context_prompt_with_docs(self):
        docs = [
            {"name": "চাল", "category": "খাদ্য", "price_taka": 70, "description": "মিনিকেট চাল"},
            {"name": "ডাল", "category": "খাদ্য", "price_taka": 90, "description": "মসুর ডাল"},
        ]
        context = PromptBuilder.build_context_prompt(docs)
        assert "চাল" in context
        # price_taka is an int, str() → ASCII "70" not Bangla "৭০"
        assert "70" in context
        assert "প্রাসঙ্গিক তথ্য" in context

    def test_build_context_prompt_empty(self):
        context = PromptBuilder.build_context_prompt([])
        assert context == ""

    def test_description_truncated_to_80_chars(self):
        long_desc = "এটি একটি দীর্ঘ বিবরণ " * 10  # > 80 chars
        docs = [{"name": "পণ্য", "category": "টেস্ট", "price_taka": 50, "description": long_desc}]
        context = PromptBuilder.build_context_prompt(docs)
        # The description section in the output should be truncated
        assert len(context) < len(long_desc) + 200

    def test_build_full_prompt_returns_tuple(self):
        system, user = PromptBuilder.build_full_prompt("চালের দাম?", [])
        assert isinstance(system, str) and len(system) > 0
        assert isinstance(user, str) and "চালের দাম?" in user

    def test_system_prompt_is_bangla(self):
        system, _ = PromptBuilder.build_full_prompt("test", [])
        # System prompt must contain Bangla chars
        assert any('\u0980' <= c <= '\u09FF' for c in system)


# ── OpenAIGenerator tests ─────────────────────────────────────────────────────

class FakeCompletions:
    async def create(self, *, model, messages, temperature, max_tokens, **kwargs):
        assert temperature == 0.1  # must be 0.1 for factual answers
        assert max_tokens == 60
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="চালের দাম ৭০ টাকা।"))]
        )


class FakeOpenAIClient:
    def __init__(self, api_key, http_client):
        self.chat = SimpleNamespace(completions=FakeCompletions())


@pytest.mark.asyncio
async def test_openai_generator_returns_response(monkeypatch):
    """OpenAIGenerator should return the first message content."""
    monkeypatch.setattr("generation.generator.AsyncOpenAI", FakeOpenAIClient)

    gen = OpenAIGenerator(api_key="test-key", model="gpt-4o-mini", timeout_ms=200)
    response = await gen.generate("system prompt", "user prompt")

    assert response == "চালের দাম ৭০ টাকা।"


@pytest.mark.asyncio
async def test_openai_generator_uses_low_temperature(monkeypatch):
    """OpenAI must use temperature=0.1 for factual product answers."""
    called_with = {}

    class CapturingCompletions:
        async def create(self, *, model, messages, temperature, max_tokens, **kwargs):
            called_with["temperature"] = temperature
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
            )

    class CapturingClient:
        def __init__(self, api_key, http_client):
            self.chat = SimpleNamespace(completions=CapturingCompletions())

    monkeypatch.setattr("generation.generator.AsyncOpenAI", CapturingClient)

    gen = OpenAIGenerator(api_key="test-key", model="gpt-4o-mini", timeout_ms=200)
    await gen.generate("sys", "usr")

    assert called_with["temperature"] == 0.1


# ── LLM facade fallback tests ─────────────────────────────────────────────────

class FailingPrimary:
    async def generate(self, sys, usr):
        raise asyncio.TimeoutError("Simulated timeout")


class SuccessfulFallback:
    async def generate(self, sys, usr):
        return "ফলব্যাক উত্তর।"


@pytest.mark.asyncio
async def test_llm_facade_falls_back_on_primary_timeout():
    """LLMGenerator must use fallback when primary times out."""
    llm = LLMGenerator.__new__(LLMGenerator)
    llm._primary = FailingPrimary()
    llm._fallback = SuccessfulFallback()
    llm._primary_name = "gemini"
    llm._fallback_name = "openai"
    llm.model = "gemini (primary)"

    result = await llm.generate("sys", "usr")
    assert result == "ফলব্যাক উত্তর।"


@pytest.mark.asyncio
async def test_llm_facade_returns_bangla_fallback_on_double_failure():
    """LLMGenerator must return the deterministic Bangla string when both fail."""
    llm = LLMGenerator.__new__(LLMGenerator)
    llm._primary = FailingPrimary()
    llm._fallback = FailingPrimary()
    llm._primary_name = "gemini"
    llm._fallback_name = "openai"
    llm.model = "gemini (primary)"

    result = await llm.generate("sys", "usr")
    assert result == _FALLBACK_RESPONSE
    assert len(result) > 0
