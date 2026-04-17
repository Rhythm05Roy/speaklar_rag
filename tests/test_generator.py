"""Tests for LLM generation helpers."""
from types import SimpleNamespace

import pytest

from generation.generator import LLMGenerator


class FakeStream:
    """Async stream stub for token streaming tests."""

    def __init__(self, contents):
        self.contents = contents

    def __aiter__(self):
        self._iter = iter(self.contents)
        return self

    async def __anext__(self):
        try:
            content = next(self._iter)
        except StopIteration:
            raise StopAsyncIteration from None

        chunk = SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=content))]
        )
        return chunk


class FakeCompletions:
    """Chat completions stub."""

    async def create(self, *, model, messages, temperature, max_tokens, stream):
        assert model == "test-model"
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert temperature == 0.7
        assert max_tokens == 150

        if stream:
            return FakeStream(["চালের ", "দাম ", "৭০ টাকা।"])

        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="চালের দাম ৭০ টাকা।"))]
        )


class FakeClient:
    """Minimal AsyncOpenAI replacement."""

    def __init__(self, api_key, http_client):
        self.api_key = api_key
        self.http_client = http_client
        self.chat = SimpleNamespace(completions=FakeCompletions())


@pytest.mark.asyncio
async def test_generate_uses_async_openai_client(monkeypatch):
    """The generator should return the first message content."""
    monkeypatch.setattr("generation.generator.AsyncOpenAI", FakeClient)

    generator = LLMGenerator(api_key="test-key", model="test-model")
    response = await generator.generate("system prompt", "user prompt")

    assert response == "চালের দাম ৭০ টাকা।"


@pytest.mark.asyncio
async def test_generate_streams_tokens(monkeypatch):
    """Streaming mode should yield content chunks in order."""
    monkeypatch.setattr("generation.generator.AsyncOpenAI", FakeClient)

    generator = LLMGenerator(api_key="test-key", model="test-model")
    stream = await generator.generate("system prompt", "user prompt", stream=True)
    chunks = [chunk async for chunk in stream]

    assert chunks == ["চালের ", "দাম ", "৭০ টাকা।"]
