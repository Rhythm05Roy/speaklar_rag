"""Tests for context processing: NER, coreference resolution, query rewriting.

Parametrized Bangla two-turn coref scenarios cover:
  - Basic price query ellipsis
  - Pronoun references (এটার, এটি)
  - Turn-window look-back
  - Non-elliptic queries (false positive prevention)
  - In-memory entity store fallback (no Redis)
"""
import pytest
from context.ner import extract_entities_bn, is_product_mentioned, contains_pronoun, normalize_bn
from context.rewriter import QueryRewriter, rewrite_query
from context.resolver import BanglaContextResolver


# ── NER tests ─────────────────────────────────────────────────────────────────

class TestBanglaNER:
    """Test Bangla NER entity extraction."""

    def test_nfc_normalization_applied(self):
        """NFC normalization should not break extraction."""
        # Both NFC and NFD forms of "চাল" should extract the same entity
        query = normalize_bn("চাল এবং ডাল কিনতে চাই")
        entities = extract_entities_bn(query)
        assert any("চাল" in e for e in entities)

    def test_extract_product_names(self):
        entities = extract_entities_bn("চাল এবং ডাল কিনতে চাই")
        assert len(entities) > 0
        assert any("চাল" in e for e in entities)

    def test_extract_noodles(self):
        entities = extract_entities_bn("নুডুলসের দাম কত?")
        assert any("নুডুলস" in e for e in entities)

    def test_no_generic_char_fallback(self):
        """Question words must NOT be extracted as entities."""
        entities = extract_entities_bn("দাম কত?")
        assert "দাম" not in entities
        assert "কত" not in entities

    def test_pronoun_detection(self):
        assert contains_pronoun("এটার দাম কত?") is True
        assert contains_pronoun("এটি পাবো কোথায়?") is True
        assert contains_pronoun("নুডুলসের দাম কত?") is False  # has explicit entity

    def test_no_entities_for_generic_query(self):
        entities = extract_entities_bn("আজ কেমন আছেন?")
        assert isinstance(entities, list)
        # No product entities expected
        assert len(entities) == 0

    def test_is_product_mentioned(self):
        assert is_product_mentioned("নুডুলসের দাম কত?") is True


# ── QueryRewriter tests ───────────────────────────────────────────────────────

class TestQueryRewriter:
    """Test query rewriting for ellipsis and pronoun resolution."""

    def test_possessive_consonant_ending(self):
        assert QueryRewriter._inject_possessive("চাল") == "চালের"

    def test_possessive_noodles(self):
        assert QueryRewriter._inject_possessive("নুডুলস") == "নুডুলসের"

    def test_rewrite_elliptic(self):
        result = QueryRewriter.rewrite_elliptic_query("দাম কত?", "চাল")
        assert "চালের" in result
        assert "দাম" in result

    def test_rewrite_pronoun(self):
        result = QueryRewriter.rewrite_pronoun_query("এটার দাম কত?", "নুডুলস")
        assert "নুডুলসের" in result
        assert "এটার" not in result

    def test_rewrite_with_none_entity(self):
        assert rewrite_query("দাম কত?", None) == "দাম কত?"

    def test_already_possessive_not_doubled(self):
        """Entity ending in 'ের' should not get double suffix."""
        result = QueryRewriter._inject_possessive("চালের")
        assert result == "চালের"


# ── Coreference resolution scenarios ─────────────────────────────────────────

# Parametrized: (Q1, Q2, expected_substring_in_rewritten_Q2)
COREF_SCENARIOS = [
    # Basic price ellipsis
    ("আপনাদের কি নুডুলস বিক্রি করেন?", "দাম কত?", "নুডুলস"),
    ("চালের দাম কত?", "কত টাকা?", "চাল"),
    ("তেলের দাম জানতে চাই", "কত দাম?", "তেল"),
    # Pronoun reference (must be price-trigger OR pronoun to fire resolution)
    ("আপনাদের কি মুরগি আছে?", "এটার দাম কত?", "মুরগি"),
    ("ডিমের দাম কত?", "এটার দাম কত?", "ডিম"),
    # Short price follow-up
    ("আলুর কেজি দর কত?", "কত টাকা?", "আলু"),
]


class TestContextResolver:
    """Test coreference resolution across conversation turns."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("q1,q2,expected", COREF_SCENARIOS)
    async def test_coref_resolution(self, session_store, q1, q2, expected):
        """Two-turn scenario: entity from Q1 injected into Q2."""
        resolver = BanglaContextResolver(session_store)
        sid = f"test_{q1[:10].replace(' ', '_')}"

        # Q1 — establish entity
        result1 = await resolver.resolve(sid, q1)
        assert len(result1.entities) > 0, f"No entities found in Q1: '{q1}'"

        # Q2 — elliptic follow-up
        result2 = await resolver.resolve(sid, q2)
        assert expected in result2.rewritten, (
            f"Expected '{expected}' in rewritten '{result2.rewritten}' for Q2='{q2}'"
        )

    @pytest.mark.asyncio
    async def test_no_false_positive_with_explicit_entity(self, session_store):
        """Query with explicit entity must NOT be rewritten."""
        resolver = BanglaContextResolver(session_store)
        sid = "test_no_fp"

        # Q1 with entity
        await resolver.resolve(sid, "নুডুলসের দাম কত?")

        # Q2 also has entity — should NOT replace চাল with নুডুলস
        result = await resolver.resolve(sid, "চালের দাম কত?")
        assert "চাল" in result.rewritten
        assert result.coref_resolved is False

    @pytest.mark.asyncio
    async def test_in_memory_entity_fallback(self):
        """Coreference resolution works without Redis (in-memory fallback)."""
        # Use a store that will definitely fall back to in-memory
        from session.store import SessionStore
        store = SessionStore(redis_url="redis://localhost:16379/0")  # wrong port
        await store.connect()  # silently falls back to in-memory

        resolver = BanglaContextResolver(store)
        sid = "test_no_redis"

        await resolver.resolve(sid, "নুডুলসের দাম কত?")
        result = await resolver.resolve(sid, "দাম কত?")

        # Should still resolve even without Redis
        assert "নুডুলস" in result.rewritten
        await store.disconnect()

    @pytest.mark.asyncio
    async def test_latency_budget(self, session_store):
        """Context resolution must complete within 5ms budget."""
        resolver = BanglaContextResolver(session_store)
        result = await resolver.resolve("latency_test", "দাম কত?")
        assert result.latency_ms < 5.0, (
            f"Context resolution took {result.latency_ms:.1f}ms — exceeds 5ms budget"
        )

    @pytest.mark.asyncio
    async def test_pronoun_resolution(self, session_store):
        """Pronoun 'এটার' in Q2 should be replaced by Q1 entity."""
        resolver = BanglaContextResolver(session_store)
        sid = "test_pronoun"

        await resolver.resolve(sid, "আপনাদের কি মুরগি পাওয়া যায়?")
        result = await resolver.resolve(sid, "এটার দাম কত টাকা?")

        assert "মুরগি" in result.rewritten
        assert "এটার" not in result.rewritten
