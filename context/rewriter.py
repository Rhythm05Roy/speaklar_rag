"""Query rewriting for coreference resolution.

Handles two cases:
1. Ellipsis: "দাম কত?" + entity "চাল" → "চালের দাম কত?"
2. Pronoun substitution: "এটার দাম?" + entity "নুডুলস" → "নুডুলসের দাম?"
"""
import re
from typing import Optional


# Pronouns that should be replaced by the resolved entity
_PRONOUN_REPLACE_RE = re.compile(
    r"\b(এটার|এটি|এটা|ওটার|ওটা|সেটার|সেটা|এর|তার|ওই|এই)\b",
    re.IGNORECASE,
)


class QueryRewriter:
    """Rewrites elliptic / pronoun queries by injecting entities from context."""

    # Characters that take genitive suffix "র" (vowel-final words)
    VOWEL_ENDINGS: frozenset[str] = frozenset({
        "া", "ি", "ী", "ু", "ূ", "ে", "ৈ", "ো", "ৌ",
        "আ", "ই", "ঈ", "উ", "ঊ", "এ", "ঐ", "ও", "ঔ",
    })

    @classmethod
    def _inject_possessive(cls, entity: str) -> str:
        """
        Convert entity to genitive (possessive) form.

        Examples:
            "চাল"    → "চালের"
            "তেল"    → "তেলের"
            "মাছ"    → "মাছের"
            "নুডুলস" → "নুডুলসের"
            "দই"     → "দইয়ের"   (vowel ending — add "য়ের")
        """
        entity = entity.strip()
        if not entity:
            return entity

        # Already possessive
        if entity.endswith("ের") or entity.endswith("এর"):
            return entity

        last_char = entity[-1]

        if last_char in cls.VOWEL_ENDINGS:
            # Short vowel endings — some words prefer "য়ের" to avoid hiatus
            # Simple rule: single-syllable vowel-final → "র", multi → "র"
            return f"{entity}র"

        return f"{entity}ের"

    @classmethod
    def rewrite_elliptic_query(cls, query: str, entity: str) -> str:
        """
        Rewrite an elliptic query by prepending the entity in possessive form.

        Examples:
            ("দাম কত?", "চাল")    → "চালের দাম কত?"
            ("কত টাকা?", "তেল")   → "তেলের কত টাকা?"

        Args:
            query:  Original elliptic query (no subject)
            entity: Entity/subject to inject from context

        Returns:
            Rewritten query with entity prepended in possessive form
        """
        possessive = cls._inject_possessive(entity)
        return f"{possessive} {query.strip()}"

    @classmethod
    def rewrite_pronoun_query(cls, query: str, entity: str) -> str:
        """
        Replace pronoun references in a query with the resolved entity.

        Examples:
            ("এটার দাম কত?", "নুডুলস")  → "নুডুলসের দাম কত?"
            ("এটি কত?", "তেল")           → "তেলের কত?"

        Args:
            query:  Query containing pronoun reference
            entity: Resolved entity to substitute

        Returns:
            Query with pronouns replaced by entity in possessive form
        """
        possessive = cls._inject_possessive(entity)
        rewritten = _PRONOUN_REPLACE_RE.sub(possessive, query.strip())
        return rewritten

    @classmethod
    def rewrite(cls, query: str, entity: str, has_pronoun: bool = False) -> str:
        """
        Unified rewrite entry point.

        Chooses pronoun-substitution if pronoun detected; otherwise ellipsis prepend.

        Args:
            query:       Original query
            entity:      Resolved entity from context
            has_pronoun: True if pronoun detected in query

        Returns:
            Fully specified, retriever-ready query
        """
        if has_pronoun:
            return cls.rewrite_pronoun_query(query, entity)
        return cls.rewrite_elliptic_query(query, entity)


# ── Convenience function ──────────────────────────────────────────────────────

def rewrite_query(
    query: str,
    context_entity: Optional[str] = None,
    has_pronoun: bool = False,
) -> str:
    """
    Rewrite a query if a context entity is provided.

    Args:
        query:          Query string
        context_entity: Entity from previous turn context (or None)
        has_pronoun:    Whether the query contains a pronoun reference

    Returns:
        Rewritten or original query
    """
    if context_entity:
        return QueryRewriter.rewrite(query, context_entity, has_pronoun=has_pronoun)
    return query
