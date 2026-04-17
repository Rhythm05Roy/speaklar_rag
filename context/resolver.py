"""BanglaContextResolver — coreference resolution for multi-turn conversations.

Three-stage resolution pipeline targeting <5ms total:
  Stage 1: NER + pronoun detection         ~1ms  (regex, in-process)
  Stage 2: Ellipsis / pronoun check         <1ms  (rule-based)
  Stage 3: Entity store lookup + rewrite   ~1ms  (Redis GET or in-memory)

Key improvement over baseline:
  - Pronoun unification ("এটার দাম?" → "নুডুলসের দাম?")
  - Turn-window look-back (scans last 3 history turns before giving up)
  - In-memory entity fallback (works even without Redis)
  - Broader ellipsis detection (catches "কত দাম?" and "দাম বলুন" patterns)
"""
import re
import time
from dataclasses import dataclass, field
from typing import Optional, List
from session.store import SessionStore
from context.ner import extract_entities_bn, contains_pronoun, NON_ENTITY_TOKENS, normalize_bn
from context.rewriter import QueryRewriter
from utils.logger import logger
from utils.metrics import metrics


# ── Ellipsis detection patterns ───────────────────────────────────────────────

# Queries starting with interrogative / price words with no explicit subject
_ELLIPSIS_START = re.compile(
    r"^(এটা|এটি|সেটা|ওটা|কি|কত|দাম|মূল্য|কোনটা|কোথায়|কিভাবে|কীভাবে"
    r"|কেমন|কী|বলুন|জানাবেন)\b",
    re.IGNORECASE,
)

# Price / quantity triggers anywhere in the query
_PRICE_TRIGGER = re.compile(
    r"(দাম|মূল্য|কত|কত টাকা|cost|price|রেট|rate)",
    re.IGNORECASE,
)

# Verb-free short queries (likely follow-up) — any query ≤ 4 tokens with no entity
_SHORT_QUERY_MAX_TOKENS = 4


@dataclass
class ResolvedQuery:
    """Result of context resolution."""
    original: str
    rewritten: str
    entities: List[str] = field(default_factory=list)
    coref_resolved: bool = False
    latency_ms: float = 0.0


class BanglaContextResolver:
    """
    Resolves coreference and ellipsis in Bangla multi-turn conversations.

    Example flow:
        Q1: "আপনাদের কি নুডুলস বিক্রি করেন?"  → entity: নুডুলস stored
        Q2: "দাম কত?"                            → rewritten: "নুডুলসের দাম কত?"
        Q3: "এটার ওজন কত?"                       → rewritten: "নুডুলসের ওজন কত?"

    Budget: <5ms total latency
    """

    def __init__(self, session_store: SessionStore):
        """Initialize context resolver with session store."""
        self.store = session_store

    async def resolve(self, session_id: str, query: str) -> ResolvedQuery:
        """
        Resolve context and ellipsis in query for a session.

        Args:
            session_id: Conversation session ID
            query:      Current turn's raw query string

        Returns:
            ResolvedQuery with original and retriever-ready rewritten query
        """
        t0 = time.perf_counter()

        try:
            # NFC normalize
            query = normalize_bn(query)

            # Stage 1: NER on current query
            entities = extract_entities_bn(query)
            has_pronoun = contains_pronoun(query)

            # Stage 2: Decide if resolution is needed
            needs_coref = self._needs_coreference(query, entities, has_pronoun)
            rewritten = query
            coref_resolved = False

            if needs_coref:
                # Stage 3a: Look up last entities from previous turn
                last_entities = await self.store.get_last_entities(session_id)

                # Stage 3b: If still none, walk back through history (max 3 turns)
                if not last_entities:
                    last_entities = await self._scan_history_for_entity(session_id)

                if last_entities:
                    entity = self._pick_best_entity(last_entities)
                    if entity:
                        rewritten = QueryRewriter.rewrite(query, entity, has_pronoun=has_pronoun)
                        entities = [entity]
                        coref_resolved = True

                        logger.debug(
                            f"Resolved coreference: '{query}' → '{rewritten}'",
                            extra={
                                "session_id": session_id,
                                "coref_entity": entity,
                                "has_pronoun": has_pronoun,
                            },
                        )

            # Persist current-turn entities for next turn
            if entities:
                await self.store.set_last_entities(session_id, entities)

            latency_ms = (time.perf_counter() - t0) * 1000
            metrics.record("context_resolve", latency_ms, success=True)

            return ResolvedQuery(
                original=query,
                rewritten=rewritten,
                entities=entities,
                coref_resolved=coref_resolved,
                latency_ms=latency_ms,
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            metrics.record("context_resolve", latency_ms, success=False, error_type="ResolutionError")
            logger.error(
                f"Context resolution failed: {e}",
                extra={"session_id": session_id, "error": str(e)},
            )
            # Safe fallback — return original query, never crash the pipeline
            return ResolvedQuery(
                original=query,
                rewritten=query,
                entities=[],
                coref_resolved=False,
                latency_ms=latency_ms,
            )

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _needs_coreference(query: str, entities: List[str], has_pronoun: bool) -> bool:
        """
        Decide whether this query requires coreference resolution.

        Triggers resolution when:
        1. Query contains a pronoun (এটা, এটার, তার, …) with no own entity
        2. Query starts with an ellipsis marker AND has no entity AND has a price trigger
        3. Query is very short (≤4 tokens) AND has no entity AND has a price trigger

        Avoids false positives:
        - "চালের দাম কত?" → has entity চাল → NOT elliptic
        - "আজ দাম কীভাবে?" → starts with আজ (not a marker) → NOT elliptic
        """
        # Pronoun present + no own entity → definitely needs resolution
        if has_pronoun and not entities:
            return True

        # Already has an explicit entity — nothing to resolve
        if entities:
            return False

        # Ellipsis marker at start + price trigger
        if _ELLIPSIS_START.match(query) and _PRICE_TRIGGER.search(query):
            return True

        # Very short query with price trigger — likely a follow-up
        tokens = query.strip().split()
        if len(tokens) <= _SHORT_QUERY_MAX_TOKENS and _PRICE_TRIGGER.search(query):
            return True

        return False

    async def _scan_history_for_entity(self, session_id: str) -> Optional[List[str]]:
        """
        Scan back through conversation history (last 3 turns) to find an entity.

        This is the turn-window look-back fallback when last_entities is empty.
        """
        try:
            history = await self.store.get_history(session_id)
            for turn in reversed(history[-3:]):
                resolved_q = turn.get("resolved_query", "")
                original_q = turn.get("original_query", "")
                for q in (resolved_q, original_q):
                    if q:
                        past_entities = extract_entities_bn(q)
                        valid = [e for e in past_entities if self._is_valid_context_entity(e)]
                        if valid:
                            return valid
        except Exception as e:
            logger.warning(f"History scan failed: {e}")
        return None

    @staticmethod
    def _pick_best_entity(entities: List[str]) -> Optional[str]:
        """Pick the most suitable entity from a list (first valid one)."""
        for e in entities:
            if BanglaContextResolver._is_valid_context_entity(e):
                return e
        return None

    @staticmethod
    def _is_valid_context_entity(entity: str) -> bool:
        """Filter out question words and other non-entities."""
        if not entity or len(entity) < 2:
            return False
        return entity not in NON_ENTITY_TOKENS
