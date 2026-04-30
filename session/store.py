"""Session management with Redis backend.

Stores per-session conversation history and entity memory for coreference.
Falls back to an in-memory dict when Redis is unavailable — both history
AND entity store use the same fallback, ensuring coreference works offline.
"""
import json
import time
from typing import Optional, List, Dict, Any, Tuple
import redis.asyncio as aioredis
from config import settings
from utils.logger import logger


class SessionStore:
    """Async Redis-backed session store for multi-turn conversations."""

    def __init__(self, redis_url: Optional[str] = None):
        """Initialize session store with Redis connection (falls back to in-memory)."""
        self.redis_url = redis_url or settings.redis_url
        self.redis: Optional[aioredis.Redis] = None
        self.session_ttl = settings.redis_session_ttl
        self.history_key_template = "session:{sid}:history"
        self.entities_key_template = "session:{sid}:entities"
        self.context_key_template = "session:{sid}:context"
        # In-memory fallback store for both history and entities
        self._in_memory_store: Dict[str, Any] = {}
        self._use_in_memory = False

    async def connect(self) -> None:
        """Establish Redis connection (falls back to in-memory if unavailable)."""
        try:
            self.redis = await aioredis.from_url(self.redis_url, decode_responses=True)
            await self.redis.ping()
            logger.info(
                "Connected to Redis",
                extra={"service": "SessionStore", "redis_url": self.redis_url},
            )
        except Exception as e:
            logger.warning(
                f"Redis unavailable, using in-memory session store: {e}",
                extra={"service": "SessionStore"},
            )
            self._use_in_memory = True
            self.redis = None

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self.redis:
            if hasattr(self.redis, "aclose"):
                await self.redis.aclose()
            else:
                await self.redis.close()
            logger.info("Disconnected from Redis", extra={"service": "SessionStore"})

    # ── History ───────────────────────────────────────────────────────────────

    async def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve conversation history for a session."""
        key = self.history_key_template.format(sid=session_id)
        try:
            if self.redis:
                data = await self.redis.get(key)
                return json.loads(data) if data else []
            # In-memory fallback
            return list(self._in_memory_store.get(key, []))
        except Exception as e:
            logger.error(
                f"Failed to get history: {e}",
                extra={"service": "SessionStore", "session_id": session_id, "error": str(e)},
            )
            return []

    async def append_to_history(self, session_id: str, turn: Dict[str, Any]) -> None:
        """Append a new turn to conversation history."""
        key = self.history_key_template.format(sid=session_id)
        try:
            history = await self.get_history(session_id)
            history.append({**turn, "timestamp": time.time()})
            if self.redis:
                await self.redis.setex(key, self.session_ttl, json.dumps(history))
            else:
                self._in_memory_store[key] = history
        except Exception as e:
            logger.error(
                f"Failed to append to history: {e}",
                extra={"service": "SessionStore", "session_id": session_id, "error": str(e)},
            )

    # ── Entity store ──────────────────────────────────────────────────────────

    async def set_last_entities(self, session_id: str, entities: List[str]) -> None:
        """Store the last turn's entities for coreference resolution.

        Previously this silently no-op'd when Redis was unavailable.
        Now it falls back to in-memory dict so coreference always works.
        """
        key = self.entities_key_template.format(sid=session_id)
        try:
            if self.redis:
                await self.redis.setex(key, self.session_ttl, json.dumps(entities))
            else:
                # In-memory fallback — same pattern as history
                self._in_memory_store[key] = entities
        except Exception as e:
            logger.error(
                f"Failed to set entities: {e}",
                extra={"service": "SessionStore", "session_id": session_id, "error": str(e)},
            )

    async def get_last_entities(self, session_id: str) -> Optional[List[str]]:
        """Retrieve the last turn's entities for coreference resolution.

        Previously this returned None on Redis unavailability.
        Now it reads from in-memory fallback.
        """
        key = self.entities_key_template.format(sid=session_id)
        try:
            if self.redis:
                data = await self.redis.get(key)
                return json.loads(data) if data else None
            # In-memory fallback
            stored = self._in_memory_store.get(key)
            return list(stored) if stored is not None else None
        except Exception as e:
            logger.error(
                f"Failed to get entities: {e}",
                extra={"service": "SessionStore", "session_id": session_id, "error": str(e)},
            )
            return None

    async def get_session_context(
        self, session_id: str
    ) -> Tuple[List[Dict[str, Any]], Optional[List[str]]]:
        """
        Load history and entities in a single pipelined Redis call.

        Returns:
            (history, entities) tuple — entities may be None if no prior turn
        """
        history_key = self.history_key_template.format(sid=session_id)
        entities_key = self.entities_key_template.format(sid=session_id)

        try:
            if self.redis:
                async with self.redis.pipeline(transaction=False) as pipe:
                    pipe.get(history_key)
                    pipe.get(entities_key)
                    h_raw, e_raw = await pipe.execute()
                history = json.loads(h_raw) if h_raw else []
                entities = json.loads(e_raw) if e_raw else None
                return history, entities
            # In-memory fallback
            history = list(self._in_memory_store.get(history_key, []))
            entities_raw = self._in_memory_store.get(entities_key)
            entities = list(entities_raw) if entities_raw is not None else None
            return history, entities
        except Exception as e:
            logger.error(
                f"Failed to get session context: {e}",
                extra={"service": "SessionStore", "session_id": session_id, "error": str(e)},
            )
            return [], None

    # ── Structured target context ────────────────────────────────────────────

    async def set_last_context(self, session_id: str, context: Dict[str, Any]) -> None:
        """Persist structured target context for deterministic follow-up resolution."""
        key = self.context_key_template.format(sid=session_id)
        try:
            if self.redis:
                await self.redis.setex(key, self.session_ttl, json.dumps(context))
            else:
                self._in_memory_store[key] = context
        except Exception as e:
            logger.error(
                f"Failed to set structured context: {e}",
                extra={"service": "SessionStore", "session_id": session_id, "error": str(e)},
            )

    async def get_last_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load the last structured target context for a session."""
        key = self.context_key_template.format(sid=session_id)
        try:
            if self.redis:
                data = await self.redis.get(key)
                return json.loads(data) if data else None
            stored = self._in_memory_store.get(key)
            return dict(stored) if stored is not None else None
        except Exception as e:
            logger.error(
                f"Failed to get structured context: {e}",
                extra={"service": "SessionStore", "session_id": session_id, "error": str(e)},
            )
            return None

    # ── Session management ────────────────────────────────────────────────────

    async def delete_session(self, session_id: str) -> None:
        """Delete all data for a session."""
        history_key = self.history_key_template.format(sid=session_id)
        entities_key = self.entities_key_template.format(sid=session_id)
        context_key = self.context_key_template.format(sid=session_id)
        try:
            if self.redis:
                await self.redis.delete(history_key, entities_key, context_key)
            else:
                self._in_memory_store.pop(history_key, None)
                self._in_memory_store.pop(entities_key, None)
                self._in_memory_store.pop(context_key, None)
            logger.info("Deleted session", extra={"service": "SessionStore", "session_id": session_id})
        except Exception as e:
            logger.error(
                f"Failed to delete session: {e}",
                extra={"service": "SessionStore", "session_id": session_id, "error": str(e)},
            )

    async def exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        key = self.history_key_template.format(sid=session_id)
        try:
            if self.redis:
                return await self.redis.exists(key) > 0
            return key in self._in_memory_store
        except Exception as e:
            logger.error(
                f"Failed to check session existence: {e}",
                extra={"service": "SessionStore", "session_id": session_id, "error": str(e)},
            )
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get Redis stats for monitoring."""
        if not self.redis:
            return {"mode": "in_memory", "keys": len(self._in_memory_store)}
        try:
            info = await self.redis.info()
            return {
                "redis_version": info.get("redis_version", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_mb": round(info.get("used_memory", 0) / (1024 * 1024), 2),
                "db_keys": info.get("db0", {}).get("keys", 0),
                "mode": "redis",
            }
        except Exception as e:
            logger.error(f"Failed to get Redis stats: {e}", extra={"service": "SessionStore", "error": str(e)})
            return {}
