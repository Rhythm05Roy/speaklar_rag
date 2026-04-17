"""Request context propagation using contextvars.

Provides a RequestContext that flows through the entire async call chain
without passing it explicitly as an argument — enables log correlation by
request_id and session_id across all pipeline stages.
"""
import uuid
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RequestContext:
    """Per-request context carrying IDs and timing."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    session_id: str = ""
    start_time: float = field(default_factory=time.perf_counter)

    @property
    def elapsed_ms(self) -> float:
        """Milliseconds since request started."""
        return (time.perf_counter() - self.start_time) * 1000


# ContextVar — async-safe, works across await boundaries without threading issues
_request_ctx: ContextVar[Optional[RequestContext]] = ContextVar(
    "request_ctx", default=None
)


def set_request_context(ctx: RequestContext) -> None:
    """Set the current request context (call at request entry point)."""
    _request_ctx.set(ctx)


def get_request_context() -> Optional[RequestContext]:
    """Get the current request context (returns None outside a request)."""
    return _request_ctx.get()


def new_request_context(session_id: str = "") -> RequestContext:
    """Create and set a new request context. Returns the context."""
    ctx = RequestContext(session_id=session_id)
    set_request_context(ctx)
    return ctx
