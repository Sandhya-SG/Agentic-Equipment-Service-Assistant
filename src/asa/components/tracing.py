"""Langfuse tracing for the graph.

Tracing is OPTIONAL: it switches on only when LANGFUSE_PUBLIC_KEY and
LANGFUSE_SECRET_KEY are set, so the app runs unchanged without them. Every
tracing call is wrapped so a Langfuse problem can never break a user request.

Privacy: spans record structure (status, hazard names, lengths, latency, token
counts), not user text. Prompt and reply text are attached to the model call only
when LANGFUSE_CAPTURE_CONTENT=true, and are redacted and truncated first.
"""

from __future__ import annotations

import functools
import logging
import os
from functools import lru_cache
from typing import Any, Callable

from asa.components.logging_sub import redact

logger = logging.getLogger(__name__)

_MAX_CONTENT_CHARS = 2000


@lru_cache(maxsize=1)
def get_tracer():
    """Return the Langfuse client, or None when tracing is not configured."""
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return None
    try:
        from langfuse import get_client

        return get_client()
    except Exception as exc:
        logger.warning("Langfuse unavailable, tracing off: %s", type(exc).__name__)
        return None


def reset_tracer() -> None:
    get_tracer.cache_clear()


def capture_content() -> bool:
    return os.getenv("LANGFUSE_CAPTURE_CONTENT", "false").lower() == "true"


def _safe_text(text: str) -> str:
    return redact(text)[:_MAX_CONTENT_CHARS]


def traced(name: str, as_type: str | None = None) -> Callable:
    """Wrap a function in a Langfuse observation. A plain call when tracing is off."""

    def decorator(fn: Callable) -> Callable:
        observed: Callable | None = None

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            nonlocal observed
            if get_tracer() is None:
                return fn(*args, **kwargs)
            if observed is None:
                from langfuse import observe

                observed = observe(name=name, as_type=as_type, capture_input=False, capture_output=False)(fn)
            return observed(*args, **kwargs)

        return wrapper

    return decorator


def annotate(**metadata: Any) -> None:
    """Attach redacted metadata to the current span."""
    client = get_tracer()
    if client is None:
        return
    try:
        client.update_current_span(metadata=redact(metadata))
    except Exception as exc:
        logger.warning("Langfuse annotate failed: %s", type(exc).__name__)


def record_generation(
    model: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    prompt: str = "",
    reply: str = "",
    **metadata: Any,
) -> None:
    """Attach model name, token usage and (optionally) redacted text to the current model call."""
    client = get_tracer()
    if client is None:
        return
    usage = {k: v for k, v in (("input", prompt_tokens), ("output", completion_tokens)) if v is not None}
    fields: dict[str, Any] = {"model": model or None, "metadata": redact(metadata)}
    if usage:
        fields["usage_details"] = usage
    if capture_content():
        fields["input"] = _safe_text(prompt)
        fields["output"] = _safe_text(reply)
    try:
        client.update_current_generation(**fields)
    except Exception as exc:
        logger.warning("Langfuse record failed: %s", type(exc).__name__)


def flush_tracing() -> None:
    """Send buffered traces. Call on shutdown."""
    client = get_tracer()
    if client is None:
        return
    try:
        client.flush()
    except Exception as exc:
        logger.warning("Langfuse flush failed: %s", type(exc).__name__)
