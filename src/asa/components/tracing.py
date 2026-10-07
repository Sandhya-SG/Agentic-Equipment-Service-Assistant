"""Langfuse tracing for the agent graph.

Tracing is OPTIONAL: it switches on only when LANGFUSE_PUBLIC_KEY and
LANGFUSE_SECRET_KEY are set, so the app runs unchanged without them. Every tracing
call is wrapped so a Langfuse problem can never break a user request.

What a trace contains
  * the node-by-node run of the LangGraph workflow (via the LangGraph callback);
  * every OpenAI call with model, latency and token usage (via the Langfuse OpenAI
    wrapper, which replaces `openai.OpenAI` before the agents import it, so the
    agent files are not touched);
  * the audit-trail run_id, session and equipment as attributes, and the outcome
    (status, escalated, hazard count, sources) as scores.

Privacy
  Langfuse applies `mask` to every input, output and span metadata it records. By
  default all text is replaced by a placeholder, so traces hold structure, timing,
  tokens and scores but not user or manual text. Set LANGFUSE_CAPTURE_CONTENT=true to
  also keep text; it is then PII-redacted and truncated first. Attributes set with
  `propagate_attributes` and scores are not masked, so they carry only short
  identifiers and numbers, never text.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from functools import lru_cache
from typing import Any, Iterator

from dotenv import load_dotenv

from asa.components.logging_sub import redact

# Local runs keep the Langfuse keys in .env. Load it now: the agents only do so when they are
# imported, which is after tracing has already checked for the keys. Real environment
# variables (Docker, CI) win over .env.
load_dotenv()

logger = logging.getLogger(__name__)

_MAX_CONTENT_CHARS = 2000
_OMITTED = "[content not captured]"


def tracing_enabled() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def capture_content() -> bool:
    return os.getenv("LANGFUSE_CAPTURE_CONTENT", "false").lower() == "true"


def _omit(value: Any) -> Any:
    """Keep numbers, booleans, None and the shape of dicts and lists; drop everything else.

    Strings AND any other object (dataclasses such as retrieved chunks, models, ...) are
    replaced, because Langfuse would serialise such an object later, after masking.
    """
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(k): _omit(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_omit(v) for v in value]
    return _OMITTED


def _plain(value: Any, depth: int = 0) -> Any:
    """Turn arbitrary objects into plain data so they can be redacted and clipped."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if depth > 8:
        return str(value)
    if isinstance(value, dict):
        return {str(k): _plain(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(v, depth + 1) for v in value]
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value), depth + 1)
    if hasattr(value, "model_dump"):
        return _plain(value.model_dump(), depth + 1)
    return str(value)


def _clip(value: Any) -> Any:
    if isinstance(value, str):
        return value[:_MAX_CONTENT_CHARS]
    if isinstance(value, dict):
        return {k: _clip(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clip(v) for v in value]
    return value


def mask(*, data: Any) -> Any:
    """Langfuse mask hook: runs on every input, output and metadata before it is stored."""
    try:
        if capture_content():
            return _clip(redact(_plain(data)))
        return _omit(data)
    except Exception:
        return _OMITTED


@lru_cache(maxsize=1)
def _client():
    if not tracing_enabled():
        return None
    try:
        from langfuse import Langfuse

        return Langfuse(
            mask=mask,
            base_url=os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST") or None,
            environment=os.getenv("ENVIRONMENT", "development"),
        )
    except Exception as exc:
        logger.warning("Langfuse unavailable, tracing off: %s", type(exc).__name__)
        return None


def get_tracer():
    """Return the Langfuse client, or None when tracing is not configured."""
    return _client()


def reset_tracer() -> None:
    _client.cache_clear()


def instrument_openai() -> bool:
    """Swap openai.OpenAI for Langfuse's wrapper so every model call reports tokens and latency.

    Call this BEFORE the agent modules are imported: they do `from openai import OpenAI`.
    """
    if get_tracer() is None:
        return False
    try:
        import openai
        from langfuse.openai import OpenAI as TracedOpenAI

        openai.OpenAI = TracedOpenAI
        return True
    except Exception as exc:
        logger.warning("Langfuse OpenAI instrumentation failed: %s", type(exc).__name__)
        return False


def graph_config() -> dict:
    """LangGraph run config that traces every node. Empty when tracing is off."""
    if get_tracer() is None:
        return {}
    try:
        from langfuse.langchain import CallbackHandler

        return {"callbacks": [CallbackHandler()]}
    except Exception as exc:
        logger.warning("Langfuse graph callback unavailable: %s", type(exc).__name__)
        return {}


@contextmanager
def trace_request(
    *, session_id: str | None = None, equipment_model: str | None = None, name: str = "chat-request"
) -> Iterator[None]:
    """Open one trace for a chat request. A no-op when tracing is off or fails to start."""
    client = get_tracer()
    entered = []
    if client is not None:
        try:
            from langfuse import propagate_attributes

            observation = client.start_as_current_observation(name=name, as_type="span")
            observation.__enter__()
            entered.append(observation)
            attributes = propagate_attributes(
                trace_name=name,
                session_id=session_id,
                metadata={"equipment_model": equipment_model} if equipment_model else None,
            )
            attributes.__enter__()
            entered.append(attributes)
        except Exception as exc:
            logger.warning("Langfuse trace not started: %s", type(exc).__name__)
    try:
        yield
    except BaseException as exc:
        _close(entered, type(exc), exc)
        raise
    else:
        _close(entered, None, None)


def _close(entered: list, exc_type, exc) -> None:
    for context in reversed(entered):
        try:
            context.__exit__(exc_type, exc, None)
        except Exception as err:
            logger.warning("Langfuse trace not closed cleanly: %s", type(err).__name__)


def record_outcome(*, status: str, hazard_count: int = 0, source_count: int = 0, escalated: bool = False) -> None:
    """Attach the outcome to the current trace as scores (structured data, never text)."""
    client = get_tracer()
    if client is None:
        return
    try:
        client.score_current_trace(name="status", value=status, data_type="CATEGORICAL")
        client.score_current_trace(name="escalated", value=1 if escalated else 0, data_type="BOOLEAN")
        client.score_current_trace(name="hazard_count", value=hazard_count, data_type="NUMERIC")
        client.score_current_trace(name="source_count", value=source_count, data_type="NUMERIC")
    except Exception as exc:
        logger.warning("Langfuse outcome not recorded: %s", type(exc).__name__)


@contextmanager
def tag_run(run_id: str | None) -> Iterator[None]:
    """Tag everything traced inside this block with the audit-trail run_id, so a trace and its
    audit-log entries can be matched. A no-op when tracing is off."""
    if get_tracer() is None or not run_id:
        yield
        return
    try:
        from langfuse import propagate_attributes

        context = propagate_attributes(tags=[f"run:{run_id}"])
        context.__enter__()
    except Exception as exc:
        logger.warning("Langfuse run tag not set: %s", type(exc).__name__)
        yield
        return
    try:
        yield
    except BaseException as exc:
        _close([context], type(exc), exc)
        raise
    else:
        _close([context], None, None)


def flush_tracing() -> None:
    """Send buffered traces. Call on shutdown."""
    client = get_tracer()
    if client is None:
        return
    try:
        client.flush()
    except Exception as exc:
        logger.warning("Langfuse flush failed: %s", type(exc).__name__)
