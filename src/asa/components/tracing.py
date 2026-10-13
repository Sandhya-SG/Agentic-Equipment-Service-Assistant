"""Langfuse tracing for the agent graph.

Tracing is OPTIONAL: it switches on only when LANGFUSE_PUBLIC_KEY and
LANGFUSE_SECRET_KEY are set, so the app runs unchanged without them. Every tracing
call is wrapped so a Langfuse problem can never break a user request.

What a trace contains
  * a root `chat-request` span whose input is the redacted question and whose output is
    the response shown to the user (both masked, see Privacy);
  * the guardrails (prompt-injection scan, PII redaction, hazard rules) as `guardrail`
    observations;
  * the node-by-node run of the LangGraph workflow (via the LangGraph callback);
  * every OpenAI call as a named generation with model, latency and token usage (via the
    Langfuse OpenAI integration; agents pass a stable name with `**llm_name(...)`);
  * the RAG specialists as `agent` observations, the document-search tool as a `tool`
    and every manual lookup as a `retriever`;
  * the audit-trail run_id, session and equipment as attributes, and the outcome
    (status, specialist, escalated, hazard count, sources, confidence) as scores.

Privacy
  Langfuse applies `mask` to every input, output and span metadata it records. By
  default all text is replaced by a placeholder, so traces hold structure, timing,
  tokens and scores but not user or manual text. Only short application identifiers
  under known keys (status, specialist, chunk_id, source_file, guardrail categories, ...;
  see _IDENTIFIER_KEYS) are kept. Set LANGFUSE_CAPTURE_CONTENT=true to
  also keep text; it is then PII-redacted and truncated first. Attributes set with
  `propagate_attributes` and scores are not masked, so they carry only short
  identifiers and numbers, never text.
"""

from __future__ import annotations

import functools
import inspect
import logging
import os
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from functools import lru_cache
from typing import Any, Callable, Iterator

from dotenv import load_dotenv

from asa.components.logging_sub import redact

# Local runs keep the Langfuse keys in .env. Load it now: the agents only do so when they are
# imported, which is after tracing has already checked for the keys. Real environment
# variables (Docker, CI) win over .env.
load_dotenv()

logger = logging.getLogger(__name__)

_MAX_CONTENT_CHARS = 2000
_OMITTED = "[content not captured]"

# Keys whose values are identifiers or labels produced by the application itself (never user,
# answer or manual text). Their short string values are kept even when content capture is off,
# so traces still show which specialist ran, which manual passages were retrieved and why a
# guardrail fired. Anything else, and any value longer than _MAX_IDENTIFIER_CHARS, is omitted.
_IDENTIFIER_KEYS = frozenset(
    {
        "status",
        "specialist",
        "equipment_model",
        "chunk_id",
        "chunk_ids",
        "source_file",
        "tool_name",
        "strategy",
        "retrieval_status",
        "senior_decision",
        "decision",
        "categories",
        "query_hazards",
        "answer_hazards",
    }
)
_MAX_IDENTIFIER_CHARS = 64

# True once `langfuse.openai` is imported. That import patches the OpenAI SDK globally, so every
# client then accepts Langfuse's extra `name=` argument; a plain client would reject it.
_openai_instrumented = False


def tracing_enabled() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def capture_content() -> bool:
    return os.getenv("LANGFUSE_CAPTURE_CONTENT", "false").lower() == "true"


def _omit(value: Any, keep_identifiers: bool = False) -> Any:
    """Keep numbers, booleans, None and the shape of dicts and lists; drop everything else.

    Strings AND any other object (dataclasses such as retrieved chunks, models, ...) are
    replaced, because Langfuse would serialise such an object later, after masking. The one
    exception is short strings under an identifier key (see _IDENTIFIER_KEYS).
    """
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str) and keep_identifiers and len(value) <= _MAX_IDENTIFIER_CHARS:
        return value
    if isinstance(value, dict):
        return {str(k): _omit(v, str(k) in _IDENTIFIER_KEYS) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_omit(v, keep_identifiers) for v in value]
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
    """Trace every OpenAI call (model, tokens, latency) as a Langfuse generation.

    Importing `langfuse.openai` patches the OpenAI SDK's methods globally, so clients created
    before or after this call are all traced. Call it before the first request.
    """
    global _openai_instrumented
    if get_tracer() is None:
        return False
    try:
        import openai
        from langfuse.openai import OpenAI as TracedOpenAI

        openai.OpenAI = TracedOpenAI
        _openai_instrumented = True
        return True
    except Exception as exc:
        logger.warning("Langfuse OpenAI instrumentation failed: %s", type(exc).__name__)
        return False


def llm_name(name: str) -> dict:
    """Extra keyword arguments that give an OpenAI call a stable generation name in Langfuse.

    Use as `client.chat.completions.create(model=..., messages=..., **llm_name("assess-retrieval"))`.
    Empty when tracing is off, because the plain OpenAI client rejects `name=`.
    """
    return {"name": name} if _openai_instrumented else {}


class _Observation:
    """Handle on an open observation. Every method is a no-op when tracing is off or failed."""

    def __init__(self, observation: Any = None):
        self._observation = observation

    def update(self, **fields: Any) -> None:
        if self._observation is None:
            return
        try:
            self._observation.update(**fields)
        except Exception as exc:
            logger.warning("Langfuse observation not updated: %s", type(exc).__name__)


@contextmanager
def observation(name: str, *, as_type: str = "span", input: Any = None, metadata: Any = None) -> Iterator[_Observation]:
    """Record one step as a typed Langfuse observation nested under the current one.

    `as_type` is one of span, agent, tool, chain, retriever, evaluator, guardrail. Input, output
    and metadata go through `mask`. Errors raised by the block are recorded on the observation and
    re-raised; Langfuse problems are logged and never break the request.
    """
    client = get_tracer()
    entered = []
    handle = _Observation()
    if client is not None:
        try:
            context = client.start_as_current_observation(name=name, as_type=as_type, input=input, metadata=metadata)
            handle = _Observation(context.__enter__())
            entered.append(context)
        except Exception as exc:
            logger.warning("Langfuse observation %s not started: %s", name, type(exc).__name__)
    try:
        yield handle
    except BaseException as exc:
        handle.update(level="ERROR", status_message=type(exc).__name__)
        _close(entered, type(exc), exc)
        raise
    else:
        _close(entered, None, None)


def traced(
    name: str,
    *,
    as_type: str = "span",
    inputs: tuple[str, ...] = (),
    output: Callable[[Any], Any] | None = None,
    metadata: Callable[[Any], Any] | None = None,
):
    """Decorator form of `observation` for functions and methods.

    Only the arguments named in `inputs` become the observation input (never `self`, clients or
    configuration). `output` and `metadata` turn the return value into what is recorded, so large
    or sensitive results can be summarised. With tracing off the function runs untouched.
    """

    def decorator(func):
        signature = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if get_tracer() is None:
                return func(*args, **kwargs)
            try:
                bound = signature.bind_partial(*args, **kwargs).arguments
                recorded_input = {key: bound[key] for key in inputs if key in bound} or None
            except Exception:
                recorded_input = None
            with observation(name, as_type=as_type, input=recorded_input) as step:
                result = func(*args, **kwargs)
                try:
                    step.update(
                        output=output(result) if output else result,
                        metadata=metadata(result) if metadata else None,
                    )
                except Exception as exc:
                    logger.warning("Langfuse output of %s not recorded: %s", name, type(exc).__name__)
                return result

        return wrapper

    return decorator


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
) -> Iterator[_Observation]:
    """Open one trace for a chat request (one conversation turn; turns share the session).

    Yields the root observation: set its input and output with `.update(...)`, since they become
    the trace input/output shown in the trace table and used by evaluators. The equipment model
    is a tag because it is a small fixed set (per-equipment dashboards). A no-op when tracing is
    off or fails to start.
    """
    client = get_tracer()
    entered = []
    root = _Observation()
    if client is not None:
        try:
            from langfuse import propagate_attributes

            context = client.start_as_current_observation(name=name, as_type="span")
            root = _Observation(context.__enter__())
            entered.append(context)
            attributes = propagate_attributes(
                trace_name=name,
                session_id=session_id,
                tags=[f"equipment:{equipment_model}"] if equipment_model else None,
                metadata={"equipment_model": equipment_model} if equipment_model else None,
            )
            attributes.__enter__()
            entered.append(attributes)
        except Exception as exc:
            logger.warning("Langfuse trace not started: %s", type(exc).__name__)
    try:
        yield root
    except BaseException as exc:
        root.update(level="ERROR", status_message=type(exc).__name__)
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


def record_outcome(
    *,
    status: str,
    hazard_count: int = 0,
    source_count: int = 0,
    escalated: bool = False,
    confidence: float | None = None,
    specialist: str | None = None,
) -> None:
    """Attach the outcome to the current trace as scores (structured data, never text).

    The specialist the planner chose is only known after the run, so it is a score rather than a
    tag (tags must be set when the trace starts)."""
    client = get_tracer()
    if client is None:
        return
    try:
        client.score_current_trace(name="status", value=status, data_type="CATEGORICAL")
        if specialist:
            client.score_current_trace(name="specialist", value=specialist, data_type="CATEGORICAL")
        client.score_current_trace(name="escalated", value=1 if escalated else 0, data_type="BOOLEAN")
        client.score_current_trace(name="hazard_count", value=hazard_count, data_type="NUMERIC")
        client.score_current_trace(name="source_count", value=source_count, data_type="NUMERIC")
        if confidence:
            client.score_current_trace(name="confidence", value=float(confidence), data_type="NUMERIC")
    except Exception as exc:
        logger.warning("Langfuse outcome not recorded: %s", type(exc).__name__)


@contextmanager
def tag_run(run_id: str | None) -> Iterator[None]:
    """Tag everything traced inside this block with the audit-trail run_id, so a trace and its
    audit-log entries can be matched. The run_id is also set as metadata (`run_id`), which is the
    filterable home for per-request identifiers; the tag is kept because the team docs search by it.
    A no-op when tracing is off."""
    if get_tracer() is None or not run_id:
        yield
        return
    try:
        from langfuse import propagate_attributes

        context = propagate_attributes(tags=[f"run:{run_id}"], metadata={"run_id": run_id})
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
