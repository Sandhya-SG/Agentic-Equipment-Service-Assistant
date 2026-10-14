"""Shared, lazily created OpenAI access with timeout, bounded retries and cost limits.

Every agent used to build its own `OpenAI(...)` client at import time. Importing an agent therefore
needed a key, and no call had a timeout, an output cap or a budget. This module replaces those clients:

* `client` is a lazy proxy. The real client is created on the first call, so importing the agents needs
  no key, and the Langfuse wrapper (`openai.OpenAI` swapped by `instrument_openai`) is picked up whenever
  it is installed before that first call.
* The real client gets a request timeout and a bounded number of SDK retries (429, 5xx and connection
  errors, with the SDK's exponential backoff). Both come from the environment.
* Every `chat.completions.create` gets a default output cap (`max_completion_tokens`) unless the caller set
  one, and its token usage is added to the budget of the current request. When the budget is used up the
  next call raises `TokenBudgetExceeded`, which the API reports as "unavailable" instead of spending more.
* `get_rag_agent()` returns one shared `AgenticRAGAgent`, created on first use. The retriever (embedding
  model and Chroma index) is loaded once, not once per module.

Settings (environment): OPENAI_TIMEOUT_S, OPENAI_CONNECT_TIMEOUT_S, OPENAI_MAX_RETRIES,
OPENAI_MAX_OUTPUT_TOKENS, OPENAI_REQUEST_TOKEN_BUDGET (0 turns the budget off).

A spend cap helps protect a small prepaid balance. It is a best-effort control on an ESTIMATED cost, not a guaranteed
financial limit (see below). With OPENAI_SPEND_CAP_USD set, the estimated cost of every call (its
tokens times the model's price) is added to a ledger file (OPENAI_SPEND_LEDGER, default `.openai_spend.json` in the
working directory), and once the ledger reaches the cap every further call raises `SpendCapReached` before anything is
sent. The estimate covers only calls made through this client (the app and the evaluation scripts), not DeepEval or
Promptfoo, which have their own clients; set the limit in the OpenAI console for those. The estimate can be lower than
the real bill (for example when a call times out on our side but completes on the provider's, or for cached-input
discounts and price changes), so keep a margin and read the real balance from the OpenAI usage page.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import os
import sys
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 60.0
DEFAULT_CONNECT_TIMEOUT_S = 10.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_MAX_OUTPUT_TOKENS = 2000
DEFAULT_REQUEST_TOKEN_BUDGET = 60000


class TokenBudgetExceeded(RuntimeError):
    """The current request has used its whole token budget; no further model call is made."""


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Per-request token budget
# ---------------------------------------------------------------------------

_usage_lock = threading.Lock()
_budget: contextvars.ContextVar[dict | None] = contextvars.ContextVar("openai_token_budget", default=None)


@contextlib.contextmanager
def request_budget(limit: int | None = None):
    """Track the tokens used by every model call made inside this block (one user request).

    Yields the counter dict (`limit`, `used`, `calls`, `input`, `output`, and `detail`: one entry per call with its
    name, model, input and output tokens and seconds). A limit of 0 or less means no limit.
    """
    if limit is None:
        limit = _env_int("OPENAI_REQUEST_TOKEN_BUDGET", DEFAULT_REQUEST_TOKEN_BUDGET)
    counter = {"limit": limit, "used": 0, "calls": 0, "input": 0, "output": 0, "detail": []}
    token = _budget.set(counter)
    try:
        yield counter
    finally:
        _budget.reset(token)


def _check_budget() -> None:
    counter = _budget.get()
    if counter and counter["limit"] > 0 and counter["used"] >= counter["limit"]:
        raise TokenBudgetExceeded(f"request token budget of {counter['limit']} used up after {counter['calls']} calls")


def _record_usage(response: Any, label: str, model: Any, seconds: float) -> None:
    counter = _budget.get()
    if counter is None:
        return
    usage = getattr(response, "usage", None)
    total = getattr(usage, "total_tokens", None)
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    with _usage_lock:
        counter["calls"] += 1
        if isinstance(total, int):
            counter["used"] += total
        if isinstance(prompt, int):
            counter["input"] += prompt
        if isinstance(completion, int):
            counter["output"] += completion
        counter["detail"].append(
            {
                "call": label,
                "model": model if isinstance(model, str) else None,
                "input": prompt if isinstance(prompt, int) else None,
                "output": completion if isinstance(completion, int) else None,
                "seconds": round(seconds, 3),
            }
        )


def _caller_label(kwargs: dict) -> str:
    """What to call this model call in the usage record: the Langfuse generation name if the caller gave one,
    else the module and function that made it (`request.request_node`)."""
    name = kwargs.get("name")
    if isinstance(name, str) and name:
        return name
    try:
        frame = sys._getframe(2)
        return f"{frame.f_globals.get('__name__', '?').rsplit('.', 1)[-1]}.{frame.f_code.co_name}"
    except (AttributeError, ValueError):
        return "unknown"


# ---------------------------------------------------------------------------
# Spend cap: a ledger of the estimated cost, shared by every process that uses this client
# ---------------------------------------------------------------------------

# US dollars per million tokens (input, output). A model that is not listed is priced like the most expensive one
# below, so the estimate errs on the side of stopping early.
PRICES_PER_MILLION = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1": (2.00, 8.00),
}
UNKNOWN_MODEL_PRICE = (2.50, 10.00)
DEFAULT_LEDGER = ".openai_spend.json"


class SpendCapReached(RuntimeError):
    """The estimated spend has reached OPENAI_SPEND_CAP_USD; no further model call is made."""


_ledger_lock = threading.Lock()


def estimate_cost(model: Any, prompt_tokens: Any, completion_tokens: Any) -> float:
    """Estimated dollars for one call. Unknown token counts count as zero."""
    name = model if isinstance(model, str) else ""
    price_in, price_out = next(
        (price for known, price in PRICES_PER_MILLION.items() if name == known or name.startswith(known + "-")),
        UNKNOWN_MODEL_PRICE,
    )
    tokens_in = prompt_tokens if isinstance(prompt_tokens, int) else 0
    tokens_out = completion_tokens if isinstance(completion_tokens, int) else 0
    return (tokens_in * price_in + tokens_out * price_out) / 1_000_000


def _ledger_path() -> str:
    return os.getenv("OPENAI_SPEND_LEDGER") or DEFAULT_LEDGER


def spend_cap() -> float | None:
    """The cap in dollars, or None when no cap is set (or the value is not a positive number)."""
    cap = _env_float("OPENAI_SPEND_CAP_USD", 0.0)
    return cap if cap > 0 else None


def read_ledger() -> dict:
    """The ledger: estimated dollars spent and calls made so far. A missing or damaged file counts as empty."""
    import json

    try:
        with open(_ledger_path(), encoding="utf-8") as handle:
            data = json.load(handle)
        return {"estimated_usd": float(data.get("estimated_usd", 0.0)), "calls": int(data.get("calls", 0))}
    except (OSError, ValueError, TypeError, AttributeError):
        return {"estimated_usd": 0.0, "calls": 0}


def _write_ledger(ledger: dict) -> None:
    import json
    import tempfile
    from datetime import datetime, timezone

    path = _ledger_path()
    payload = {**ledger, "updated": datetime.now(timezone.utc).isoformat()}
    directory = os.path.dirname(os.path.abspath(path))
    handle, temp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            json.dump(payload, out)
        os.replace(temp, path)
    except OSError:
        with contextlib.suppress(OSError):
            os.remove(temp)


def add_spend(dollars: float, calls: int = 1) -> dict:
    """Add to the ledger (used after each call, and by `scripts/openai_spend.py --add` for spend made before)."""
    with _ledger_lock:
        ledger = read_ledger()
        ledger["estimated_usd"] = round(ledger["estimated_usd"] + dollars, 6)
        ledger["calls"] += calls
        _write_ledger(ledger)
        return ledger


def reset_spend() -> None:
    with _ledger_lock:
        _write_ledger({"estimated_usd": 0.0, "calls": 0})


def _check_spend_cap() -> None:
    cap = spend_cap()
    if cap is None:
        return
    spent = read_ledger()["estimated_usd"]
    if spent >= cap:
        raise SpendCapReached(
            f"estimated OpenAI spend ${spent:.2f} has reached the cap of ${cap:.2f} (OPENAI_SPEND_CAP_USD); "
            "raise the cap or reset the ledger with scripts/openai_spend.py to continue"
        )


def _record_spend(response: Any, model: Any) -> None:
    if spend_cap() is None:
        return
    usage = getattr(response, "usage", None)
    cost = estimate_cost(model, getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None))
    add_spend(cost)


# ---------------------------------------------------------------------------
# Lazy client
# ---------------------------------------------------------------------------

_client_lock = threading.Lock()
_real_client: Any = None


def _create_client() -> Any:
    """Build the real client. Looks `openai.OpenAI` up now so Langfuse's wrapper is used when installed."""
    import httpx
    import openai
    from dotenv import load_dotenv

    load_dotenv()
    timeout = httpx.Timeout(
        _env_float("OPENAI_TIMEOUT_S", DEFAULT_TIMEOUT_S),
        connect=_env_float("OPENAI_CONNECT_TIMEOUT_S", DEFAULT_CONNECT_TIMEOUT_S),
    )
    return openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        timeout=timeout,
        max_retries=_env_int("OPENAI_MAX_RETRIES", DEFAULT_MAX_RETRIES),
    )


def _get_real_client() -> Any:
    global _real_client
    if _real_client is None:
        with _client_lock:
            if _real_client is None:
                _real_client = _create_client()
    return _real_client


def reset_client() -> None:
    """Forget the cached client (tests, or after the key or settings changed)."""
    global _real_client
    with _client_lock:
        _real_client = None


class _Completions:
    """Stable `client.chat.completions` object: applies the output cap and budget around `create`."""

    def create(self, *args: Any, **kwargs: Any) -> Any:
        _check_budget()
        _check_spend_cap()
        if "max_completion_tokens" not in kwargs and "max_tokens" not in kwargs:
            cap = _env_int("OPENAI_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS)
            if cap > 0:
                kwargs["max_completion_tokens"] = cap
        label = _caller_label(kwargs)
        started = time.monotonic()
        response = _get_real_client().chat.completions.create(*args, **kwargs)
        _record_usage(response, label, kwargs.get("model"), time.monotonic() - started)
        _record_spend(response, kwargs.get("model"))
        return response


class _Chat:
    def __init__(self) -> None:
        self.completions = _Completions()


class LazyOpenAI:
    """Drop-in for `OpenAI(...)` that creates the real client on first use (see the module docstring)."""

    def __init__(self) -> None:
        self.chat = _Chat()

    def __getattr__(self, name: str) -> Any:  # anything other than chat: delegate to the real client
        return getattr(_get_real_client(), name)


client = LazyOpenAI()


def get_client() -> LazyOpenAI:
    return client


# ---------------------------------------------------------------------------
# Concurrent calls
# ---------------------------------------------------------------------------


def run_parallel(tasks: list, max_workers: int = 6) -> list:
    """Run independent tasks concurrently and return their results in task order.

    Each task runs in a copy of the caller's context, so the request's token budget and the tracing context carry
    over to the worker threads. The first exception raised by a task is raised here, as it would be one by one.
    """
    if not tasks:
        return []
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=min(max_workers, len(tasks))) as pool:
        futures = [pool.submit(contextvars.copy_context().run, task) for task in tasks]
        return [future.result() for future in futures]


# ---------------------------------------------------------------------------
# Shared RAG agent
# ---------------------------------------------------------------------------

_rag_lock = threading.Lock()
_rag_agent: Any = None


def get_rag_agent() -> Any:
    """The one shared `AgenticRAGAgent`, created on first use (loads the embedding model and the index)."""
    global _rag_agent
    if _rag_agent is None:
        with _rag_lock:
            if _rag_agent is None:
                from asa.agents.rag import AgenticRAGAgent

                _rag_agent = AgenticRAGAgent()
    return _rag_agent


class LazyRagAgent:
    """Stands in for a module-level `AgenticRAGAgent()`: attribute access reaches the shared instance."""

    def __getattr__(self, name: str) -> Any:
        return getattr(get_rag_agent(), name)

    # Reads and writes both reach the shared instance, so `monkeypatch.setattr(rag_agent, "retriever", fake)` in a
    # test changes the agent the code really uses, exactly as it did when each module held its own instance.
    def __setattr__(self, name: str, value: Any) -> None:
        setattr(get_rag_agent(), name, value)

    def __delattr__(self, name: str) -> None:
        delattr(get_rag_agent(), name)


rag_agent = LazyRagAgent()
