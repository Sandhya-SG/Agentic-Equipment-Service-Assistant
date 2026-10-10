"""Pending clarification state for multi-turn chat requests."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True)
class PendingClarification:
    original_query: str
    clarification_question: str
    equipment_model: str | None
    clarification_count: int


_pending: dict[
    str,
    PendingClarification,
] = {}

_lock = Lock()


def get_pending(
    conversation_id: str,
) -> PendingClarification | None:
    with _lock:
        return _pending.get(
            conversation_id
        )


def set_pending(
    conversation_id: str,
    pending: PendingClarification,
) -> None:
    with _lock:
        _pending[conversation_id] = pending


def clear_pending(
    conversation_id: str,
) -> None:
    with _lock:
        _pending.pop(
            conversation_id,
            None,
        )


def clear_all() -> None:
    """
    Test helper.

    Production code should normally clear individual
    conversations rather than the whole store.
    """
    with _lock:
        _pending.clear()