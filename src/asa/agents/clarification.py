"""Clarification continuation helpers."""

from __future__ import annotations


def resolve_clarification(
    original_query: str,
    clarification_response: str,
) -> str:
    """
    Combine the original engineer request with the
    clarification response without inventing information.

    Both inputs are engineer-provided text. The combined
    query is passed back through the Request Agent.
    """

    original = (
        original_query
        or ""
    ).strip()

    clarification = (
        clarification_response
        or ""
    ).strip()

    if not original:
        return clarification

    if not clarification:
        return original

    return (
        f"Original engineer request:\n"
        f"{original}\n\n"
        f"Engineer clarification:\n"
        f"{clarification}"
    )