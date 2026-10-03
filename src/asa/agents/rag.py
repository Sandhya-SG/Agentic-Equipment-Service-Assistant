"""Retrieval hook for the graph.

`retrieve` is the single place the graph asks for evidence. It returns no chunks
until RAG is connected. To plug in real retrieval, replace the body and keep the
signature: the graph, the safety check and the API response already handle chunks
(citations come from the chunks' provenance fields).
"""

from __future__ import annotations

from asa.graph.state import Chunk


def retrieve(query: str) -> list[Chunk]:
    return []
