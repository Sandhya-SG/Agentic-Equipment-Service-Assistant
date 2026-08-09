"""Section-aware chunking."""

from __future__ import annotations

import re
from dataclasses import dataclass

from asa.ingestion.metadata import DocMetadata

SECTION_RE = re.compile(r"^##\s+(.*)$", re.MULTILINE)
TARGET_WORDS = 220
OVERLAP_WORDS = 25


@dataclass
class RawChunk:
    chunk_id: str
    doc_id: str
    section_id: str
    revision: str
    equipment_model: str
    text: str


def _split_long(text: str) -> list[str]:
    words = text.split()
    if len(words) <= TARGET_WORDS:
        return [text]
    pieces, start = [], 0
    while start < len(words):
        end = start + TARGET_WORDS
        pieces.append(" ".join(words[start:end]))
        start = end - OVERLAP_WORDS
    return pieces


def chunk_document(meta: DocMetadata, body: str) -> list[RawChunk]:
    """Split a document body into section-aware chunks."""
    matches = list(SECTION_RE.finditer(body))
    chunks: list[RawChunk] = []

    if not matches:
        sections = [("body", body)]
    else:
        sections = []
        for i, m in enumerate(matches):
            title = m.group(1).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
            sections.append((title, body[start:end].strip()))

    for s_idx, (title, content) in enumerate(sections):
        section_id = f"{meta.doc_id}#S{s_idx:02d}"
        for p_idx, piece in enumerate(_split_long(content)):
            if not piece.strip():
                continue
            chunks.append(RawChunk(
                chunk_id=f"{section_id}-{p_idx:02d}",
                doc_id=meta.doc_id,
                section_id=section_id,
                revision=meta.revision,
                equipment_model=meta.equipment_model,
                text=f"[{title}] {piece}",
            ))
    return chunks
