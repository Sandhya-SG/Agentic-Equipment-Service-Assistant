"""Page-aware chunking for real AEM PDF manuals."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from asa.ingestion.metadata import (
    DocMetadata,
    metadata_for_pdf,
)


TARGET_WORDS = 350
OVERLAP_WORDS = 60


@dataclass
class RawChunk:
    """One retrievable passage with full document provenance."""

    chunk_id: str

    doc_id: str
    title: str

    section_id: str
    section_title: str

    revision: str
    equipment_model: str

    source_file: str
    page: int
    chunk_index: int

    text: str


def clean_text(text: str) -> str:
    """
    Normalize whitespace while preserving readable text.
    """

    text = re.sub(r"[ \t]+", " ", text)

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def detect_section(page_text: str) -> str:
    """
    Detect a likely section heading from the beginning of a PDF page.

    This is intentionally conservative for v1.0.0.
    """

    lines = [
        line.strip()
        for line in page_text.splitlines()
        if line.strip()
    ]

    if not lines:
        return "Unknown"

    # Examine only the first part of the page.
    candidates = lines[:12]

    patterns = [
        # 1 Introduction
        r"^\d+\s+[A-Z].+",

        # 1. Introduction
        r"^\d+\.\s+[A-Z].+",

        # 1.1 Overview
        r"^\d+\.\d+\s+[A-Z].+",

        # 1.1.1 Detail
        r"^\d+\.\d+\.\d+\s+[A-Z].+",
    ]

    for line in candidates:

        if len(line) > 120:
            continue

        for pattern in patterns:

            if re.match(pattern, line):
                return line

    return "Unknown"


def split_words(
    text: str,
    target_words: int = TARGET_WORDS,
    overlap_words: int = OVERLAP_WORDS,
) -> list[str]:
    """
    Split page text into overlapping word chunks.
    """

    words = text.split()

    if not words:
        return []

    if len(words) <= target_words:
        return [text]

    chunks = []

    start = 0

    while start < len(words):

        end = min(
            start + target_words,
            len(words),
        )

        piece = " ".join(
            words[start:end]
        )

        chunks.append(piece)

        if end >= len(words):
            break

        start = max(
            end - overlap_words,
            start + 1,
        )

    return chunks


def chunk_pdf(
    path: Path,
    meta: DocMetadata | None = None,
) -> list[RawChunk]:
    """
    Extract and chunk one AEM PDF while preserving physical page numbers.
    """

    if meta is None:
        meta = metadata_for_pdf(path)

    document = pymupdf.open(path)

    chunks: list[RawChunk] = []

    try:

        for page_number, page in enumerate(
            document,
            start=1,
        ):

            raw_text = page.get_text("text")

            if not raw_text.strip():
                continue

            cleaned_page = clean_text(
                raw_text
            )

            if not cleaned_page:
                continue

            section_title = detect_section(
                raw_text
            )

            section_slug = re.sub(
                r"[^a-zA-Z0-9]+",
                "_",
                section_title,
            ).strip("_").lower()

            if not section_slug:
                section_slug = "unknown"

            section_id = (
                f"{meta.doc_id}"
                f"#p{page_number}"
                f"#{section_slug[:60]}"
            )

            pieces = split_words(
                cleaned_page
            )

            for chunk_index, piece in enumerate(
                pieces
            ):

                chunk_id = (
                    f"{meta.doc_id}"
                    f"_p{page_number}"
                    f"_c{chunk_index}"
                )

                chunks.append(
                    RawChunk(
                        chunk_id=chunk_id,

                        doc_id=meta.doc_id,
                        title=meta.title,

                        section_id=section_id,
                        section_title=section_title,

                        revision=meta.revision,
                        equipment_model=meta.equipment_model,

                        source_file=meta.source_file,
                        page=page_number,
                        chunk_index=chunk_index,

                        text=piece,
                    )
                )

    finally:

        document.close()

    return chunks