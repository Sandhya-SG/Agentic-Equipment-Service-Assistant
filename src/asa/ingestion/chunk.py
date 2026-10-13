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


UNKNOWN_SECTION = "Unknown"
CONTENTS_SECTION = "Table of Contents"
FRONT_MATTER_SECTION = "Front Matter"
_ROMAN_PAGE = re.compile(r"^[ivxl]{1,6}$")  # front matter is numbered i, ii, ... vii; the body 1-1, 3-2, ...

# "2 Safety", "2.5 Safety Hazard", "2.5.3 Hazardous Energies Listing". The number has no trailing
# full stop: "5. The tag must be filled ..." is a numbered procedure step or legal clause, not a heading.
_HEADING = re.compile(r"^\d{1,2}(?:\.\d{1,2}){0,3}\s+(?P<title>\S.*)$")
_TITLE_FIRST_WORD = re.compile(r"[A-Z][A-Za-z&'\-]+")  # a real word: rejects units such as "L/min"
_SECTION_LINE = re.compile(r"^\d{1,2}(?:\.\d{1,2}){1,3}\s+\S")
_PAGE_REFERENCE = re.compile(r"^\d{1,2}-\d{1,3}$")  # "11-2": a manual page number on its own line
MAX_TITLE_WORDS = 8
MIN_CONTENTS_LINES = 8  # a page with this many section-number lines is a table of contents ...
MIN_CONTENTS_PAGE_REFS = 5  # ... as is one with this many page numbers on their own lines


def _heading(line: str) -> str | None:
    """The line if it looks like a section heading, else None.

    Table values ("83.33 L/min @ 36.2psi") and sentences ("6. Remove 4x screws on kit and remove the
    substrate pusher plate.") also start with numbers, so the title must look like a title: a real
    first word, at most MAX_TITLE_WORDS words, no "@", and no closing punctuation.
    """
    if len(line) > 120:
        return None
    match = _HEADING.match(line)
    if not match:
        return None
    title = match.group("title").strip()
    words = title.split()
    if not _TITLE_FIRST_WORD.fullmatch(words[0]):
        return None
    if len(words) > MAX_TITLE_WORDS or "@" in title or title.endswith((".", ",", ";", ":")):
        return None
    return line


def detect_section(page_text: str) -> str:
    """
    Detect a likely section heading from the beginning of a PDF page.

    Returns CONTENTS_SECTION for a table-of-contents page and UNKNOWN_SECTION when no heading
    starts near the top (the page continues the previous section; see `page_sections`).
    """

    lines = [
        line.strip()
        for line in page_text.splitlines()
        if line.strip()
    ]

    if not lines:
        return UNKNOWN_SECTION

    section_lines = sum(1 for line in lines if _SECTION_LINE.match(line))
    page_references = sum(1 for line in lines if _PAGE_REFERENCE.match(line))
    if section_lines >= MIN_CONTENTS_LINES or page_references >= MIN_CONTENTS_PAGE_REFS:
        return CONTENTS_SECTION

    # Headings listed on a front-matter page (the end of the contents, a revision record) are not
    # the section that page belongs to.
    if _ROMAN_PAGE.match(lines[0]):
        return FRONT_MATTER_SECTION

    # Examine only the first part of the page.
    for line in lines[:12]:
        heading = _heading(line)
        if heading:
            return heading

    return UNKNOWN_SECTION


def page_sections(page_texts: list[str]) -> list[str]:
    """The section of every page, in order.

    A page without a heading near its top continues the section of the page before it. The table of
    contents and other front matter are labelled as such but not carried forward.
    """
    sections = []
    current = UNKNOWN_SECTION
    for text in page_texts:
        detected = detect_section(text)
        if detected in (CONTENTS_SECTION, FRONT_MATTER_SECTION):
            sections.append(detected)
            continue
        if detected != UNKNOWN_SECTION:
            current = detected
        sections.append(current)
    return sections


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

        raw_pages = [page.get_text("text") for page in document]
        sections = page_sections(raw_pages)

        for page_number, raw_text in enumerate(
            raw_pages,
            start=1,
        ):

            if not raw_text.strip():
                continue

            cleaned_page = clean_text(
                raw_text
            )

            if not cleaned_page:
                continue

            section_title = sections[page_number - 1]

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