"""Corpus ingestion CLI."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from asa.ingestion.metadata import parse_document
from asa.ingestion.chunk import chunk_document
from asa.ingestion.embed import write_chunks, CHROMA_DIR

load_dotenv()

CORPUS_PATH = os.getenv("CORPUS_PATH", "corpus/synthetic")
DOC_GLOB = "**/*.md"


def main() -> None:
    corpus_dir = Path(CORPUS_PATH)
    if not corpus_dir.exists():
        raise SystemExit(f"Corpus path not found: {corpus_dir.resolve()}")

    doc_paths = sorted(corpus_dir.glob(DOC_GLOB))
    if not doc_paths:
        raise SystemExit(f"No .md documents found under {corpus_dir.resolve()}")

    total_chunks = 0
    print(f"Ingesting from: {corpus_dir.resolve()}")
    print("-" * 60)
    for path in doc_paths:
        meta, body = parse_document(path)
        chunks = chunk_document(meta, body)
        written = write_chunks(chunks)
        total_chunks += written
        print(f"  {meta.doc_id:<18} rev {meta.revision:<4} "
              f"{meta.equipment_model:<10} -> {written:>3} chunks")

    print("-" * 60)
    print(f"Done. {len(doc_paths)} documents, {total_chunks} chunks -> {CHROMA_DIR}/")


if __name__ == "__main__":
    main()
