"""Ingest authoritative AEM PDF manuals into ChromaDB."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from asa.ingestion.chunk import chunk_pdf
from asa.ingestion.embed import (
    CHROMA_DIR,
    reset_collection,
    write_chunks,
)
from asa.ingestion.metadata import (
    metadata_for_pdf,
)
from asa.versioning import write_corpus_manifest

load_dotenv()


DOCUMENT_PATH = os.getenv(
    "AEM_DOCUMENT_PATH",
    "aem_documents",
)


def main() -> None:

    document_dir = Path(DOCUMENT_PATH)

    if not document_dir.exists():
        raise SystemExit(f"AEM document path not found: {document_dir.resolve()}")

    pdf_paths = sorted(document_dir.glob("*.pdf"))

    if not pdf_paths:
        raise SystemExit(f"No PDF documents found under {document_dir.resolve()}")

    print(f"Ingesting AEM manuals from: {document_dir.resolve()}")

    print("-" * 80)

    # Rebuild from authoritative source documents.
    reset_collection()

    total_chunks = 0

    for path in pdf_paths:
        meta = metadata_for_pdf(path)

        print(f"\nDocument: {meta.title}")

        print(f"Equipment: {meta.equipment_model}")

        print(f"Source: {path.name}")

        chunks = chunk_pdf(
            path,
            meta,
        )

        written = write_chunks(chunks)

        total_chunks += written

        print(f"Chunks written: {written}")

    print("\n" + "-" * 80)

    print(f"Documents ingested: {len(pdf_paths)}")

    print(f"Total chunks: {total_chunks}")

    print(f"Chroma store: {CHROMA_DIR}/")

    # Record which manuals this index was built from (names and sha256), so a run can say what it answered from.
    manifest = write_corpus_manifest(document_dir, CHROMA_DIR)
    print(f"Corpus manifest: {manifest}")


if __name__ == "__main__":
    main()
