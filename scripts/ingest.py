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


load_dotenv()


DOCUMENT_PATH = os.getenv(
    "AEM_DOCUMENT_PATH",
    "aem_documents",
)


def main() -> None:

    document_dir = Path(
        DOCUMENT_PATH
    )

    if not document_dir.exists():

        raise SystemExit(
            "AEM document path not found: "
            f"{document_dir.resolve()}"
        )

    pdf_paths = sorted(
        document_dir.glob("*.pdf")
    )

    if not pdf_paths:

        raise SystemExit(
            "No PDF documents found under "
            f"{document_dir.resolve()}"
        )

    print(
        f"Ingesting AEM manuals from: "
        f"{document_dir.resolve()}"
    )

    print("-" * 80)

    # Rebuild from authoritative source documents.
    reset_collection()

    total_chunks = 0

    for path in pdf_paths:

        meta = metadata_for_pdf(
            path
        )

        print(
            f"\nDocument: "
            f"{meta.title}"
        )

        print(
            f"Equipment: "
            f"{meta.equipment_model}"
        )

        print(
            f"Source: "
            f"{path.name}"
        )

        chunks = chunk_pdf(
            path,
            meta,
        )

        written = write_chunks(
            chunks
        )

        total_chunks += written

        print(
            f"Chunks written: "
            f"{written}"
        )

    print("\n" + "-" * 80)

    print(
        f"Documents ingested: "
        f"{len(pdf_paths)}"
    )

    print(
        f"Total chunks: "
        f"{total_chunks}"
    )

    print(
        f"Chroma store: "
        f"{CHROMA_DIR}/"
    )


if __name__ == "__main__":
    main()