import json
import re
import fitz

from config import DOCS_DIR, DATA_DIR, CHUNKS_FILE


def clean_text(text):
    """Clean unnecessary whitespace from extracted PDF text."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_section(text):
    """
    Try to identify a section heading from the beginning of a text block.

    This is intentionally simple for v0.1.1.
    We can make this more sophisticated later.
    """

    lines = [line.strip() for line in text.split("\n") if line.strip()]

    if not lines:
        return "Unknown"

    first_line = lines[0]

    # Common heading patterns
    heading_patterns = [
        r"^\d+\.\s+.+",          # 1. Introduction
        r"^\d+\.\d+\s+.+",       # 1.1 Overview
        r"^[A-Z][A-Za-z\s/&-]{3,60}$",
    ]

    for pattern in heading_patterns:
        if re.match(pattern, first_line):
            return first_line

    return "Unknown"


def split_into_blocks(text):
    """
    Split page text into logical blocks using blank lines.
    """
    blocks = re.split(r"\n\s*\n", text)

    return [
        block.strip()
        for block in blocks
        if block.strip()
    ]


def chunk_text(text, max_words=500, overlap_words=80):
    """
    Create chunks while trying to preserve logical blocks.
    """

    blocks = split_into_blocks(text)

    chunks = []
    current_words = []

    for block in blocks:

        block_words = block.split()

        # If adding this block exceeds the limit,
        # save the current chunk first.
        if (
            current_words
            and len(current_words) + len(block_words) > max_words
        ):
            chunks.append(" ".join(current_words))

            # Keep overlap from the previous chunk
            current_words = current_words[-overlap_words:]

        current_words.extend(block_words)

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def process_pdf(pdf_path):

    document = fitz.open(pdf_path)

    records = []

    for page_number, page in enumerate(document, start=1):

        raw_text = page.get_text()

        if not raw_text.strip():
            continue

        section = detect_section(raw_text)

        chunks = chunk_text(raw_text)

        for chunk_index, chunk in enumerate(chunks):

            chunk = clean_text(chunk)

            if not chunk:
                continue

            record = {
                "chunk_id": (
                    f"{pdf_path.stem}_"
                    f"p{page_number}_"
                    f"c{chunk_index}"
                ),

                "source": pdf_path.name,

                "document": pdf_path.stem,

                "page": page_number,

                "section": section,

                "chunk_index": chunk_index,

                "text": chunk,
            }

            records.append(record)

    document.close()

    return records


def main():

    DATA_DIR.mkdir(exist_ok=True)

    all_chunks = []

    pdf_files = sorted(DOCS_DIR.glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF documents.")

    for pdf_path in pdf_files:

        print(f"Processing: {pdf_path.name}")

        records = process_pdf(pdf_path)

        print(f"  Created {len(records)} chunks.")

        all_chunks.extend(records)

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:

        json.dump(
            all_chunks,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Saved to: {CHUNKS_FILE}")


if __name__ == "__main__":
    main()