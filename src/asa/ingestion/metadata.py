"""Document metadata schema and front-matter parsing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DocMetadata:
    doc_id: str
    title: str
    revision: str
    equipment_model: str
    doc_type: str
    source_path: str


def parse_document(path: Path) -> tuple[DocMetadata, str]:
    """Split a document into its metadata header and body."""
    raw = path.read_text(encoding="utf-8")
    if raw.startswith("---"):
        _, header, body = raw.split("---", 2)
        meta = {}
        for line in header.strip().splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                meta[key.strip()] = val.strip()
        return (
            DocMetadata(
                doc_id=meta.get("doc_id", path.stem),
                title=meta.get("title", path.stem),
                revision=meta.get("revision", "unknown"),
                equipment_model=meta.get("equipment_model", "unknown"),
                doc_type=meta.get("doc_type", "manual"),
                source_path=str(path),
            ),
            body.strip(),
        )
    return (
        DocMetadata(
            doc_id=path.stem, title=path.stem, revision="unknown",
            equipment_model="unknown", doc_type="manual", source_path=str(path),
        ),
        raw.strip(),
    )
