"""Metadata definitions for real AEM service documentation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Supported equipment identifiers
# ---------------------------------------------------------------------------

THERMAL_STATION = "thermal_station"
THERMAL_RETROFIT_1KW = "thermal_retrofit_1kw"


@dataclass(frozen=True)
class DocMetadata:
    """Normalized metadata attached to every document and chunk."""

    doc_id: str
    title: str
    revision: str
    equipment_model: str
    doc_type: str
    source_path: str
    source_file: str


# ---------------------------------------------------------------------------
# Sponsor-document registry
# ---------------------------------------------------------------------------

DOCUMENT_REGISTRY = {
    "aem_thermal_station.pdf": {
        "doc_id": "aem_thermal_station",
        "title": "AEM Thermal Station Operation Manual",
        "revision": "unknown",
        "equipment_model": THERMAL_STATION,
        "doc_type": "operation_manual",
    },

    "aem_1kw_thermal_retrofit_system.pdf": {
        "doc_id": "aem_1kw_thermal_retrofit_system",
        "title": "AEM 1kW Thermal Retrofit System Operation Manual",
        "revision": "unknown",
        "equipment_model": THERMAL_RETROFIT_1KW,
        "doc_type": "operation_manual",
    },
}


def metadata_for_pdf(path: Path) -> DocMetadata:
    """
    Return normalized metadata for a supported sponsor PDF.

    Unknown PDFs are rejected intentionally so that a document cannot
    silently enter the knowledge base under the wrong equipment model.
    """

    filename = path.name.lower()

    if filename not in DOCUMENT_REGISTRY:
        raise ValueError(
            "Unsupported AEM document: "
            f"{path.name}. Add it to DOCUMENT_REGISTRY first."
        )

    registered = DOCUMENT_REGISTRY[filename]

    return DocMetadata(
        doc_id=registered["doc_id"],
        title=registered["title"],
        revision=registered["revision"],
        equipment_model=registered["equipment_model"],
        doc_type=registered["doc_type"],
        source_path=str(path),
        source_file=path.name,
    )