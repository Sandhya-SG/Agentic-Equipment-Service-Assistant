"""Embedding and ChromaDB persistence."""

from __future__ import annotations

import chromadb
from chromadb.utils import embedding_functions

from asa.ingestion.chunk import RawChunk

EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "service_docs"
CHROMA_DIR = "chroma_store"


def get_collection(persist_dir: str = CHROMA_DIR):
    """Return (creating if needed) the persistent Chroma collection."""
    client = chromadb.PersistentClient(path=persist_dir)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def write_chunks(chunks: list[RawChunk], persist_dir: str = CHROMA_DIR) -> int:
    """Embed and upsert chunks. Returns the number written."""
    if not chunks:
        return 0
    col = get_collection(persist_dir)
    col.upsert(
        ids=[c.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[
            {
                "doc_id": c.doc_id,
                "section_id": c.section_id,
                "revision": c.revision,
                "equipment_model": c.equipment_model,
            }
            for c in chunks
        ],
    )
    return len(chunks)
