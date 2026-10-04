"""Embedding and ChromaDB persistence for AEM service documentation."""

from __future__ import annotations

import chromadb

from chromadb.utils import embedding_functions

from asa.ingestion.chunk import RawChunk


EMBED_MODEL = "all-MiniLM-L6-v2"

COLLECTION_NAME = "aem_service_docs"

CHROMA_DIR = "chroma_store"


def get_collection(
    persist_dir: str = CHROMA_DIR,
):
    """
    Return the persistent AEM service-document collection.
    """

    client = chromadb.PersistentClient(
        path=persist_dir
    )

    embedding_function = (
        embedding_functions
        .SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
    )

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={
            "hnsw:space": "cosine"
        },
    )


def reset_collection(
    persist_dir: str = CHROMA_DIR,
):
    """
    Delete the current service-document collection.

    Useful when rebuilding the corpus from authoritative manuals.
    """

    client = chromadb.PersistentClient(
        path=persist_dir
    )

    try:
        client.delete_collection(
            COLLECTION_NAME
        )

    except Exception:
        pass


def write_chunks(
    chunks: list[RawChunk],
    persist_dir: str = CHROMA_DIR,
) -> int:
    """
    Embed and upsert chunks into ChromaDB.
    """

    if not chunks:
        return 0

    collection = get_collection(
        persist_dir
    )

    collection.upsert(

        ids=[
            chunk.chunk_id
            for chunk in chunks
        ],

        documents=[
            chunk.text
            for chunk in chunks
        ],

        metadatas=[
            {
                "doc_id":
                    chunk.doc_id,

                "title":
                    chunk.title,

                "section_id":
                    chunk.section_id,

                "section_title":
                    chunk.section_title,

                "revision":
                    chunk.revision,

                "equipment_model":
                    chunk.equipment_model,

                "source_file":
                    chunk.source_file,

                "page":
                    chunk.page,

                "chunk_index":
                    chunk.chunk_index,
            }

            for chunk in chunks
        ],
    )

    return len(chunks)