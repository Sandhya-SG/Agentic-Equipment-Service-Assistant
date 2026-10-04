"""Equipment-aware hybrid retrieval for AEM service manuals."""

from __future__ import annotations

import re

from collections import defaultdict

from rank_bm25 import BM25Okapi

from asa.graph.state import Chunk

from asa.ingestion.embed import (
    CHROMA_DIR,
    get_collection,
)


RRF_K = 60


def tokenize(text: str) -> list[str]:
    """
    Lightweight tokenizer for BM25 retrieval.
    """

    return re.findall(
        r"[a-zA-Z0-9]+",
        text.lower(),
    )


class AEMHybridRetriever:
    """
    Hybrid vector + BM25 retrieval with strict equipment isolation.
    """

    def __init__(
        self,
        persist_dir: str = CHROMA_DIR,
    ):

        self.collection = get_collection(
            persist_dir
        )

    def _load_equipment_chunks(
        self,
        equipment_model: str,
    ) -> list[dict]:
        """
        Load only chunks belonging to the selected equipment.
        """

        result = self.collection.get(
            where={
                "equipment_model":
                    equipment_model
            },
            include=[
                "documents",
                "metadatas",
            ],
        )

        records = []

        ids = result.get(
            "ids",
            []
        )

        documents = result.get(
            "documents",
            []
        )

        metadatas = result.get(
            "metadatas",
            []
        )

        for chunk_id, text, metadata in zip(
            ids,
            documents,
            metadatas,
        ):

            records.append({
                "chunk_id":
                    chunk_id,

                "text":
                    text,

                "metadata":
                    metadata,
            })

        return records

    def vector_search(
        self,
        query: str,
        equipment_model: str,
        k: int = 10,
    ) -> list[tuple[str, float]]:
        """
        Vector search restricted to one equipment model.
        """

        result = self.collection.query(

            query_texts=[
                query
            ],

            n_results=k,

            where={
                "equipment_model":
                    equipment_model
            },

            include=[
                "distances",
            ],
        )

        ids = (
            result.get("ids")
            or [[]]
        )[0]

        distances = (
            result.get("distances")
            or [[]]
        )[0]

        return [
            (
                chunk_id,
                float(distance),
            )

            for chunk_id, distance in zip(
                ids,
                distances,
            )
        ]

    def bm25_search(
        self,
        query: str,
        equipment_model: str,
        k: int = 10,
    ) -> list[tuple[str, float]]:
        """
        BM25 search over only the selected equipment corpus.
        """

        records = (
            self._load_equipment_chunks(
                equipment_model
            )
        )

        if not records:
            return []

        corpus = [
            tokenize(record["text"])
            for record in records
        ]

        bm25 = BM25Okapi(
            corpus
        )

        scores = bm25.get_scores(
            tokenize(query)
        )

        ranked = sorted(
            zip(records, scores),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            (
                record["chunk_id"],
                float(score),
            )

            for record, score in ranked[:k]
        ]

    def hybrid_search(
        self,
        query: str,
        equipment_model: str,
        k: int = 5,
        candidate_k: int = 10,
    ) -> list[Chunk]:
        """
        Reciprocal Rank Fusion over vector and BM25 rankings.
        """

        vector_results = self.vector_search(
            query,
            equipment_model,
            k=candidate_k,
        )

        bm25_results = self.bm25_search(
            query,
            equipment_model,
            k=candidate_k,
        )

        fused_scores = defaultdict(
            float
        )

        for ranking in [
            vector_results,
            bm25_results,
        ]:

            for rank, (
                chunk_id,
                _raw_score,
            ) in enumerate(
                ranking,
                start=1,
            ):

                fused_scores[
                    chunk_id
                ] += (
                    1.0
                    / (
                        RRF_K
                        + rank
                    )
                )

        ranked_ids = sorted(
            fused_scores,
            key=fused_scores.get,
            reverse=True,
        )[:k]

        if not ranked_ids:
            return []

        fetched = self.collection.get(

            ids=ranked_ids,

            include=[
                "documents",
                "metadatas",
            ],
        )

        by_id = {}

        for chunk_id, text, metadata in zip(
            fetched["ids"],
            fetched["documents"],
            fetched["metadatas"],
        ):

            by_id[chunk_id] = (
                text,
                metadata,
            )

        results = []

        for chunk_id in ranked_ids:

            if chunk_id not in by_id:
                continue

            text, metadata = (
                by_id[chunk_id]
            )

            # Defence in depth:
            # never return cross-equipment evidence.
            if (
                metadata.get(
                    "equipment_model"
                )
                != equipment_model
            ):
                continue

            results.append(
                Chunk(
                    chunk_id=chunk_id,

                    doc_id=metadata[
                        "doc_id"
                    ],

                    section_id=metadata[
                        "section_id"
                    ],

                    revision=metadata[
                        "revision"
                    ],

                    equipment_model=metadata[
                        "equipment_model"
                    ],

                    source_file=metadata[
                        "source_file"
                    ],

                    page=int(
                        metadata[
                            "page"
                        ]
                    ),

                    section_title=metadata.get(
                        "section_title",
                        "Unknown"
                    ),

                    text=text,

                    score=float(
                        fused_scores[
                            chunk_id
                        ]
                    ),
                )
            )

        return results