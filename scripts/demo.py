"""Quick retrieval smoke test."""

from __future__ import annotations

import sys

from asa.ingestion.embed import get_collection


def retrieve(query: str, k: int = 5):
    col = get_collection()
    res = col.query(query_texts=[query], n_results=k)
    hits = []
    for i in range(len(res["ids"][0])):
        hits.append({
            "chunk_id": res["ids"][0][i],
            "doc_id": res["metadatas"][0][i]["doc_id"],
            "section_id": res["metadatas"][0][i]["section_id"],
            "revision": res["metadatas"][0][i]["revision"],
            "distance": round(res["distances"][0][i], 4),
            "text": res["documents"][0][i][:160],
        })
    return hits


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "unit stuck in test socket"
    print(f"Query: {query}\n" + "=" * 70)
    for h in retrieve(query):
        print(f"[{h['distance']}] {h['doc_id']} rev {h['revision']} "
              f"({h['section_id']})")
        print(f"    {h['text']}...\n")


if __name__ == "__main__":
    main()
