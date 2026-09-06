import json, numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from config import CHUNKS_FILE, EMBEDDINGS_FILE, EMBEDDING_MODEL, TOP_K

class LocalRAG:
    def __init__(self):
        with open(CHUNKS_FILE, encoding="utf-8") as f:
            self.records = json.load(f)
        self.embeddings = np.load(EMBEDDINGS_FILE)
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.bm25 = BM25Okapi([r["text"].lower().split() for r in self.records])

    def vector_search(self, query, k=10):
        q = self.model.encode([query], normalize_embeddings=True)
        scores = cosine_similarity(q, self.embeddings)[0]
        idx = np.argsort(scores)[::-1][:k]
        return [(i, float(scores[i])) for i in idx]

    def bm25_search(self, query, k=10):
        scores = self.bm25.get_scores(query.lower().split())
        idx = np.argsort(scores)[::-1][:k]
        return [(i, float(scores[i])) for i in idx]

    def hybrid_search(self, query, k=5):
        # Reciprocal Rank Fusion: simple, robust v0.1 hybrid retrieval.
        rankings = [self.vector_search(query, 10), self.bm25_search(query, 10)]
        fused = {}
        for ranking in rankings:
            for rank, (idx, _) in enumerate(ranking, 1):
                fused[idx] = fused.get(idx, 0) + 1 / (60 + rank)
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        return [{**self.records[i], "rrf_score": fused[i]} for i in top]

if __name__ == "__main__":
    rag = LocalRAG()
    while True:
        q = input("\nQuestion (or quit): ").strip()
        if q.lower() in {"quit", "exit"}: break
        for n, r in enumerate(rag.hybrid_search(q), 1):
            print(f"\n[{n}] {r['source']} | page {r['page']} | {r['rrf_score']:.4f}")
            print(r["text"])
