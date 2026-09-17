import json, numpy as np
from sentence_transformers import SentenceTransformer
from config import CHUNKS_FILE, EMBEDDINGS_FILE, EMBEDDING_MODEL

with open(CHUNKS_FILE, encoding="utf-8") as f:
    records = json.load(f)

model = SentenceTransformer(EMBEDDING_MODEL)
embeddings = model.encode(
    [r["text"] for r in records],
    normalize_embeddings=True,
    show_progress_bar=True
)
np.save(EMBEDDINGS_FILE, embeddings)
print("Embedding matrix:", embeddings.shape)
