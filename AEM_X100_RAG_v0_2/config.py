from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "synthetic_docs"
DATA_DIR = ROOT / "data"
CHUNKS_FILE = DATA_DIR / "chunks.json"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.npy"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5
