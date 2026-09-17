# AEM-X100 Servicing Equipment Assistant — RAG v0.1

First milestone: local PDF ingestion, chunking, embeddings, vector retrieval,
BM25 retrieval and hybrid retrieval.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\\Scripts\\activate    # Windows

pip install -r requirements.txt
```

## Run

```bash
python chunking.py
python embed.py
python test_rag.py
```

Interactive retrieval:

```bash
python rag.py
```

## Architecture

Synthetic PDFs
 -> PDF extraction
 -> chunking + metadata
 -> local embeddings + BM25
 -> hybrid retrieval
 -> top-K evidence

No LLM or agents yet.

## Next versions

v0.2 Agentic RAG
v0.3 Request + Planning
v0.4 Diagnostic
v0.5 Safety & Compliance
v0.6 AI Security / Guardrails
v0.7 Explainability + Logging
v0.8 End-to-end evaluation

IMPORTANT: all supplied AEM-X100 documents are fictional synthetic training data.
They are not real AEM procedures and must not be used for equipment servicing.
