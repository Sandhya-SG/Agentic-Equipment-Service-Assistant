from rag import LocalRAG

TESTS = [
    "The wafer is not detected after loading. What should I check?",
    "What does error E101 mean?",
    "The transfer position is not confirmed. What should I do?",
    "Can I open the panel while the equipment is powered?",
    "What are the monthly preventive maintenance activities?",
    "What should happen if the required procedure is not found?"
]

rag = LocalRAG()
for q in TESTS:
    print("\n" + "=" * 90)
    print(q)
    print("=" * 90)
    for i, r in enumerate(rag.hybrid_search(q), 1):
        print(f"{i}. {r['source']} p.{r['page']} | RRF={r['rrf_score']:.4f}")
        print(r["text"][:400].replace("\n", " ") + "...")
