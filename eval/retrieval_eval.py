"""Retrieval quality: does the right manual section come back, and how high in the ranking?

    python eval/retrieval_eval.py                       # run the judgements, print the metrics
    python eval/retrieval_eval.py --out retrieval.json  # also save every number and miss

Each judgement in `eval/datasets/retrieval_judgements.jsonl` is a question, the equipment it is about and the manual
section that answers it (the same 16 questions as the gold set, with room for more). The hybrid retriever (vector search
plus BM25, fused by reciprocal rank) is asked for the top `MAX_K` passages of the selected manual, and:

  * Hit Rate@k  the share of questions where a passage of the right section is among the first k (k = 1, 3, 5);
  * MRR         the mean of 1 / (rank of the first right passage), 0 when it is not in the top MAX_K;
  * isolation   the share of questions where EVERY returned passage belongs to the selected equipment (must be 100%).

Each is reported at two levels, because the chunks carry only coarse section titles (a passage of 6.2.3 is often labelled
"6.2 Machine Operating Procedure"):
  * strict  the passage's section is the judged section or one inside it ("2.5.4" also accepts "2.5.4.1");
  * area    as strict, and also a passage labelled with a section that CONTAINS the judged one, at least two levels deep
            ("6.2" for "6.2.3", but not the whole chapter "6"): the retriever reached the right part of the manual.
If the judged section has no number, the titles must match. It uses the local embedding model and the index built by
`scripts/ingest.py`; it makes no OpenAI call and costs nothing. Without an index the suite reports `available: false`.

The question is searched as the engineer wrote it. The agents rewrite the question into a search query first, so this
measures the retriever itself, not the whole pipeline.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

DATASET = REPO_ROOT / "eval" / "datasets" / "retrieval_judgements.jsonl"
KS = (1, 3, 5)
MAX_K = 10
_NUMBER = re.compile(r"^\d+(?:\.\d+)*")


def load_cases(path: Path = DATASET) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _normal(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _number_of(text: str) -> str | None:
    match = _NUMBER.match(text)
    return match.group(0).rstrip(".") if match else None


def section_matches(section_title: str, relevant: str, area: bool = False) -> bool:
    """True when a passage's section is the judged section (or one inside it); with `area`, also when it is a
    section that contains the judged one, at least two levels deep."""
    got, want = _normal(section_title), _normal(relevant)
    if not got or not want:
        return False
    wanted = _number_of(want)
    if wanted is None:
        return got == want or want in got
    have = _number_of(got)
    if have is None:
        return False
    if have == wanted or have.startswith(wanted + "."):
        return True
    return area and have.count(".") >= 1 and wanted.startswith(have + ".")


def rank_of_first_hit(chunks: list, relevant: list[str], area: bool = False) -> int | None:
    """1-based rank of the first passage that belongs to any relevant section, or None."""
    for rank, chunk in enumerate(chunks, start=1):
        title = getattr(chunk, "section_title", None) if not isinstance(chunk, dict) else chunk.get("section_title")
        if any(section_matches(title, section, area) for section in relevant):
            return rank
    return None


def _equipment_of(chunk) -> str | None:
    return chunk.get("equipment_model") if isinstance(chunk, dict) else getattr(chunk, "equipment_model", None)


def evaluate(cases: list[dict], search) -> dict:
    """Metrics over the cases. `search(query, equipment_model, k)` returns the ranked passages."""
    rows = []
    for case in cases:
        chunks = list(search(case["query"], case["equipment_model"], MAX_K))
        relevant = case["relevant_sections"]
        rank = rank_of_first_hit(chunks, relevant)
        area_rank = rank_of_first_hit(chunks, relevant, area=True)
        rows.append(
            {
                "id": case["id"],
                "equipment_model": case["equipment_model"],
                "category": case.get("category"),
                "query": case["query"],
                "relevant_sections": relevant,
                "rank": rank,
                "area_rank": area_rank,
                "isolated": all(_equipment_of(c) == case["equipment_model"] for c in chunks),
                "returned": [
                    (c.get("section_title") if isinstance(c, dict) else getattr(c, "section_title", None))
                    for c in chunks[:5]
                ],
            }
        )

    def rate(values) -> float:
        values = list(values)
        return round(sum(values) / len(values), 3) if values else 0.0

    report = {
        "available": True,
        "questions": len(rows),
        "mrr": rate(1 / r["rank"] if r["rank"] else 0.0 for r in rows),
        "area_mrr": rate(1 / r["area_rank"] if r["area_rank"] else 0.0 for r in rows),
        "isolation": rate(r["isolated"] for r in rows),
        "misses": [r for r in rows if r["area_rank"] is None or r["area_rank"] > max(KS)],
        "strict_misses": [r for r in rows if r["rank"] is None or r["rank"] > max(KS)],
        "by_equipment": {},
        "by_category": {},
        "rows": rows,
    }
    for k in KS:
        report[f"hit_at_{k}"] = rate(r["rank"] is not None and r["rank"] <= k for r in rows)
        report[f"area_hit_at_{k}"] = rate(r["area_rank"] is not None and r["area_rank"] <= k for r in rows)
    for key, name in (("equipment_model", "by_equipment"), ("category", "by_category")):
        groups: dict[str, list] = {}
        for row in rows:
            groups.setdefault(row[key] or "n/a", []).append(row)
        report[name] = {
            group: {
                "questions": len(items),
                f"hit_at_{max(KS)}": rate(r["rank"] is not None and r["rank"] <= max(KS) for r in items),
                f"area_hit_at_{max(KS)}": rate(r["area_rank"] is not None and r["area_rank"] <= max(KS) for r in items),
                "mrr": rate(1 / r["rank"] if r["rank"] else 0.0 for r in items),
                "area_mrr": rate(1 / r["area_rank"] if r["area_rank"] else 0.0 for r in items),
            }
            for group, items in sorted(groups.items())
        }
    return report


def unavailable(reason: str) -> dict:
    return {"available": False, "reason": reason}


def make_search():
    """The real retriever over the local index, or None when there is no index to search."""
    try:
        from asa.ingestion.embed import CHROMA_DIR
        from asa.ingestion.retrieval import AEMHybridRetriever

        # No index on disk (CI, a fresh clone): skip, without loading the embedding model or creating an empty index.
        if not (Path(CHROMA_DIR) / "chroma.sqlite3").exists():
            return None
        retriever = AEMHybridRetriever()
        if retriever.collection.count() == 0:
            return None
    except Exception:  # noqa: BLE001 - no index, no embedding model, missing packages: the suite is skipped
        return None
    return lambda query, equipment, k: retriever.hybrid_search(query, equipment, k=k, candidate_k=max(k, 10))


def run() -> dict:
    search = make_search()
    if search is None:
        return unavailable("no manual index (run scripts/ingest.py) or no embedding model")
    return evaluate(load_cases(), search)


def print_report(report: dict) -> None:
    if not report.get("available"):
        print(f"retrieval evaluation skipped: {report.get('reason')}")
        return
    print(f"{report['questions']} questions; isolation {report['isolation']:.0%}")
    print("  strict  " + "  ".join(f"hit@{k} {report[f'hit_at_{k}']:.1%}" for k in KS) + f"  MRR {report['mrr']}")
    print(
        "  area    "
        + "  ".join(f"hit@{k} {report[f'area_hit_at_{k}']:.1%}" for k in KS)
        + f"  MRR {report['area_mrr']}"
    )
    top = max(KS)
    for group, values in report["by_equipment"].items():
        print(
            f"  {group:<24} area hit@{top} {values[f'area_hit_at_{top}']:.1%}  area MRR {values['area_mrr']}"
            f"  ({values['questions']} questions)"
        )
    for row in report["misses"]:
        print(
            f"  right area not in the top {top}: {row['id']} {row['query'][:70]!r} expected {row['relevant_sections']}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", help="write the full report as JSON to this file")
    args = parser.parse_args(argv)
    report = run()
    print_report(report)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"report written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
