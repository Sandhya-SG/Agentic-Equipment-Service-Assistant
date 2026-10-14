"""Measure how often the Request Agent asks the technician for more detail.

The Request Agent decides whether a question is clear enough to search the manuals. If it is too strict, the
technician is asked again and again. This script runs ONLY that agent (no retrieval, no answer) over
`eval/datasets/clarification.jsonl` and reports how often it behaves as labelled.

It calls OpenAI (about 55 short calls per run, a few cents), so it needs a real key.

    python eval/clarification_eval.py                    # one run, prints the report
    python eval/clarification_eval.py --repeat 3         # runs each case 3 times (model output can vary)
    python eval/clarification_eval.py --out report.json  # also write the numbers to a file

Case kinds in the dataset:
    ready                 a clear question: must go straight through (READY)
    clarify               a vague request: should ask (CLARIFY)
    safety_ready          a safety-critical request: must never be held back by a question (READY)
    continuation_ready    the vague question plus the technician's useful reply: must now be accepted (READY)
    continuation_clarify  the vague question plus a reply that says nothing: may ask again (CLARIFY)

The labels marked "judgement" in the file are opinions; a subject-matter expert should confirm them.
"""

import argparse
import contextlib
import io
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

DATASET = REPO_ROOT / "eval" / "datasets" / "clarification.jsonl"
EXPECTED = {
    "ready": "READY",
    "clarify": "CLARIFY",
    "safety_ready": "READY",
    "continuation_ready": "READY",
    "continuation_clarify": "CLARIFY",
}


def load_cases(path: Path = DATASET) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_state(case: dict) -> dict:
    """The state the Request Agent receives: a fresh question, or a question plus the technician's reply."""
    from asa.agents.clarification import resolve_clarification

    state = {"raw_query": case["query"], "equipment_model": case["equipment_model"]}
    if "clarification" in case:
        state["resolved_query"] = resolve_clarification(case["query"], case["clarification"])
    return state


def run_case(case: dict) -> dict:
    from asa.agents.request import request_node

    # The agent prints its progress; keep the report readable.
    with contextlib.redirect_stdout(io.StringIO()):
        result = request_node(build_state(case))
    return {"status": result["request_status"], "question": result.get("clarification_question")}


def evaluate(cases: list[dict], repeat: int = 1) -> list[dict]:
    """Run every case `repeat` times; each result is the case plus the status the agent returned."""
    results = []
    for case in cases:
        for _ in range(repeat):
            results.append({**case, **run_case(case)})
    return results


def summarise(results: list[dict]) -> dict:
    """Pass rate per kind. A case passes when the agent returned the labelled status."""
    per_kind: dict[str, list[bool]] = defaultdict(list)
    for item in results:
        per_kind[item["kind"]].append(item["status"] == EXPECTED[item["kind"]])
    rates = {kind: sum(passed) / len(passed) for kind, passed in per_kind.items()}
    # What the technician feels: clear questions that were still turned back.
    asked_when_clear = [
        item
        for item in results
        if item["kind"] in {"ready", "safety_ready", "continuation_ready"} and item["status"] == "CLARIFY"
    ]
    clear_total = sum(1 for item in results if item["kind"] in {"ready", "safety_ready", "continuation_ready"})
    return {
        "pass_rate_by_kind": {kind: round(rate, 3) for kind, rate in sorted(rates.items())},
        "clear_questions_turned_back": len(asked_when_clear),
        "clear_questions_total": clear_total,
        "turned_back_rate": round(len(asked_when_clear) / clear_total, 3) if clear_total else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeat", type=int, default=1, help="run each case this many times")
    parser.add_argument("--out", help="write the numbers and every answer to this JSON file")
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set: this evaluation calls OpenAI. Put the key in .env.")
        return 2

    results = evaluate(load_cases(), args.repeat)

    failures = [item for item in results if item["status"] != EXPECTED[item["kind"]]]
    for item in failures:
        shown = item["query"] + (f"  +  {item['clarification']}" if "clarification" in item else "")
        asked = f"  -> asked: {item['question']!r}" if item["status"] == "CLARIFY" else ""
        print(f"FAIL [{item['kind']}] expected {EXPECTED[item['kind']]}, got {item['status']}: {shown!r}{asked}")

    summary = summarise(results)
    print(f"\n{len(results)} runs, {len(failures)} not as labelled")
    for kind, rate in summary["pass_rate_by_kind"].items():
        print(f"  {kind:<22} {rate:>6.1%}")
    print(
        f"  clear questions turned back: {summary['clear_questions_turned_back']} of {summary['clear_questions_total']}"
        f" ({summary['turned_back_rate']:.1%})"
    )

    if args.out:
        report = {"created": datetime.now(timezone.utc).isoformat(), "summary": summary, "results": results}
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"report written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
