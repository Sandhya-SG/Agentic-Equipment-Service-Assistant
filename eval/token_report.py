"""Tokens and latency per question and per agent, measured on real model calls.

    python eval/token_report.py                      # run the questions of eval/datasets/token_questions.jsonl
    python eval/token_report.py --out before.json    # also save the numbers
    python eval/token_report.py --compare before.json after.json

Each question goes through the real graph (`run_graph`), which records every model call of the request: the agent that
made it, the model, the input and output tokens and the seconds. The table shows where the tokens and the time go, so a
change to a prompt or to the number of calls can be judged by numbers. It calls OpenAI (about 10 questions, a few
cents) and needs the key and the manual index. It does not need Langfuse; the same numbers can be checked there.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

DATASET = REPO_ROOT / "eval" / "datasets" / "token_questions.jsonl"

# The model call names (Langfuse generation names or `module.function`) grouped by the agent that makes them.
AGENT_OF = {
    "assess-request-readiness": "request",
    "plan-specialist-route": "planning",
    "select-search-query": "rag",
    "assess-retrieval": "rag",
    "reformulate-search-query": "rag",
    "review-failed-retrieval": "rag",
    "generate-grounded-answer": "rag",
    "build-safety-search-query": "safety",
    "validate-safety-claim": "safety",
    "validate-safety-claims": "safety",
    "validate-safety-verdict": "safety",
    "assess-safety": "safety",
    "validate-diagnostic-cause": "diagnostic",
    "validate-troubleshooting-step": "diagnostic",
    "generate-diagnosis": "diagnostic",
}


def agent_of(call: str) -> str:
    if call in AGENT_OF:
        return AGENT_OF[call]
    return call.split(".", 1)[0]  # `module.function`: the module is the agent


def load_questions(path: Path = DATASET) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarise(calls: list[dict]) -> dict:
    """Totals of one question's model calls, and the same per agent."""
    per_agent: dict[str, dict] = defaultdict(lambda: {"calls": 0, "input": 0, "output": 0, "seconds": 0.0})
    for call in calls:
        row = per_agent[agent_of(call["call"])]
        row["calls"] += 1
        row["input"] += call.get("input") or 0
        row["output"] += call.get("output") or 0
        row["seconds"] = round(row["seconds"] + (call.get("seconds") or 0.0), 3)
    return {
        "calls": len(calls),
        "input": sum(row["input"] for row in per_agent.values()),
        "output": sum(row["output"] for row in per_agent.values()),
        "per_agent": dict(per_agent),
    }


def run_questions(questions: list[dict]) -> list[dict]:
    from app.services.graph_runner import run_graph

    rows = []
    for item in questions:
        started = time.monotonic()
        with contextlib.redirect_stdout(io.StringIO()):  # the agents print their progress
            state = run_graph(item["question"], item["equipment_model"])
        seconds = round(time.monotonic() - started, 2)
        clarify = state.get("request_status") == "CLARIFY"
        row = {
            "id": item["id"],
            "kind": item.get("kind"),
            "question": item["question"],
            "status": "clarification" if clarify else state.get("current_step"),
            "escalated": bool(state.get("escalated")),
            "verdict": state.get("safety_verdict"),
            "seconds": seconds,
            **summarise(state.get("llm_calls") or []),
        }
        rows.append(row)
        print(
            f"{row['id']} {row['kind']:<13} {row['calls']:>2} calls "
            f"{row['input']:>6} in {row['output']:>5} out {seconds:>6.1f}s"
        )
    return rows


def aggregate(rows: list[dict]) -> dict:
    per_agent: dict[str, dict] = defaultdict(lambda: {"calls": 0, "input": 0, "output": 0, "seconds": 0.0})
    for row in rows:
        for agent, values in row["per_agent"].items():
            for key in ("calls", "input", "output", "seconds"):
                per_agent[agent][key] += values[key]
    times = sorted(row["seconds"] for row in rows)
    return {
        "questions": len(rows),
        "calls": sum(row["calls"] for row in rows),
        "input": sum(row["input"] for row in rows),
        "output": sum(row["output"] for row in rows),
        "seconds_median": statistics.median(times),
        "seconds_max": times[-1],
        "per_agent": {agent: {**v, "seconds": round(v["seconds"], 2)} for agent, v in sorted(per_agent.items())},
    }


def print_summary(total: dict) -> None:
    print(
        f"\n{total['questions']} questions, {total['calls']} model calls, {total['input']} input and "
        f"{total['output']} output tokens; latency median {total['seconds_median']}s, max {total['seconds_max']}s"
    )
    print(f"{'agent':<12}{'calls':>6}{'input':>9}{'output':>8}{'seconds':>9}")
    for agent, v in total["per_agent"].items():
        print(f"{agent:<12}{v['calls']:>6}{v['input']:>9}{v['output']:>8}{v['seconds']:>9}")


def compare(before: dict, after: dict) -> list[str]:
    """Lines comparing two saved reports (totals and per agent)."""

    def change(old: float, new: float) -> str:
        return f"{(new - old) / old:+.0%}" if old else "n/a"

    b, a = before["total"], after["total"]
    lines = [f"{'':<12}{'before':>10}{'after':>10}{'change':>9}"]
    for key in ("calls", "input", "output"):
        lines.append(f"{key:<12}{b[key]:>10}{a[key]:>10}{change(b[key], a[key]):>9}")
    median_b, median_a = b["seconds_median"], a["seconds_median"]
    lines.append(f"{'median s':<12}{median_b:>10}{median_a:>10}{change(median_b, median_a):>9}")
    lines.append("")
    lines.append(f"{'input/agent':<12}{'before':>10}{'after':>10}{'change':>9}")
    for agent in sorted(set(b["per_agent"]) | set(a["per_agent"])):
        x = b["per_agent"].get(agent, {}).get("input", 0)
        y = a["per_agent"].get(agent, {}).get("input", 0)
        lines.append(f"{agent:<12}{x:>10}{y:>10}{change(x, y):>9}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", help="write the numbers to this file")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"), help="compare two saved reports")
    args = parser.parse_args(argv)

    if args.compare:
        before, after = (json.loads(Path(name).read_text(encoding="utf-8")) for name in args.compare)
        print("\n".join(compare(before, after)))
        return 0

    from asa.versioning import build_info

    rows = run_questions(load_questions())
    total = aggregate(rows)
    print_summary(total)
    if args.out:
        report = {
            "created": datetime.now(timezone.utc).isoformat(),
            "versions": build_info(include_datasets=True),
            "total": total,
            "rows": rows,
        }
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"report written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
