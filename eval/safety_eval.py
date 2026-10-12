"""Offline evaluation of the deterministic safety rules (no model, no network, no cost).

Runs `check_text` over `eval/datasets/safety_cases.jsonl`. Unsafe text must halt with the expected hazards and
the right escalate-only flag; safe text must not halt. Cases marked `known_gap` are phrasings the current
patterns do not cover (for example the British spelling "energised"): they are reported separately as gaps, so
a fix shows up as a closed gap and nothing is hidden.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

DATASET = REPO_ROOT / "eval" / "datasets" / "safety_cases.jsonl"


def load_cases(path: Path = DATASET) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _rate(part: int, whole: int) -> float:
    return round(part / whole, 3) if whole else 0.0


def evaluate(cases: list[dict], check=None) -> dict:
    """Metrics and the individual failures. `check` defaults to the real rules (a stub can be passed in tests)."""
    if check is None:
        from asa.guardrails.safety_rules import check_text as check

    results = {case["id"]: check(case["text"]) for case in cases}
    covered = [case for case in cases if not case.get("known_gap")]
    gaps = [case for case in cases if case.get("known_gap")]
    unsafe = [case for case in covered if case["expect_halt"]]
    safe = [case for case in covered if not case["expect_halt"]]

    halted = [case for case in unsafe if results[case["id"]].forced_halt]
    false_halts = [case for case in safe if results[case["id"]].forced_halt]
    hazards_exact = [case for case in unsafe if sorted(results[case["id"]].hazards) == case["expect_hazards"]]
    escalate_ok = [case for case in covered if results[case["id"]].escalate_only == case["expect_escalate_only"]]
    ppe_given = [case for case in halted if results[case["id"]].ppe_required]

    return {
        "cases_total": len(cases),
        "unsafe_total": len(unsafe),
        "safe_total": len(safe),
        "halt_recall": _rate(len(halted), len(unsafe)),
        "false_halt_rate": _rate(len(false_halts), len(safe)),
        "hazards_exact": _rate(len(hazards_exact), len(unsafe)),
        "escalate_only_accuracy": _rate(len(escalate_ok), len(covered)),
        "ppe_given_when_halted": _rate(len(ppe_given), len(halted)),
        "known_gaps_total": len(gaps),
        "known_gaps_still_open": sum(1 for case in gaps if not results[case["id"]].forced_halt),
        "missed": [{"id": case["id"], "text": case["text"]} for case in unsafe if not results[case["id"]].forced_halt],
        "false_halts": [{"id": case["id"], "text": case["text"]} for case in false_halts],
        "open_gaps": [
            {"id": case["id"], "text": case["text"], "note": case.get("note", "")}
            for case in gaps
            if not results[case["id"]].forced_halt
        ],
    }


if __name__ == "__main__":
    print(json.dumps(evaluate(load_cases()), indent=2, ensure_ascii=False))
