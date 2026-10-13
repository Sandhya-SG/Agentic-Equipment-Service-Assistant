"""Offline evaluation of the response firewall (no model, no network, no cost).

Runs `guard_output` over `eval/datasets/output_cases.jsonl`. Each case is an assistant answer plus the manual passages it
was built from, and says what the firewall must do:

  action "none"      a clean answer (equipment values, citations, the manual's own contact details): untouched
  action "redacted"  secrets, personal data, executable markup or over-long text: cleaned
  action "withheld"  a leaked prompt or an internal structure: replaced whole

`must_not_contain` lists strings that must be gone from the returned text (the secret itself, the e-mail address).

Metrics: catch_rate (answers that had to be acted on and were), false_action_rate (clean answers that were touched),
leak_free_rate (cases where none of the listed strings survive), action_accuracy and findings_exact.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

DATASET = REPO_ROOT / "eval" / "datasets" / "output_cases.jsonl"


def load_cases(path: Path = DATASET) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _rate(part: int, whole: int) -> float:
    return round(part / whole, 3) if whole else 0.0


def evaluate(cases: list[dict], guard=None) -> dict:
    """Metrics and the individual failures. `guard` defaults to the real firewall (a stub can be passed in tests)."""
    if guard is None:
        from asa.guardrails.schema_guard import guard_output as guard

    results = {case["id"]: guard(case["answer"], case.get("evidence", [])) for case in cases}
    must_act = [case for case in cases if case["expect_action"] != "none"]
    clean = [case for case in cases if case["expect_action"] == "none"]
    with_strings = [case for case in cases if case.get("must_not_contain")]
    with_findings = [case for case in cases if "expect_findings" in case]

    caught = [case for case in must_act if results[case["id"]].action == case["expect_action"]]
    touched = [
        case for case in clean if results[case["id"]].action != "none" or results[case["id"]].text != case["answer"]
    ]
    leak_free = [
        case
        for case in with_strings
        if not any(text.lower() in results[case["id"]].text.lower() for text in case["must_not_contain"])
    ]
    action_ok = [case for case in cases if results[case["id"]].action == case["expect_action"]]
    findings_ok = [
        case for case in with_findings if sorted(results[case["id"]].findings) == sorted(case["expect_findings"])
    ]

    return {
        "cases_total": len(cases),
        "must_act_total": len(must_act),
        "clean_total": len(clean),
        "catch_rate": _rate(len(caught), len(must_act)),
        "false_action_rate": _rate(len(touched), len(clean)),
        "leak_free_rate": _rate(len(leak_free), len(with_strings)),
        "action_accuracy": _rate(len(action_ok), len(cases)),
        "findings_exact": _rate(len(findings_ok), len(with_findings)),
        "missed": [
            {"id": case["id"], "expected": case["expect_action"], "got": results[case["id"]].action}
            for case in must_act
            if results[case["id"]].action != case["expect_action"]
        ],
        "false_actions": [{"id": case["id"], "answer": case["answer"][:80]} for case in touched],
        "leaks": [{"id": case["id"]} for case in with_strings if case not in leak_free],
    }


if __name__ == "__main__":
    print(json.dumps(evaluate(load_cases()), indent=2, ensure_ascii=False))
