"""Offline evaluation of the prompt-injection guard (no model, no network, no cost).

Runs `scan_user_input` over `eval/datasets/adversarial.jsonl` and reports how many attacks are blocked, how
many benign equipment questions are wrongly blocked, and whether the category is the expected one.

Attacks marked `known_miss` are techniques a pattern-based guard cannot see (encodings, homoglyphs, other
languages, paraphrases). They are kept in the data and reported separately, so the numbers stay honest:
`block_rate_catalogued` is the rate on what the guard is designed for, `block_rate_all` includes the rest.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

DATASET = REPO_ROOT / "eval" / "datasets" / "adversarial.jsonl"


def load_cases(path: Path = DATASET) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _rate(part: int, whole: int) -> float:
    return round(part / whole, 3) if whole else 0.0


def evaluate(cases: list[dict], scan=None) -> dict:
    """Metrics and the individual failures. `scan` defaults to the real guard (a stub can be passed in tests)."""
    if scan is None:
        from asa.guardrails.injection import scan_user_input as scan

    attacks = [case for case in cases if case["label"] == "attack"]
    benign = [case for case in cases if case["label"] == "benign"]
    catalogued = [case for case in attacks if not case.get("known_miss")]
    known_miss = [case for case in attacks if case.get("known_miss")]

    results = {case["id"]: scan(case["text"]) for case in cases}

    blocked_catalogued = [case for case in catalogued if results[case["id"]].flagged]
    blocked_all = [case for case in attacks if results[case["id"]].flagged]
    right_category = [case for case in blocked_catalogued if case.get("category") in results[case["id"]].categories]
    false_positives = [case for case in benign if results[case["id"]].flagged]

    return {
        "attacks_total": len(attacks),
        "attacks_catalogued": len(catalogued),
        "attacks_known_miss": len(known_miss),
        "benign_total": len(benign),
        "block_rate_catalogued": _rate(len(blocked_catalogued), len(catalogued)),
        "block_rate_all": _rate(len(blocked_all), len(attacks)),
        "category_accuracy": _rate(len(right_category), len(blocked_catalogued)),
        "false_positive_rate": _rate(len(false_positives), len(benign)),
        "known_miss_blocked": len(blocked_all) - len(blocked_catalogued),
        "missed_catalogued": [
            {"id": case["id"], "text": case["text"]} for case in catalogued if not results[case["id"]].flagged
        ],
        "false_positives": [
            {"id": case["id"], "text": case["text"], "categories": results[case["id"]].categories}
            for case in false_positives
        ],
    }


if __name__ == "__main__":
    print(json.dumps(evaluate(load_cases()), indent=2, ensure_ascii=False))
