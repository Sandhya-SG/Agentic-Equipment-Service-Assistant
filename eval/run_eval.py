"""Run the offline evaluations (no OpenAI calls, no network, no cost) and compare them with the targets.

    python eval/run_eval.py                       # print the report; exit 0 even if a target is missed
    python eval/run_eval.py --enforce             # exit 1 if any target is missed (a CI gate)
    python eval/run_eval.py --out report.json     # also write every number and failure to a file

Suites: the prompt-injection guard (`injection_eval.py`), the deterministic safety rules (`safety_eval.py`) the
response firewall (`output_eval.py`) and the explainability layer (`explanation_eval.py`, offline part).
Targets live in `eval/thresholds.json`. The suites that call OpenAI (clarification, grounding) are separate and
opt-in; see `eval/clarification_eval.py`.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent
for extra in (EVAL_DIR, REPO_ROOT, REPO_ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

THRESHOLDS = EVAL_DIR / "thresholds.json"


def run_offline() -> dict:
    import explanation_eval
    import injection_eval
    import output_eval
    import retrieval_eval
    import safety_eval

    return {
        "injection": injection_eval.evaluate(injection_eval.load_cases()),
        "safety_rules": safety_eval.evaluate(safety_eval.load_cases()),
        "output_guard": output_eval.evaluate(output_eval.load_cases()),
        "explanation": explanation_eval.evaluate(explanation_eval.load_cases()),
        # Needs the manual index and the local embedding model; reported as unavailable without them.
        "retrieval": retrieval_eval.run(),
    }


def check_targets(report: dict, targets: dict) -> list[dict]:
    """One row per target: the value, the target and whether it is met."""
    rows = []
    for suite, metrics in targets.items():
        if suite.startswith("_") or report.get(suite, {}).get("available") is False:
            continue  # a suite that could not run (no index) has no value to judge
        for metric, bound in metrics.items():
            value = report[suite][metric]
            if "min" in bound:
                met, target = value >= bound["min"], f">= {bound['min']}"
            else:
                met, target = value <= bound["max"], f"<= {bound['max']}"
            rows.append({"suite": suite, "metric": metric, "value": value, "target": target, "met": met})
    return rows


def print_report(report: dict, rows: list[dict]) -> None:
    print(f"{'suite':<14}{'metric':<26}{'value':>8}  {'target':<9}result")
    for row in rows:
        print(
            f"{row['suite']:<14}{row['metric']:<26}{row['value']:>8}  {row['target']:<9}{'ok' if row['met'] else 'MISSED'}"
        )

    injection = report["injection"]
    print(
        f"\ninjection: {injection['attacks_catalogued']} catalogued attacks, {injection['attacks_known_miss']} obfuscated"
        f" (a pattern guard cannot see them); block rate on everything: {injection['block_rate_all']}"
    )
    for item in injection["missed_catalogued"]:
        print(f"  attack not blocked: {item['text']!r}")
    for item in injection["false_positives"]:
        print(f"  benign wrongly blocked ({', '.join(item['categories'])}): {item['text']!r}")

    safety = report["safety_rules"]
    print(f"\nsafety rules: {safety['known_gaps_still_open']} of {safety['known_gaps_total']} known gaps still open")
    for item in safety["missed"]:
        print(f"  unsafe text not halted: {item['text']!r}")
    for item in safety["false_halts"]:
        print(f"  safe text halted: {item['text']!r}")
    for item in safety["open_gaps"]:
        print(f"  known gap: {item['text']!r} ({item['note']})")

    explanation = report["explanation"]
    print(
        f"\nexplainability: {explanation['cases_total']} cases; expectations met "
        f"{explanation['expectation_pass_rate']:.0%} (the near-miss claims are ordinary cases); "
        f"{explanation['known_gaps_open']} known gap(s) open"
    )
    for item in explanation["failures"]:
        print(f"  explanation case failed ({item['id']}): {item['problems']}")
    for gap in explanation["open_gaps"]:
        print(f"  known gap ({gap['id']}): {gap['note']}")

    retrieval = report["retrieval"]
    if retrieval.get("available") is False:
        print(f"\nretrieval: skipped ({retrieval['reason']})")
    else:
        print(
            f"\nretrieval: {retrieval['questions']} questions; right area of the manual in the top 5 "
            f"{retrieval['area_hit_at_5']:.1%} (exact section {retrieval['hit_at_5']:.1%}), MRR {retrieval['area_mrr']} "
            f"(exact {retrieval['mrr']}), equipment isolation {retrieval['isolation']:.0%}"
        )
        for item in retrieval["misses"]:
            print(f"  right area not in the top 5 ({item['id']}): {item['query'][:70]!r}")

    output = report["output_guard"]
    print(
        f"\nresponse firewall: {output['must_act_total']} answers that must be cleaned or withheld, {output['clean_total']} clean ones"
    )
    for item in output["missed"]:
        print(f"  not acted on ({item['id']}): expected {item['expected']}, got {item['got']}")
    for item in output["false_actions"]:
        print(f"  clean answer touched ({item['id']}): {item['answer']!r}")
    for item in output["leaks"]:
        print(f"  text survived the firewall ({item['id']})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--enforce", action="store_true", help="exit with status 1 if any target is missed")
    parser.add_argument("--out", help="write the full report as JSON to this file")
    args = parser.parse_args()

    report = run_offline()
    rows = check_targets(report, json.loads(THRESHOLDS.read_text(encoding="utf-8")))
    print_report(report, rows)

    missed = [row for row in rows if not row["met"]]
    print(f"\n{len(rows) - len(missed)} of {len(rows)} targets met" + (" (report only)" if not args.enforce else ""))

    if args.out:
        from asa.versioning import build_info

        payload = {
            "created": datetime.now(timezone.utc).isoformat(),
            "versions": build_info(include_datasets=True),  # code, models, prompts, manuals, datasets
            "targets": rows,
            "report": report,
        }
        Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"report written to {args.out}")

    return 1 if (args.enforce and missed) else 0


if __name__ == "__main__":
    sys.exit(main())
