"""Evaluation of the explainability layer (src/asa/components/explanation.py).

The layer attaches citations with exact quotes and an evidence confidence to every answer. This evaluation asks two
questions about it: does it keep its promises (offline, no model), and can it be fooled (offline and live)?

OFFLINE (no model, no network, no cost): `eval/datasets/explanation_cases.jsonl`, 33 hand-built answers with the manual
passages they came from.
  Invariants, checked on every explanation produced:
    quote_exactness       every quote is an exact stretch of the passage it is attributed to (nothing invented)
    citation_provenance   every citation names a passage that was really retrieved
    level_consistency     the level matches the confidence bands, and the confidence is within 0 and 1
    counts_consistent     claims verified is never more than claims checked; at most 8 citations
    non_mutation          the explanation never changes the state it reads
    deterministic         the same input gives the same explanation
  Expectations (what a faithful, an unsupported, a contradicting or an escalated answer must get):
    expectation_pass_rate  the share of cases that get what they must
  Known gaps (reported, not hidden): near-miss claims whose wording overlaps the manual but whose meaning differs are
  verified by the word-overlap matcher. `false_verification_rate` is the share of those cases it wrongly verified.

LIVE (`--live`, calls the models, a few cents): the 16 gold questions go through the real pipeline, then
    * the same invariants are checked on the real explanations, and
    * each real answer is corrupted in four ways (an invented claim with a real citation marker, a changed number, an
      inserted negation, a swapped entity) and the layer must notice. Reported at the answer level and, more
      precisely, at the claim level (only claims the layer had verified are corrupted, one at a time).

    python eval/explanation_eval.py                 # offline
    python eval/explanation_eval.py --live          # also run the gold set through the real pipeline
    python eval/explanation_eval.py --from-records  # repeat the live analysis on the answers saved by the last --live
"""

import copy
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

DATASET = REPO_ROOT / "eval" / "datasets" / "explanation_cases.jsonl"
GOLD_SET = REPO_ROOT / "eval" / "datasets" / "gold_set.jsonl"
REPORT_DIR = REPO_ROOT / "eval" / "reports"


def load_cases(path: Path = DATASET) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _rate(part: int, whole: int) -> float:
    return round(part / whole, 3) if whole else 0.0


def _norm(text: str) -> str:
    return " ".join(str(text).split())


def _to_state(case_state: dict) -> dict:
    """A fresh state for explain(): the passages as Chunk objects, everything else copied."""
    from asa.graph.state import Chunk

    state = copy.deepcopy(case_state)
    state["retrieved_chunks"] = [Chunk(**chunk) for chunk in state.get("retrieved_chunks", [])]
    return state


def _snapshot(state: dict) -> str:
    return json.dumps(state, default=lambda o: o.__dict__, sort_keys=True)


# --------------------------------------------------------------------------- #
# Invariants: true of EVERY explanation                                       #
# --------------------------------------------------------------------------- #


def check_invariants(state: dict, explanation: dict, bands: tuple[float, float], max_citations: int) -> dict[str, bool]:
    low, floor = bands
    chunks = {c.chunk_id: c for c in state["retrieved_chunks"]}
    citations = explanation.get("citations", [])

    def exact(citation: dict) -> bool:
        chunk = chunks.get(citation.get("chunk_id"))
        return bool(chunk) and bool(citation.get("quote")) and _norm(citation["quote"]) in _norm(chunk.text)

    def provenance(citation: dict) -> bool:
        chunk = chunks.get(citation.get("chunk_id"))
        return bool(chunk) and chunk.page == citation.get("page") and chunk.source_file == citation.get("source_file")

    confidence = explanation.get("confidence", -1)
    level = "low" if confidence < low else "medium" if confidence < floor else "high"
    try:
        json.dumps(explanation)
        serializable = True
    except (TypeError, ValueError):
        serializable = False
    return {
        "quote_exactness": all(exact(c) for c in citations),
        "citation_provenance": all(provenance(c) for c in citations),
        "level_consistency": 0 <= confidence <= 1 and explanation.get("confidence_level") == level,
        "counts_consistent": explanation.get("claims_verified", 0) <= explanation.get("claims_checked", 0)
        and len(citations) <= max_citations,
        "serializable": serializable,
    }


# --------------------------------------------------------------------------- #
# Expectations                                                                #
# --------------------------------------------------------------------------- #


def check_expectations(explanation: dict | None, expect: dict) -> list[str]:
    """What is wrong with this explanation compared with what the case requires (empty = fine)."""
    problems: list[str] = []
    if expect.get("none") is True:
        return ["expected no explanation"] if explanation is not None else []
    if explanation is None:
        return ["no explanation"]
    e = explanation
    if "verified" in expect:
        want = e["claims_checked"] if expect["verified"] == "all" else expect["verified"]
        if e["claims_verified"] != want or (expect["verified"] == "all" and e["claims_checked"] == 0):
            problems.append(f"verified {e['claims_verified']} of {e['claims_checked']}")
    if "citations" in expect and len(e["citations"]) != expect["citations"]:
        problems.append(f"{len(e['citations'])} citations")
    if "min_confidence" in expect and e["confidence"] < expect["min_confidence"]:
        problems.append(f"confidence {e['confidence']} below {expect['min_confidence']}")
    if "max_confidence" in expect and e["confidence"] > expect["max_confidence"]:
        problems.append(f"confidence {e['confidence']} above {expect['max_confidence']}")
    if "confidence" in expect and abs(e["confidence"] - expect["confidence"]) > 0.001:
        problems.append(f"confidence {e['confidence']}, expected {expect['confidence']}")
    if "level" in expect and e["confidence_level"] != expect["level"]:
        problems.append(f"level {e['confidence_level']}")
    if "note_contains" in expect and not any(expect["note_contains"] in note for note in e["notes"]):
        problems.append(f"no note mentioning {expect['note_contains']!r}")
    if "cites_pages" in expect and sorted({c["page"] for c in e["citations"]}) != sorted(expect["cites_pages"]):
        problems.append("cited pages differ")
    if expect.get("quote_has_no_newline") and any("\n" in c["quote"] for c in e["citations"]):
        problems.append("a quote contains a line break")
    if "max_citations" in expect and len(e["citations"]) > expect["max_citations"]:
        problems.append(f"{len(e['citations'])} citations")
    if "decision_verdict" in expect and e["decision"]["verdict"] != expect["decision_verdict"]:
        problems.append(f"verdict {e['decision']['verdict']}")
    return problems


def evaluate(cases: list[dict], explain=None) -> dict:
    """Metrics and the individual failures. `explain` defaults to the real layer (a stub can be passed in tests)."""
    if explain is None:
        from asa.components.explanation import CONFIDENCE_LOW, MAX_CITATIONS, explain
        from asa.graph.state import CONFIDENCE_FLOOR

        bands, max_citations = (CONFIDENCE_LOW, CONFIDENCE_FLOOR), MAX_CITATIONS
    else:
        bands, max_citations = (0.5, 0.7), 8

    invariant_totals: dict[str, list[bool]] = {}
    deterministic, non_mutating = [], []
    failures, gaps = [], []
    for case in cases:
        state = _to_state(case["state"])
        before = _snapshot(state)
        explanation = explain(state, case["status"])
        non_mutating.append(_snapshot(state) == before)
        again = explain(_to_state(case["state"]), case["status"])
        deterministic.append(json.dumps(explanation, sort_keys=True) == json.dumps(again, sort_keys=True))
        if explanation is not None:
            for name, ok in check_invariants(state, explanation, bands, max_citations).items():
                invariant_totals.setdefault(name, []).append(ok)
        problems = check_expectations(explanation, case["expect"])
        if case.get("known_gap"):
            gaps.append({"id": case["id"], "open": bool(problems), "note": case.get("note", ""), "problems": problems})
        elif problems:
            failures.append({"id": case["id"], "description": case["description"], "problems": problems})

    covered = [c for c in cases if not c.get("known_gap")]
    metrics = {name: _rate(sum(values), len(values)) for name, values in invariant_totals.items()}
    metrics["non_mutation"] = _rate(sum(non_mutating), len(non_mutating))
    metrics["deterministic"] = _rate(sum(deterministic), len(deterministic))
    return {
        "cases_total": len(cases),
        **metrics,
        "expectation_pass_rate": _rate(len(covered) - len(failures), len(covered)),
        "known_gaps_total": len(gaps),
        "known_gaps_open": sum(1 for g in gaps if g["open"]),
        "false_verification_rate": _rate(sum(1 for g in gaps if g["open"]), len(gaps)),
        "failures": failures,
        "open_gaps": [g for g in gaps if g["open"]],
    }


# --------------------------------------------------------------------------- #
# Live: real answers, and corrupted copies of them                            #
# --------------------------------------------------------------------------- #

_MARKER = re.compile(r"\[Source:[^\]]+\]")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _first_marker(answer: str) -> str | None:
    match = _MARKER.search(answer)
    return match.group(0) if match else None


_LIST_PREFIX = re.compile(r"^\s*(?:[-*\u2022]\s*)?\d+[.)]\s", re.MULTILINE)
_MODAL = re.compile(r"\b(must|should|shall|may|can|will)\b", re.IGNORECASE)


def _outside_markers(answer: str, pattern: re.Pattern[str], skip_list_prefix: bool = False):
    """Matches of `pattern` that are not inside a citation marker (and, optionally, not a list number like '2.')."""
    protected = [m.span() for m in _MARKER.finditer(answer)]
    if skip_list_prefix:
        protected += [m.span() for m in _LIST_PREFIX.finditer(answer)]
    return [m for m in pattern.finditer(answer) if not any(a <= m.start() < b for a, b in protected)]


def _replace_spans(text: str, matches, make) -> str:
    out, last = [], 0
    for m in matches:
        out.append(text[last : m.start()])
        out.append(make(m))
        last = m.end()
    out.append(text[last:])
    return "".join(out)


def mutations(answer: str) -> dict[str, str | None]:
    """Corrupted copies of a real answer. None when the answer has nothing to corrupt that way.

    Each corruption touches every place it applies (outside the citation markers and list numbers), so that if the
    layer checks a corrupted sentence at all it has to notice:
      invented_claim     an extra sentence with a real citation marker that the passage does not support
      number_changed     every quantity in the answer changed to a different one
      negation_inserted  'not' inserted after every modal verb (must, should, may, can ...)
      entity_swapped     'EGW' replaced by 'air' (or 'air' by 'EGW'): a near miss that keeps the wording
    """
    marker = _first_marker(answer)
    out: dict[str, str | None] = {}
    out["invented_claim"] = (
        f"{answer}\nThe recommended tightening torque for this part is 45 Nm. {marker}" if marker else None
    )

    numbers = _outside_markers(answer, _NUMBER, skip_list_prefix=True)
    out["number_changed"] = (
        _replace_spans(answer, numbers, lambda m: str(int(float(m.group(0))) + 7)) if numbers and marker else None
    )

    modals = _outside_markers(answer, _MODAL)
    out["negation_inserted"] = (
        _replace_spans(answer, modals, lambda m: m.group(0) + " not") if modals and marker else None
    )

    egw = _outside_markers(answer, re.compile(r"\bEGW\b"))
    air = _outside_markers(answer, re.compile(r"\bair\b", re.IGNORECASE))
    if egw and marker:
        out["entity_swapped"] = _replace_spans(answer, egw, lambda m: "air")
    elif air and marker:
        out["entity_swapped"] = _replace_spans(answer, air, lambda m: "EGW")
    else:
        out["entity_swapped"] = None
    return out


def _detected(kind: str, original: dict, mutated: dict) -> bool:
    """Did the layer notice? Fewer claims verified than before, or a lower confidence that is no longer 'high'."""
    if kind == "invented_claim":
        return (
            mutated["claims_checked"] > original["claims_checked"]
            and mutated["claims_verified"] <= original["claims_verified"]
            and mutated["confidence_level"] != "high"
        )
    return mutated["claims_verified"] < original["claims_verified"] or mutated["confidence"] < original["confidence"]


CLAIM_KINDS = ("number_changed", "negation_inserted", "entity_swapped")


def _claim_level_detection(state: dict, status: str, explain, collect: dict[str, list[bool]]) -> dict[str, list[bool]]:
    """Corrupt each verified, cited claim of the answer on its own and see whether it is still verified."""
    from asa.components.explanation import _marked_sentences

    result: dict[str, list[bool]] = {}
    for sentence in _marked_sentences(str(state.get("final_answer") or "")):
        if not _MARKER.search(sentence):
            continue
        alone = explain({**state, "final_answer": sentence}, status)
        if not alone or alone["claims_checked"] != 1 or alone["claims_verified"] != 1:
            continue  # only claims the layer verified can be wrongly verified after a corruption
        for kind, corrupted in mutations(sentence).items():
            if kind not in CLAIM_KINDS or corrupted is None:
                continue
            after = explain({**state, "final_answer": corrupted}, status)
            caught = bool(after) and after["claims_verified"] == 0
            result.setdefault(kind, []).append(caught)
            collect.setdefault(kind, []).append(caught)
    return {kind: values for kind, values in result.items()}


def evaluate_live_records(records: list[dict], explain=None) -> dict:
    """`records`: one per gold question, with `state` (as returned by the pipeline) and `status`."""
    if explain is None:
        from asa.components.explanation import CONFIDENCE_LOW, MAX_CITATIONS, explain
        from asa.graph.state import CONFIDENCE_FLOOR

        bands, max_citations = (CONFIDENCE_LOW, CONFIDENCE_FLOOR), MAX_CITATIONS
    else:
        bands, max_citations = (0.5, 0.7), 8

    invariants: dict[str, list[bool]] = {}
    detection: dict[str, list[bool]] = {}
    claim_level: dict[str, list[bool]] = {}
    rows = []
    for record in records:
        state, status = record["state"], record["status"]
        explanation = explain(copy.copy(state), status)
        if explanation is None:
            rows.append({"id": record["id"], "explained": False})
            continue
        for name, ok in check_invariants(state, explanation, bands, max_citations).items():
            invariants.setdefault(name, []).append(ok)
        row = {
            "id": record["id"],
            "explained": True,
            "confidence": explanation["confidence"],
            "level": explanation["confidence_level"],
            "claims_checked": explanation["claims_checked"],
            "claims_verified": explanation["claims_verified"],
            "citations": len(explanation["citations"]),
            "detected": {},
        }
        for kind, corrupted in mutations(str(state.get("final_answer") or "")).items():
            if corrupted is None:
                continue
            mutated = explain({**state, "final_answer": corrupted}, status)
            caught = bool(mutated) and _detected(kind, explanation, mutated)
            row["detected"][kind] = caught
            detection.setdefault(kind, []).append(caught)
        row["claim_level"] = _claim_level_detection(state, status, explain, claim_level)
        rows.append(row)

    explained = [r for r in rows if r["explained"]]
    levels = {level: sum(1 for r in explained if r["level"] == level) for level in ("low", "medium", "high")}
    return {
        "questions": len(records),
        "explained": len(explained),
        "with_a_citation_rate": _rate(sum(1 for r in explained if r["citations"] > 0), len(explained)),
        "claims_verified_rate": _rate(
            sum(r["claims_verified"] for r in explained), sum(r["claims_checked"] for r in explained)
        ),
        "confidence_levels": levels,
        **{name: _rate(sum(values), len(values)) for name, values in invariants.items()},
        # Answer level: the whole answer is corrupted and the layer must notice something. A lower bound, because a
        # corruption can land in a sentence the layer never checked.
        "answer_level_detection": {
            kind: {"rate": _rate(sum(v), len(v)), "tested": len(v)} for kind, v in detection.items()
        },
        # Claim level: only claims the layer had VERIFIED are corrupted one at a time, and the corrupted claim must no
        # longer be verified. Every miss here is a claim that was wrong but still verified.
        "claim_level_detection": {
            kind: {"rate": _rate(sum(v), len(v)), "tested": len(v)} for kind, v in claim_level.items()
        },
        "rows": rows,
    }


def run_live() -> dict:
    """Run the gold set through the real pipeline (needs OPENAI_API_KEY and the ChromaDB index)."""
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    from app.services.graph_runner import run_graph, warm_up

    if not warm_up():
        raise SystemExit("the agents or the manual index could not be loaded")
    records = []
    for line in GOLD_SET.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        state = run_graph(item["question"], item["equipment_model"])
        status = "escalated" if state.get("escalated") else "halted" if state.get("safety_verdict") == "halt" else "ok"
        if state.get("request_status") == "CLARIFY":
            status = "clarification"
        records.append({"id": item["id"], "state": state, "status": status})
    _save_records(records)
    return evaluate_live_records(records)


def _save_records(records: list[dict]) -> None:
    """Keep the real answers and passages so the analysis can be repeated without calling the models again."""
    from dataclasses import asdict, is_dataclass

    def plain(value):
        if is_dataclass(value):
            return asdict(value)
        raise TypeError(type(value).__name__)

    keep = (
        "final_answer",
        "retrieved_chunks",
        "sufficiency",
        "retry_count",
        "current_step",
        "escalated",
        "safety_verdict",
        "root_causes",
        "troubleshooting_steps",
        "safety_claims",
        "request_status",
    )
    slim = [{"id": r["id"], "status": r["status"], "state": {k: r["state"].get(k) for k in keep}} for r in records]
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "explanation-live-records.json").write_text(
        json.dumps(slim, default=plain, indent=1, ensure_ascii=False), encoding="utf-8"
    )


def load_records(path: Path) -> list[dict]:
    """Records saved by a live run, with the passages turned back into Chunk objects."""
    from asa.graph.state import Chunk

    records = json.loads(path.read_text(encoding="utf-8"))
    for record in records:
        record["state"]["retrieved_chunks"] = [Chunk(**c) for c in record["state"].get("retrieved_chunks") or []]
    return records


def main() -> int:
    offline = evaluate(load_cases())
    print(json.dumps({k: v for k, v in offline.items() if k not in ("failures", "open_gaps")}, indent=2))
    for item in offline["failures"]:
        print(f"FAILED {item['id']}: {item['description']}: {item['problems']}")
    for gap in offline["open_gaps"]:
        print(f"known gap {gap['id']}: {gap['note']}")
    from asa.versioning import build_info

    report = {"versions": build_info(include_datasets=True), "offline": offline}
    if "--from-records" in sys.argv:
        live = evaluate_live_records(load_records(REPORT_DIR / "explanation-live-records.json"))
        print(json.dumps({k: v for k, v in live.items() if k != "rows"}, indent=2))
        report["live"] = live
    elif "--live" in sys.argv:
        live = run_live()
        print(json.dumps({k: v for k, v in live.items() if k != "rows"}, indent=2))
        report["live"] = live
    if "live" in report:
        REPORT_DIR.mkdir(exist_ok=True)
        (REPORT_DIR / "explanation-live.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
