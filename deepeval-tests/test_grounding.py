"""DeepEval grounding and answer-quality evaluation (PRD: grounding at least 90%).

For each gold question the real pipeline answers; a judge model then scores
  - faithfulness: every claim in the answer is supported by the retrieved manual passages,
  - no invented specifics: a G-Eval check that numbers, part names and steps come from the passages,
  - answer relevancy: the answer addresses the question.

How the numbers are made steadier:
  - the judge is gpt-4.1 at temperature 0 (stronger than the agents' own gpt-4o-mini),
  - every metric is measured REPEATS times (default 3) and the median is used,
  - the spread between the repeats is recorded as the measured judge noise.

Two separate rates, each with a 90% target (see conftest.py for the definitions):
  - grounding = faithfulness AND no invented specifics,
  - relevancy = answer relevancy.
The older single rate that needed all three metrics is still reported as strict_rate.

Questions that the Request agent sends back for clarification, or that end in an escalation (evidence judged
insufficient), have no grounded answer to score. They are skipped and counted, because the rate of those is a
finding in itself (over-asking, retrieval misses), reported in the summary.
"""

import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from conftest import (
    FAITHFULNESS_THRESHOLD,
    GROUNDING_TARGET,
    INVENTED_THRESHOLD,
    JUDGE_MODEL,
    RELEVANCY_TARGET,
    RELEVANCY_THRESHOLD,
    REPEATS,
    RESULTS_FILE,
    load_gold_set,
)
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

GOLD = load_gold_set()
SCORES: dict = {"cases": {}, "skipped": {}, "errors": {}, "judge_fallbacks": {}, "resumed": []}

# A transient judge error (timeout, rate limit, dropped connection) is retried with a growing pause before the
# question is called unscored.
JUDGE_ATTEMPTS = 3
RETRY_PAUSES = (15, 45, 90)  # seconds before the 2nd, 3rd ... attempt

# If the main judge keeps failing on one question (it can loop on a long answer), the question is judged once by this
# model instead. The substitution is recorded per question and in the summary, never hidden. Empty to disable.
FALLBACK_JUDGE = os.getenv("DEEPEVAL_FALLBACK_JUDGE", "gpt-4o")

# DEEPEVAL_RESUME=1 keeps the questions already scored in deepeval-results.json and only runs the missing ones, so an
# interrupted run does not have to pay for the whole set again. Resumed questions are listed in the summary.
RESUME = os.getenv("DEEPEVAL_RESUME") == "1"
if RESUME and RESULTS_FILE.exists():
    _previous = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    for _id, _case_result in _previous.get("cases", {}).items():
        SCORES["cases"][_id] = _case_result
        SCORES["resumed"].append(_id)
        if _case_result.get("judge") and _case_result["judge"] != JUDGE_MODEL:
            SCORES["judge_fallbacks"][_id] = _case_result["judge"]

FAITHFULNESS = "FaithfulnessMetric"
RELEVANCY = "AnswerRelevancyMetric"
INVENTED = "No invented specifics"
THRESHOLDS = {FAITHFULNESS: FAITHFULNESS_THRESHOLD, RELEVANCY: RELEVANCY_THRESHOLD, INVENTED: INVENTED_THRESHOLD}


def _case(item: dict, state: dict) -> LLMTestCase:
    chunks = state.get("retrieved_chunks") or []
    return LLMTestCase(
        input=item["question"],
        actual_output=state.get("final_answer") or "",
        retrieval_context=[chunk.text for chunk in chunks],
    )


def _gradable(item: dict, state: dict) -> LLMTestCase:
    """The test case, or a skip with the reason when there is no grounded answer to score."""
    if state.get("request_status") == "CLARIFY":
        SCORES["skipped"][item["id"]] = "clarification (the Request agent asked instead of answering)"
        pytest.skip(f"{item['id']}: clarification asked: {state.get('clarification_question')}")
    if state.get("escalated"):
        SCORES["skipped"][item["id"]] = f"escalated: {state.get('escalation_reason')}"
        pytest.skip(f"{item['id']}: escalated: {state.get('escalation_reason')}")
    case = _case(item, state)
    if not case.actual_output or not case.retrieval_context:
        SCORES["skipped"][item["id"]] = "no answer or no retrieved passages"
        pytest.skip(f"{item['id']}: no answer or no retrieved passages")
    return case


def _metrics(model: str):
    """Fresh metric objects: a metric keeps state from its last measurement, so each repeat gets its own."""
    return [
        FaithfulnessMetric(threshold=FAITHFULNESS_THRESHOLD, model=model, include_reason=True),
        AnswerRelevancyMetric(threshold=RELEVANCY_THRESHOLD, model=model, include_reason=True),
        GEval(
            name=INVENTED,
            criteria=(
                "Every specific value, part name, tool, warning and procedure step in the actual output must "
                "appear in, or follow directly from, the retrieval context (the manual passages). Penalise any "
                "number, torque, temperature, time, step or safety instruction that the context does not "
                "contain. General connecting words are fine."
            ),
            evaluation_params=[
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.RETRIEVAL_CONTEXT,
            ],
            threshold=INVENTED_THRESHOLD,
            model=model,
        ),
    ]


def _measure_once(case: LLMTestCase, model: str) -> dict[str, tuple[float, str]]:
    """One full set of measurements: metric name -> (score, the judge's reason).

    A judge error is retried with a growing pause. If it keeps failing the error is raised: an unscored question
    must never be counted as scored, nor quietly dropped from the denominator.
    """
    last_error: Exception | None = None
    for attempt in range(JUDGE_ATTEMPTS):
        try:
            out = {}
            for metric in _metrics(model):
                metric.measure(case)
                name = metric.name if isinstance(metric, GEval) else metric.__class__.__name__
                out[name] = (float(metric.score), str(getattr(metric, "reason", "") or ""))
            return out
        except Exception as error:  # noqa: BLE001 - any judge failure is retried, then reported
            last_error = error
            if attempt < JUDGE_ATTEMPTS - 1:
                time.sleep(RETRY_PAUSES[min(attempt, len(RETRY_PAUSES) - 1)])
    raise RuntimeError(f"the judge failed {JUDGE_ATTEMPTS} times: {type(last_error).__name__}: {str(last_error)[:160]}")


def _measure_with(case: LLMTestCase, model: str, repeats: int) -> dict:
    """Measure the case `repeats` times (concurrently) with one judge and summarise each metric by its median."""
    with ThreadPoolExecutor(max_workers=repeats) as pool:
        runs = list(pool.map(lambda _: _measure_once(case, model), range(repeats)))
    summary = {}
    for name in THRESHOLDS:
        values = [run[name][0] for run in runs]
        lowest = min(range(len(runs)), key=lambda i: runs[i][name][0])
        summary[name] = {
            "median": round(statistics.median(values), 3),
            "runs": [round(v, 3) for v in values],
            "spread": round(max(values) - min(values), 3),
            "reason_of_lowest": runs[lowest][name][1],
        }
    return summary


def measure(case: LLMTestCase) -> tuple[dict, str]:
    """Measure with the main judge; if it keeps failing, once with the fallback judge. Returns (summary, judge used)."""
    try:
        return _measure_with(case, JUDGE_MODEL, REPEATS), JUDGE_MODEL
    except RuntimeError as main_error:
        if not FALLBACK_JUDGE or FALLBACK_JUDGE == JUDGE_MODEL:
            raise
        try:
            return _measure_with(case, FALLBACK_JUDGE, 1), FALLBACK_JUDGE
        except RuntimeError as fallback_error:
            raise RuntimeError(f"{main_error}; fallback {FALLBACK_JUDGE}: {fallback_error}") from fallback_error


@pytest.mark.parametrize("item", GOLD, ids=[g["id"] for g in GOLD])
def test_answer_is_grounded(item, pipeline):
    """The answer agrees with the manual: median faithfulness and median no-invented-specifics both pass."""
    if item["id"] in SCORES["resumed"]:
        # Already scored in an earlier session (DEEPEVAL_RESUME=1): keep it, and still fail if it was not grounded.
        assert SCORES["cases"][item["id"]]["grounded"], "; ".join(SCORES["cases"][item["id"]]["failures"])
        return

    state = pipeline(item)
    case = _gradable(item, state)
    try:
        summary, judge = measure(case)
    except RuntimeError as error:
        SCORES["errors"][item["id"]] = str(error)
        pytest.fail(f"{item['id']} could not be scored: {error}")
    if judge != JUDGE_MODEL:
        SCORES["judge_fallbacks"][item["id"]] = judge

    scores = {name: summary[name]["median"] for name in summary}
    grounded = scores[FAITHFULNESS] >= FAITHFULNESS_THRESHOLD and scores[INVENTED] >= INVENTED_THRESHOLD
    relevant = scores[RELEVANCY] >= RELEVANCY_THRESHOLD
    failures = [
        f"{name} median {summary[name]['median']} (runs {summary[name]['runs']}) < {THRESHOLDS[name]}: "
        f"{summary[name]['reason_of_lowest']}"
        for name in (FAITHFULNESS, INVENTED, RELEVANCY)
        if summary[name]["median"] < THRESHOLDS[name]
    ]
    SCORES["cases"][item["id"]] = {
        "question": item["question"],
        "judge": judge,
        "repeats": REPEATS if judge == JUDGE_MODEL else 1,
        "scores": scores,
        "runs": {name: summary[name]["runs"] for name in summary},
        "spread": {name: summary[name]["spread"] for name in summary},
        "grounded": grounded,
        "relevant": relevant,
        "passed": grounded and relevant,  # the earlier combined rule, kept for comparison (strict_rate)
        "failures": failures,
    }
    assert grounded, "; ".join(f for f in failures if not f.startswith(RELEVANCY))


def _summary() -> dict:
    cases = SCORES["cases"]
    n = len(cases)
    spreads = [c["spread"][name] for c in cases.values() for name in THRESHOLDS]
    return {
        "judge_model": JUDGE_MODEL,
        "repeats": REPEATS,
        "judge_fallbacks": dict(SCORES["judge_fallbacks"]),  # question -> the other judge used for it
        "resumed": list(SCORES["resumed"]),  # questions kept from an earlier session of the same run
        "gradable": n,
        "judge_errors": len(SCORES["errors"]),
        "skipped": len(SCORES["skipped"]),
        "total_questions": len(GOLD),
        "answer_rate": round(n / len(GOLD), 3),
        "grounding_rate": round(sum(c["grounded"] for c in cases.values()) / n, 3),
        "grounding_target": GROUNDING_TARGET,
        "relevancy_rate": round(sum(c["relevant"] for c in cases.values()) / n, 3),
        "relevancy_target": RELEVANCY_TARGET,
        "strict_rate": round(sum(c["passed"] for c in cases.values()) / n, 3),
        "mean_scores": {
            name: round(statistics.mean(c["scores"][name] for c in cases.values()), 3) for name in THRESHOLDS
        },
        "mean_judge_spread": round(statistics.mean(spreads), 3),
        "max_judge_spread": round(max(spreads), 3),
        # Kept so the report and older tooling can still read the single rate and the target.
        "target": GROUNDING_TARGET,
    }


def test_zz_grounding_rate_meets_the_prd_target():
    """At least 90% of the gradable answers must be grounded (faithful and with no invented specifics)."""
    if not SCORES["cases"]:
        pytest.skip("no gradable answers were scored in this session")
    SCORES["summary"] = summary = _summary()
    assert not SCORES["errors"], (
        f"{len(SCORES['errors'])} answers could not be scored, so the rate is not valid: {SCORES['errors']}"
    )
    assert summary["grounding_rate"] >= GROUNDING_TARGET, (
        f"grounding rate {summary['grounding_rate']:.1%} is below the {GROUNDING_TARGET:.0%} target: {summary}"
    )


def test_zz_relevancy_rate_meets_its_target():
    """At least 90% of the gradable answers must address the question (reported separately from grounding)."""
    if not SCORES["cases"]:
        pytest.skip("no gradable answers were scored in this session")
    SCORES["summary"] = summary = _summary()
    assert not SCORES["errors"], f"{len(SCORES['errors'])} answers could not be scored, so the rate is not valid"
    assert summary["relevancy_rate"] >= RELEVANCY_TARGET, (
        f"relevancy rate {summary['relevancy_rate']:.1%} is below the {RELEVANCY_TARGET:.0%} target: {summary}"
    )
