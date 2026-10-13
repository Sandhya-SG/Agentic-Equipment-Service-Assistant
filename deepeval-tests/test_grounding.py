"""DeepEval grounding and answer-quality evaluation (PRD: grounding at least 90%).

For each gold question the real pipeline answers; the judge model then scores
  - faithfulness: every claim in the answer is supported by the retrieved manual passages,
  - answer relevancy: the answer addresses the question,
  - no invented specifics: a G-Eval check that numbers, part names and steps come from the passages.

Questions that the Request agent sends back for clarification, or that end in an escalation (evidence judged
insufficient), have no grounded answer to score. They are skipped and counted, because the rate of those is a
finding in itself (over-asking, retrieval misses), reported in the summary test.
"""

import pytest
from conftest import (
    FAITHFULNESS_THRESHOLD,
    GROUNDING_TARGET,
    JUDGE_MODEL,
    RELEVANCY_THRESHOLD,
    load_gold_set,
)
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

GOLD = load_gold_set()
SCORES: dict = {"cases": {}, "skipped": {}}


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


def _metrics():
    return [
        FaithfulnessMetric(threshold=FAITHFULNESS_THRESHOLD, model=JUDGE_MODEL, include_reason=True),
        AnswerRelevancyMetric(threshold=RELEVANCY_THRESHOLD, model=JUDGE_MODEL, include_reason=True),
        GEval(
            name="No invented specifics",
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
            threshold=0.7,
            model=JUDGE_MODEL,
        ),
    ]


@pytest.mark.parametrize("item", GOLD, ids=[g["id"] for g in GOLD])
def test_answer_is_grounded_and_relevant(item, pipeline):
    state = pipeline(item)
    case = _gradable(item, state)
    passed = True
    scores = {}
    failures = []
    for metric in _metrics():
        # Measured once: the judge is a model, so measuring again would give a different score than the one recorded.
        metric.measure(case)
        name = metric.name if isinstance(metric, GEval) else metric.__class__.__name__
        scores[name] = round(metric.score, 3)
        if not metric.is_successful():
            passed = False
            failures.append(f"{name} {metric.score:.2f} < {metric.threshold}: {metric.reason}")
    SCORES["cases"][item["id"]] = {"scores": scores, "passed": passed, "question": item["question"], "failures": failures}
    assert passed, "; ".join(failures)


def test_zz_grounding_rate_meets_the_prd_target():
    """At least 90% of the gradable answers must be grounded (every metric above its threshold)."""
    cases = SCORES["cases"]
    if not cases:
        pytest.skip("no gradable answers were scored in this session")
    rate = sum(1 for c in cases.values() if c["passed"]) / len(cases)
    SCORES["summary"] = {
        "gradable": len(cases),
        "skipped": len(SCORES["skipped"]),
        "total_questions": len(GOLD),
        "grounding_rate": round(rate, 3),
        "target": GROUNDING_TARGET,
        "answer_rate": round(len(cases) / len(GOLD), 3),
    }
    assert rate >= GROUNDING_TARGET, f"grounding rate {rate:.0%} is below the {GROUNDING_TARGET:.0%} target: {SCORES['summary']}"
