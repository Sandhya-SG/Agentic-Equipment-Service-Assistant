"""The clarification evaluation's dataset and runner, checked with a stub model (no OpenAI calls).

The real measurement (`python eval/clarification_eval.py`) needs a key and costs a little credit, so it is not
run here. These tests only make sure the cases are well formed and the runner and its arithmetic are right,
so that a broken dataset line cannot silently distort the numbers.
"""

import importlib.util
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from asa.agents import request as request_agent

EVAL_FILE = Path(__file__).resolve().parents[2] / "eval" / "clarification_eval.py"
KNOWN_EQUIPMENT = {"thermal_station", "thermal_retrofit_1kw"}
CONTINUATION_KINDS = {"continuation_ready", "continuation_clarify"}

READY = "STATUS: READY\nQUESTION: NONE"
CLARIFY = "STATUS: CLARIFY\nQUESTION: What specific issue are you seeing?"


@pytest.fixture(scope="module")
def ce():
    spec = importlib.util.spec_from_file_location("clarification_eval", EVAL_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StubClient:
    """Answers every Request Agent call with a fixed decision and counts the calls."""

    def __init__(self, reply: str):
        self.reply = reply
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, *args, **kwargs):
        self.calls += 1
        message = SimpleNamespace(content=self.reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


# ---------------------------------------------------------------------------
# The dataset
# ---------------------------------------------------------------------------


def test_every_case_is_well_formed(ce):
    cases = ce.load_cases()
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids)), "case ids must be unique"
    for case in cases:
        assert case["kind"] in ce.EXPECTED, case["id"]
        assert case["equipment_model"] in KNOWN_EQUIPMENT, case["id"]
        assert case["query"].strip(), case["id"]
        assert case["label_confidence"] in {"firm", "judgement"}, case["id"]
        assert ("clarification" in case) == (case["kind"] in CONTINUATION_KINDS), case["id"]
        if "clarification" in case:
            assert case["clarification"].strip(), case["id"]


def test_every_kind_has_enough_cases_to_mean_something(ce):
    counts = Counter(case["kind"] for case in ce.load_cases())
    assert set(counts) == set(ce.EXPECTED)
    assert counts["ready"] >= 20
    assert counts["clarify"] >= 5
    assert counts["safety_ready"] >= 5
    assert counts["continuation_ready"] >= 5


def test_the_safety_critical_cases_include_the_prompts_own_examples(ce):
    queries = {case["query"] for case in ce.load_cases() if case["kind"] == "safety_ready"}
    assert "Can I bypass the safety interlock?" in queries
    assert "Can I open the panel while the equipment is powered?" in queries


# ---------------------------------------------------------------------------
# The state the agent receives
# ---------------------------------------------------------------------------


def test_a_fresh_question_has_no_resolved_query(ce):
    case = {"query": "What is the maximum operating temperature?", "equipment_model": "thermal_station"}
    state = ce.build_state(case)
    assert state["raw_query"] == "What is the maximum operating temperature?"
    assert state["equipment_model"] == "thermal_station"
    assert "resolved_query" not in state


def test_a_follow_up_combines_the_question_and_the_reply(ce):
    case = {"query": "It's not working.", "clarification": "The pump is noisy.", "equipment_model": "thermal_station"}
    state = ce.build_state(case)
    assert state["raw_query"] == "It's not working."
    assert "It's not working." in state["resolved_query"]
    assert "The pump is noisy." in state["resolved_query"]


# ---------------------------------------------------------------------------
# Running the cases with a stub model
# ---------------------------------------------------------------------------


def test_the_runner_reports_the_status_the_model_returned(ce, monkeypatch):
    monkeypatch.setattr(request_agent, "client", StubClient(CLARIFY))
    outcome = ce.run_case({"query": "Fix it.", "equipment_model": "thermal_station"})
    assert outcome == {"status": "CLARIFY", "question": "What specific issue are you seeing?"}


def test_a_model_that_always_says_ready_fails_only_the_clarify_cases(ce, monkeypatch):
    stub = StubClient(READY)
    monkeypatch.setattr(request_agent, "client", stub)
    cases = ce.load_cases()
    summary = ce.summarise(ce.evaluate(cases))
    assert stub.calls == len(cases)
    assert summary["pass_rate_by_kind"]["ready"] == 1.0
    assert summary["pass_rate_by_kind"]["safety_ready"] == 1.0
    assert summary["pass_rate_by_kind"]["continuation_ready"] == 1.0
    assert summary["pass_rate_by_kind"]["clarify"] == 0.0
    assert summary["pass_rate_by_kind"]["continuation_clarify"] == 0.0
    assert summary["clear_questions_turned_back"] == 0


def test_a_model_that_always_asks_turns_back_every_clear_question(ce, monkeypatch):
    monkeypatch.setattr(request_agent, "client", StubClient(CLARIFY))
    summary = ce.summarise(ce.evaluate(ce.load_cases()))
    assert summary["pass_rate_by_kind"]["clarify"] == 1.0
    assert summary["pass_rate_by_kind"]["ready"] == 0.0
    assert summary["turned_back_rate"] == 1.0
    assert summary["clear_questions_turned_back"] == summary["clear_questions_total"]


def test_repeat_runs_every_case_that_many_times(ce, monkeypatch):
    stub = StubClient(READY)
    monkeypatch.setattr(request_agent, "client", stub)
    results = ce.evaluate(ce.load_cases()[:5], repeat=3)
    assert len(results) == 15
    assert stub.calls == 15


def test_summarise_arithmetic(ce):
    results = [
        {"kind": "ready", "status": "READY"},
        {"kind": "ready", "status": "CLARIFY"},
        {"kind": "safety_ready", "status": "READY"},
        {"kind": "clarify", "status": "CLARIFY"},
    ]
    summary = ce.summarise(results)
    assert summary["pass_rate_by_kind"] == {"clarify": 1.0, "ready": 0.5, "safety_ready": 1.0}
    assert summary["clear_questions_turned_back"] == 1
    assert summary["clear_questions_total"] == 3
    assert summary["turned_back_rate"] == pytest.approx(0.333, abs=0.001)
