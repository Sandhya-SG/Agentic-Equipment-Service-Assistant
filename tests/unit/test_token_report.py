"""The token and latency report: grouping by agent, totals and the before and after comparison (no model calls)."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def tr():
    spec = importlib.util.spec_from_file_location("token_report", ROOT / "eval" / "token_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CALLS = [
    {"call": "assess-request-readiness", "input": 700, "output": 7, "seconds": 0.5},
    {"call": "plan-specialist-route", "input": 750, "output": 30, "seconds": 1.0},
    {"call": "validate-safety-claims", "input": 2500, "output": 80, "seconds": 2.0},
    {"call": "validate-safety-claims", "input": 2500, "output": 70, "seconds": 2.5},
    {"call": "rag.some_function", "input": 100, "output": None, "seconds": None},
]


def test_calls_are_grouped_by_the_agent_that_made_them(tr):
    summary = tr.summarise(CALLS)
    assert summary["calls"] == 5 and summary["input"] == 6550 and summary["output"] == 187
    assert set(summary["per_agent"]) == {"request", "planning", "safety", "rag"}
    assert summary["per_agent"]["safety"] == {"calls": 2, "input": 5000, "output": 150, "seconds": 4.5}


def test_unknown_names_are_grouped_by_their_module(tr):
    assert tr.agent_of("diagnostic.some_function") == "diagnostic"
    assert tr.agent_of("generate-diagnosis") == "diagnostic"


def test_every_named_call_of_the_agents_has_an_agent(tr):
    """A new `llm_name("...")` in an agent must be added to the table, or its tokens land in a wrong group."""
    import re

    for agent in ("request", "planning", "rag", "safety", "diagnostic"):
        source = (ROOT / "src" / "asa" / "agents" / f"{agent}.py").read_text(encoding="utf-8")
        for name in re.findall(r'llm_name\("([^"]+)"\)', source):
            assert tr.agent_of(name) == agent, f"{name} in {agent}.py is not listed in AGENT_OF"


def test_the_totals_and_the_latency_summary(tr):
    rows = [
        {
            "calls": 5,
            "input": 1000,
            "output": 10,
            "seconds": 2.0,
            "per_agent": {"safety": {"calls": 5, "input": 1000, "output": 10, "seconds": 2.0}},
        },
        {
            "calls": 1,
            "input": 700,
            "output": 7,
            "seconds": 4.0,
            "per_agent": {"request": {"calls": 1, "input": 700, "output": 7, "seconds": 0.5}},
        },
        {
            "calls": 2,
            "input": 300,
            "output": 5,
            "seconds": 9.0,
            "per_agent": {"safety": {"calls": 2, "input": 300, "output": 5, "seconds": 1.0}},
        },
    ]
    total = tr.aggregate(rows)
    assert total["questions"] == 3 and total["calls"] == 8 and total["input"] == 2000
    assert total["seconds_median"] == 4.0 and total["seconds_max"] == 9.0
    assert total["per_agent"]["safety"]["input"] == 1300 and total["per_agent"]["request"]["calls"] == 1


def test_the_comparison_shows_the_change_per_agent(tr):
    before = {
        "total": {
            "calls": 10,
            "input": 1000,
            "output": 100,
            "seconds_median": 8.0,
            "per_agent": {"safety": {"input": 600}, "request": {"input": 400}},
        }
    }
    after = {
        "total": {
            "calls": 6,
            "input": 500,
            "output": 100,
            "seconds_median": 6.0,
            "per_agent": {"safety": {"input": 100}, "request": {"input": 400}},
        }
    }
    text = "\n".join(tr.compare(before, after))
    assert "-50%" in text and "-83%" in text and "-40%" in text and "+0%" in text


def test_the_comparison_survives_an_agent_that_appears_or_disappears(tr):
    before = {"total": {"calls": 0, "input": 0, "output": 0, "seconds_median": 0, "per_agent": {"a": {"input": 0}}}}
    after = {"total": {"calls": 1, "input": 5, "output": 1, "seconds_median": 1, "per_agent": {"b": {"input": 5}}}}
    assert "n/a" in "\n".join(tr.compare(before, after))


def test_the_question_set_covers_every_kind_of_request(tr):
    questions = tr.load_questions()
    assert len(questions) >= 10 and len({q["id"] for q in questions}) == len(questions)
    assert {"procedure", "lookup", "diagnostic", "safety", "clarification"} <= {q["kind"] for q in questions}
    assert {q["equipment_model"] for q in questions} == {"thermal_station", "thermal_retrofit_1kw"}


def test_a_saved_report_can_be_compared(tr, tmp_path, capsys):
    total = {"calls": 4, "input": 400, "output": 40, "seconds_median": 2.0, "per_agent": {"rag": {"input": 400}}}
    for name in ("a.json", "b.json"):
        (tmp_path / name).write_text(json.dumps({"total": total}), encoding="utf-8")
    assert tr.main(["--compare", str(tmp_path / "a.json"), str(tmp_path / "b.json")]) == 0
    assert "input" in capsys.readouterr().out
