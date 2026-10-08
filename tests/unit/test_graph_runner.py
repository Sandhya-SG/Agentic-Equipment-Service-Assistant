import pytest

from app.services import graph_runner


class StreamingGraph:
    """A stand-in graph that streams node updates and full states like LangGraph does."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.config = None

    def stream(self, state, config=None, stream_mode=None):
        self.config = config
        yield from self.chunks


@pytest.fixture
def run(monkeypatch):
    def _run(chunks):
        graph = StreamingGraph(chunks)
        monkeypatch.setattr(graph_runner, "_graph", lambda: graph)
        monkeypatch.setattr(graph_runner, "graph_config", lambda: {})
        return graph_runner.run_graph("question", "thermal_station")

    return _run


CHUNKS = [
    ("values", {"raw_query": "question"}),
    ("updates", {"request": {"request_status": "READY"}}),
    ("values", {"request_status": "READY"}),
    ("updates", {"planning": {"current_step": "safety", "plan": ["safety"]}}),
    ("values", {"request_status": "READY", "current_step": "safety"}),
    (
        "updates",
        {"safety": {"safety_verdict": "halt", "escalated": False, "final_answer": "PRIVATE MANUAL TEXT " * 5}},
    ),
    ("values", {"current_step": "safety", "safety_verdict": "halt", "final_answer": "answer"}),
]


def test_the_final_state_is_the_last_full_state(run):
    state = run(CHUNKS)
    assert state["safety_verdict"] == "halt"
    assert state["final_answer"] == "answer"


def test_trace_lists_the_agents_in_the_order_they_ran(run):
    state = run(CHUNKS)
    assert [event.agent for event in state["trace"]] == ["request", "planning", "safety"]
    assert all(event.action == "completed" and event.timestamp for event in state["trace"])
    assert all(isinstance(event.detail["duration_ms"], int) for event in state["trace"])


def test_trace_keeps_only_short_categorical_details_never_text(run):
    state = run(CHUNKS)
    details = {event.agent: event.detail for event in state["trace"]}
    assert details["request"]["request_status"] == "READY"
    assert details["planning"]["current_step"] == "safety"
    assert details["safety"]["safety_verdict"] == "halt"
    assert "final_answer" not in details["safety"] and "plan" not in details["planning"]
    assert "PRIVATE MANUAL TEXT" not in str(state["trace"])


def test_a_graph_that_produces_no_state_is_an_error(run):
    with pytest.raises(RuntimeError):
        run([("updates", {"request": {"request_status": "READY"}})])


def test_a_node_that_returns_nothing_is_still_traced(run):
    state = run([("updates", {"request": None}), ("values", {"final_answer": "x"})])
    assert [event.agent for event in state["trace"]] == ["request"]
