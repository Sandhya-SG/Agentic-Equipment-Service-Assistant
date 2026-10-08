from contextlib import contextmanager

import pytest

import asa.components.tracing as tracing
from app.services import graph_runner
from asa.graph.state import Chunk


class FakeObservation:
    def __init__(self):
        self.entered = False
        self.exit_args = None

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc, tb):
        self.exit_args = (exc_type, exc)
        return False


class FakeTracer:
    def __init__(self):
        self.observation = FakeObservation()
        self.scores = []

    def start_as_current_observation(self, **kwargs):
        return self.observation

    def score_current_trace(self, **kwargs):
        self.scores.append(kwargs)


@pytest.fixture
def tracer(monkeypatch):
    fake = FakeTracer()
    monkeypatch.setattr(tracing, "get_tracer", lambda: fake)

    @contextmanager
    def no_attributes(**kwargs):
        yield

    monkeypatch.setattr("langfuse.propagate_attributes", no_attributes)
    return fake


def test_everything_is_a_noop_without_keys():
    assert tracing.tracing_enabled() is False
    assert tracing.get_tracer() is None
    assert tracing.graph_config() == {}
    assert tracing.instrument_openai() is False
    tracing.record_outcome(status="ok")
    tracing.flush_tracing()
    with tracing.trace_request(session_id="s"), tracing.tag_run("abc"):
        pass


def test_tracing_turns_on_only_with_both_keys(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    assert tracing.tracing_enabled() is False
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    assert tracing.tracing_enabled() is True


def test_mask_drops_all_text_by_default_but_keeps_numbers_and_shape():
    masked = tracing.mask(data={"question": "how do I fix it?", "pages": [1, 2], "nested": {"text": "secret"}})
    assert masked == {
        "question": tracing._OMITTED,
        "pages": [1, 2],
        "nested": {"text": tracing._OMITTED},
    }
    assert tracing.mask(data="some answer") == tracing._OMITTED
    assert tracing.mask(data=None) is None


def test_mask_redacts_and_truncates_when_capture_is_on(monkeypatch):
    monkeypatch.setenv("LANGFUSE_CAPTURE_CONTENT", "true")
    masked = tracing.mask(data={"q": "mail alice@example.com " + "x" * 5000})
    assert "alice@example.com" not in masked["q"]
    assert len(masked["q"]) <= tracing._MAX_CONTENT_CHARS


def test_mask_never_raises(monkeypatch):
    monkeypatch.setenv("LANGFUSE_CAPTURE_CONTENT", "true")
    monkeypatch.setattr(tracing, "redact", lambda value: 1 / 0)
    assert tracing.mask(data="anything") == tracing._OMITTED


def test_trace_request_opens_and_closes_one_observation(tracer):
    with tracing.trace_request(session_id="s", equipment_model="thermal_station"):
        assert tracer.observation.entered
    assert tracer.observation.exit_args == (None, None)


def test_trace_request_closes_the_trace_and_reraises_errors(tracer):
    with pytest.raises(ValueError):
        with tracing.trace_request(session_id="s"):
            raise ValueError("boom")
    assert tracer.observation.exit_args[0] is ValueError


def test_trace_request_still_runs_the_body_if_tracing_cannot_start(monkeypatch):
    class Broken:
        def start_as_current_observation(self, **kwargs):
            raise RuntimeError("langfuse down")

    monkeypatch.setattr(tracing, "get_tracer", lambda: Broken())
    ran = []
    with tracing.trace_request(session_id="s"):
        ran.append(True)
    assert ran == [True]


def test_record_outcome_writes_scores_not_text(tracer):
    tracing.record_outcome(status="escalated", hazard_count=2, source_count=3, escalated=True)
    scores = {s["name"]: s for s in tracer.scores}
    assert scores["status"]["value"] == "escalated"
    assert scores["escalated"]["value"] == 1
    assert scores["hazard_count"]["value"] == 2
    assert scores["source_count"]["value"] == 3


def test_record_outcome_errors_never_propagate(monkeypatch):
    class Broken:
        def score_current_trace(self, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(tracing, "get_tracer", lambda: Broken())
    tracing.record_outcome(status="ok")


def test_run_graph_passes_the_tracing_config_to_the_graph(monkeypatch):
    calls = []

    class FakeGraph:
        def stream(self, state, config=None, stream_mode=None):
            calls.append((state["raw_query"], config, stream_mode))
            yield "values", {"final_answer": "ok"}

    monkeypatch.setattr(graph_runner, "_graph", lambda: FakeGraph())
    monkeypatch.setattr(graph_runner, "graph_config", lambda: {"callbacks": ["handler"]})
    graph_runner.run_graph("question", "thermal_station")
    assert calls == [("question", {"callbacks": ["handler"]}, ["updates", "values"])]


def _chunk():
    return Chunk(
        chunk_id="c1",
        doc_id="d",
        section_id="1",
        revision="r",
        equipment_model="thermal_station",
        text="Warning Label Check: replace peeling labels",
    )


def test_mask_drops_objects_such_as_retrieved_chunks_by_default():
    masked = tracing.mask(data={"retrieved_chunks": [_chunk()], "retry_count": 1})
    assert "Warning Label" not in str(masked)
    assert masked["retry_count"] == 1


def test_mask_redacts_text_inside_objects_when_capture_is_on(monkeypatch):
    monkeypatch.setenv("LANGFUSE_CAPTURE_CONTENT", "true")
    chunk = _chunk()
    chunk.text = "contact bob@example.com about the label"
    masked = tracing.mask(data={"retrieved_chunks": [chunk]})
    assert "bob@example.com" not in str(masked)
    assert "label" in str(masked)
