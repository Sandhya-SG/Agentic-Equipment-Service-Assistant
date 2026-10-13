from contextlib import contextmanager

import pytest

import asa.components.tracing as tracing
from app.services import graph_runner
from asa.graph.state import Chunk


class FakeObservation:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.entered = False
        self.exit_args = None
        self.updates = []

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc, tb):
        self.exit_args = (exc_type, exc)
        return False

    def update(self, **fields):
        self.updates.append(fields)


class FakeTracer:
    def __init__(self):
        self.observation = FakeObservation()
        self.started = []
        self.scores = []

    def start_as_current_observation(self, **kwargs):
        # The first observation is `self.observation`, so existing tests can inspect the root.
        observation = self.observation if not self.started else FakeObservation()
        observation.kwargs = kwargs
        self.started.append(observation)
        return observation

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


def test_mask_keeps_short_application_identifiers_by_default():
    masked = tracing.mask(
        data={
            "status": "ok",
            "specialist": "agentic_rag",
            "categories": ["override", "exfiltration"],
            "chunks": [{"chunk_id": "c1", "source_file": "aem.pdf", "section_title": "Fan check", "page": 4}],
            "response": "Inspect the fan",
        }
    )
    assert masked == {
        "status": "ok",
        "specialist": "agentic_rag",
        "categories": ["override", "exfiltration"],
        "chunks": [{"chunk_id": "c1", "source_file": "aem.pdf", "section_title": tracing._OMITTED, "page": 4}],
        "response": tracing._OMITTED,
    }


def test_mask_drops_long_or_nested_text_even_under_an_identifier_key():
    masked = tracing.mask(data={"status": "x" * 200, "decision": {"reason": "free text"}})
    assert masked == {"status": tracing._OMITTED, "decision": {"reason": tracing._OMITTED}}


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


def test_llm_name_is_empty_until_openai_is_instrumented(monkeypatch):
    # A plain OpenAI client rejects `name=`, so it is only passed once Langfuse patched the SDK.
    assert tracing.llm_name("assess-retrieval") == {}
    monkeypatch.setattr(tracing, "_openai_instrumented", True)
    assert tracing.llm_name("assess-retrieval") == {"name": "assess-retrieval"}


def test_observation_is_a_noop_handle_without_keys():
    with tracing.observation("detect-prompt-injection", as_type="guardrail") as step:
        step.update(output={"flagged": False})


def test_observation_starts_a_typed_observation_and_records_output(tracer):
    with tracing.observation("redact-pii", as_type="guardrail", input={"n": 1}) as step:
        step.update(output={"redacted": True})
    assert tracer.observation.kwargs == {
        "name": "redact-pii",
        "as_type": "guardrail",
        "input": {"n": 1},
        "metadata": None,
    }
    assert tracer.observation.updates == [{"output": {"redacted": True}}]
    assert tracer.observation.exit_args == (None, None)


def test_observation_marks_errors_and_reraises(tracer):
    with pytest.raises(ValueError):
        with tracing.observation("check-hazard-rules", as_type="guardrail"):
            raise ValueError("boom")
    assert {"level": "ERROR", "status_message": "ValueError"} in tracer.observation.updates
    assert tracer.observation.exit_args[0] is ValueError


def test_traced_runs_the_function_untouched_without_keys():
    @tracing.traced("retrieve-manual-evidence", as_type="retriever", inputs=("query",))
    def retrieve(query, k=5):
        return [query, k]

    assert retrieve("fan noise", k=2) == ["fan noise", 2]


def test_traced_records_only_the_named_inputs_and_the_summarised_output(tracer):
    class Agent:
        @tracing.traced(
            "retrieve-manual-evidence",
            as_type="retriever",
            inputs=("query", "k"),
            output=len,
            metadata=lambda result: {"chunk_count": len(result)},
        )
        def retrieve(self, query, api_key="secret", k=5):
            return ["chunk-1", "chunk-2"]

    assert Agent().retrieve("fan noise", k=2) == ["chunk-1", "chunk-2"]
    assert tracer.observation.kwargs["as_type"] == "retriever"
    assert tracer.observation.kwargs["input"] == {"query": "fan noise", "k": 2}
    assert tracer.observation.updates == [{"output": 2, "metadata": {"chunk_count": 2}}]


def test_trace_request_yields_the_root_for_input_and_output(tracer):
    with tracing.trace_request(session_id="s", equipment_model="thermal_station") as root:
        root.update(input={"message": "q"})
        root.update(output={"status": "ok"})
    assert tracer.observation.kwargs == {"name": "chat-request", "as_type": "span"}
    assert tracer.observation.updates == [{"input": {"message": "q"}}, {"output": {"status": "ok"}}]


def test_record_outcome_adds_the_specialist_as_a_categorical_score(tracer):
    tracing.record_outcome(status="ok", specialist="agentic_rag")
    scores = {s["name"]: s for s in tracer.scores}
    assert scores["specialist"] == {"name": "specialist", "value": "agentic_rag", "data_type": "CATEGORICAL"}


def test_mask_redacts_text_inside_objects_when_capture_is_on(monkeypatch):
    monkeypatch.setenv("LANGFUSE_CAPTURE_CONTENT", "true")
    chunk = _chunk()
    chunk.text = "contact bob@example.com about the label"
    masked = tracing.mask(data={"retrieved_chunks": [chunk]})
    assert "bob@example.com" not in str(masked)
    assert "label" in str(masked)
