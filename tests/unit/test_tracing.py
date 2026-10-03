import pytest

import asa.components.tracing as tracing


class FakeTracer:
    def __init__(self):
        self.spans = []
        self.generations = []

    def update_current_span(self, **fields):
        self.spans.append(fields)

    def update_current_generation(self, **fields):
        self.generations.append(fields)


@pytest.fixture
def tracer(monkeypatch):
    fake = FakeTracer()
    monkeypatch.setattr(tracing, "get_tracer", lambda: fake)
    return fake


def test_tracing_is_off_without_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    tracing.reset_tracer()
    assert tracing.get_tracer() is None
    tracing.annotate(anything=1)  # no-op, must not raise
    tracing.flush_tracing()


def test_traced_calls_the_function_when_tracing_is_off(monkeypatch):
    monkeypatch.setattr(tracing, "get_tracer", lambda: None)

    @tracing.traced("step")
    def add(a, b):
        return a + b

    assert add(1, 2) == 3


def test_annotate_redacts_sensitive_values(tracer):
    tracing.annotate(note="contact alice@example.com", api_key="x")
    assert "alice@example.com" not in str(tracer.spans[0])


def test_generation_records_usage_but_not_text_by_default(tracer, monkeypatch):
    monkeypatch.delenv("LANGFUSE_CAPTURE_CONTENT", raising=False)
    tracing.record_generation("m", 10, 4, prompt="secret question", reply="answer")
    fields = tracer.generations[0]
    assert fields["model"] == "m"
    assert fields["usage_details"] == {"input": 10, "output": 4}
    assert "input" not in fields and "output" not in fields


def test_generation_text_is_redacted_when_capture_is_on(tracer, monkeypatch):
    monkeypatch.setenv("LANGFUSE_CAPTURE_CONTENT", "true")
    tracing.record_generation("m", None, None, prompt="mail bob@example.com", reply="ok")
    fields = tracer.generations[0]
    assert "bob@example.com" not in fields["input"]
    assert "usage_details" not in fields


def test_tracing_errors_never_propagate(monkeypatch):
    class Broken:
        def update_current_span(self, **fields):
            raise RuntimeError("boom")

    monkeypatch.setattr(tracing, "get_tracer", lambda: Broken())
    tracing.annotate(x=1)
