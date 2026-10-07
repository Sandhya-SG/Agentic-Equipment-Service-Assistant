import os

import pytest

from asa.components import logging_sub, tracing


def pytest_addoption(parser):
    parser.addoption(
        "--run-openai",
        action="store_true",
        default=False,
        help="Run live tests that call the OpenAI API (they cost money).",
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests marked `openai` unless the user opted in with --run-openai or RUN_OPENAI_TESTS=1."""
    if config.getoption("--run-openai") or os.getenv("RUN_OPENAI_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="live OpenAI test: run with --run-openai or RUN_OPENAI_TESTS=1")
    for item in items:
        if "openai" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def tracing_off(monkeypatch):
    """Tests never send traces, even when real Langfuse keys are in .env."""
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_CAPTURE_CONTENT"):
        monkeypatch.delenv(name, raising=False)
    tracing.reset_tracer()
    yield
    tracing.reset_tracer()


@pytest.fixture(autouse=True)
def isolated_logs(tmp_path, monkeypatch):
    """Keep audit, feedback and metrics logs out of the real logs/ folder."""
    monkeypatch.setattr(logging_sub, "LOG_DIR", tmp_path)
    monkeypatch.setattr(logging_sub, "AUDIT_LOG", tmp_path / "audit.jsonl")
    monkeypatch.setattr(logging_sub, "FEEDBACK_LOG", tmp_path / "feedback.jsonl")
    monkeypatch.setattr(logging_sub, "METRICS_LOG", tmp_path / "metrics.jsonl")
