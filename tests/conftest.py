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


def pytest_configure(config):
    """Let the agent modules be imported with no credentials (as in CI).

    Several agents create their OpenAI client when the module is imported, which fails without a key. Every model call
    in these tests is a fake, so a placeholder key is enough. It is only set when there is no real key, and never when
    the live tests were requested (--run-openai or RUN_OPENAI_TESTS=1), which need the real one.
    """
    live = config.getoption("--run-openai") or os.getenv("RUN_OPENAI_TESTS") == "1"
    if not live and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "sk-placeholder-not-used-by-the-tests"


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


@pytest.fixture(autouse=True)
def cost_controls_off(tmp_path, monkeypatch):
    """No request limit, no spend cap and a throw-away ledger, unless a test turns them on itself."""
    from app.services import rate_limit

    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "0")
    monkeypatch.delenv("TRUST_FORWARDED_FOR", raising=False)
    # Empty, not deleted: the application calls load_dotenv(), which would bring a real token back from a developer's .env
    # (CI has no .env). A test that needs a token sets its own.
    monkeypatch.setenv("MONITOR_TOKEN", "")
    monkeypatch.setenv("REVIEWER_TOKEN", "")
    monkeypatch.setenv("VERSION_TOKEN", "")
    monkeypatch.setenv("OPENAI_SPEND_CAP_USD", "")  # empty, so load_dotenv() cannot bring back a cap from .env
    monkeypatch.setenv("OPENAI_SPEND_LEDGER", str(tmp_path / "openai_spend.json"))
    rate_limit.reset()
