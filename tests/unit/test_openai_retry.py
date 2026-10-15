"""OpenAI retries, proven against a real (local) HTTP server that imitates the API.

The shared client (`asa.components.llm`) gives the OpenAI SDK a retry budget (`OPENAI_MAX_RETRIES`, default 2). These
tests run the REAL SDK against a local server that answers with a scripted sequence of HTTP statuses, so they show the
behaviour and not just the settings:

  * a 429 (rate limit) is retried automatically and a success on the third attempt is returned;
  * when every attempt fails, the SDK gives up after 1 + max_retries attempts, and the application answers "unavailable"
    (HTTP 503) without leaking the error;
  * errors that must not be retried (400, 401) are not;
  * the retry count follows the environment variable.

No OpenAI call is made and nothing costs anything. The server sends `retry-after-ms` so the tests do not wait.
"""

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import openai
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import chat_service
from asa.agents.request import request_node
from asa.components import llm

COMPLETION = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "STATUS: READY\nQUESTION: NONE"},
        }
    ],
    "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19},
}
ERRORS = {
    400: ("invalid_request_error", "bad request"),
    401: ("invalid_api_key", "Incorrect API key provided: sk-secret-key-value"),
    429: ("rate_limit_exceeded", "Rate limit reached for gpt-4o-mini"),
    500: ("server_error", "The server had an error"),
    503: ("server_error", "The engine is currently overloaded"),
}


class FakeOpenAI:
    """A local stand-in for the chat completions endpoint that replays a script of HTTP statuses."""

    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.requests = []
        self._lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                with outer._lock:
                    outer.requests.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
                    index = min(len(outer.requests), len(outer.statuses)) - 1
                    status = outer.statuses[index]
                payload = (
                    COMPLETION
                    if status == 200
                    else {"error": {"type": ERRORS[status][0], "message": ERRORS[status][1]}}
                )
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                if status in (429, 500, 503):
                    self.send_header("retry-after-ms", "5")  # the SDK honours it, so the test does not wait
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def attempts(self):
        return len(self.requests)

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def serve(monkeypatch):
    """`serve([429, 429, 200])` starts the fake API and points the shared client at it."""
    servers = []

    def start(statuses, max_retries=None):
        fake = FakeOpenAI(statuses)
        servers.append(fake)
        monkeypatch.setenv("OPENAI_BASE_URL", fake.url)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("OPENAI_SPEND_CAP_USD", "")  # empty, so load_dotenv() cannot bring back a cap from .env
        for name in ("OPENAI_REQUEST_TOKEN_BUDGET", "OPENAI_MAX_OUTPUT_TOKENS"):
            monkeypatch.delenv(name, raising=False)
        if max_retries is None:
            monkeypatch.delenv("OPENAI_MAX_RETRIES", raising=False)
        else:
            monkeypatch.setenv("OPENAI_MAX_RETRIES", str(max_retries))
        llm.reset_client()
        return fake

    yield start
    for fake in servers:
        fake.stop()
    llm.reset_client()


def ask():
    return llm.client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])


# --- 1 to 3: a 429 is retried and the third attempt succeeds ---


def test_a_rate_limit_is_retried_and_the_third_attempt_succeeds(serve):
    fake = serve([429, 429, 200])
    response = ask()
    assert response.choices[0].message.content == "STATUS: READY\nQUESTION: NONE"
    assert fake.attempts == 3  # two 429s, then the success: the SDK retried by itself


def test_every_retry_sends_the_same_request_with_the_same_key(serve):
    fake = serve([429, 429, 200])
    ask()
    assert {r["path"] for r in fake.requests} == {"/v1/chat/completions"}
    assert {r["auth"] for r in fake.requests} == {"Bearer sk-test-key"}
    assert len({r["body"] for r in fake.requests}) == 1


def test_a_success_on_the_first_attempt_is_not_retried(serve):
    fake = serve([200])
    ask()
    assert fake.attempts == 1


def test_a_server_error_is_retried_too(serve):
    fake = serve([500, 503, 200])
    assert ask().choices[0].message.content
    assert fake.attempts == 3


def test_the_retries_of_one_call_count_as_one_call_in_the_request_budget(serve):
    fake = serve([429, 429, 200])
    with llm.request_budget(limit=0) as counter:
        ask()
    assert fake.attempts == 3 and counter["calls"] == 1
    assert counter["input"] == 12 and counter["output"] == 7  # only the successful attempt is billed


def test_the_retry_honours_the_servers_retry_after(serve):
    serve([429, 429, 200])
    started = time.monotonic()
    ask()
    assert time.monotonic() - started < 1.5  # the SDK's own backoff alone would be about 1.5 s or more


# --- 4: when every attempt fails, it gives up after 1 + max_retries attempts ---


def test_it_gives_up_after_the_retries_are_used_up(serve):
    fake = serve([429, 429, 429, 429, 429])
    with pytest.raises(openai.RateLimitError):
        ask()
    assert fake.attempts == 3  # one attempt and two retries (the default OPENAI_MAX_RETRIES=2), not more


@pytest.mark.parametrize("retries, attempts", [(0, 1), (1, 2), (4, 5)])
def test_the_retry_count_follows_the_environment_variable(serve, retries, attempts):
    fake = serve([429] * 10, max_retries=retries)
    with pytest.raises(openai.RateLimitError):
        ask()
    assert fake.attempts == attempts


def test_an_invalid_retry_setting_falls_back_to_the_default(serve, monkeypatch):
    fake = serve([429] * 10)
    monkeypatch.setenv("OPENAI_MAX_RETRIES", "not-a-number")
    llm.reset_client()
    with pytest.raises(openai.RateLimitError):
        ask()
    assert fake.attempts == 1 + llm.DEFAULT_MAX_RETRIES


@pytest.mark.parametrize("status, error", [(400, openai.BadRequestError), (401, openai.AuthenticationError)])
def test_errors_that_retrying_cannot_fix_are_not_retried(serve, status, error):
    fake = serve([status] * 5)
    with pytest.raises(error):
        ask()
    assert fake.attempts == 1


def test_a_refused_connection_is_retried_then_reported(monkeypatch):
    """Nothing is listening: the SDK retries the connection error and then raises."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]  # a port nobody listens on
    monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{port}/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    monkeypatch.setenv("OPENAI_MAX_RETRIES", "1")
    monkeypatch.setenv("OPENAI_SPEND_CAP_USD", "")  # empty, so load_dotenv() cannot bring back a cap from .env
    llm.reset_client()
    try:
        with pytest.raises(openai.APIConnectionError):
            ask()
    finally:
        llm.reset_client()


# --- the application: success after retries, and safe handling when they are exhausted ---

client = TestClient(app)


@pytest.fixture
def through_the_request_agent(monkeypatch):
    """Run the real Request agent (and so the real shared client) instead of the whole graph, no index needed."""
    monkeypatch.setattr(chat_service, "_explain", lambda state, status: None)
    monkeypatch.setattr(chat_service, "_is_standalone_request", lambda message, equipment_model: False)

    def run_graph(message, equipment_model, **kwargs):
        update = request_node({"raw_query": message, "equipment_model": equipment_model})
        return {**update, "current_step": "request", "final_answer": "The request is clear.", "retrieved_chunks": []}

    monkeypatch.setattr(chat_service, "run_graph", run_graph)


def chat():
    return client.post("/api/chat", json={"message": "What PPE is required?", "equipment_model": "thermal_station"})


def test_the_application_answers_normally_when_the_third_attempt_succeeds(serve, through_the_request_agent):
    fake = serve([429, 429, 200])
    response = chat()
    assert response.status_code == 200 and response.json()["status"] == "ok"
    assert fake.attempts == 3


def test_the_application_reports_unavailable_when_the_retries_are_exhausted(serve, through_the_request_agent):
    fake = serve([429] * 5)
    response = chat()
    body = response.json()
    assert response.status_code == 503 and body["status"] == "unavailable"
    assert fake.attempts == 3  # it did retry before giving up
    assert body["response"] == chat_service.UNAVAILABLE_MESSAGE


def test_the_unavailable_answer_leaks_nothing_about_the_failure(serve, through_the_request_agent):
    serve([401] * 5)  # the worst case: the error text names the key
    text = json.dumps(chat().json())
    for secret in ("sk-secret-key-value", "sk-test-key", "Incorrect API key", "rate limit", "RateLimitError", "429"):
        assert secret.lower() not in text.lower()


def test_a_failed_request_is_still_audited(serve, through_the_request_agent, tmp_path, monkeypatch):
    from asa.components import logging_sub

    monkeypatch.setattr(logging_sub, "LOG_DIR", tmp_path)
    monkeypatch.setattr(logging_sub, "AUDIT_LOG", tmp_path / "audit.jsonl")
    monkeypatch.setattr(logging_sub, "METRICS_LOG", tmp_path / "metrics.jsonl")
    serve([429] * 5)
    assert chat().status_code == 503
    records = [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["type"] for r in records][-1] == "run_end" and records[-1]["status"] == "unavailable"
    assert records[-1].get("error_type") == "RateLimitError"  # the kind of error, never its text


def test_the_next_request_works_once_the_provider_recovers(serve, through_the_request_agent):
    fake = serve([429, 429, 429, 200])  # the first request exhausts its retries; the second succeeds at once
    assert chat().status_code == 503
    assert chat().status_code == 200
    assert fake.attempts == 4
