"""The per-client chat limit and the spend figure in monitoring (cost controls that need no model call)."""

import io
import json
import urllib.error

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.main import app
from app.schemas.response import ChatResponse
from app.services import rate_limit

client = TestClient(app)


@pytest.fixture(autouse=True)
def fake_chat(monkeypatch):
    calls = []

    def handle(message, conversation_id, equipment_model, clarification_state):
        calls.append(message)
        return ChatResponse(response="ok", conversation_id=conversation_id, status="ok")

    monkeypatch.setattr(main, "handle_chat", handle)
    return calls


def post(headers=None):
    return client.post("/api/chat", json={"message": "How do I reset the unit?"}, headers=headers or {})


# --- the limiter ---


def test_requests_up_to_the_limit_pass_and_the_next_is_refused(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "3")
    assert [rate_limit.check("a", now=100.0 + i) for i in range(3)] == [0.0, 0.0, 0.0]
    wait = rate_limit.check("a", now=103.0)
    assert 56.0 < wait <= 57.0  # the oldest request leaves the window 60 s after it was made


def test_the_window_slides_so_the_client_can_send_again(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "2")
    rate_limit.check("a", now=0.0)
    rate_limit.check("a", now=1.0)
    assert rate_limit.check("a", now=2.0) > 0
    assert rate_limit.check("a", now=60.5) == 0.0  # the first request has left the window


def test_each_client_has_its_own_allowance(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1")
    assert rate_limit.check("a", now=0.0) == 0.0
    assert rate_limit.check("b", now=0.0) == 0.0
    assert rate_limit.check("a", now=1.0) > 0


def test_a_refused_request_does_not_extend_the_wait(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1")
    rate_limit.check("a", now=0.0)
    for second in range(1, 30):
        rate_limit.check("a", now=float(second))  # hammering while blocked
    assert rate_limit.check("a", now=60.5) == 0.0


@pytest.mark.parametrize("value, expected", [("0", 0), ("-5", 0), ("abc", 30), ("12", 12)])
def test_the_limit_setting_is_read_safely(monkeypatch, value, expected):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", value)
    assert rate_limit.limit_per_minute() == expected


def test_zero_turns_the_limit_off(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "0")
    assert all(rate_limit.check("a", now=float(i)) == 0.0 for i in range(500))


def test_memory_is_bounded_when_many_addresses_connect(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "5")
    monkeypatch.setattr(rate_limit, "MAX_TRACKED_CLIENTS", 50)
    for i in range(500):
        rate_limit.check(f"10.0.{i // 250}.{i % 250}", now=float(i))
    assert len(rate_limit._hits) <= 50


def test_the_forwarded_address_is_used_only_when_trusted(monkeypatch):
    assert rate_limit.client_key("10.0.0.5", "203.0.113.9") == "10.0.0.5"  # not trusted: could be faked
    monkeypatch.setenv("TRUST_FORWARDED_FOR", "true")
    assert rate_limit.client_key("10.0.0.5", "203.0.113.9, 10.0.0.1") == "203.0.113.9"
    assert rate_limit.client_key("10.0.0.5", "  ") == "10.0.0.5"
    assert rate_limit.client_key(None, None) == "unknown"


# --- the endpoint ---


def test_the_chat_endpoint_answers_429_with_retry_after_and_does_no_work(monkeypatch, fake_chat):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "2")
    assert post().status_code == 200 and post().status_code == 200
    refused = post()
    body = refused.json()
    assert refused.status_code == 429
    assert int(refused.headers["retry-after"]) >= 1
    assert body["status"] == "unavailable" and body["error"] == "rate_limited" and "Please wait" in body["response"]
    assert len(fake_chat) == 2  # the third request never reached the agents (no model call, no cost)


def test_different_forwarded_clients_do_not_share_an_allowance(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1")
    monkeypatch.setenv("TRUST_FORWARDED_FOR", "true")
    assert post({"X-Forwarded-For": "198.51.100.1"}).status_code == 200
    assert post({"X-Forwarded-For": "198.51.100.2"}).status_code == 200
    assert post({"X-Forwarded-For": "198.51.100.1"}).status_code == 429


def test_a_faked_forwarded_address_does_not_get_a_fresh_allowance_when_not_trusted(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1")
    assert post({"X-Forwarded-For": "198.51.100.1"}).status_code == 200
    assert post({"X-Forwarded-For": "198.51.100.2"}).status_code == 429  # same real peer


def test_the_other_endpoints_are_not_limited(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1")
    assert all(client.get("/health").status_code == 200 for _ in range(5))


def test_the_frontend_passes_the_clients_address_and_the_limit_reaches_the_page(monkeypatch):
    from frontend import app as frontend_app

    sent = {}

    def fake_urlopen(request, timeout=None):
        sent["xff"] = request.get_header("X-forwarded-for")
        body = json.dumps({"status": "unavailable", "response": "wait"}).encode()
        raise urllib.error.HTTPError("u", 429, "Too Many Requests", {}, io.BytesIO(body))

    monkeypatch.setattr(frontend_app.urllib_request, "urlopen", fake_urlopen)
    response = frontend_app.app.test_client().post(
        "/api/ask", json={"message": "hi"}, environ_overrides={"REMOTE_ADDR": "203.0.113.7"}
    )
    assert sent["xff"] == "203.0.113.7"
    assert response.status_code == 429 and response.get_json()["response"] == "wait"


# --- the spend figure in monitoring ---


def test_monitoring_shows_the_estimated_spend_and_the_cap(monkeypatch):
    from asa.components import llm

    llm.reset_spend()
    llm.add_spend(0.05, calls=3)
    monkeypatch.setenv("OPENAI_SPEND_CAP_USD", "0.10")
    spend = client.get("/api/monitoring", params={"verify": "false"}).json()["spend"]
    assert spend == {"estimated_usd": 0.05, "calls": 3, "cap_usd": 0.1, "cap_reached": False}
    llm.add_spend(0.06)
    assert client.get("/api/monitoring", params={"verify": "false"}).json()["spend"]["cap_reached"] is True


def test_monitoring_with_no_cap_says_so():
    spend = client.get("/api/monitoring", params={"verify": "false"}).json()["spend"]
    assert spend["cap_usd"] is None and spend["cap_reached"] is False
