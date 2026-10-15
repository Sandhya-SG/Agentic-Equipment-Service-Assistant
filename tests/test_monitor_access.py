"""Who may read the operational numbers: the monitoring endpoint is protected by the backend, not by the page."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
PARAMS = {"verify": "false"}


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in ("MONITOR_TOKEN", "ENVIRONMENT", "HOSTNAME"):
        monkeypatch.delenv(name, raising=False)


def test_in_development_with_no_token_the_endpoint_is_open_as_before():
    assert client.get("/api/monitoring", params=PARAMS).status_code == 200


@pytest.mark.parametrize("environment", ["kubernetes", "production", "staging", "anything-else"])
def test_in_a_deployed_environment_with_no_token_there_is_no_open_mode(monkeypatch, environment):
    monkeypatch.setenv("ENVIRONMENT", environment)
    response = client.get("/api/monitoring", params=PARAMS)
    assert response.status_code == 503 and "not configured" in response.json()["detail"]


@pytest.mark.parametrize("environment", [None, "development", "dev", "local", "test", "TEST"])
def test_the_open_environments_stay_open(monkeypatch, environment):
    if environment:
        monkeypatch.setenv("ENVIRONMENT", environment)
    assert client.get("/api/monitoring", params=PARAMS).status_code == 200


def test_with_a_token_set_the_right_token_is_needed_even_in_development(monkeypatch):
    monkeypatch.setenv("MONITOR_TOKEN", "s3cret-operator-token")
    assert client.get("/api/monitoring", params=PARAMS).status_code == 401
    assert client.get("/api/monitoring", params=PARAMS, headers={"X-Monitor-Token": "wrong"}).status_code == 401
    assert client.get("/api/monitoring", params=PARAMS, headers={"X-Monitor-Token": ""}).status_code == 401
    ok = client.get("/api/monitoring", params=PARAMS, headers={"X-Monitor-Token": "s3cret-operator-token"})
    assert ok.status_code == 200


def test_a_deployed_environment_with_the_token_works_and_without_it_is_refused(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "kubernetes")
    monkeypatch.setenv("MONITOR_TOKEN", "t" * 32)
    assert client.get("/api/monitoring", params=PARAMS).status_code == 401
    assert client.get("/api/monitoring", params=PARAMS, headers={"X-Monitor-Token": "t" * 32}).status_code == 200


def test_the_reviewer_token_does_not_open_the_monitoring_endpoint(monkeypatch):
    monkeypatch.setenv("MONITOR_TOKEN", "monitor-token")
    monkeypatch.setenv("REVIEWER_TOKEN", "reviewer-token")
    assert (
        client.get("/api/monitoring", params=PARAMS, headers={"X-Monitor-Token": "reviewer-token"}).status_code == 401
    )
    assert (
        client.get("/api/monitoring", params=PARAMS, headers={"X-Reviewer-Token": "reviewer-token"}).status_code == 401
    )


def test_a_refusal_never_echoes_the_token(monkeypatch):
    monkeypatch.setenv("MONITOR_TOKEN", "do-not-leak-this-value")
    body = client.get("/api/monitoring", params=PARAMS, headers={"X-Monitor-Token": "guess"}).text
    assert "do-not-leak-this-value" not in body and "guess" not in body


def test_the_response_names_the_pod_the_numbers_come_from(monkeypatch):
    monkeypatch.setenv("HOSTNAME", "backend-1")
    data = client.get("/api/monitoring", params=PARAMS).json()
    assert data["instance"] == "backend-1"
    assert {"summary", "alerts", "audit_integrity", "review", "spend", "instance"} <= set(data)


def test_the_other_endpoints_are_not_affected_by_the_monitoring_token(monkeypatch):
    monkeypatch.setenv("MONITOR_TOKEN", "x" * 20)
    assert client.get("/health").status_code == 200
    assert client.get("/api/equipment").status_code == 200
