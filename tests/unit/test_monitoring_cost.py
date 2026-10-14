"""Monitoring: tokens, model calls and an estimated cost per request, and the alert on a token surge."""

import json

import pytest

from asa.components import logging_sub as L


@pytest.fixture(autouse=True)
def temp_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "LOG_DIR", tmp_path)
    monkeypatch.setattr(L, "AUDIT_LOG", tmp_path / "audit.jsonl")
    monkeypatch.setattr(L, "FEEDBACK_LOG", tmp_path / "feedback.jsonl")
    monkeypatch.setattr(L, "METRICS_LOG", tmp_path / "metrics.jsonl")
    for key in ("ALERT_AVG_INPUT_TOKENS", "ALERT_MIN_RUNS"):
        monkeypatch.delenv(key, raising=False)


def run(i, usage=None, status="ok"):
    state = {"status": status, "confidence": 0.8}
    if usage is not None:
        state["llm_usage"] = usage
    L.emit_metrics(f"r{i}", state, 1.0)


def test_the_token_use_of_a_request_is_kept_in_its_metrics():
    usage = {"calls": 5, "input_tokens": 5600, "output_tokens": 150}
    run(1, usage)
    record = json.loads(L.METRICS_LOG.read_text(encoding="utf-8").splitlines()[0])
    assert record["llm_usage"] == usage


def test_the_summary_reports_averages_totals_and_an_estimated_cost():
    run(1, {"calls": 5, "input_tokens": 6000, "output_tokens": 200})
    run(2, {"calls": 1, "input_tokens": 700, "output_tokens": 20})
    run(3, {"calls": 9, "input_tokens": 8000, "output_tokens": 400})
    summary = L.monitoring_summary()
    assert summary["llm_runs_measured"] == 3
    assert summary["avg_model_calls"] == 5.0
    assert summary["avg_input_tokens"] == 4900 and summary["avg_output_tokens"] == 207
    assert summary["total_input_tokens"] == 14700 and summary["total_output_tokens"] == 620
    # 14,700 input at $0.15 and 620 output at $0.60 per million tokens
    assert summary["estimated_cost_usd"] == pytest.approx(0.0026, abs=0.0001)


def test_runs_without_usage_are_left_out_of_the_cost_figures():
    """A blocked request makes no model call; an older record has no usage."""
    run(1, {"calls": 5, "input_tokens": 6000, "output_tokens": 200})
    run(2, None, status="blocked")
    summary = L.monitoring_summary()
    assert summary["runs"] == 2 and summary["llm_runs_measured"] == 1
    assert summary["avg_input_tokens"] == 6000


def test_with_no_usage_at_all_the_cost_figures_are_empty_not_zero():
    run(1, None)
    summary = L.monitoring_summary()
    assert summary["avg_input_tokens"] is None and summary["estimated_cost_usd"] is None
    assert summary["total_input_tokens"] == 0


def test_a_damaged_usage_record_does_not_break_the_summary():
    run(1, {"calls": "x", "input_tokens": None, "output_tokens": 5})
    L._append(
        L.METRICS_LOG, {"type": "metrics", "run_id": "r9", "timestamp": "2026-10-14T00:00:00", "llm_usage": "nope"}
    )
    summary = L.monitoring_summary()
    assert summary["llm_runs_measured"] == 1


def test_a_token_surge_raises_an_alert_once_there_are_enough_runs():
    for i in range(12):
        run(i, {"calls": 20, "input_tokens": 30000, "output_tokens": 500})
    alerts = L.check_alerts(L.monitoring_summary())
    surge = [a for a in alerts if a["name"] == "avg_input_tokens"]
    assert surge and surge[0]["severity"] == "warning" and surge[0]["value"] == 30000


def test_normal_token_use_raises_no_alert():
    for i in range(12):
        run(i, {"calls": 5, "input_tokens": 6500, "output_tokens": 200})
    assert [a for a in L.check_alerts(L.monitoring_summary()) if a["name"] == "avg_input_tokens"] == []


def test_the_token_alert_threshold_can_be_changed(monkeypatch):
    monkeypatch.setenv("ALERT_AVG_INPUT_TOKENS", "5000")
    for i in range(12):
        run(i, {"calls": 5, "input_tokens": 6500, "output_tokens": 200})
    assert [a["name"] for a in L.check_alerts(L.monitoring_summary())] == ["avg_input_tokens"]


def test_too_few_runs_never_alert():
    for i in range(3):
        run(i, {"calls": 20, "input_tokens": 90000, "output_tokens": 500})
    assert L.check_alerts(L.monitoring_summary()) == []


def test_the_monitoring_endpoint_returns_the_cost_figures(monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import app

    run(1, {"calls": 5, "input_tokens": 6000, "output_tokens": 200})
    body = TestClient(app).get("/api/monitoring?verify=false").json()
    assert body["summary"]["avg_input_tokens"] == 6000 and body["summary"]["estimated_cost_usd"] is not None
