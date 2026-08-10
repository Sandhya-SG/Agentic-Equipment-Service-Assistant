"""Tests for the Logging & Monitoring subsystem.

Run with: pytest tests/unit/test_logging.py -v
These use a temporary log directory so they don't touch real logs.
"""

import json

import pytest

from asa.graph.state import TraceEvent
from asa.components import logging_sub as L


@pytest.fixture(autouse=True)
def temp_logs(tmp_path, monkeypatch):
    """Redirect all log files into a temp dir for each test."""
    monkeypatch.setattr(L, "LOG_DIR", tmp_path)
    monkeypatch.setattr(L, "AUDIT_LOG", tmp_path / "audit.jsonl")
    monkeypatch.setattr(L, "FEEDBACK_LOG", tmp_path / "feedback.jsonl")
    monkeypatch.setattr(L, "METRICS_LOG", tmp_path / "metrics.jsonl")
    yield


def test_start_run_returns_id():
    run_id = L.start_run("test query")
    assert isinstance(run_id, str) and len(run_id) == 12


def test_full_run_lifecycle_writes_records():
    run_id = L.start_run("handler fault")
    trace = [TraceEvent("rag", "retrieve", "t", {"chunks": 3})]
    assert L.log_trace(run_id, trace)
    assert L.end_run(run_id, {"escalated": False, "confidence": 0.8})
    records = L.AUDIT_LOG.read_text().splitlines()
    types = [json.loads(r)["type"] for r in records]
    assert types == ["run_start", "trace", "run_end"]


def test_pii_is_redacted_before_write():
    run_id = L.start_run("contact tanya@example.com password=hunter2")
    content = L.AUDIT_LOG.read_text()
    assert "tanya@example.com" not in content
    assert "hunter2" not in content
    assert "[EMAIL]" in content


def test_metrics_emitted_and_returned():
    run_id = L.start_run("q")
    m = L.emit_metrics(run_id, {"escalated": True, "confidence": 0.0,
                                "safety_verdict": "halt", "retrieved_chunks": [1, 2]},
                       latency_s=3.5)
    assert m["latency_s"] == 3.5
    assert m["escalated"] is True
    assert m["num_chunks"] == 2


def test_feedback_logged():
    run_id = L.start_run("q")
    assert L.log_feedback(run_id, helpful=True, comment="great")
    rec = json.loads(L.FEEDBACK_LOG.read_text().splitlines()[0])
    assert rec["helpful"] is True and rec["comment"] == "great"


def test_monitoring_summary_aggregates():
    L.emit_metrics("r1", {"escalated": False, "confidence": 0.9,
                          "safety_verdict": "allow", "retrieved_chunks": [1]}, 2.0)
    L.emit_metrics("r2", {"escalated": True, "confidence": 0.0,
                          "safety_verdict": "halt", "retrieved_chunks": []}, 4.0)
    s = L.monitoring_summary()
    assert s["runs"] == 2
    assert s["escalation_rate"] == 0.5


def test_audit_integrity_intact_for_clean_log():
    run_id = L.start_run("q")
    L.end_run(run_id, {"escalated": False})
    result = L.verify_audit_integrity()
    assert result["intact"] is True


def test_audit_integrity_detects_tampering():
    run_id = L.start_run("original")
    L.end_run(run_id, {"escalated": False})
    # tamper with the first record
    lines = L.AUDIT_LOG.read_text().splitlines()
    rec = json.loads(lines[0])
    rec["query"] = "tampered"
    lines[0] = json.dumps(rec)
    L.AUDIT_LOG.write_text("\n".join(lines) + "\n")
    result = L.verify_audit_integrity()
    assert result["intact"] is False
    assert result["broken_at"] == 0


def test_logging_never_raises_on_bad_input():
    # Even with something odd, public calls must not raise.
    run_id = L.start_run("q")
    assert L.emit_metrics(run_id, {}, 1.0)          # empty state
    assert L.log_trace(run_id, [])                  # empty trace
