"""Tests for the Logging & Monitoring subsystem.

Run with: pytest tests/unit/test_logging.py -v
These use a temporary log directory so they don't touch real logs.
"""

import json

import pytest

from asa.components import logging_sub as L
from asa.graph.state import TraceEvent


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
    L.start_run("contact tanya@example.com password=hunter2")
    content = L.AUDIT_LOG.read_text()
    assert "tanya@example.com" not in content
    assert "hunter2" not in content
    assert "[EMAIL]" in content


def test_metrics_emitted_and_returned():
    run_id = L.start_run("q")
    m = L.emit_metrics(
        run_id,
        {"escalated": True, "confidence": 0.0, "safety_verdict": "halt", "retrieved_chunks": [1, 2]},
        latency_s=3.5,
    )
    assert m["latency_s"] == 3.5
    assert m["escalated"] is True
    assert m["num_chunks"] == 2


def test_feedback_logged():
    run_id = L.start_run("q")
    assert L.log_feedback(run_id, helpful=True, comment="great")
    rec = json.loads(L.FEEDBACK_LOG.read_text().splitlines()[0])
    assert rec["helpful"] is True and rec["comment"] == "great"


def test_monitoring_summary_aggregates():
    L.emit_metrics(
        "r1", {"escalated": False, "confidence": 0.9, "safety_verdict": "allow", "retrieved_chunks": [1]}, 2.0
    )
    L.emit_metrics("r2", {"escalated": True, "confidence": 0.0, "safety_verdict": "halt", "retrieved_chunks": []}, 4.0)
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
    assert L.emit_metrics(run_id, {}, 1.0)  # empty state
    assert L.log_trace(run_id, [])  # empty trace


# --------------------------------------------------------------------------- #
# Concurrency and multiple replicas                                           #
# --------------------------------------------------------------------------- #


def test_concurrent_writes_keep_every_record_and_the_chain_intact():
    """FastAPI runs the chat route in a thread pool, so requests write the audit log at once."""
    import threading

    def work():
        for i in range(25):
            L.start_run(f"query {i}")

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    result = L.verify_audit_integrity()
    assert result["intact"] is True
    assert result["checked"] == 200
    assert len(L.AUDIT_LOG.read_text().splitlines()) == 200


def test_append_does_not_reread_the_whole_log_every_time(monkeypatch):
    calls = []
    real = L._last_hash
    monkeypatch.setattr(L, "_last_hash", lambda path: calls.append(path) or real(path))
    for i in range(30):
        L.start_run(f"query {i}")
    assert len(calls) == 1  # only the first write reads the file
    assert L.verify_audit_integrity()["intact"] is True


def test_append_notices_when_the_file_was_changed_outside_the_process():
    L.start_run("first")
    L.start_run("second")
    # Someone truncates the log to its first record; the next write must chain to that record.
    first_line = L.AUDIT_LOG.read_text().splitlines()[0]
    L.AUDIT_LOG.write_text(first_line + "\n")
    L.start_run("third")
    assert L.verify_audit_integrity()["intact"] is True


def test_each_replica_writes_its_own_chain_file(monkeypatch):
    monkeypatch.setenv("AUDIT_LOG_PER_INSTANCE", "true")
    monkeypatch.setenv("HOSTNAME", "pod-a")
    L.start_run("from a 1")
    L.start_run("from a 2")
    monkeypatch.setenv("HOSTNAME", "pod-b")
    L.start_run("from b 1")

    names = sorted(p.name for p in L.LOG_DIR.glob("audit*.jsonl"))
    assert names == ["audit-pod-a.jsonl", "audit-pod-b.jsonl"]
    result = L.verify_audit_integrity()
    assert result["intact"] is True
    assert result["checked"] == 3
    assert result["files"] == 2


def test_tampering_is_traced_to_the_replica_file_it_happened_in(monkeypatch):
    monkeypatch.setenv("AUDIT_LOG_PER_INSTANCE", "true")
    monkeypatch.setenv("HOSTNAME", "pod-a")
    L.start_run("a")
    monkeypatch.setenv("HOSTNAME", "pod-b")
    L.start_run("b 1")
    L.start_run("b 2")

    target = L.LOG_DIR / "audit-pod-b.jsonl"
    lines = target.read_text().splitlines()
    record = json.loads(lines[0])
    record["query"] = "tampered"
    lines[0] = json.dumps(record)
    target.write_text("\n".join(lines) + "\n")

    result = L.verify_audit_integrity()
    assert result["intact"] is False
    assert result["broken_file"] == "audit-pod-b.jsonl"
    assert result["broken_at"] == 0


def test_monitoring_summary_reads_the_metrics_of_every_replica(monkeypatch):
    monkeypatch.setenv("AUDIT_LOG_PER_INSTANCE", "true")
    state = {"escalated": False, "safety_verdict": "allow", "confidence": 0.8, "retrieved_chunks": [], "retry_count": 0}
    monkeypatch.setenv("HOSTNAME", "pod-a")
    L.emit_metrics("run-a", state, 1.0)
    monkeypatch.setenv("HOSTNAME", "pod-b")
    L.emit_metrics("run-b", state, 2.0)
    assert L.monitoring_summary()["runs"] == 2


# --------------------------------------------------------------------------- #
# Provenance, blocked requests, monitoring rates and alerts                   #
# --------------------------------------------------------------------------- #


def _metrics(
    status="ok", confidence=0.0, latency=1.0, escalated=False, explained=False, sources=1, equipment="thermal_station"
):
    run_id = L.start_run("q")
    L.emit_metrics(
        run_id,
        {
            "status": status,
            "confidence": confidence,
            "escalated": escalated,
            "explained": explained,
            "source_count": sources,
            "equipment_model": equipment,
        },
        latency,
    )
    return run_id


def test_log_blocked_records_categories_and_length_but_not_the_text():
    run_id = L.log_blocked("sess-1", ["prompt_leak", "instruction_override"], 61)
    records = [r for r in L._read(L.AUDIT_LOG) if r["run_id"] == run_id]
    assert [r["type"] for r in records] == ["run_start", "run_end"]
    end = records[1]
    assert end["status"] == "blocked"
    assert end["blocked_categories"] == ["instruction_override", "prompt_leak"]
    assert end["input_length"] == 61
    assert records[0]["query"] == "[withheld: blocked input]"
    assert L.verify_audit_integrity()["intact"] is True


def test_end_run_records_the_full_provenance_fields():
    run_id = L.start_run("q")
    L.end_run(
        run_id,
        {
            "final_answer": "answer",
            "status": "halted",
            "specialist": "safety",
            "equipment_model": "thermal_station",
            "source_refs": ["a.pdf, p.1, Safety"],
            "hazard_categories": ["high_voltage"],
            "ppe_required": ["insulated gloves"],
            "requires_human_review": True,
            "escalation_reason": "x" * 500,
            "confidence_level": "high",
            "explained": True,
            "answer_sha256": "abc",
            "trace_summary": ["request", "planning", "safety"],
        },
    )
    record = [r for r in L._read(L.AUDIT_LOG) if r["type"] == "run_end"][0]
    assert record["status"] == "halted" and record["specialist"] == "safety"
    assert record["source_refs"] == ["a.pdf, p.1, Safety"]
    assert record["hazard_categories"] == ["high_voltage"]
    assert record["requires_human_review"] is True and record["explained"] is True
    assert record["trace_summary"] == ["request", "planning", "safety"]
    assert len(record["escalation_reason"]) == 200


def test_run_exists_finds_real_runs_only():
    run_id = L.start_run("q")
    assert L.run_exists(run_id) is True
    assert L.run_exists("0" * 12) is False


def test_confidence_levels_match_the_explanation_spec():
    assert L.confidence_level(0.2) == "low"
    assert L.confidence_level(0.5) == "medium"
    assert L.confidence_level(0.69) == "medium"
    assert L.confidence_level(0.7) == "high"


def test_summary_reports_status_rates_and_excludes_blocked_runs_from_latency():
    _metrics("ok", latency=2.0)
    _metrics("ok", latency=4.0)
    _metrics("blocked", latency=0.0)
    _metrics("unavailable", latency=1.0)
    summary = L.monitoring_summary()
    assert summary["runs"] == 4
    assert summary["status_counts"] == {"ok": 2, "blocked": 1, "unavailable": 1}
    assert summary["blocked_rate"] == 0.25 and summary["unavailable_rate"] == 0.25
    assert summary["p50_latency_s"] in (2.0, 4.0)  # the 0.0 of the blocked run is not counted
    assert summary["by_equipment"] == {"thermal_station": 4}


def test_summary_limit_keeps_only_the_most_recent_runs():
    for _ in range(5):
        _metrics("ok")
    for _ in range(3):
        _metrics("unavailable")
    assert L.monitoring_summary(limit=3)["status_counts"] == {"unavailable": 3}


def test_low_confidence_rate_counts_only_scored_runs():
    _metrics("ok", confidence=0.9)
    _metrics("ok", confidence=0.4)
    _metrics("ok", confidence=0.0)  # no explanation yet: not scored
    summary = L.monitoring_summary()
    assert summary["confidence_scored_runs"] == 2
    assert summary["low_confidence_rate"] == 0.5


def test_low_confidence_rate_is_none_when_nothing_is_scored():
    _metrics("ok")
    assert L.monitoring_summary()["low_confidence_rate"] is None


def test_explainability_coverage_is_the_share_of_answers_with_an_explanation_and_sources():
    _metrics("ok", explained=True, sources=2)
    _metrics("ok", explained=True, sources=0)  # explained but no source: not covered
    _metrics("ok", explained=False, sources=3)
    _metrics("blocked")  # not an answer
    assert L.monitoring_summary()["explainability_coverage"] == round(1 / 3, 3)


def test_helpfulness_is_reported_by_confidence_level():
    high = _metrics("ok", confidence=0.9)
    low = _metrics("ok", confidence=0.3)
    L.log_feedback(high, True)
    L.log_feedback(low, False)
    L.log_feedback(low, False)
    result = L.monitoring_summary()["helpful_rate_by_confidence"]
    assert result == {"high": 1.0, "low": 0.0}


def test_no_alerts_below_the_minimum_sample():
    for _ in range(5):
        _metrics("unavailable")
    assert L.check_alerts(L.monitoring_summary()) == []


def test_alerts_fire_for_error_spikes_and_blocked_anomalies():
    for _ in range(6):
        _metrics("ok")
    for _ in range(3):
        _metrics("unavailable")  # 3 of 13 = 0.231, above the 0.20 threshold
    for _ in range(4):
        _metrics("blocked")  # 4 of 13 = 0.308, above the 0.30 threshold
    alerts = {a["name"]: a for a in L.check_alerts(L.monitoring_summary())}
    assert alerts["unavailable_rate"]["severity"] == "critical"
    assert alerts["blocked_rate"]["severity"] == "warning"
    assert alerts["unavailable_rate"]["value"] == round(3 / 13, 3)
    assert alerts["unavailable_rate"]["threshold"] == 0.2


def test_alert_thresholds_can_be_overridden_by_environment_and_by_argument(monkeypatch):
    for _ in range(10):
        _metrics("ok")
    for _ in range(2):
        _metrics("blocked")
    summary = L.monitoring_summary()  # blocked rate 2/12 = 0.167
    assert L.check_alerts(summary) == []
    monkeypatch.setenv("ALERT_BLOCKED_RATE", "0.1")
    assert [a["name"] for a in L.check_alerts(summary)] == ["blocked_rate"]
    assert L.check_alerts(summary, {"blocked_rate": 0.5}) == []
    monkeypatch.setenv("ALERT_BLOCKED_RATE", "not-a-number")  # ignored, default 0.3 applies
    assert L.check_alerts(summary) == []
