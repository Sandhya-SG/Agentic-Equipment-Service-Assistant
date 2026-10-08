"""Logging & Monitoring subsystem.

Provides the traceability, auditability, and feedback loop the proposal requires.
Three responsibilities:

  1. Audit trail   -- an append-only, tamper-evident record of every run and the
                      agent events within it (from state["trace"]).
  2. Feedback      -- captures user helpful/not-helpful feedback against a run.
  3. Metrics       -- emits operational metrics (latency, confidence, escalation
                      rate) for monitoring and the evaluation report.

Design decisions:
  * Logging is NON-BLOCKING in spirit: every public function is wrapped so a
    logging failure never breaks the user-facing pipeline. The agent path must
    never crash because the audit log had a problem.
  * The audit log is APPEND-ONLY and tamper-evident: each record carries a hash
    chained to the previous record, so after-the-fact edits are detectable. This
    is what makes the trail trustworthy for the 'accountability' guardrail.
  * PII/secret REDACTION runs before anything is written.
  * Storage is JSON Lines on disk here (simple, inspectable, good enough for the
    prototype). The interface is small, so a swap to a database later is trivial.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import re
import socket
import threading
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from asa.graph.state import TraceEvent

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

LOG_DIR = Path("logs")
AUDIT_LOG = LOG_DIR / "audit.jsonl"  # append-only run + event records
FEEDBACK_LOG = LOG_DIR / "feedback.jsonl"  # user feedback records
METRICS_LOG = LOG_DIR / "metrics.jsonl"  # per-run operational metrics


# --------------------------------------------------------------------------- #
# Redaction — runs before any write                                           #
# --------------------------------------------------------------------------- #

_REDACTION_PATTERNS = [
    (re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b"), "[EMAIL]"),
    (re.compile(r"\b(?:\d[ -]*?){13,16}\b"), "[CARD]"),  # card-like digit runs
    (re.compile(r"(?i)\b(api[_-]?key|token|password|secret)\s*[:=]\s*\S+"), r"\1=[REDACTED]"),
]


def _redact(value: Any) -> Any:
    """Recursively redact strings inside dicts/lists before persisting."""
    if isinstance(value, str):
        out = value
        for pat, repl in _REDACTION_PATTERNS:
            out = pat.sub(repl, out)
        return out
    if isinstance(value, dict):
        return {k: _redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


def redact(value: Any) -> Any:
    """Public wrapper so other components (e.g. tracing) reuse the same redaction."""
    return _redact(value)


# --------------------------------------------------------------------------- #
# Low-level append (tamper-evident, non-blocking)                             #
# --------------------------------------------------------------------------- #


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _last_hash(path: Path) -> str:
    """Return the hash of the last record in an append-only log ('' if empty)."""
    if not path.exists():
        return ""
    try:
        last_line = ""
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last_line = line
        if not last_line:
            return ""
        return json.loads(last_line).get("_hash", "")
    except Exception:
        return ""


# Appends are serialised, and the last hash is cached per file, so concurrent requests
# (FastAPI runs the chat route in a thread pool) cannot both chain to the same previous
# record, and each record no longer re-reads the whole log.
_APPEND_LOCK = threading.Lock()
_LAST_HASH: dict[str, tuple[str, int]] = {}  # path -> (last hash, file size when cached)


def _per_instance() -> bool:
    return os.getenv("AUDIT_LOG_PER_INSTANCE", "false").lower() == "true"


def _instance_path(path: Path) -> Path:
    """With AUDIT_LOG_PER_INSTANCE=true each replica (pod) writes its own chain file, because
    several processes appending to one hash chain would corrupt it."""
    if not _per_instance():
        return path
    instance = os.getenv("HOSTNAME") or socket.gethostname()
    return path.with_name(f"{path.stem}-{instance}{path.suffix}")


def _log_files(path: Path) -> list[Path]:
    """The log itself plus any per-instance files written by other replicas."""
    files = [path] if path.exists() else []
    files += sorted(path.parent.glob(f"{path.stem}-*{path.suffix}")) if path.parent.exists() else []
    return files


def _append(path: Path, record: dict) -> bool:
    """Append a redacted, hash-chained record. Never raises — returns success bool.

    Each record's _hash = sha256(prev_hash + payload). Chaining to the previous
    hash means altering any earlier record breaks every hash after it, making
    tampering detectable.
    """
    try:
        path = _instance_path(path)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        clean = _redact(record)
        with _APPEND_LOCK:
            key = str(path)
            size = path.stat().st_size if path.exists() else 0
            cached = _LAST_HASH.get(key)
            # Trust the cache only if the file has not changed since we last wrote it.
            prev = cached[0] if cached and cached[1] == size else _last_hash(path)
            payload = json.dumps(clean, sort_keys=True, default=str)
            clean["_prev"] = prev
            clean["_hash"] = hashlib.sha256((prev + payload).encode("utf-8")).hexdigest()
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(clean, default=str) + "\n")
            _LAST_HASH[key] = (clean["_hash"], path.stat().st_size)
        return True
    except Exception as exc:  # logging must never break the pipeline
        # Last-resort: print to stderr, but do not propagate.
        print(f"[logging] WARN: failed to write to {path}: {exc}")
        return False


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #


def start_run(raw_query: str) -> str:
    """Open a new run, returning a run_id used to correlate all later records."""
    run_id = uuid.uuid4().hex[:12]
    _append(
        AUDIT_LOG,
        {
            "type": "run_start",
            "run_id": run_id,
            "timestamp": _now_iso(),
            "query": raw_query,
        },
    )
    return run_id


def log_blocked(conversation_id: str, categories: list[str], length: int) -> str:
    """Record a request rejected by the input guard and return its run_id.

    Only the guard categories and the input length are stored, never the text, so a blocked
    attack is part of the tamper-evident trail without the trail holding the attack.
    """
    run_id = uuid.uuid4().hex[:12]
    now = _now_iso()
    _append(
        AUDIT_LOG,
        {
            "type": "run_start",
            "run_id": run_id,
            "timestamp": now,
            "query": "[withheld: blocked input]",
            "conversation_id": conversation_id,
        },
    )
    _append(
        AUDIT_LOG,
        {
            "type": "run_end",
            "run_id": run_id,
            "timestamp": now,
            "status": "blocked",
            "blocked_categories": sorted(categories),
            "input_length": length,
            "escalated": False,
            "safety_verdict": None,
            "confidence": None,
            "answer_preview": "",
        },
    )
    emit_metrics(run_id, {"status": "blocked", "safety_verdict": "n/a"}, 0.0)
    return run_id


def log_trace(run_id: str, trace: list[TraceEvent]) -> bool:
    """Persist the agent trace events accumulated in state['trace'] for a run."""
    events = [asdict(e) if isinstance(e, TraceEvent) else dict(e) for e in trace]
    return _append(
        AUDIT_LOG,
        {
            "type": "trace",
            "run_id": run_id,
            "timestamp": _now_iso(),
            "events": events,
        },
    )


# Optional fields a caller can pass in `final_state`; they complete the provenance of a run.
_RUN_END_FIELDS = (
    "status",
    "specialist",
    "equipment_model",
    "source_refs",
    "hazard_categories",
    "hazard_count",
    "ppe_required",
    "requires_human_review",
    "escalation_reason",
    "confidence_level",
    "explained",
    "answer_sha256",
    "error_type",
    "trace_summary",
)


def end_run(run_id: str, final_state: dict) -> bool:
    """Close a run, recording the outcome (answer/escalation) and key signals.

    Besides the answer preview, the record can carry the full provenance of the run: status,
    the specialist that handled it, the equipment, references to the sources used (never their
    text), the hazard categories, PPE, the escalation reason, the confidence level, whether an
    explanation was produced, and a hash of the answer the user saw.
    """
    record = {
        "type": "run_end",
        "run_id": run_id,
        "timestamp": _now_iso(),
        "escalated": final_state.get("escalated", False),
        "safety_verdict": final_state.get("safety_verdict"),
        "confidence": final_state.get("confidence"),
        "answer_preview": (final_state.get("final_answer") or "")[:300],
    }
    for key in _RUN_END_FIELDS:
        if key in final_state:
            record[key] = final_state[key]
    if isinstance(record.get("escalation_reason"), str):
        record["escalation_reason"] = record["escalation_reason"][:200]
    return _append(AUDIT_LOG, record)


def log_feedback(run_id: str, helpful: bool, comment: str = "") -> bool:
    """Capture user feedback against a run (feeds the improvement loop)."""
    return _append(
        FEEDBACK_LOG,
        {
            "type": "feedback",
            "run_id": run_id,
            "timestamp": _now_iso(),
            "helpful": helpful,
            "comment": comment,
        },
    )


def emit_metrics(run_id: str, final_state: dict, latency_s: float) -> dict:
    """Record per-run operational metrics and return them for immediate use."""
    metrics = {
        "type": "metrics",
        "run_id": run_id,
        "timestamp": _now_iso(),
        "latency_s": round(latency_s, 3),
        "confidence": final_state.get("confidence", 0.0) or 0.0,
        "escalated": bool(final_state.get("escalated", False)),
        "safety_verdict": final_state.get("safety_verdict", "unknown"),
        "num_chunks": len(final_state.get("retrieved_chunks", []) or []),
        "retry_count": final_state.get("retry_count", 0),
    }
    for key in (
        "status",
        "specialist",
        "equipment_model",
        "source_count",
        "hazard_count",
        "confidence_level",
        "explained",
    ):
        if key in final_state:
            metrics[key] = final_state[key]
    _append(METRICS_LOG, metrics)
    return metrics


def run_exists(run_id: str) -> bool:
    """True if an audit record for this run was written (used to validate feedback)."""
    return any(rec.get("run_id") == run_id and rec.get("type") == "run_start" for rec in _read(AUDIT_LOG))


# --------------------------------------------------------------------------- #
# Monitoring aggregation (for dashboards / the evaluation report)             #
# --------------------------------------------------------------------------- #


def _read_file(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def _read(path: Path) -> list[dict]:
    """All records of a log, including the per-instance files of other replicas."""
    out: list[dict] = []
    for file in _log_files(path):
        out.extend(_read_file(file))
    return out


CONFIDENCE_LOW = 0.50  # below this the confidence level is "low"
CONFIDENCE_FLOOR = 0.70  # below this an answer is flagged as low confidence (PRD escalation gate)


def confidence_level(confidence: float) -> str:
    """Map a confidence score to the levels used by the explanation layer."""
    if confidence < CONFIDENCE_LOW:
        return "low"
    if confidence < CONFIDENCE_FLOOR:
        return "medium"
    return "high"


def _rate(count: int, total: int) -> float:
    return round(count / total, 3) if total else 0.0


def monitoring_summary(limit: int | None = None) -> dict:
    """Aggregate metrics across runs (all replicas) - the numbers a monitor or dashboard shows.

    `limit` keeps only the most recent runs. Beyond latency and escalation it reports the rates
    the PRD monitors (blocked, unavailable, low confidence), the explainability coverage
    (answered runs that carry an explanation) and, when feedback exists, how often answers at
    each confidence level were rated helpful (a check that confidence means something).
    """
    metrics = [m for m in _read(METRICS_LOG) if m.get("type") == "metrics"]
    metrics.sort(key=lambda m: m.get("timestamp", ""))
    if limit:
        metrics = metrics[-limit:]
    feedback = [f for f in _read(FEEDBACK_LOG) if f.get("type") == "feedback"]
    n = len(metrics)
    if n == 0:
        return {"runs": 0}

    statuses = collections.Counter(m.get("status", "unknown") for m in metrics)
    escalations = sum(1 for m in metrics if m.get("escalated"))
    # Blocked requests never reach the agents, so they are left out of the latency figures.
    latencies = sorted(m.get("latency_s", 0.0) for m in metrics if m.get("status") != "blocked") or [0.0]
    confidences = [m.get("confidence", 0.0) or 0.0 for m in metrics]
    scored = [c for c in confidences if c > 0]
    helpful = sum(1 for f in feedback if f.get("helpful"))

    answered = [m for m in metrics if m.get("status") == "ok"]
    explained = sum(1 for m in answered if m.get("explained") and (m.get("source_count") or 0) >= 1)

    by_run = {m.get("run_id"): m for m in metrics}
    helpful_by_level: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])  # level -> [helpful, total]
    for f in feedback:
        m = by_run.get(f.get("run_id"))
        if m and (m.get("confidence") or 0) > 0:
            level = confidence_level(m["confidence"])
            helpful_by_level[level][1] += 1
            helpful_by_level[level][0] += 1 if f.get("helpful") else 0

    return {
        "runs": n,
        "status_counts": dict(statuses),
        "escalation_rate": round(escalations / n, 3),
        "blocked_rate": _rate(statuses.get("blocked", 0), n),
        "unavailable_rate": _rate(statuses.get("unavailable", 0), n),
        "clarification_rate": _rate(statuses.get("clarification", 0), n),
        "halted_rate": _rate(statuses.get("halted", 0), n),
        "avg_confidence": round(sum(confidences) / n, 3),
        "confidence_scored_runs": len(scored),
        "low_confidence_rate": _rate(sum(1 for c in scored if c < CONFIDENCE_FLOOR), len(scored)) if scored else None,
        "explainability_coverage": _rate(explained, len(answered)) if answered else None,
        "p50_latency_s": latencies[len(latencies) // 2],
        "p95_latency_s": latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)],
        "feedback_count": len(feedback),
        "helpful_rate": round(helpful / len(feedback), 3) if feedback else None,
        "helpful_rate_by_confidence": {k: _rate(v[0], v[1]) for k, v in helpful_by_level.items()},
        "by_equipment": dict(collections.Counter(m.get("equipment_model") or "none" for m in metrics)),
    }


# Alert thresholds (PRD 10.3: error spikes, low-confidence surges, safety-block anomalies).
# Override with environment variables, for example ALERT_BLOCKED_RATE=0.2.
DEFAULT_ALERT_THRESHOLDS = {
    "min_runs": 10,  # do not alert on tiny samples
    "unavailable_rate": 0.20,
    "blocked_rate": 0.30,
    "escalation_rate": 0.50,
    "low_confidence_rate": 0.40,
}
_ALERT_RULES = (
    ("unavailable_rate", "critical", "Error spike: too many requests could not be answered"),
    ("blocked_rate", "warning", "Safety-block anomaly: many requests blocked by the input guard"),
    ("escalation_rate", "warning", "Escalation surge: many requests escalated to a human"),
    ("low_confidence_rate", "warning", "Low-confidence surge: many answers below the confidence floor"),
)


def _alert_thresholds(overrides: dict | None) -> dict:
    thresholds = dict(DEFAULT_ALERT_THRESHOLDS)
    for key in thresholds:
        raw = os.getenv(f"ALERT_{key.upper()}")
        if raw:
            try:
                thresholds[key] = float(raw)
            except ValueError:
                pass
    thresholds.update(overrides or {})
    return thresholds


def check_alerts(summary: dict, thresholds: dict | None = None) -> list[dict]:
    """Compare a monitoring summary with the thresholds and return the alerts that fired."""
    limits = _alert_thresholds(thresholds)
    if summary.get("runs", 0) < limits["min_runs"]:
        return []
    alerts = []
    for name, severity, message in _ALERT_RULES:
        value = summary.get(name)
        if value is not None and value > limits[name]:
            alerts.append(
                {"name": name, "severity": severity, "value": value, "threshold": limits[name], "message": message}
            )
    return alerts


def _verify_chain(records: list[dict]) -> int | None:
    """Return the index of the first broken record, or None if the chain is intact."""
    prev = ""
    for i, rec in enumerate(records):
        stored_hash = rec.get("_hash", "")
        stored_prev = rec.get("_prev", "")
        recomputed = dict(rec)
        recomputed.pop("_hash", None)
        recomputed.pop("_prev", None)
        payload = json.dumps(recomputed, sort_keys=True, default=str)
        expected = hashlib.sha256((prev + payload).encode("utf-8")).hexdigest()
        if stored_prev != prev or stored_hash != expected:
            return i
        prev = stored_hash
    return None


def verify_audit_integrity() -> dict:
    """Re-walk the audit log and confirm the hash chain is intact.

    Each file (the main log and one per replica) is its own chain. Returns
    {'intact': bool, 'checked': int, 'broken_at': index|None, 'files': int,
    'broken_file': name|None}. A broken chain means a record was altered or removed after
    being written.
    """
    checked = 0
    files = _log_files(AUDIT_LOG)
    for file in files:
        records = _read_file(file)
        broken = _verify_chain(records)
        if broken is not None:
            return {
                "intact": False,
                "checked": checked + broken,
                "broken_at": broken,
                "files": len(files),
                "broken_file": file.name,
            }
        checked += len(records)
    return {"intact": True, "checked": checked, "broken_at": None, "files": len(files), "broken_file": None}
