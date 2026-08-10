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

import hashlib
import json
import re
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from asa.graph.state import TraceEvent

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

LOG_DIR = Path("logs")
AUDIT_LOG = LOG_DIR / "audit.jsonl"        # append-only run + event records
FEEDBACK_LOG = LOG_DIR / "feedback.jsonl"  # user feedback records
METRICS_LOG = LOG_DIR / "metrics.jsonl"    # per-run operational metrics


# --------------------------------------------------------------------------- #
# Redaction — runs before any write                                           #
# --------------------------------------------------------------------------- #

_REDACTION_PATTERNS = [
    (re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b"), "[EMAIL]"),
    (re.compile(r"\b(?:\d[ -]*?){13,16}\b"), "[CARD]"),          # card-like digit runs
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


def _append(path: Path, record: dict) -> bool:
    """Append a redacted, hash-chained record. Never raises — returns success bool.

    Each record's _hash = sha256(prev_hash + payload). Chaining to the previous
    hash means altering any earlier record breaks every hash after it, making
    tampering detectable.
    """
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        clean = _redact(record)
        prev = _last_hash(path)
        payload = json.dumps(clean, sort_keys=True, default=str)
        clean["_prev"] = prev
        clean["_hash"] = hashlib.sha256((prev + payload).encode("utf-8")).hexdigest()
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(clean, default=str) + "\n")
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
    _append(AUDIT_LOG, {
        "type": "run_start",
        "run_id": run_id,
        "timestamp": _now_iso(),
        "query": raw_query,
    })
    return run_id


def log_trace(run_id: str, trace: list[TraceEvent]) -> bool:
    """Persist the agent trace events accumulated in state['trace'] for a run."""
    events = [asdict(e) if isinstance(e, TraceEvent) else dict(e) for e in trace]
    return _append(AUDIT_LOG, {
        "type": "trace",
        "run_id": run_id,
        "timestamp": _now_iso(),
        "events": events,
    })


def end_run(run_id: str, final_state: dict) -> bool:
    """Close a run, recording the outcome (answer/escalation) and key signals."""
    return _append(AUDIT_LOG, {
        "type": "run_end",
        "run_id": run_id,
        "timestamp": _now_iso(),
        "escalated": final_state.get("escalated", False),
        "safety_verdict": final_state.get("safety_verdict"),
        "confidence": final_state.get("confidence"),
        "answer_preview": (final_state.get("final_answer") or "")[:300],
    })


def log_feedback(run_id: str, helpful: bool, comment: str = "") -> bool:
    """Capture user feedback against a run (feeds the improvement loop)."""
    return _append(FEEDBACK_LOG, {
        "type": "feedback",
        "run_id": run_id,
        "timestamp": _now_iso(),
        "helpful": helpful,
        "comment": comment,
    })


def emit_metrics(run_id: str, final_state: dict, latency_s: float) -> dict:
    """Record per-run operational metrics and return them for immediate use."""
    metrics = {
        "type": "metrics",
        "run_id": run_id,
        "timestamp": _now_iso(),
        "latency_s": round(latency_s, 3),
        "confidence": final_state.get("confidence", 0.0),
        "escalated": bool(final_state.get("escalated", False)),
        "safety_verdict": final_state.get("safety_verdict", "unknown"),
        "num_chunks": len(final_state.get("retrieved_chunks", []) or []),
        "retry_count": final_state.get("retry_count", 0),
    }
    _append(METRICS_LOG, metrics)
    return metrics


# --------------------------------------------------------------------------- #
# Monitoring aggregation (for dashboards / the evaluation report)             #
# --------------------------------------------------------------------------- #

def _read(path: Path) -> list[dict]:
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


def monitoring_summary() -> dict:
    """Aggregate metrics across all runs — the numbers a monitor/dashboard shows."""
    metrics = [m for m in _read(METRICS_LOG) if m.get("type") == "metrics"]
    feedback = [f for f in _read(FEEDBACK_LOG) if f.get("type") == "feedback"]
    n = len(metrics)
    if n == 0:
        return {"runs": 0}

    escalations = sum(1 for m in metrics if m.get("escalated"))
    latencies = sorted(m.get("latency_s", 0.0) for m in metrics)
    confidences = [m.get("confidence", 0.0) for m in metrics]
    helpful = sum(1 for f in feedback if f.get("helpful"))

    return {
        "runs": n,
        "escalation_rate": round(escalations / n, 3),
        "avg_confidence": round(sum(confidences) / n, 3),
        "p50_latency_s": latencies[n // 2],
        "p95_latency_s": latencies[min(int(n * 0.95), n - 1)],
        "feedback_count": len(feedback),
        "helpful_rate": round(helpful / len(feedback), 3) if feedback else None,
    }


def verify_audit_integrity() -> dict:
    """Re-walk the audit log and confirm the hash chain is intact.

    Returns {'intact': bool, 'checked': int, 'broken_at': index|None}. A broken
    chain means a record was altered or removed after being written.
    """
    records = _read(AUDIT_LOG)
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
            return {"intact": False, "checked": i, "broken_at": i}
        prev = stored_hash
    return {"intact": True, "checked": len(records), "broken_at": None}
