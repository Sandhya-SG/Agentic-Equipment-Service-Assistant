"""Human review queue: answers that must not stand alone go to a reviewer, whose decision is recorded.

An answer is opened for review when the assistant stopped the request, escalated it, hit an "escalate-only" hazard, was
not confident (below the confidence floor), or (optionally, `REVIEW_SAMPLE_RATE`) as a random quality sample. A qualified
reviewer then records one decision per item:

  * confirm   the assistant's decision was right. It never releases a stopped answer.
  * correct   the reviewer writes corrected guidance and cites the manual. The text must pass the response firewall and
              the deterministic safety checks (it may not tell the engineer to defeat a safety control), and it is
              shown to the engineer as the reviewer's guidance, never as the assistant's.
  * reject    the assistant's answer is withdrawn; a reason is required.

Storage is an append-only, hash-chained event log (`review_queue.jsonl`, next to the audit log, on the persistent volume):
`review_created` and `review_decided` events. The state of an item is rebuilt from the events, so nothing is ever
overwritten, and a decision cannot be changed quietly. Every decision is also written into the audit chain
(`review_decision`, with the reviewer, the outcome and a hash of the text, never the text). One process works on the log
at a time: creating an item and deciding it run under a lock, so two simultaneous decisions on the same item cannot both
succeed.

LIMIT: the queue lives in the log of ONE backend pod. With several pods each has its own queue, so run one backend
replica while using it (see docs/runbook.md); a shared database would remove the limit.
"""

from __future__ import annotations

import logging
import os
import random
import re
import statistics
import threading
import uuid
from datetime import datetime, timezone

from asa.components import logging_sub
from asa.guardrails.safety_rules import check_defeat_request
from asa.guardrails.schema_guard import guard_output

TRIGGERS = ("halted", "escalated", "escalate_only", "low_confidence", "sample")
DECISIONS = ("confirm", "correct", "reject")
OUTCOMES = {"confirm": "confirmed", "correct": "corrected", "reject": "rejected"}
REVIEWER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._@-]{0,63}$")
REVIEW_ID = re.compile(r"^[0-9a-f]{32}$")

MIN_REASON_CHARS = 5
MIN_CORRECTION_CHARS = 20
MAX_CORRECTION_CHARS = 4000
MAX_REASON_CHARS = 1000
MAX_EVIDENCE = 5
MAX_EVIDENCE_CHARS = 300

# What the engineer is told (wording to be approved by the subject-matter expert).
PENDING_MESSAGE = "This answer is pending review by a qualified engineer. Do not rely on it until it is confirmed."
CONFIRMED_MESSAGE = "Reviewed and confirmed by a qualified engineer."
CORRECTED_MESSAGE = "Reviewed by a qualified engineer, who corrected this guidance:"
REJECTED_MESSAGE = "A qualified engineer reviewed this answer and withdrew it. Please escalate to your supervisor."

logger = logging.getLogger(__name__)
_LOCK = threading.RLock()


class ReviewError(Exception):
    """Base class: a review request that cannot be carried out."""


class NotFound(ReviewError):
    pass


class AlreadyDecided(ReviewError):
    pass


class InvalidDecision(ReviewError):
    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


class UnsafeCorrection(InvalidDecision):
    """The corrected guidance failed a safety check; nothing was recorded."""


# --------------------------------------------------------------------------- #
# The event log                                                               #
# --------------------------------------------------------------------------- #


def _path():
    return logging_sub._instance_path(logging_sub.LOG_DIR / "review_queue.jsonl")


def _events() -> list[dict]:
    return logging_sub._read_file(_path())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seconds_between(start: str, end: str) -> float | None:
    try:
        return round((datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds(), 1)
    except (TypeError, ValueError):
        return None


def _replay(events: list[dict]) -> dict[str, dict]:
    """Rebuild every item from the events, oldest first. The first decision of an item is the only one that counts."""
    items: dict[str, dict] = {}
    for event in events:
        kind, review_id = event.get("type"), event.get("review_id")
        if kind == "review_created" and review_id and review_id not in items:
            item = {k: v for k, v in event.items() if not k.startswith("_") and k != "type"}
            item.update({"status": "pending", "created_at": event.get("ts")})
            items[review_id] = item
        elif kind == "review_decided" and review_id in items and items[review_id]["status"] == "pending":
            item = items[review_id]
            item.update(
                {
                    "status": event.get("outcome"),
                    "decision": event.get("decision"),
                    "reviewer": event.get("reviewer"),
                    "reason": event.get("reason"),
                    "corrected_text": event.get("corrected_text"),
                    "evidence": event.get("evidence") or [],
                    "decided_at": event.get("ts"),
                    "seconds_to_decide": event.get("seconds_to_decide"),
                }
            )
    return items


def _state() -> dict[str, dict]:
    return _replay(_events())


# --------------------------------------------------------------------------- #
# Opening an item                                                             #
# --------------------------------------------------------------------------- #


def open_review(
    *,
    run_id: str,
    trigger: str,
    question: str,
    answer: str,
    status: str,
    equipment_model: str | None,
    specialist: str | None,
    verdict: str | None,
    hazards: list[str],
    ppe_required: list[str],
    sources: list[str],
    explanation: dict | None,
    confidence: float | None,
    trace: list[str],
) -> dict | None:
    """Put an answer on the reviewer's list. Returns {review_id, status, trigger}, or None if it could not be saved
    (the answer is still returned to the engineer; opening a review never breaks a request)."""
    try:
        if trigger not in TRIGGERS:
            raise ValueError(f"unknown review trigger {trigger!r}")
        with _LOCK:
            for existing in _state().values():
                if existing.get("run_id") == run_id:  # one item per run
                    return {
                        "review_id": existing["review_id"],
                        "status": existing["status"],
                        "trigger": existing["trigger"],
                    }
            review_id = uuid.uuid4().hex
            citations = [
                {k: c.get(k) for k in ("source_file", "page", "section", "quote")}
                for c in ((explanation or {}).get("citations") or [])[:8]
                if isinstance(c, dict)
            ]
            saved = logging_sub._append(
                _path_for_append(),
                {
                    "type": "review_created",
                    "review_id": review_id,
                    "ts": _now(),
                    "run_id": run_id,
                    "trigger": trigger,
                    "answer_status": status,
                    "equipment_model": equipment_model,
                    "specialist": specialist,
                    "verdict": verdict,
                    "question": question,
                    "answer": answer,
                    "hazards": list(hazards or []),
                    "ppe_required": list(ppe_required or []),
                    "sources": list(sources or []),
                    "citations": citations,
                    "confidence": confidence,
                    "confidence_level": (explanation or {}).get("confidence_level"),
                    "trace": list(trace or []),
                },
            )
            if not saved:
                return None
        logging_sub.log_review_opened(run_id, review_id, trigger)
        return {"review_id": review_id, "status": "pending", "trigger": trigger}
    except Exception as exc:  # noqa: BLE001 - a review that cannot be opened must not break the request
        logger.warning("review not opened: %s", type(exc).__name__)
        return None


def _path_for_append():
    """The log path to hand to `logging_sub._append`, which adds the per-replica suffix itself."""
    return logging_sub.LOG_DIR / "review_queue.jsonl"


def should_sample() -> bool:
    """True for a random share of ordinary answers (REVIEW_SAMPLE_RATE, 0 to 1; off by default)."""
    try:
        rate = float(os.getenv("REVIEW_SAMPLE_RATE", "0") or 0)
    except ValueError:
        return False
    return rate > 0 and random.random() < min(rate, 1.0)


# --------------------------------------------------------------------------- #
# Reading                                                                     #
# --------------------------------------------------------------------------- #


def _summary_row(item: dict) -> dict:
    return {
        "review_id": item["review_id"],
        "status": item["status"],
        "trigger": item.get("trigger"),
        "created_at": item.get("created_at"),
        "decided_at": item.get("decided_at"),
        "equipment_model": item.get("equipment_model"),
        "specialist": item.get("specialist"),
        "verdict": item.get("verdict"),
        "confidence_level": item.get("confidence_level"),
        "question": (item.get("question") or "")[:160],
        "reviewer": item.get("reviewer"),
    }


def list_items(status: str = "pending", limit: int = 100) -> list[dict]:
    """Summaries: pending items oldest first (the order to work through them), decided items newest first."""
    items = list(_state().values())
    if status == "pending":
        rows = [i for i in items if i["status"] == "pending"]
    elif status == "decided":
        rows = sorted(
            (i for i in items if i["status"] != "pending"), key=lambda i: i.get("decided_at") or "", reverse=True
        )
    else:
        rows = sorted(items, key=lambda i: i.get("created_at") or "", reverse=True)
    return [_summary_row(i) for i in rows[: max(1, limit)]]


def counts() -> dict[str, int]:
    tally = {"pending": 0, "confirmed": 0, "corrected": 0, "rejected": 0}
    for item in _state().values():
        tally[item["status"]] = tally.get(item["status"], 0) + 1
    return tally


def get_item(review_id: str) -> dict:
    item = _state().get(review_id)
    if item is None:
        raise NotFound(review_id)
    return item


def engineer_view(review_id: str) -> dict:
    """What the engineer who asked may see: the state and the outcome in plain words, never who decided or the
    internal case data."""
    item = get_item(review_id)
    status = item["status"]
    view = {"review_id": review_id, "status": status}
    if status == "pending":
        view["message"] = PENDING_MESSAGE
    elif status == "confirmed":
        view["message"] = CONFIRMED_MESSAGE
    elif status == "corrected":
        view["message"] = CORRECTED_MESSAGE
        view["corrected_guidance"] = item.get("corrected_text")
        view["evidence"] = item.get("evidence") or []
    elif status == "rejected":
        view["message"] = REJECTED_MESSAGE
        view["reason"] = item.get("reason")
    return view


# --------------------------------------------------------------------------- #
# Deciding                                                                    #
# --------------------------------------------------------------------------- #


_DEFEAT_VERBS = (
    r"bypass\w*|defeat\w*|disabl\w*|overrid\w*|circumvent\w*|jumper\w*|tamper\w*|trick\w*|work(?:ing)? around|"
    r"get(?:ting)? around|remov\w*|unplug\w*|disconnect\w*|tak(?:e|ing) off|switch(?:ing)? off|turn(?:ing)? off"
)
# A sentence that FORBIDS defeating a control ("Never bypass the interlock", "Bypassing the guard is not permitted") is
# the kind of guidance a reviewer should write; it is not an instruction to do it. The "do not" must be in the same
# clause as the verb (no comma, colon or semicolon between them), so "Do not wait, bypass the interlock" still counts.
_FORBIDS = re.compile(
    r"\b(?:never|do not|don't|must not|mustn't|should not|shouldn't|shall not|cannot|can't|may not|"
    r"not allowed to|not permitted to|forbidden to|prohibited from)\b[^.!?;:,\n]{0,40}\b(?:" + _DEFEAT_VERBS + r")\b"
    r"|\b(?:"
    + _DEFEAT_VERBS
    + r")\b[^.!?\n]{0,80}\b(?:is|are) (?:strictly )?(?:prohibited|forbidden|not permitted|not allowed)\b",
    re.IGNORECASE,
)


def _defeat_instruction(text: str) -> str | None:
    """The safety control the text tells the engineer to defeat, or None. Sentences that forbid defeating a control are
    skipped; a sentence that mixes a harmless "do not" with an instruction to defeat one still counts."""
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        if sentence.strip() and not _FORBIDS.search(sentence):
            control = check_defeat_request(sentence)
            if control:
                return control
    return None


def validate_correction(text: str, evidence: list[str]) -> list[str]:
    """Problems that stop corrected guidance from being released (empty list: it may be). These are deterministic
    safety nets, not a replacement for the reviewer's judgement: they check what the response firewall checks on every
    answer, that the text does not tell the engineer to defeat a safety control, and that it cites the manual."""
    problems = []
    if len(text) < MIN_CORRECTION_CHARS:
        problems.append(f"the corrected guidance must be at least {MIN_CORRECTION_CHARS} characters")
    if len(text) > MAX_CORRECTION_CHARS:
        problems.append(f"the corrected guidance must be at most {MAX_CORRECTION_CHARS} characters")
    if not evidence:
        problems.append("cite at least one manual source (document, page or section) that supports the guidance")
    checked = guard_output(text, [])
    if checked.action != "none":
        problems.append(
            "the guidance contains text the response firewall removes or withholds "
            f"({', '.join(checked.findings) or checked.action}): remove it and try again"
        )
    control = _defeat_instruction(text)
    if control:
        problems.append(
            f"the guidance reads as an instruction to bypass or defeat a safety control ({control}): "
            "rephrase it so that it does not"
        )
    return problems


def decide(
    review_id: str,
    decision: str,
    reviewer: str,
    reason: str | None = None,
    corrected_text: str | None = None,
    evidence: list[str] | None = None,
) -> dict:
    """Record the reviewer's decision. Raises NotFound, AlreadyDecided, InvalidDecision or UnsafeCorrection; nothing is
    recorded then. An item can be decided once: the check and the write happen under one lock."""
    reason = (reason or "").strip()
    corrected_text = (corrected_text or "").strip()
    evidence = [e.strip() for e in (evidence or []) if isinstance(e, str) and e.strip()]
    problems = []
    if decision not in DECISIONS:
        problems.append(f"decision must be one of {', '.join(DECISIONS)}")
    if not isinstance(reviewer, str) or not REVIEWER_ID.fullmatch(reviewer.strip()):
        problems.append("a reviewer name or id is required (letters, digits, space . _ @ -)")
    if decision in ("correct", "reject") and len(reason) < MIN_REASON_CHARS:
        problems.append(f"a reason of at least {MIN_REASON_CHARS} characters is required to {decision}")
    if len(reason) > MAX_REASON_CHARS:
        problems.append(f"the reason must be at most {MAX_REASON_CHARS} characters")
    if decision != "correct" and (corrected_text or evidence):
        problems.append("corrected guidance and sources belong to a 'correct' decision only")
    if len(evidence) > MAX_EVIDENCE or any(len(e) > MAX_EVIDENCE_CHARS for e in evidence):
        problems.append(f"at most {MAX_EVIDENCE} sources of at most {MAX_EVIDENCE_CHARS} characters each")
    if problems:
        raise InvalidDecision(problems)
    if decision == "correct":
        unsafe = validate_correction(corrected_text, evidence)
        if unsafe:
            raise UnsafeCorrection(unsafe)

    with _LOCK:
        item = _state().get(review_id)
        if item is None:
            raise NotFound(review_id)
        if item["status"] != "pending":
            raise AlreadyDecided(review_id)
        now = _now()
        saved = logging_sub._append(
            _path_for_append(),
            {
                "type": "review_decided",
                "review_id": review_id,
                "run_id": item.get("run_id"),
                "ts": now,
                "decision": decision,
                "outcome": OUTCOMES[decision],
                "reviewer": reviewer.strip(),
                "reason": reason or None,
                "corrected_text": corrected_text or None,
                "evidence": evidence,
                "seconds_to_decide": _seconds_between(item.get("created_at"), now),
            },
        )
        if not saved:
            raise RuntimeError("the decision could not be recorded")
    logging_sub.log_review_decision(
        item.get("run_id"), review_id, decision, OUTCOMES[decision], reviewer.strip(), reason, corrected_text
    )
    return get_item(review_id)


# --------------------------------------------------------------------------- #
# Monitoring                                                                  #
# --------------------------------------------------------------------------- #


def summary(now: datetime | None = None) -> dict:
    """Backlog, time to decide and override rate, for /api/monitoring (aggregates only, no text)."""
    items = list(_state().values())
    pending = [i for i in items if i["status"] == "pending"]
    decided = [i for i in items if i["status"] != "pending"]
    clock = now or datetime.now(timezone.utc)
    ages = []
    for item in pending:
        try:
            ages.append((clock - datetime.fromisoformat(item["created_at"])).total_seconds())
        except (KeyError, TypeError, ValueError):
            continue
    times = [i["seconds_to_decide"] for i in decided if isinstance(i.get("seconds_to_decide"), (int, float))]
    overridden = sum(1 for i in decided if i["status"] in ("corrected", "rejected"))
    by_trigger: dict[str, int] = {}
    for item in items:
        by_trigger[item.get("trigger") or "unknown"] = by_trigger.get(item.get("trigger") or "unknown", 0) + 1
    return {
        "opened": len(items),
        "pending": len(pending),
        "oldest_pending_age_s": round(max(ages)) if ages else None,
        "decided": len(decided),
        "confirmed": sum(1 for i in decided if i["status"] == "confirmed"),
        "corrected": sum(1 for i in decided if i["status"] == "corrected"),
        "rejected": sum(1 for i in decided if i["status"] == "rejected"),
        # How often the assistant was wrong where it asked for help: corrected or rejected out of decided.
        "override_rate": round(overridden / len(decided), 3) if decided else None,
        "median_seconds_to_decide": round(statistics.median(times), 1) if times else None,
        "by_trigger": by_trigger,
    }
