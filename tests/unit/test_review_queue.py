"""The human review queue: opening items, one decision per item, safety checks on corrected guidance, persistence,
the hash chain and the audit link. (Fake data only; no model and no network.)"""

import json
import threading
from datetime import datetime, timedelta, timezone

import pytest

from app.services import review_queue as rq
from asa.components import logging_sub

GUIDANCE = "Power the equipment off and lock it out before opening the panel, as the manual requires."
SOURCE = "aem_thermal_station.pdf, p.19, 2.2.2 General Safety Guidelines"


def make_item(run_id="aaaaaaaaaaaa", trigger="halted", **overrides):
    case = {
        "run_id": run_id,
        "trigger": trigger,
        "question": "Can I open the panel while the equipment is powered?",
        "answer": "Request halted: opening the panel while powered is not permitted.",
        "status": "halted",
        "equipment_model": "thermal_station",
        "specialist": "safety",
        "verdict": "halt",
        "hazards": ["high_voltage"],
        "ppe_required": ["insulated gloves"],
        "sources": [SOURCE],
        "explanation": {
            "confidence_level": "high",
            "citations": [
                {"source_file": "aem_thermal_station.pdf", "page": 19, "section": "2.2.2", "quote": "Power off."}
            ],
        },
        "confidence": 0.85,
        "trace": ["request", "planning", "safety"],
    }
    case.update(overrides)
    result = rq.open_review(**case)
    assert result is not None
    return result["review_id"]


# --- opening ---


def test_an_item_is_opened_pending_with_the_case_data():
    review_id = make_item()
    item = rq.get_item(review_id)
    assert item["status"] == "pending" and item["trigger"] == "halted"
    assert item["question"].startswith("Can I open the panel") and item["verdict"] == "halt"
    assert item["hazards"] == ["high_voltage"] and item["sources"] == [SOURCE]
    assert item["citations"][0]["quote"] == "Power off."
    assert len(review_id) == 32


def test_the_same_run_is_never_opened_twice():
    first = make_item(run_id="bbbbbbbbbbbb")
    second = rq.open_review(
        run_id="bbbbbbbbbbbb",
        trigger="halted",
        question="q",
        answer="a",
        status="halted",
        equipment_model=None,
        specialist=None,
        verdict=None,
        hazards=[],
        ppe_required=[],
        sources=[],
        explanation=None,
        confidence=None,
        trace=[],
    )
    assert second["review_id"] == first and rq.counts()["pending"] == 1


def test_an_unknown_trigger_is_refused_without_raising():
    assert (
        rq.open_review(
            run_id="cccccccccccc",
            trigger="because",
            question="q",
            answer="a",
            status="ok",
            equipment_model=None,
            specialist=None,
            verdict=None,
            hazards=[],
            ppe_required=[],
            sources=[],
            explanation=None,
            confidence=None,
            trace=[],
        )
        is None
    )
    assert rq.counts()["pending"] == 0


def test_a_failed_write_never_raises(monkeypatch):
    monkeypatch.setattr(logging_sub, "_append", lambda *a, **k: False)
    assert (
        rq.open_review(
            run_id="dddddddddddd",
            trigger="halted",
            question="q",
            answer="a",
            status="halted",
            equipment_model=None,
            specialist=None,
            verdict=None,
            hazards=[],
            ppe_required=[],
            sources=[],
            explanation=None,
            confidence=None,
            trace=[],
        )
        is None
    )


def test_the_list_is_oldest_first_for_pending_and_newest_first_for_decided():
    ids = [make_item(run_id=f"{i:012x}") for i in range(3)]
    assert [r["review_id"] for r in rq.list_items("pending")] == ids
    rq.decide(ids[0], "confirm", "Nazir")
    rq.decide(ids[2], "confirm", "Nazir")
    assert [r["review_id"] for r in rq.list_items("pending")] == [ids[1]]
    assert [r["review_id"] for r in rq.list_items("decided")] == [ids[2], ids[0]]
    assert {r["review_id"] for r in rq.list_items("all")} == set(ids)
    assert rq.counts() == {"pending": 1, "confirmed": 2, "corrected": 0, "rejected": 0}
    assert rq.list_items("pending", limit=1) == rq.list_items("pending")[:1]


def test_the_list_shows_a_short_question_only():
    make_item(question="x" * 500)
    assert len(rq.list_items("pending")[0]["question"]) == 160


# --- deciding ---


def test_confirm_records_who_when_and_how_long_it_took():
    review_id = make_item()
    item = rq.decide(review_id, "confirm", "Nazir Zakaria")
    assert item["status"] == "confirmed" and item["reviewer"] == "Nazir Zakaria" and item["decision"] == "confirm"
    assert item["decided_at"] and item["seconds_to_decide"] >= 0


def test_a_confirmed_stop_stays_a_stop():
    """Confirm never releases anything: the answer is unchanged and the engineer gets no new guidance."""
    review_id = make_item()
    before = rq.get_item(review_id)["answer"]
    rq.decide(review_id, "confirm", "Nazir")
    assert rq.get_item(review_id)["answer"] == before
    assert "corrected_guidance" not in rq.engineer_view(review_id)


def test_correct_needs_guidance_sources_and_a_reason():
    review_id = make_item()
    item = rq.decide(review_id, "correct", "Nazir", "The assistant omitted lock out", GUIDANCE, [SOURCE])
    assert item["status"] == "corrected" and item["corrected_text"] == GUIDANCE and item["evidence"] == [SOURCE]
    assert item["reason"] == "The assistant omitted lock out"


def test_reject_needs_a_reason():
    review_id = make_item()
    item = rq.decide(review_id, "reject", "Nazir", "The answer cites the wrong section")
    assert item["status"] == "rejected" and item["reason"].startswith("The answer cites")


@pytest.mark.parametrize(
    "args, expected",
    [
        (("approve", "Nazir"), "decision must be one of"),
        (("confirm", ""), "reviewer name or id is required"),
        (("confirm", "<script>"), "reviewer name or id is required"),
        (("confirm", "x" * 80), "reviewer name or id is required"),
        (("reject", "Nazir", "no"), "reason of at least"),
        (("reject", "Nazir", None), "reason of at least"),
        (("correct", "Nazir", "reason here", None, [SOURCE]), "at least 20 characters"),
        (("correct", "Nazir", "reason here", GUIDANCE, None), "cite at least one manual source"),
        (("correct", "Nazir", "reason here", GUIDANCE, []), "cite at least one manual source"),
        (("confirm", "Nazir", None, GUIDANCE, None), "belong to a 'correct' decision only"),
        (("reject", "Nazir", "wrong section", None, [SOURCE]), "belong to a 'correct' decision only"),
        (("correct", "Nazir", "reason here", GUIDANCE, ["s"] * 6), "at most 5 sources"),
        (("correct", "Nazir", "reason here", GUIDANCE, ["s" * 301]), "at most 5 sources"),
        (("correct", "Nazir", "reason here", "g" * 4001, [SOURCE]), "at most 4000 characters"),
    ],
)
def test_an_incomplete_or_invalid_decision_is_refused_and_nothing_is_recorded(args, expected):
    review_id = make_item()
    with pytest.raises(rq.InvalidDecision, match=expected):
        rq.decide(review_id, *args)
    assert rq.get_item(review_id)["status"] == "pending"


# --- the safety checks on corrected guidance ---


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "Bypass the safety interlock and keep the equipment running while you work on the panel.",
            "bypass or defeat a safety control",
        ),
        (
            "You may disable the door interlock to finish the job faster, then re-enable it afterwards.",
            "bypass or defeat a safety control",
        ),
        (
            "Use the service key sk-abcdefghijklmnopqrstuvwxyz123456 to unlock the maintenance mode first.",
            "response firewall",
        ),
        (
            "Open the panel after isolating the supply. <script>alert(1)</script> Then continue as documented.",
            "response firewall",
        ),
        (
            "Email the reading to jane.doe@example.com before you start, then power the equipment off first.",
            "response firewall",
        ),
    ],
)
def test_unsafe_corrected_guidance_is_refused_and_nothing_is_recorded(text, expected):
    review_id = make_item()
    with pytest.raises(rq.UnsafeCorrection, match=expected) as caught:
        rq.decide(review_id, "correct", "Nazir", "reason here", text, [SOURCE])
    assert caught.value.problems
    assert rq.get_item(review_id)["status"] == "pending"


def test_guidance_that_forbids_bypassing_is_not_mistaken_for_an_instruction_to_bypass():
    problems = rq.validate_correction(
        "Never bypass the safety interlock. Power the equipment off and lock it out before opening the panel.", [SOURCE]
    )
    assert problems == []


@pytest.mark.parametrize(
    "text",
    [
        "Do not disable the door interlock. Power the equipment off and lock it out before opening the panel.",
        "You must not bypass the emergency stop. Isolate the supply and apply your personal lock first.",
        "Bypassing the safety interlock is strictly prohibited. Power off and lock out before servicing the unit.",
        "Never override the guard, and never remove it while the machine is energised. Isolate the supply first.",
    ],
)
def test_guidance_that_forbids_defeating_a_control_is_accepted(text):
    assert rq.validate_correction(text, [SOURCE]) == []


@pytest.mark.parametrize(
    "text",
    [
        "Do not wait for the supervisor, bypass the safety interlock and carry on with the work.",
        "First power off. Then bypass the door interlock so the heater keeps running while you work.",
        "Never mind the lock out: disable the emergency stop and open the panel.",
    ],
)
def test_guidance_that_mixes_a_do_not_with_an_instruction_to_defeat_a_control_is_refused(text):
    problems = rq.validate_correction(text, [SOURCE])
    assert any("bypass or defeat a safety control" in p for p in problems)


def test_the_firewall_and_defeat_checks_both_report():
    problems = rq.validate_correction("Bypass the interlock. Contact jane.doe@example.com", [])
    assert len(problems) == 3  # no source, the firewall, the defeat rule


# --- one decision per item, also when two arrive at once ---


def test_an_item_cannot_be_decided_twice_and_the_first_decision_stands():
    review_id = make_item()
    rq.decide(review_id, "confirm", "Nazir")
    with pytest.raises(rq.AlreadyDecided):
        rq.decide(review_id, "reject", "Someone Else", "I disagree with this one")
    item = rq.get_item(review_id)
    assert item["status"] == "confirmed" and item["reviewer"] == "Nazir"


def test_two_simultaneous_decisions_cannot_both_succeed():
    review_id = make_item()
    barrier = threading.Barrier(8)
    outcomes = []

    def attempt(index):
        barrier.wait()
        try:
            rq.decide(review_id, "reject", f"Reviewer {index}", "Each reviewer thinks it is wrong")
            outcomes.append("decided")
        except rq.AlreadyDecided:
            outcomes.append("refused")

    threads = [threading.Thread(target=attempt, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert outcomes.count("decided") == 1 and outcomes.count("refused") == 7
    decided_events = [e for e in rq._events() if e["type"] == "review_decided" and e["review_id"] == review_id]
    assert len(decided_events) == 1  # exactly one decision is in the log


def test_simultaneous_decisions_on_different_items_all_succeed():
    ids = [make_item(run_id=f"{i:012x}") for i in range(10)]
    errors = []

    def attempt(review_id):
        try:
            rq.decide(review_id, "confirm", "Nazir")
        except Exception as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=attempt, args=(i,)) for i in ids]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == [] and rq.counts()["confirmed"] == 10


def test_an_unknown_item_is_not_found():
    with pytest.raises(rq.NotFound):
        rq.decide("0" * 32, "confirm", "Nazir")
    with pytest.raises(rq.NotFound):
        rq.get_item("0" * 32)


# --- persistence, the chain and the audit link ---


def test_the_queue_survives_a_restart():
    pending = make_item(run_id="111111111111")
    decided = make_item(run_id="222222222222")
    rq.decide(decided, "confirm", "Nazir")
    logging_sub._LAST_HASH.clear()  # a new process knows nothing in memory: the file is the only memory
    assert rq.get_item(pending)["status"] == "pending"
    assert rq.get_item(decided)["status"] == "confirmed"
    assert rq.counts() == {"pending": 1, "confirmed": 1, "corrected": 0, "rejected": 0}
    rq.decide(pending, "reject", "Nazir", "Wrong section cited")  # and the chain carries on
    assert logging_sub._verify_chain(rq._events()) is None


def test_the_log_is_a_hash_chain_and_an_edit_is_detected():
    review_id = make_item()
    rq.decide(review_id, "confirm", "Nazir")
    path = rq._path()
    assert logging_sub._verify_chain(rq._events()) is None
    lines = path.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[-1])
    tampered["outcome"] = "rejected"
    lines[-1] = json.dumps(tampered)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert logging_sub._verify_chain(rq._events()) == len(lines) - 1


def test_a_damaged_line_is_skipped_not_fatal():
    review_id = make_item()
    with rq._path().open("a", encoding="utf-8") as handle:
        handle.write("{this is not json\n")
    assert rq.get_item(review_id)["status"] == "pending"


def test_each_decision_is_in_the_audit_chain_without_the_free_text():
    review_id = make_item(run_id="333333333333")
    rq.decide(review_id, "correct", "Nazir Zakaria", "Lock out was missing", GUIDANCE, [SOURCE])
    audit = logging_sub._read_file(logging_sub.AUDIT_LOG)
    opened = [r for r in audit if r.get("type") == "review_opened"]
    decided = [r for r in audit if r.get("type") == "review_decision"]
    assert opened and opened[0]["run_id"] == "333333333333" and opened[0]["trigger"] == "halted"
    [record] = decided
    assert (
        record["reviewer"] == "Nazir Zakaria" and record["outcome"] == "corrected" and record["review_id"] == review_id
    )
    assert record["correction_chars"] == len(GUIDANCE) and len(record["correction_sha256"]) == 64
    assert record["reason_chars"] == len("Lock out was missing")
    assert GUIDANCE not in json.dumps(audit) and "Lock out was missing" not in json.dumps(audit)
    assert logging_sub.verify_audit_integrity()["intact"] is True


def test_with_one_log_per_replica_the_queue_reads_what_it_wrote(monkeypatch):
    monkeypatch.setenv("AUDIT_LOG_PER_INSTANCE", "true")
    monkeypatch.setenv("HOSTNAME", "backend-0")
    review_id = make_item()
    assert rq._path().name == "review_queue-backend-0.jsonl" and rq._path().exists()
    rq.decide(review_id, "confirm", "Nazir")
    assert rq.get_item(review_id)["status"] == "confirmed"


# --- what the engineer may see ---


def test_the_engineer_sees_the_state_and_the_outcome_but_nothing_internal():
    pending, corrected, rejected = make_item(run_id="a" * 12), make_item(run_id="b" * 12), make_item(run_id="c" * 12)
    rq.decide(corrected, "correct", "Nazir", "Lock out was missing", GUIDANCE, [SOURCE])
    rq.decide(rejected, "reject", "Nazir", "The answer cites the wrong section")
    assert rq.engineer_view(pending) == {"review_id": pending, "status": "pending", "message": rq.PENDING_MESSAGE}
    view = rq.engineer_view(corrected)
    assert view["status"] == "corrected" and view["message"] == rq.CORRECTED_MESSAGE
    assert view["corrected_guidance"] == GUIDANCE and view["evidence"] == [SOURCE]
    assert rq.engineer_view(rejected)["message"] == rq.REJECTED_MESSAGE
    assert rq.engineer_view(rejected)["reason"] == "The answer cites the wrong section"
    for review_id in (pending, corrected, rejected):
        text = json.dumps(rq.engineer_view(review_id))
        assert "Nazir" not in text and "reviewer" not in text and "Can I open the panel" not in text


def test_the_outcome_message_for_a_confirmation():
    review_id = make_item()
    rq.decide(review_id, "confirm", "Nazir")
    assert rq.engineer_view(review_id)["message"] == rq.CONFIRMED_MESSAGE


# --- monitoring numbers ---


def test_the_summary_with_no_items_is_empty_not_zero_division():
    summary = rq.summary()
    assert summary["opened"] == 0 and summary["override_rate"] is None and summary["median_seconds_to_decide"] is None
    assert summary["oldest_pending_age_s"] is None


def test_the_summary_reports_backlog_override_rate_and_time_to_decide():
    ids = [
        make_item(run_id=f"{i:012x}", trigger=t)
        for i, t in enumerate(["halted", "escalated", "low_confidence", "halted", "sample"])
    ]
    rq.decide(ids[0], "confirm", "Nazir")
    rq.decide(ids[1], "correct", "Nazir", "Lock out was missing", GUIDANCE, [SOURCE])
    rq.decide(ids[2], "reject", "Nazir", "The answer is not supported")
    later = datetime.now(timezone.utc) + timedelta(hours=2)
    summary = rq.summary(now=later)
    assert summary["opened"] == 5 and summary["pending"] == 2 and summary["decided"] == 3
    assert (summary["confirmed"], summary["corrected"], summary["rejected"]) == (1, 1, 1)
    assert summary["override_rate"] == 0.667  # two of the three decided were corrected or rejected
    assert summary["by_trigger"] == {"halted": 2, "escalated": 1, "low_confidence": 1, "sample": 1}
    assert 7100 <= summary["oldest_pending_age_s"] <= 7300
    assert summary["median_seconds_to_decide"] is not None


# --- the sample setting ---


@pytest.mark.parametrize("value, expected", [("0", False), ("", False), ("abc", False), ("1", True), ("5", True)])
def test_the_sample_rate_setting(monkeypatch, value, expected):
    monkeypatch.setenv("REVIEW_SAMPLE_RATE", value)
    assert all(rq.should_sample() is expected for _ in range(20))


def test_the_sample_is_a_share_of_answers(monkeypatch):
    monkeypatch.setenv("REVIEW_SAMPLE_RATE", "0.5")
    hits = sum(rq.should_sample() for _ in range(2000))
    assert 800 < hits < 1200
