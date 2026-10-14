"""The human review API: the backend itself authenticates the reviewer, one decision per case (also under concurrent
requests), unsafe corrected guidance is refused, and the engineer's status endpoint shows only the outcome."""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import review_queue as rq

client = TestClient(app)
TOKEN = "reviewer-token-for-tests-0123456789"
HEADERS = {"X-Reviewer-Token": TOKEN, "X-Reviewer-Id": "Nazir Zakaria"}
GUIDANCE = "Power the equipment off and lock it out before opening the panel, as the manual requires."
SOURCE = "aem_thermal_station.pdf, p.19, 2.2.2 General Safety Guidelines"


@pytest.fixture(autouse=True)
def reviewer_token(monkeypatch):
    monkeypatch.setenv("REVIEWER_TOKEN", TOKEN)


def make_item(run_id="aaaaaaaaaaaa", trigger="halted"):
    result = rq.open_review(
        run_id=run_id,
        trigger=trigger,
        question="Can I open the panel while the equipment is powered?",
        answer="Request halted: opening the panel while powered is not permitted.",
        status="halted",
        equipment_model="thermal_station",
        specialist="safety",
        verdict="halt",
        hazards=["high_voltage"],
        ppe_required=["insulated gloves"],
        sources=[SOURCE],
        explanation={"confidence_level": "high", "citations": []},
        confidence=0.85,
        trace=["request", "planning", "safety"],
    )
    return result["review_id"]


def decide(review_id, body, headers=HEADERS):
    return client.post(f"/api/reviews/{review_id}/decision", json=body, headers=headers)


# --- authentication is the backend's job ---


@pytest.mark.parametrize(
    "method, path",
    [("GET", "/api/reviews"), ("GET", "/api/reviews/" + "a" * 32), ("POST", "/api/reviews/" + "a" * 32 + "/decision")],
)
def test_without_a_configured_token_every_reviewer_endpoint_is_off(monkeypatch, method, path):
    monkeypatch.delenv("REVIEWER_TOKEN")
    response = client.request(method, path, headers=HEADERS, json={"decision": "confirm"} if method == "POST" else None)
    assert response.status_code == 503  # no open mode, even with a token in the request


def test_an_empty_token_setting_is_the_same_as_none(monkeypatch):
    monkeypatch.setenv("REVIEWER_TOKEN", "")
    assert client.get("/api/reviews", headers=HEADERS).status_code == 503


@pytest.mark.parametrize("token", [None, "", "wrong", TOKEN + "x", TOKEN[:-1]])
def test_a_missing_or_wrong_token_is_refused(token):
    headers = {"X-Reviewer-Id": "Nazir"}
    if token is not None:
        headers["X-Reviewer-Token"] = token.encode("utf-8").decode("latin-1")
    response = client.get("/api/reviews", headers=headers)
    assert response.status_code == 401
    assert TOKEN not in response.text


@pytest.mark.parametrize("reviewer", [None, "", "   ", "<script>", "x" * 70])
def test_a_reviewer_id_is_required_and_must_be_plain(reviewer):
    headers = {"X-Reviewer-Token": TOKEN}
    if reviewer is not None:
        headers["X-Reviewer-Id"] = reviewer
    assert client.get("/api/reviews", headers=headers).status_code == 400


def test_a_token_with_non_ascii_characters_is_refused_without_an_error():
    """The comparison works on bytes, so an odd token cannot crash the check."""
    from fastapi import HTTPException

    from app.routes import reviews

    with pytest.raises(HTTPException) as refused:
        reviews.reviewer(x_reviewer_token="héllo-ünïcode", x_reviewer_id="Nazir")
    assert refused.value.status_code == 401


def test_a_refused_decision_records_nothing():
    review_id = make_item()
    response = decide(
        review_id, {"decision": "confirm"}, headers={"X-Reviewer-Token": "wrong", "X-Reviewer-Id": "Nazir"}
    )
    assert response.status_code == 401 and rq.get_item(review_id)["status"] == "pending"


# --- listing and reading ---


def test_the_list_shows_pending_cases_the_counts_and_which_pod_it_is():
    make_item("a" * 12)
    make_item("b" * 12)
    body = client.get("/api/reviews", headers=HEADERS).json()
    assert [i["status"] for i in body["items"]] == ["pending", "pending"]
    assert body["counts"]["pending"] == 2 and body["queue_instance"]


def test_the_list_can_show_decided_and_all_cases():
    first, second = make_item("a" * 12), make_item("b" * 12)
    decide(first, {"decision": "confirm"})
    assert [i["review_id"] for i in client.get("/api/reviews?status=decided", headers=HEADERS).json()["items"]] == [
        first
    ]
    assert len(client.get("/api/reviews?status=all", headers=HEADERS).json()["items"]) == 2
    assert client.get("/api/reviews?status=nonsense", headers=HEADERS).status_code == 422
    assert client.get("/api/reviews?limit=0", headers=HEADERS).status_code == 422
    assert second  # still pending


def test_one_case_can_be_read_in_full():
    review_id = make_item()
    item = client.get(f"/api/reviews/{review_id}", headers=HEADERS).json()
    assert item["question"].startswith("Can I open the panel") and item["hazards"] == ["high_voltage"]
    assert item["sources"] == [SOURCE] and item["status"] == "pending"


def test_an_unknown_or_malformed_id_is_refused():
    assert client.get("/api/reviews/" + "0" * 32, headers=HEADERS).status_code == 404
    assert client.get("/api/reviews/not-an-id", headers=HEADERS).status_code == 422
    assert client.get("/api/reviews/../../etc/passwd", headers=HEADERS).status_code in (404, 422)
    assert decide("0" * 32, {"decision": "confirm"}).status_code == 404


# --- deciding ---


def test_confirm_correct_and_reject_through_the_api():
    ids = [make_item(f"{i:012x}") for i in range(3)]
    assert decide(ids[0], {"decision": "confirm"}).json()["status"] == "confirmed"
    corrected = decide(
        ids[1],
        {"decision": "correct", "reason": "Lock out was missing", "corrected_text": GUIDANCE, "evidence": [SOURCE]},
    ).json()
    assert corrected["status"] == "corrected" and corrected["reviewer"] == "Nazir Zakaria"
    assert decide(ids[2], {"decision": "reject", "reason": "Wrong section cited"}).json()["status"] == "rejected"


def test_the_reviewer_is_the_authenticated_one_not_whatever_the_body_says():
    review_id = make_item()
    body = {"decision": "confirm", "reviewer": "Somebody Else"}
    item = decide(review_id, body).json()
    assert item["reviewer"] == "Nazir Zakaria"


def test_a_decided_case_cannot_be_decided_again():
    review_id = make_item()
    assert decide(review_id, {"decision": "confirm"}).status_code == 200
    again = decide(review_id, {"decision": "reject", "reason": "Changed my mind about it"})
    assert again.status_code == 409
    assert rq.get_item(review_id)["status"] == "confirmed"


def test_simultaneous_decision_requests_produce_exactly_one_decision():
    review_id = make_item()
    body = {"decision": "reject", "reason": "Both reviewers think it is wrong"}
    with ThreadPoolExecutor(max_workers=8) as pool:
        codes = list(pool.map(lambda _: decide(review_id, body).status_code, range(8)))
    assert sorted(codes) == [200] + [409] * 7
    assert len([e for e in rq._events() if e["type"] == "review_decided"]) == 1


@pytest.mark.parametrize(
    "body, fragment",
    [
        ({"decision": "reject"}, "reason of at least"),
        (
            {"decision": "correct", "reason": "Missing lock out", "corrected_text": GUIDANCE},
            "cite at least one manual source",
        ),
        ({"decision": "correct", "reason": "Missing lock out", "evidence": [SOURCE]}, "at least 20 characters"),
        (
            {
                "decision": "correct",
                "reason": "Faster",
                "corrected_text": "Bypass the safety interlock and keep working on it.",
                "evidence": [SOURCE],
            },
            "bypass or defeat a safety control",
        ),
        (
            {
                "decision": "correct",
                "reason": "Fix the text",
                "corrected_text": GUIDANCE + " <script>x</script>",
                "evidence": [SOURCE],
            },
            "response firewall",
        ),
    ],
)
def test_incomplete_or_unsafe_decisions_are_refused_with_the_reasons(body, fragment):
    review_id = make_item()
    response = decide(review_id, body)
    assert response.status_code == 422
    assert any(fragment in problem for problem in response.json()["detail"]["problems"])
    assert rq.get_item(review_id)["status"] == "pending"  # nothing was recorded


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"decision": "approve"},
        {"decision": "confirm", "evidence": ["a"] * 6},
        {"decision": "confirm", "reason": "x" * 1001},
    ],
)
def test_a_malformed_body_is_refused(body):
    review_id = make_item()
    assert decide(review_id, body).status_code == 422
    assert rq.get_item(review_id)["status"] == "pending"


# --- the engineer's status ---


def test_the_engineer_status_needs_no_token_and_shows_only_the_outcome():
    review_id = make_item()
    pending = client.get(f"/api/review-status/{review_id}").json()
    assert pending["status"] == "pending" and pending["message"] == rq.PENDING_MESSAGE
    decide(
        review_id,
        {"decision": "correct", "reason": "Missing lock out", "corrected_text": GUIDANCE, "evidence": [SOURCE]},
    )
    view = client.get(f"/api/review-status/{review_id}").json()
    assert view["status"] == "corrected" and view["corrected_guidance"] == GUIDANCE
    text = json.dumps(view)
    assert "Nazir" not in text and "Can I open the panel" not in text and "reviewer" not in text


def test_the_status_of_an_unknown_review_is_not_found():
    assert client.get("/api/review-status/" + "0" * 32).status_code == 404
    assert client.get("/api/review-status/short").status_code == 422


def test_the_status_endpoint_works_even_without_a_reviewer_token_configured(monkeypatch):
    review_id = make_item()
    monkeypatch.delenv("REVIEWER_TOKEN")
    assert client.get(f"/api/review-status/{review_id}").status_code == 200


# --- monitoring ---


def test_monitoring_reports_the_review_backlog_and_override_rate():
    first, second, third = make_item("a" * 12), make_item("b" * 12), make_item("c" * 12)
    decide(first, {"decision": "confirm"})
    decide(second, {"decision": "reject", "reason": "Wrong section cited"})
    review = client.get("/api/monitoring?verify=false").json()["review"]
    assert review["opened"] == 3 and review["pending"] == 1 and review["decided"] == 2
    assert review["override_rate"] == 0.5 and review["median_seconds_to_decide"] is not None
    assert third


def test_the_review_decisions_are_in_the_audit_chain_and_the_chain_is_intact():
    review_id = make_item()
    decide(review_id, {"decision": "confirm"})
    body = client.get("/api/monitoring").json()
    assert body["audit_integrity"]["intact"] is True
    from asa.components import logging_sub

    kinds = [r.get("type") for r in logging_sub._read_file(logging_sub.AUDIT_LOG)]
    assert "review_opened" in kinds and "review_decision" in kinds
