"""The reviewer page and the engineer's banner (Flask frontend). The frontend only forwards: the backend checks the token."""

import io
import json
from pathlib import Path

import pytest

from frontend import app as frontend_app

ROOT = Path(__file__).resolve().parents[1]
REVIEW_ID = "a" * 32
JS = ROOT / "frontend" / "static" / "js"


class FakeResponse(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@pytest.fixture
def backend(monkeypatch):
    """Record what the frontend sends to the backend and answer with a canned reply."""
    seen = {"requests": [], "reply": {"items": []}, "error": None}

    def fake_urlopen(request, timeout=None):
        seen["requests"].append(
            {
                "url": request.full_url,
                "method": request.get_method(),
                "headers": {k.lower(): v for k, v in request.header_items()},
                "body": json.loads(request.data) if request.data else None,
            }
        )
        if seen["error"]:
            raise seen["error"]
        return FakeResponse(json.dumps(seen["reply"]).encode())

    monkeypatch.setattr(frontend_app.urllib_request, "urlopen", fake_urlopen)
    return seen


client = frontend_app.app.test_client()
REVIEWER = {"X-Reviewer-Token": "tok-123", "X-Reviewer-Id": "Nazir"}


def test_the_reviewer_page_is_served_and_asks_for_a_sign_in():
    response = client.get("/review")
    page = response.get_data(as_text=True)
    assert response.status_code == 200 and "Expert review" in page
    assert 'id="reviewer-token"' in page and 'type="password"' in page
    assert "js/review.js" in page and "css/review.css" in page


def test_the_page_and_its_assets_exist():
    for path in ("static/js/review.js", "static/css/review.css", "templates/review.html"):
        assert (ROOT / "frontend" / path).is_file()
    assert client.get("/static/js/review.js").status_code == 200


def test_the_proxy_forwards_the_reviewer_headers_and_the_status_filter(backend):
    client.get("/api/review/items?status=decided", headers=REVIEWER)
    sent = backend["requests"][0]
    assert sent["url"].endswith("/api/reviews?status=decided") and sent["method"] == "GET"
    assert sent["headers"]["x-reviewer-token"] == "tok-123" and sent["headers"]["x-reviewer-id"] == "Nazir"


def test_the_proxy_does_not_invent_a_token(backend):
    client.get("/api/review/items")
    assert "x-reviewer-token" not in backend["requests"][0]["headers"]  # no token in, none out: the backend refuses


def test_the_proxy_passes_the_backends_refusals_through(backend):
    import urllib.error

    backend["error"] = urllib.error.HTTPError(
        "u", 401, "Unauthorized", {}, io.BytesIO(b'{"detail": "a valid X-Reviewer-Token is required"}')
    )
    response = client.get("/api/review/items", headers=REVIEWER)
    assert response.status_code == 401 and "X-Reviewer-Token" in response.get_json()["detail"]


@pytest.mark.parametrize(
    "code, body",
    [(409, b'{"detail": "this case has already been decided"}'), (422, b'{"detail": {"problems": ["x"]}}')],
)
def test_decision_errors_reach_the_page_unchanged(backend, code, body):
    import urllib.error

    backend["error"] = urllib.error.HTTPError("u", code, "err", {}, io.BytesIO(body))
    response = client.post(f"/api/review/items/{REVIEW_ID}/decision", json={"decision": "confirm"}, headers=REVIEWER)
    assert response.status_code == code and response.get_json() == json.loads(body)


def test_a_decision_is_forwarded_with_its_body_and_headers(backend):
    payload = {"decision": "correct", "reason": "Missing lock out", "corrected_text": "x" * 30, "evidence": ["p.19"]}
    client.post(f"/api/review/items/{REVIEW_ID}/decision", json=payload, headers=REVIEWER)
    sent = backend["requests"][0]
    assert sent["method"] == "POST" and sent["url"].endswith(f"/api/reviews/{REVIEW_ID}/decision")
    assert sent["body"] == payload and sent["headers"]["x-reviewer-token"] == "tok-123"


def test_a_single_case_is_read_through_the_proxy(backend):
    client.get(f"/api/review/items/{REVIEW_ID}", headers=REVIEWER)
    assert backend["requests"][0]["url"].endswith(f"/api/reviews/{REVIEW_ID}")


@pytest.mark.parametrize("bad", ["short", "../../etc/passwd", "A" * 32, "g" * 32, "a" * 33, "%2e%2e"])
def test_a_malformed_id_never_reaches_the_backend(backend, bad):
    for response in (
        client.get(f"/api/review/items/{bad}", headers=REVIEWER),
        client.post(f"/api/review/items/{bad}/decision", json={"decision": "confirm"}, headers=REVIEWER),
        client.get(f"/api/review/status/{bad}"),
    ):
        assert response.status_code == 404
    assert backend["requests"] == []


def test_an_unknown_status_filter_is_refused_before_the_backend(backend):
    assert client.get("/api/review/items?status=everything", headers=REVIEWER).status_code == 400
    assert backend["requests"] == []


def test_the_engineer_status_sends_no_reviewer_headers(backend):
    backend["reply"] = {"status": "pending"}
    response = client.get(f"/api/review/status/{REVIEW_ID}", headers=REVIEWER)
    sent = backend["requests"][0]
    assert response.get_json()["status"] == "pending"
    assert sent["url"].endswith(f"/api/review-status/{REVIEW_ID}")
    assert "x-reviewer-token" not in sent["headers"] and "x-reviewer-id" not in sent["headers"]


# --- the scripts ---


def test_the_reviewer_script_never_puts_server_text_into_markup():
    script = (JS / "review.js").read_text(encoding="utf-8")
    assert "innerHTML" not in script and "outerHTML" not in script and "insertAdjacentHTML" not in script
    assert "document.write" not in script and "eval(" not in script
    assert "textContent" in script


def test_the_reviewer_script_sends_the_token_to_the_backend_only_through_the_proxy():
    script = (JS / "review.js").read_text(encoding="utf-8")
    assert '"X-Reviewer-Token"' in script and "/api/review/items" in script
    assert "localStorage" not in script  # the token is kept in this tab only (sessionStorage)
    assert "sessionStorage" in script


def test_the_engineers_page_shows_a_banner_and_follows_the_outcome():
    script = (JS / "app.js").read_text(encoding="utf-8")
    assert "addReviewBanner(" in script and "/api/review/status/" in script
    assert "pending review by a qualified engineer" in script
    assert "Reviewed and confirmed by a qualified engineer." in script
    assert "who corrected this guidance" in script and "withdrew it" in script
    assert "innerHTML" not in script.split("function addReviewBanner")[1].split("function addResponseIndicators")[0]


def test_the_banner_wording_matches_the_backend():
    from app.services import review_queue

    script = (JS / "app.js").read_text(encoding="utf-8")
    for message in (
        review_queue.PENDING_MESSAGE,
        review_queue.CONFIRMED_MESSAGE,
        review_queue.CORRECTED_MESSAGE,
        review_queue.REJECTED_MESSAGE,
    ):
        assert message in script, message


# --- the reviewer help ---


def test_the_review_page_has_a_help_button_and_a_closed_help_panel():
    page = client.get("/review").get_data(as_text=True)
    assert 'id="help-toggle"' in page and 'aria-expanded="false"' in page and 'aria-controls="help"' in page
    panel = page.split('id="help"')[1].split("</section>")[0]
    assert "hidden" in page.split('id="help"')[1].split(">")[0]  # closed until the reviewer asks for it
    for heading in (
        "1. Why a case is here",
        "2. Understanding the assistant's decision",
        "3. Reviewing the evidence",
        "4. Making a decision",
        "5. Safety and accountability",
        "Before you record a decision",
    ):
        assert heading in panel, heading


def test_the_help_explains_the_three_decisions_without_overstating_them():
    panel = client.get("/review").get_data(as_text=True).split('id="help"')[1].split("</section>")[0]
    assert "does not authorise the engineer to proceed" in panel  # confirming a stop is not permission to act
    assert "Never use Correct to get around a mandatory safety prohibition" in panel
    assert "at least 5 characters" in panel  # matches the backend's rule for a reason
    assert "cite the manual" in panel
    assert "None recorded" in panel and "does not mean a citation was checked" in panel


def test_the_help_says_the_checklist_is_a_reminder_and_not_a_control():
    panel = client.get("/review").get_data(as_text=True).split('id="help"')[1].split("</section>")[0]
    assert panel.count('<input type="checkbox">') == 6
    assert "does not check these boxes or enforce them" in panel


def test_the_help_matches_what_the_backend_enforces():
    from app.services import review_queue

    panel = client.get("/review").get_data(as_text=True).split('id="help"')[1].split("</section>")[0]
    assert f"at least {review_queue.MIN_REASON_CHARS} characters" in panel  # the help quotes the backend's own limit


def test_every_hazard_the_rules_can_find_is_explained_in_the_page():
    """If a hazard category is added to the safety rules, the reviewer help must explain it."""
    import re

    from asa.guardrails.safety_rules import HARD_HALT_TRIGGERS

    script = (JS / "review.js").read_text(encoding="utf-8")
    block = script.split("const HAZARDS = {")[1].split("};")[0]
    explained = set(re.findall(r"^\s+(\w+):", block, flags=re.MULTILINE))
    assert set(HARD_HALT_TRIGGERS) | {"safety_control_defeat"} <= explained


def test_only_the_hazards_that_withhold_an_answer_are_described_as_escalate_only():
    from asa.guardrails.safety_rules import ESCALATE_ONLY_HAZARDS

    script = (JS / "review.js").read_text(encoding="utf-8")
    block = script.split("const HAZARDS = {")[1].split("};")[0]
    for line in block.splitlines():
        if "Escalate-only" in line:
            assert line.strip().split(":")[0] in ESCALATE_ONLY_HAZARDS, line


def test_the_case_view_explains_an_empty_citation_list_without_markup():
    script = (JS / "review.js").read_text(encoding="utf-8")
    assert "A safety rule can stop a request before the manual is searched" in script
    assert "innerHTML" not in script  # the explanations are plain text too
