"""The agent trace under an answer and the helpful / not helpful buttons: the frontend route that forwards a rating, and
the page script that shows the trace and sends the rating. The frontend only forwards: the backend decides."""

import io
import json
import re
import urllib.error
from pathlib import Path

import pytest

from frontend import app as frontend_app

ROOT = Path(__file__).resolve().parents[1]
JS = ROOT / "frontend" / "static" / "js"
client = frontend_app.app.test_client()
RUN = "0123456789ab"


class FakeResponse(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@pytest.fixture
def backend(monkeypatch):
    """Answer each backend call from a list of outcomes: an int is an HTTP error code, a dict is a 200 body."""
    seen = {"requests": [], "outcomes": [{"recorded": True}]}

    def fake_urlopen(request, timeout=None):
        seen["requests"].append(
            {
                "url": request.full_url,
                "method": request.get_method(),
                "body": json.loads(request.data) if request.data else None,
            }
        )
        outcome = seen["outcomes"][min(len(seen["requests"]) - 1, len(seen["outcomes"]) - 1)]
        if isinstance(outcome, int):
            raise urllib.error.HTTPError(
                "u", outcome, "err", {}, io.BytesIO(json.dumps({"detail": f"HTTP {outcome}"}).encode())
            )
        return FakeResponse(json.dumps(outcome).encode())

    monkeypatch.setattr(frontend_app.urllib_request, "urlopen", fake_urlopen)
    return seen


def rate(body):
    return client.post("/api/feedback", json=body)


# --- the feedback route ---


def test_a_rating_is_forwarded_exactly_as_the_backend_accepts_it(backend):
    response = rate({"run_id": RUN, "helpful": True})
    assert response.status_code == 200 and response.get_json() == {"recorded": True}
    [sent] = backend["requests"]
    assert sent["method"] == "POST" and sent["url"].endswith("/api/feedback")
    assert sent["body"] == {"run_id": RUN, "helpful": True}


def test_a_not_helpful_rating_with_a_comment_is_forwarded(backend):
    rate({"run_id": RUN, "helpful": False, "comment": "Cited the wrong section"})
    assert backend["requests"][0]["body"] == {"run_id": RUN, "helpful": False, "comment": "Cited the wrong section"}


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"helpful": True},
        {"run_id": RUN},
        {"run_id": "short", "helpful": True},
        {"run_id": "0123456789abc", "helpful": True},  # one character too long
        {"run_id": "0123456789AB", "helpful": True},  # not lower-case hex
        {"run_id": "../../etc/pass", "helpful": True},
        {"run_id": 123456789012, "helpful": True},
        {"run_id": RUN, "helpful": "true"},  # a string is not a boolean
        {"run_id": RUN, "helpful": 1},
        {"run_id": RUN, "helpful": None},
        {"run_id": RUN, "helpful": True, "comment": "x" * 501},
        {"run_id": RUN, "helpful": True, "comment": 5},
        [RUN, True],
    ],
)
def test_anything_the_backend_would_not_accept_is_turned_away_before_it(backend, body):
    assert rate(body).status_code == 400
    assert backend["requests"] == []


def test_a_body_that_is_not_json_is_refused(backend):
    assert client.post("/api/feedback", data="not json", content_type="text/plain").status_code == 400
    assert backend["requests"] == []


def test_extra_fields_are_not_passed_on(backend):
    rate({"run_id": RUN, "helpful": True, "role": "admin", "reviewer": "x"})
    assert backend["requests"][0]["body"] == {"run_id": RUN, "helpful": True}


def test_an_unknown_run_is_tried_again_so_another_pod_can_answer_and_then_succeeds(backend):
    backend["outcomes"] = [404, 404, {"recorded": True}]
    response = rate({"run_id": RUN, "helpful": True})
    assert response.status_code == 200 and len(backend["requests"]) == 3


def test_a_run_no_pod_knows_is_reported_as_404_after_the_attempts_are_used_up(backend):
    backend["outcomes"] = [404]
    assert rate({"run_id": RUN, "helpful": True}).status_code == 404
    assert len(backend["requests"]) == frontend_app.FEEDBACK_ATTEMPTS


@pytest.mark.parametrize("code", [422, 503])
def test_other_backend_errors_are_not_retried_and_reach_the_page(backend, code):
    backend["outcomes"] = [code]
    response = rate({"run_id": RUN, "helpful": True})
    assert response.status_code == code and len(backend["requests"]) == 1


def test_the_route_accepts_post_only():
    assert client.get("/api/feedback").status_code == 405


def test_the_run_id_pattern_matches_the_backends_own_schema():
    from app.schemas.feedback import FeedbackRequest

    assert FeedbackRequest.model_fields["run_id"].metadata[0].pattern == frontend_app.RUN_ID.pattern


# --- the page script ---


SCRIPT = (JS / "app.js").read_text(encoding="utf-8")


def function_body(name):
    start = SCRIPT.index(f"function {name}(")
    return SCRIPT[start : SCRIPT.index("\n}\n", start)]


def test_the_trace_and_feedback_code_never_puts_server_text_into_markup():
    for name in ("addAgentTrace", "addTraceList", "addFeedback", "stepDetail", "traceSteps"):
        assert not re.search(r"innerHTML|outerHTML|insertAdjacentHTML|document\.write|eval\(", function_body(name)), (
            name
        )
    assert "innerHTML" not in SCRIPT.split("function addReviewBanner")[1].split("function addResponseIndicators")[0]


def test_every_agent_the_graph_can_emit_has_a_short_name_and_a_role_in_the_page():
    graph = (ROOT / "src" / "asa" / "graph" / "orchestrator.py").read_text(encoding="utf-8")
    nodes = set(re.findall(r'add_node\(\s*"(\w+)"', graph))
    assert nodes, "no graph nodes found: the pattern needs updating"
    short = set(
        re.findall(r"^\s+(\w+): \"", SCRIPT.split("const AGENT_SHORT = {")[1].split("};")[0], flags=re.MULTILINE)
    )
    role = set(re.findall(r"^\s+(\w+): \"", SCRIPT.split("const AGENT_ROLE = {")[1].split("};")[0], flags=re.MULTILINE))
    assert nodes <= short and nodes <= role, (nodes - short, nodes - role)


def test_the_trace_is_shown_under_the_answer_and_listed_inside_the_details():
    body = function_body("appendAssistantMessage")
    assert (
        body.index("addAgentTrace(")
        < body.index("addEvidenceDetails(")
        < body.index("addReviewBanner(")
        < body.index("addFeedback(")
    )
    assert "addTraceList(" in function_body("addEvidenceDetails")
    assert "hasTrace" in function_body("addEvidenceDetails")  # a trace alone is enough to show the details


def test_a_request_stopped_by_a_rule_says_so_instead_of_showing_an_empty_trace():
    assert "Stopped by a safety rule before any agent ran." in function_body("addAgentTrace")


def test_feedback_is_offered_only_for_answers_with_a_run_id_and_saved_with_the_conversation():
    body = function_body("addFeedback")
    assert "!data.run_id" in body and "FEEDBACK_STATUSES" in body
    assert '"/api/feedback"' in body and "saveConversations()" in body and "data.feedback" in body
    assert "Could not save your feedback" in body
    assert 'FEEDBACK_STATUSES = ["ok", "halted", "escalated"]' in SCRIPT  # not clarifications, blocks or errors


def test_the_feedback_buttons_are_real_buttons_with_a_live_status_for_screen_readers():
    body = function_body("addFeedback")
    assert 'yes.type = "button"' in body and 'no.type = "button"' in body
    assert '"role"' in body and '"status"' in body and "aria-pressed" in body


def test_the_styles_exist_for_the_new_elements():
    css = (ROOT / "frontend" / "static" / "css" / "app.css").read_text(encoding="utf-8")
    for selector in (".trace-line", ".trace-chip", ".feedback-row", ".feedback-button", ".feedback-note"):
        assert selector in css, selector
