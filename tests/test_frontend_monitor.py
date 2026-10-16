"""The monitoring page, its proxy and the navigation shared by the three pages. The frontend only forwards: the backend
checks the token."""

import io
import json
import urllib.error
from pathlib import Path

import pytest

from frontend import app as frontend_app

ROOT = Path(__file__).resolve().parents[1]
JS = ROOT / "frontend" / "static" / "js"
client = frontend_app.app.test_client()
OPERATOR = {"X-Monitor-Token": "tok-ops"}


class FakeResponse(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@pytest.fixture
def backend(monkeypatch):
    seen = {"requests": [], "reply": {"summary": {"runs": 0}}, "error": None}

    def fake_urlopen(request, timeout=None):
        seen["requests"].append(
            {
                "url": request.full_url,
                "method": request.get_method(),
                "headers": {k.lower(): v for k, v in request.header_items()},
            }
        )
        if seen["error"]:
            raise seen["error"]
        return FakeResponse(json.dumps(seen["reply"]).encode())

    monkeypatch.setattr(frontend_app.urllib_request, "urlopen", fake_urlopen)
    return seen


# --- the page ---


def test_the_page_is_served_with_a_sign_in_and_its_assets():
    page = client.get("/monitor").get_data(as_text=True)
    assert "Monitoring" in page and 'id="monitor-token"' in page and 'type="password"' in page
    assert "js/monitor.js" in page and "css/monitor.css" in page
    for path in ("static/js/monitor.js", "static/css/monitor.css", "static/css/nav.css", "templates/monitor.html"):
        assert (ROOT / "frontend" / path).is_file()
    assert client.get("/static/js/monitor.js").status_code == 200


def test_the_page_does_not_pretend_to_show_demo_data():
    page = client.get("/monitor").get_data(as_text=True) + (JS / "monitor.js").read_text(encoding="utf-8")
    assert "Demo data" not in page and "demo data" not in page


def test_the_page_says_it_shows_one_pod_and_no_text_of_any_question():
    page = client.get("/monitor").get_data(as_text=True)
    script = (JS / "monitor.js").read_text(encoding="utf-8")
    assert "no question or answer is shown" in page
    assert "only: each pod keeps its own logs" in script and "data.instance" in script


def test_the_langfuse_link_appears_only_for_an_http_address(monkeypatch):
    monkeypatch.delenv("LANGFUSE_URL", raising=False)
    assert "Langfuse" not in client.get("/monitor").get_data(as_text=True)
    monkeypatch.setenv("LANGFUSE_URL", "https://langfuse.example.org/project/1")
    page = client.get("/monitor").get_data(as_text=True)
    assert 'href="https://langfuse.example.org/project/1"' in page and 'rel="noopener noreferrer"' in page
    for bad in ("javascript:alert(1)", "data:text/html,<script>1</script>", "ftp://x/y", "//evil.example", "not a url"):
        monkeypatch.setenv("LANGFUSE_URL", bad)
        assert "Langfuse" not in client.get("/monitor").get_data(as_text=True), bad


# --- the proxy ---


def test_the_proxy_forwards_the_token_and_the_validated_query(backend):
    client.get("/api/monitor/data?limit=50&verify=false", headers=OPERATOR)
    sent = backend["requests"][0]
    assert sent["url"].endswith("/api/monitoring?limit=50&verify=false") and sent["method"] == "GET"
    assert sent["headers"]["x-monitor-token"] == "tok-ops"


def test_the_proxy_uses_safe_defaults(backend):
    client.get("/api/monitor/data", headers=OPERATOR)
    assert backend["requests"][0]["url"].endswith("/api/monitoring?limit=200&verify=true")


def test_the_proxy_does_not_invent_a_token(backend):
    client.get("/api/monitor/data")
    assert "x-monitor-token" not in backend["requests"][0]["headers"]  # no token in, none out: the backend refuses


@pytest.mark.parametrize("limit", ["0", "5001", "-1", "abc", "1;rm", "1.5", "", "99999999999999999999"])
def test_a_bad_limit_never_reaches_the_backend(backend, limit):
    assert client.get("/api/monitor/data", query_string={"limit": limit}, headers=OPERATOR).status_code == 400
    assert backend["requests"] == []


@pytest.mark.parametrize("verify", ["yes", "1", "TRUE", "true&limit=1", ""])
def test_a_bad_verify_value_never_reaches_the_backend(backend, verify):
    assert client.get("/api/monitor/data", query_string={"verify": verify}, headers=OPERATOR).status_code == 400
    assert backend["requests"] == []


@pytest.mark.parametrize("code", [401, 503])
def test_the_backends_refusals_reach_the_page_unchanged(backend, code):
    body = json.dumps({"detail": "refused"}).encode()
    backend["error"] = urllib.error.HTTPError("u", code, "err", {}, io.BytesIO(body))
    response = client.get("/api/monitor/data", headers=OPERATOR)
    assert response.status_code == code and response.get_json() == {"detail": "refused"}


def test_the_backends_data_reaches_the_page(backend):
    backend["reply"] = {"summary": {"runs": 7}, "instance": "backend-0", "alerts": []}
    assert client.get("/api/monitor/data", headers=OPERATOR).get_json()["summary"]["runs"] == 7


# --- the script ---


def test_the_script_never_puts_server_text_into_markup_and_keeps_the_token_in_this_tab_only():
    script = (JS / "monitor.js").read_text(encoding="utf-8")
    for banned in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "localStorage"):
        assert banned not in script, banned
    assert "textContent" in script and "sessionStorage" in script
    assert '"X-Monitor-Token"' in script and "/api/monitor/data" in script


def test_the_script_handles_a_wrong_token_an_unconfigured_server_and_an_outage():
    script = (JS / "monitor.js").read_text(encoding="utf-8")
    assert "res.status === 401" in script and "res.status === 503" in script
    assert "Could not reach the server" in script and "last ones received" in script
    assert "REFRESH_MS = 30000" in script


def test_the_overall_state_comes_only_from_what_the_backend_reported():
    script = (JS / "monitor.js").read_text(encoding="utf-8")
    body = script.split("function overall(data)")[1].split("function render")[0]
    assert "audit.intact === false" in body and "alerts" in body
    assert "System healthy" not in script  # the label says what was checked: no alerts, audit intact


# --- the shared navigation ---


@pytest.mark.parametrize(
    "path, current", [("/", "Assistant"), ("/review", "Expert review"), ("/monitor", "Monitoring")]
)
def test_every_page_links_to_the_other_two_and_marks_where_you_are(path, current):
    page = client.get(path).get_data(as_text=True)
    assert 'class="site-nav"' in page and 'href="/"' in page and 'href="/review"' in page and 'href="/monitor"' in page
    marked = [line for line in page.splitlines() if 'aria-current="page"' in line]
    assert len(marked) == 1 and current in marked[0]
    assert "css/nav.css" in page


def test_the_review_page_keeps_its_help_and_sign_in_after_the_navigation_change():
    page = client.get("/review").get_data(as_text=True)
    assert 'id="help-toggle"' in page and 'id="reviewer-token"' in page


def test_the_cards_do_not_show_a_misleading_zero_confidence_and_say_how_many_requests_used_the_model():
    script = (JS / "monitor.js").read_text(encoding="utf-8")
    assert "summary.confidence_scored_runs ? num(summary.avg_confidence, 2) : " in script  # a dash when none was scored
    assert '"Used the model"' in script and '"Stopped (halted)"' in script
