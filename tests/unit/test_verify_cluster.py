"""scripts/verify_cluster.py: each check passes on a healthy deployment and FAILS when the deployment is broken in the way
that check is about. A small fake stands in for the cluster; nothing real is contacted."""

import importlib.util
import json
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("verify_cluster", ROOT / "scripts" / "verify_cluster.py")
vc = importlib.util.module_from_spec(spec)
sys.modules["verify_cluster"] = vc
spec.loader.exec_module(vc)

MONITOR, REVIEWER, VERSION = "monitor-token-AAA", "reviewer-token-BBB", "version-token-CCC"
COMMIT = "7795493226164f1c5c0e3e7b4e5e49b50a1b64a1"


def page(label, help_button=False):
    marks = {"Assistant": "/", "Expert review": "/review", "Monitoring": "/monitor"}
    current = ' aria-current="page"'
    newline = chr(10)
    links = newline.join(
        '<a href="' + href + '"' + (current if name == label else "") + ">" + name + "</a>"
        for name, href in marks.items()
    )
    help_html = '<button id="help-toggle">Reviewer help</button>' if help_button else ""
    return '<nav class="site-nav">' + newline + links + newline + "</nav>" + help_html


class FakeDeployment:
    """The frontend as a browser sees it. Flags break one behaviour each."""

    def __init__(self, **broken):
        self.broken = broken
        self.cases = {}
        self.asked = 0

    def call(self, method, path, body=None, headers=None):
        headers = headers or {}
        route = path.split("?")[0]
        if method == "GET" and route in ("/", "/review", "/monitor"):
            label = {"/": "Assistant", "/review": "Expert review", "/monitor": "Monitoring"}[route]
            if self.broken.get("no_nav") and route == "/monitor":
                return 200, "<html>no navigation</html>"
            return 200, page(label, help_button=route == "/review" and not self.broken.get("no_help"))
        if route == "/api/monitor/data":
            if "limit=0" in path:
                return (200, {}) if self.broken.get("no_validation") else (400, {"detail": "limit"})
            allowed = headers.get("X-Monitor-Token") == MONITOR or self.broken.get("open_monitoring")
            if not allowed:
                return 401, {"detail": "token"}
            audit = {"intact": not self.broken.get("broken_chain"), "checked": 150}
            data = {
                "summary": {},
                "alerts": [{"name": "blocked_rate"}],
                "audit_integrity": audit,
                "review": {},
                "spend": {"estimated_usd": 0.0, "cap_usd": 0.25},
                "instance": "backend-0",
            }
            if self.broken.get("missing_fields"):
                del data["spend"]
            return 200, data
        if route == "/version":
            if headers.get("X-Version-Token") != VERSION:
                return 404, {}
            return 200, {"git": {"commit": "unknown" if self.broken.get("unknown_commit") else COMMIT}}
        if route == "/api/ask":
            return self.ask(body)
        if route.startswith("/api/review/status/"):
            case = self.cases.get(route.rsplit("/", 1)[1])
            return (200, case["view"]) if case else (404, {})
        if route.startswith("/api/review/items"):
            return self.review(method, route, body, headers)
        return 404, {}

    def ask(self, body):
        message = body["message"]
        if not message:
            return 422, {}
        if message == vc.PROBE:
            if self.broken.get("probe_answered"):
                return 200, {"status": "ok"}
            if self.broken.get("probe_unavailable"):
                return 503, {"status": "unavailable"}
            return 400, {"status": "blocked"}
        case = uuid.uuid4().hex
        self.cases[case] = {"decided": False, "view": {"status": "pending"}}
        if self.broken.get("not_halted"):
            return 200, {"status": "ok"}
        return 200, {"status": "halted", "review": {"review_id": case, "status": "pending"}}

    def review(self, method, route, body, headers):
        if headers.get("X-Reviewer-Token") != REVIEWER and not self.broken.get("open_review"):
            return 401, {}
        if method == "GET":
            return 200, {
                "items": [
                    {"review_id": c, "status": "pending" if not v["decided"] else "decided"}
                    for c, v in self.cases.items()
                ]
            }
        case = route.split("/")[-2]
        state = self.cases[case]
        if state["decided"] and not self.broken.get("second_decision_allowed"):
            return 409, {}
        decision = body["decision"]
        if decision == "correct":
            text = body.get("corrected_text", "")
            if "Bypass" in text and not self.broken.get("bypass_accepted"):
                return 422, {}
            if not body.get("evidence") and not self.broken.get("no_source_needed"):
                return 422, {}
            state["view"] = {"status": "corrected", "corrected_guidance": text}
        elif decision == "reject":
            if not body.get("reason"):
                return 422, {}
            state["view"] = {"status": "rejected"}
        else:
            state["view"] = {
                "status": "confirmed",
                **({"corrected_guidance": "x"} if self.broken.get("confirm_adds_guidance") else {}),
            }
        state["decided"] = True
        return 200, {}


def run(check, fake, *args):
    report = vc.Report()
    check(report, fake.call, *args)
    return report


# --- a healthy deployment passes every check ---


def test_a_healthy_deployment_passes_every_check_and_nothing_fails():
    fake = FakeDeployment()
    report = vc.Report()
    vc.check_pages(report, fake.call)
    vc.check_monitoring(report, fake.call, MONITOR)
    vc.check_version(report, fake.call, COMMIT[:7], VERSION)
    vc.check_injection_is_blocked(report, fake.call)
    created = vc.check_hitl(report, fake.call, REVIEWER)
    assert report.failed == [] and report.passed == 27 and len(created) == 3


# --- each check fails when its behaviour breaks ---


@pytest.mark.parametrize(
    "check, args, broken, failing",
    [
        (vc.check_pages, (), {"no_nav": True}, "/monitor is served with the shared navigation"),
        (vc.check_pages, (), {"no_help": True}, "the reviewer help is on the review page"),
        (vc.check_monitoring, (MONITOR,), {"open_monitoring": True}, "monitoring without a token is refused"),
        (vc.check_monitoring, (MONITOR,), {"open_monitoring": True}, "monitoring with a wrong token is refused"),
        (vc.check_monitoring, (MONITOR,), {"no_validation": True}, "a bad limit is refused by the frontend"),
        (vc.check_monitoring, (MONITOR,), {"broken_chain": True}, "the audit hash chain is intact"),
        (
            vc.check_monitoring,
            (MONITOR,),
            {"missing_fields": True},
            "monitoring with the right token returns the numbers",
        ),
        (vc.check_version, (COMMIT[:7], VERSION), {"unknown_commit": True}, "the build reports its commit"),
        (
            vc.check_injection_is_blocked,
            (),
            {"probe_answered": True},
            "a prompt-injection attempt is blocked (not answered, not just unavailable)",
        ),
        (
            vc.check_injection_is_blocked,
            (),
            {"probe_unavailable": True},
            "a prompt-injection attempt is blocked (not answered, not just unavailable)",
        ),
        (
            vc.check_hitl,
            (REVIEWER,),
            {"not_halted": True},
            "a request to defeat a safety control is halted and opens a pending case",
        ),
        (vc.check_hitl, (REVIEWER,), {"open_review": True}, "the review list without a token is refused"),
        (
            vc.check_hitl,
            (REVIEWER,),
            {"bypass_accepted": True},
            "guidance that tells the engineer to bypass a control is refused",
        ),
        (vc.check_hitl, (REVIEWER,), {"no_source_needed": True}, "guidance without a manual source is refused"),
        (
            vc.check_hitl,
            (REVIEWER,),
            {"second_decision_allowed": True},
            "a second decision on the same case is refused",
        ),
        (
            vc.check_hitl,
            (REVIEWER,),
            {"confirm_adds_guidance": True},
            "confirm leaves the stop a stop and adds no guidance",
        ),
    ],
)
def test_the_check_fails_when_the_deployment_is_broken_in_its_way(check, args, broken, failing):
    report = run(check, FakeDeployment(**broken), *args)
    assert failing in report.failed, (failing, report.failed)


def test_a_wrong_commit_is_a_failure_and_an_unreadable_version_is_a_skip_not_a_pass():
    report = run(vc.check_version, FakeDeployment(), "0000000", VERSION)
    assert "the running commit is the expected one" in report.failed
    skipped = run(vc.check_version, FakeDeployment(), None, "wrong-token")
    assert skipped.skipped == ["the running commit"] and skipped.failed == []


def test_checks_that_need_a_token_are_skipped_not_passed_when_there_is_none():
    assert run(vc.check_monitoring, FakeDeployment(), None).skipped == ["monitoring with the right token"]
    report = run(vc.check_hitl, FakeDeployment(), None)
    assert report.skipped == ["the HITL flow"] and report.passed == 0


# --- the rate limit ---


def test_the_rate_limit_check_passes_when_a_429_with_a_wait_time_appears():
    count = {"n": 0}

    def call(method, path, body=None, headers=None):
        count["n"] += 1
        return (429, {"retry_after_s": 42}) if count["n"] > 30 else (400, {"status": "blocked"})

    report = vc.Report()
    vc.check_rate_limit(report, call)
    assert report.failed == [] and report.passed == 1


def test_the_rate_limit_check_fails_when_the_limit_never_answers_429():
    report = vc.Report()
    vc.check_rate_limit(report, lambda *a, **k: (400, {"status": "blocked"}), ceiling=40)
    assert len(report.failed) == 1


# --- the pods ---


def pod(name, ready=True, image_id="ghcr.io/o/asa-backend@sha256:" + "a" * 64, restarts=0):
    return {
        "metadata": {"name": name},
        "status": {"containerStatuses": [{"ready": ready, "imageID": image_id, "restartCount": restarts}]},
    }


def test_ready_pods_on_registry_digests_pass():
    report = vc.Report()
    vc.check_pods(report, {"items": [pod("backend-0"), pod("backend-1")]}, "backend", "sha256:" + "a" * 64)
    assert report.failed == [] and report.passed == 7


def test_a_pod_that_is_not_ready_a_locally_built_image_or_the_wrong_digest_each_fail():
    report = vc.Report()
    vc.check_pods(report, {"items": [pod("b-0", ready=False)]}, "backend")
    assert report.failed == ["b-0 is ready"]
    report = vc.Report()
    vc.check_pods(report, {"items": [pod("b-0", image_id="sha256:" + "f" * 64)]}, "backend")
    assert report.failed == ["b-0 runs an image identified by a registry digest"]
    report = vc.Report()
    vc.check_pods(report, {"items": [pod("b-0")]}, "backend", "sha256:" + "b" * 64)
    assert report.failed == ["b-0 runs the expected digest"]


def test_no_pods_at_all_is_a_failure():
    report = vc.Report()
    vc.check_pods(report, {"items": []}, "backend")
    assert report.failed == ["backend: pods exist"]


# --- tokens and the command line ---


def test_the_env_file_is_read_without_comments_or_quotes(tmp_path):
    env = tmp_path / ".env"
    env.write_text('# comment\nREVIEWER_TOKEN="abc123"\n\nMONITOR_TOKEN=xyz\nBAD LINE\nEMPTY=\n', encoding="utf-8")
    assert vc.read_env_file(env) == {"REVIEWER_TOKEN": "abc123", "MONITOR_TOKEN": "xyz", "EMPTY": ""}
    assert vc.read_env_file(tmp_path / "missing") == {}


def test_no_token_is_ever_printed_even_when_checks_fail(capsys):
    fake = FakeDeployment(open_monitoring=True, bypass_accepted=True, second_decision_allowed=True)
    report = vc.Report()
    vc.check_monitoring(report, fake.call, MONITOR)
    vc.check_hitl(report, fake.call, REVIEWER)
    out = capsys.readouterr().out
    for secret in (MONITOR, REVIEWER, VERSION):
        assert secret not in out
    assert "FAIL" in out


class Result:
    def __init__(self, stdout="", code=0):
        self.stdout, self.returncode = stdout, code


@pytest.fixture
def cluster(monkeypatch, tmp_path):
    state = {"replicas": "1", "fake": FakeDeployment()}
    pods = json.dumps({"items": [pod("backend-0")]})

    def kubectl(*args):
        if "statefulset" in args:
            return Result(state["replicas"])
        if "secret" in args:
            return Result("")
        return Result(pods)

    class Process:
        def terminate(self):
            pass

    (tmp_path / ".env").write_text(f"MONITOR_TOKEN={MONITOR}\nREVIEWER_TOKEN={REVIEWER}\n", encoding="utf-8")
    monkeypatch.setattr(vc, "ROOT", tmp_path)
    monkeypatch.setattr(vc, "kubectl", kubectl)
    monkeypatch.setattr(vc, "forward", lambda *a, **k: Process())
    monkeypatch.setattr(vc, "make_caller", lambda base: state["fake"].call)
    monkeypatch.setattr(vc, "secret_value", lambda key: VERSION if key == "VERSION_TOKEN" else None)
    return state


def test_main_runs_everything_with_one_replica_and_exits_zero_when_all_pass(cluster, capsys):
    assert vc.main([]) == 0
    out = capsys.readouterr().out
    assert "RESULT" in out and " 0 failed" in out and "SKIP" not in out and "opened and decided 3 review cases" in out


def test_main_skips_the_hitl_flow_with_more_than_one_replica_and_says_why(cluster, capsys):
    cluster["replicas"] = "2"
    assert vc.main([]) == 0
    out = capsys.readouterr().out
    assert "SKIP the HITL flow" in out and "2 replicas" in out and "runbook 7A" in out
    assert not cluster["fake"].cases  # nothing was opened


def test_main_can_leave_the_review_queue_alone_on_request(cluster, capsys):
    assert vc.main(["--skip-hitl"]) == 0
    assert not cluster["fake"].cases and "--skip-hitl" in capsys.readouterr().out


def test_main_exits_one_when_a_check_fails(cluster, capsys):
    cluster["fake"] = FakeDeployment(open_monitoring=True)
    assert vc.main(["--skip-hitl"]) == 1
    assert "FAIL monitoring without a token is refused" in capsys.readouterr().out


def test_main_compares_the_expected_commit(cluster, capsys):
    assert vc.main(["--skip-hitl", "--expect-commit", "0000000"]) == 1
    assert "FAIL the running commit is the expected one" in capsys.readouterr().out
