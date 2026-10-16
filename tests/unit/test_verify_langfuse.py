"""scripts/verify_langfuse.py: each check passes on good traces and FAILS when the traces are bad in the way that check
is about (no release stamp, a different release, no agents, no token counts, captured text). A fake Langfuse stands in."""

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("verify_langfuse", ROOT / "scripts" / "verify_langfuse.py")
vl = importlib.util.module_from_spec(spec)
sys.modules["verify_langfuse"] = vl
spec.loader.exec_module(vl)

COMMIT = "7795493226164f1c5c0e3e7b4e5e49b50a1b64a1"
SHORT = COMMIT[:12]
MASKED = "[content not captured]"


def trace(trace_id="t1", commit=SHORT, version=None, latency=4.2, stamp=True, **extra):
    metadata = (
        {"commit": commit, "prompts": "f64929ebb44b", "corpus": "29e675ad724f", "models": "gpt-4o-mini"}
        if stamp
        else {}
    )
    return {
        "id": trace_id,
        "name": "chat-request",
        "version": commit if version is None else version,
        "latency": latency,
        "metadata": metadata,
        "totalCost": 0.0012,
        "environment": "kubernetes",
        "input": None,
        "output": None,
        **extra,
    }


def chain(name, start):
    return {"type": "CHAIN", "name": name, "startTime": start}


def generation(tokens_in=700, tokens_out=20, model="gpt-4o-mini-2024-07-18", text=MASKED):
    return {
        "type": "GENERATION",
        "name": "assess",
        "model": model,
        "usageDetails": {"input": tokens_in, "output": tokens_out},
        "input": [{"role": text, "content": text}],
        "output": {"role": text, "content": text},
    }


GOOD_OBSERVATIONS = [
    chain("request", "2026-10-16T08:00:00Z"),
    chain("planning", "2026-10-16T08:00:01Z"),
    chain("safety", "2026-10-16T08:00:03Z"),
    chain("LangGraph", "2026-10-16T08:00:00Z"),
    generation(),
]


class FakeLangfuse:
    def __init__(self, traces, observations=None):
        self.traces, self.observations, self.paths = traces, observations or {}, []

    def get(self, path):
        self.paths.append(path)
        if path.startswith("/api/public/traces"):
            return {"data": self.traces}
        trace_id = path.split("traceId=")[1].split("&")[0]
        return {"data": self.observations.get(trace_id, GOOD_OBSERVATIONS)}


def run(traces, observations=None, expect=None, minimum=1, allow_content=False):
    fake = FakeLangfuse(traces, observations)
    summaries = [
        vl.summarise(t, vl.fetch_observations(fake.get, t["id"]))
        for t in vl.fetch_traces(fake.get, datetime.now(timezone.utc), 50)
    ]
    report = vl.Report()
    vl.verify(report, summaries, expect, minimum, allow_content)
    return report


# --- good traces pass ---


def test_good_traces_pass_every_check_and_nothing_fails():
    report = run([trace("a"), trace("b")], expect=COMMIT[:7], minimum=2)
    assert report.failed == [] and report.passed == 8


def test_a_trace_summary_names_the_agents_in_order_the_tokens_the_model_and_the_release():
    summary = vl.summarise(trace(), GOOD_OBSERVATIONS)
    assert summary["agents"] == ["request", "planning", "safety"]  # LangGraph is the wrapper, not an agent
    assert (
        summary["tokens_in"] == 700 and summary["tokens_out"] == 20 and summary["models"] == ["gpt-4o-mini-2024-07-18"]
    )
    assert summary["commit"] == SHORT and summary["release_fields"] == list(vl.RELEASE_FIELDS)
    assert summary["latency_s"] == 4.2 and summary["text_captured"] is False


def test_agents_are_listed_in_the_order_they_started_not_the_order_returned():
    shuffled = [
        chain("safety", "2026-10-16T08:00:03Z"),
        chain("request", "2026-10-16T08:00:00Z"),
        chain("planning", "2026-10-16T08:00:01Z"),
    ]
    assert vl.summarise(trace(), shuffled)["agents"] == ["request", "planning", "safety"]


# --- each check fails when its trace property is broken ---


@pytest.mark.parametrize(
    "traces, observations, kwargs, failing",
    [
        ([], {}, {}, "at least 1 trace(s) in the window"),
        ([trace(stamp=False)], {}, {}, "every trace carries the release stamp (commit, prompts, corpus, models)"),
        ([trace(version="deadbeef0000")], {}, {}, "the trace version equals the commit"),
        ([trace(commit="0000000aaaaa")], {}, {"expect": COMMIT[:7]}, "every trace is from the expected release"),
        (
            [trace()],
            {"t1": [chain("request", "2026-10-16T08:00:00Z")]},
            {},
            "agent execution is visible (at least two agents in order)",
        ),
        ([trace(latency=0)], {}, {}, "every trace has a latency"),
        (
            [trace()],
            {"t1": GOOD_OBSERVATIONS[:-1] + [generation(tokens_in=0, tokens_out=0)]},
            {},
            "token usage and the model are recorded for model calls",
        ),
        (
            [trace()],
            {"t1": GOOD_OBSERVATIONS[:-1] + [generation(model=None)]},
            {},
            "token usage and the model are recorded for model calls",
        ),
        (
            [trace()],
            {"t1": GOOD_OBSERVATIONS[:-1] + [generation(text="Can I bypass the safety interlock?")]},
            {},
            "no question or answer text is captured",
        ),
        ([trace(input="What is the power off procedure?")], {}, {}, "no question or answer text is captured"),
    ],
)
def test_the_check_fails_when_the_traces_are_bad_in_its_way(traces, observations, kwargs, failing):
    assert failing in run(traces, observations, **kwargs).failed


def test_one_bad_trace_among_good_ones_is_enough_to_fail_the_release_stamp_check():
    report = run([trace("a"), trace("b", stamp=False), trace("c")], minimum=3)
    assert report.failed == ["every trace carries the release stamp (commit, prompts, corpus, models)"]


def test_the_privacy_check_can_be_switched_off_when_content_capture_is_deliberately_on():
    leaky = {"t1": GOOD_OBSERVATIONS[:-1] + [generation(text="Can I bypass the safety interlock?")]}
    assert "no question or answer text is captured" not in run([trace()], leaky, allow_content=True).failed


# --- what counts as captured text ---


@pytest.mark.parametrize(
    "value, captured",
    [
        (None, False),
        ("", False),
        ([], False),
        (MASKED, False),
        ({"role": MASKED, "content": MASKED}, False),
        ("thermal_station", False),  # an identifier is not free text
        ({"question": MASKED, "equipment_model": "thermal_station"}, False),
        ("Can I bypass the safety interlock?", True),
        ({"content": "reset the controller"}, True),
        ([{"role": "user", "content": "How do I drain the line?"}], True),
        (42, False),
    ],
)
def test_only_free_text_counts_as_captured(value, captured):
    assert vl._text_captured(value) is captured


def test_two_commit_spellings_match_when_one_is_a_prefix_of_the_other():
    assert vl.same_commit(SHORT, COMMIT) and vl.same_commit(COMMIT[:7], SHORT)
    assert not vl.same_commit("", COMMIT) and not vl.same_commit("abc1234", COMMIT)


# --- reading Langfuse ---


def test_the_traces_asked_for_are_the_backends_request_traces_in_the_window_newest_first():
    fake = FakeLangfuse([])
    vl.fetch_traces(fake.get, datetime(2026, 10, 16, 8, 0, tzinfo=timezone.utc), 20)
    path = fake.paths[0]
    assert (
        "name=chat-request" in path
        and "fromTimestamp=2026-10-16T08%3A00%3A00Z" in path
        and "orderBy=timestamp.desc" in path
        and "limit=20" in path
    )
    vl.fetch_traces(fake.get, datetime.now(timezone.utc), 20, None)
    assert "name=" not in fake.paths[1]  # --all-traces


def test_the_keys_are_sent_as_basic_authentication_and_the_trace_id_is_encoded(monkeypatch):
    seen = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"data": []}'

    def fake_urlopen(request, timeout=None):
        seen["auth"], seen["url"] = request.get_header("Authorization"), request.full_url
        return Response()

    monkeypatch.setattr(vl.urllib.request, "urlopen", fake_urlopen)
    get = vl.make_getter("http://localhost:3000/", "pk-test", "sk-test")
    vl.fetch_observations(get, "a b&c")
    assert (
        seen["auth"].startswith("Basic ")
        and "traceId=a%20b%26c" in seen["url"]
        and seen["url"].startswith("http://localhost:3000/api/")
    )


# --- the command line ---


@pytest.fixture
def cli(monkeypatch, tmp_path):
    fake = FakeLangfuse([trace("a"), trace("b")])
    (tmp_path / ".env").write_text(
        "LANGFUSE_PUBLIC_KEY=pk-test-111\nLANGFUSE_SECRET_KEY=sk-test-222\nLANGFUSE_HOST=http://langfuse.invalid:3000\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(vl, "ROOT", tmp_path)
    monkeypatch.setattr(vl, "make_getter", lambda host, public, secret: (setattr(fake, "host", host), fake.get)[1])
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"):
        monkeypatch.setenv(name, "")
    return fake


def test_the_command_exits_zero_on_good_traces_and_never_prints_a_key(cli, capsys):
    assert vl.main(["--min-traces", "2", "--expect-commit", COMMIT[:7]]) == 0
    out = capsys.readouterr().out
    assert "RESULT 9 passed, 0 failed" in out and "pk-test-111" not in out and "sk-test-222" not in out
    assert cli.host == "http://langfuse.invalid:3000"  # the host from .env is used


def test_the_command_exits_one_when_a_check_fails(cli, capsys):
    assert vl.main(["--expect-commit", "0000000"]) == 1
    assert "FAIL every trace is from the expected release" in capsys.readouterr().out


def test_without_keys_the_command_says_so_and_fails(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(vl, "ROOT", tmp_path)
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert vl.main([]) == 1
    assert "LANGFUSE_PUBLIC_KEY" in capsys.readouterr().out


def test_an_unreachable_langfuse_is_a_failure_not_a_crash(monkeypatch, tmp_path, capsys):
    (tmp_path / ".env").write_text("LANGFUSE_PUBLIC_KEY=p\nLANGFUSE_SECRET_KEY=s\n", encoding="utf-8")
    monkeypatch.setattr(vl, "ROOT", tmp_path)

    def broken(host, public, secret):
        def get(path):
            raise vl.urllib.error.URLError("connection refused")

        return get

    monkeypatch.setattr(vl, "make_getter", broken)
    assert vl.main([]) == 1
    assert "FAIL Langfuse can be read with these keys" in capsys.readouterr().out
