"""The shared OpenAI access layer: lazy creation, timeout and retries, output cap, per-request budget."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import openai
import pytest

from asa.components import llm

REPO_ROOT = Path(__file__).resolve().parents[2]


class FakeOpenAI:
    """Records how it was built and what it was asked, and reports a fixed token usage."""

    instances: list["FakeOpenAI"] = []
    tokens_per_call = 100

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls: list[dict] = []
        FakeOpenAI.instances.append(self)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, *args, **kwargs):
        self.calls.append(kwargs)
        total = FakeOpenAI.tokens_per_call
        usage = SimpleNamespace(total_tokens=total, prompt_tokens=total - 10, completion_tokens=10)
        return SimpleNamespace(usage=usage)


@pytest.fixture(autouse=True)
def fake_openai(monkeypatch):
    FakeOpenAI.instances = []
    FakeOpenAI.tokens_per_call = 100
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    for name in (
        "OPENAI_TIMEOUT_S",
        "OPENAI_CONNECT_TIMEOUT_S",
        "OPENAI_MAX_RETRIES",
        "OPENAI_MAX_OUTPUT_TOKENS",
        "OPENAI_REQUEST_TOKEN_BUDGET",
    ):
        monkeypatch.delenv(name, raising=False)
    llm.reset_client()
    yield
    llm.reset_client()


def ask():
    return llm.client.chat.completions.create(model="m", messages=[])


def test_client_is_created_only_on_first_use_and_only_once():
    assert FakeOpenAI.instances == []
    ask()
    ask()
    assert len(FakeOpenAI.instances) == 1


def test_timeout_and_retries_are_applied_with_defaults():
    ask()
    kwargs = FakeOpenAI.instances[0].kwargs
    assert kwargs["max_retries"] == llm.DEFAULT_MAX_RETRIES
    assert kwargs["timeout"].read == llm.DEFAULT_TIMEOUT_S
    assert kwargs["timeout"].connect == llm.DEFAULT_CONNECT_TIMEOUT_S


def test_timeout_and_retries_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_TIMEOUT_S", "15")
    monkeypatch.setenv("OPENAI_MAX_RETRIES", "5")
    ask()
    kwargs = FakeOpenAI.instances[0].kwargs
    assert kwargs["max_retries"] == 5
    assert kwargs["timeout"].read == 15


def test_invalid_environment_values_fall_back_to_defaults(monkeypatch):
    monkeypatch.setenv("OPENAI_MAX_RETRIES", "many")
    ask()
    assert FakeOpenAI.instances[0].kwargs["max_retries"] == llm.DEFAULT_MAX_RETRIES


def test_default_output_cap_is_added():
    ask()
    assert FakeOpenAI.instances[0].calls[0]["max_completion_tokens"] == llm.DEFAULT_MAX_OUTPUT_TOKENS


def test_a_cap_set_by_the_caller_is_kept():
    llm.client.chat.completions.create(model="m", messages=[], max_completion_tokens=50)
    llm.client.chat.completions.create(model="m", messages=[], max_tokens=60)
    first, second = FakeOpenAI.instances[0].calls
    assert first["max_completion_tokens"] == 50
    assert "max_completion_tokens" not in second and second["max_tokens"] == 60


def test_the_output_cap_can_be_turned_off(monkeypatch):
    monkeypatch.setenv("OPENAI_MAX_OUTPUT_TOKENS", "0")
    ask()
    assert "max_completion_tokens" not in FakeOpenAI.instances[0].calls[0]


def test_budget_stops_further_calls_once_used_up():
    with llm.request_budget(250) as counter:
        ask()
        ask()
        ask()  # 300 tokens used now, over the limit of 250
        with pytest.raises(llm.TokenBudgetExceeded):
            ask()
    assert counter["calls"] == 3
    assert counter["used"] == 300
    assert len(FakeOpenAI.instances[0].calls) == 3  # the fourth call never reached the model


def test_budget_of_zero_means_no_limit():
    with llm.request_budget(0):
        for _ in range(20):
            ask()
    assert len(FakeOpenAI.instances[0].calls) == 20


def test_budget_default_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_REQUEST_TOKEN_BUDGET", "100")
    with llm.request_budget() as counter:
        ask()
        with pytest.raises(llm.TokenBudgetExceeded):
            ask()
    assert counter["limit"] == 100


def test_each_request_gets_its_own_budget():
    with llm.request_budget(150):
        ask()
        ask()
        with pytest.raises(llm.TokenBudgetExceeded):
            ask()
    with llm.request_budget(150):
        ask()  # the earlier request's usage does not carry over


def test_calls_outside_a_budget_are_not_limited():
    for _ in range(5):
        ask()


def test_other_client_attributes_reach_the_real_client():
    FakeOpenAI.embeddings = "embeddings-api"
    assert llm.client.embeddings == "embeddings-api"


def test_importing_the_agents_needs_no_key_and_creates_no_client():
    """The point of the lazy client: `import` must work without credentials (CI, health checks)."""
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT), str(REPO_ROOT / "src")])
    code = (
        "import asa.agents.request, asa.agents.planning, asa.agents.safety, asa.agents.diagnostic\n"
        "from asa.components import llm\n"
        "print('client_created=%s rag_created=%s' % (llm._real_client is not None, llm._rag_agent is not None))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=REPO_ROOT)
    assert result.returncode == 0, result.stderr
    assert "client_created=False rag_created=False" in result.stdout


# --- the shared RAG agent proxy ---


class FakeRagAgent:
    retriever = "real-retriever"


def test_the_rag_agent_proxy_reads_and_writes_the_shared_instance(monkeypatch):
    """Tests (and code) that set `rag_agent.retriever` must change the agent that is really used."""
    shared = FakeRagAgent()
    monkeypatch.setattr(llm, "_rag_agent", shared)
    assert llm.rag_agent.retriever == "real-retriever"
    llm.rag_agent.retriever = "fake"
    assert shared.retriever == "fake"
    del llm.rag_agent.retriever
    assert shared.retriever == "real-retriever"  # back to the class attribute


def test_one_rag_agent_is_shared_by_every_module(monkeypatch):
    created = []

    class Counting(FakeRagAgent):
        def __init__(self):
            created.append(1)

    monkeypatch.setattr(llm, "_rag_agent", None)
    monkeypatch.setattr("asa.agents.rag.AgenticRAGAgent", Counting)
    first, second = llm.get_rag_agent(), llm.get_rag_agent()
    assert first is second and len(created) == 1


# --- the budget follows the request into worker threads ---


def test_calls_made_in_a_copied_context_count_against_the_request_budget():
    """The Safety Agent validates items in a thread pool inside a copied context: those calls must be counted."""
    import contextvars
    from concurrent.futures import ThreadPoolExecutor

    with llm.request_budget(limit=1000) as counter:
        context = contextvars.copy_context()
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda _: context.copy().run(ask), range(3)))
    assert counter["calls"] == 3 and counter["used"] == 300


def test_a_spent_budget_surfaces_as_unavailable_not_as_more_spending(monkeypatch):
    """run_graph wraps the whole request in a budget, so a runaway loop ends with an error the API reports."""
    from app.services import graph_runner

    class Loop:
        def stream(self, initial, config=None, stream_mode=None):
            while True:
                ask()
                yield "updates", {}

    monkeypatch.setattr(graph_runner, "_graph", lambda: Loop())
    monkeypatch.setenv("OPENAI_REQUEST_TOKEN_BUDGET", "250")
    with pytest.raises(llm.TokenBudgetExceeded):
        graph_runner.run_graph("question", "thermal_station")
    assert len(FakeOpenAI.instances[0].calls) == 3  # 100 + 100 + 100 tokens, then it stops


# --- what every call used: agent, model, input and output tokens, seconds ---


def test_every_call_is_recorded_with_its_name_model_tokens_and_time():
    with llm.request_budget(limit=0) as counter:
        llm.client.chat.completions.create(model="gpt-4o-mini", messages=[], name="assess-retrieval")
    assert counter["calls"] == 1 and counter["input"] == 90 and counter["output"] == 10 and counter["used"] == 100
    [entry] = counter["detail"]
    assert entry["call"] == "assess-retrieval" and entry["model"] == "gpt-4o-mini"
    assert entry["input"] == 90 and entry["output"] == 10 and entry["seconds"] >= 0


def test_a_call_without_a_name_is_labelled_with_the_module_and_function_that_made_it():
    def request_node():
        return llm.client.chat.completions.create(model="m", messages=[])

    with llm.request_budget(limit=0) as counter:
        request_node()
    assert counter["detail"][0]["call"].endswith(".request_node")


def test_calls_from_worker_threads_are_all_recorded_once():
    import contextvars
    from concurrent.futures import ThreadPoolExecutor

    with llm.request_budget(limit=0) as counter:
        context = contextvars.copy_context()
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda _: context.copy().run(ask), range(40)))
    assert counter["calls"] == 40 and len(counter["detail"]) == 40
    assert counter["input"] == 40 * 90 and counter["output"] == 400 and counter["used"] == 4000


def test_a_response_without_usage_is_counted_as_a_call_with_unknown_tokens(monkeypatch):
    monkeypatch.setattr(FakeOpenAI, "_create", lambda self, *a, **k: SimpleNamespace(usage=None))
    llm.reset_client()
    with llm.request_budget(limit=0) as counter:
        ask()
    assert counter["calls"] == 1 and counter["used"] == 0
    assert counter["detail"][0]["input"] is None


def test_run_graph_hands_the_usage_of_the_request_to_the_caller(monkeypatch):
    from app.services import graph_runner

    class Graph:
        def stream(self, initial, config=None, stream_mode=None):
            ask()
            ask()
            yield "values", {"request_status": "READY"}

    monkeypatch.setattr(graph_runner, "_graph", lambda: Graph())
    state = graph_runner.run_graph("question", "thermal_station")
    assert state["llm_usage"] == {"calls": 2, "input_tokens": 180, "output_tokens": 20}
    assert len(state["llm_calls"]) == 2


def test_the_token_use_of_a_request_reaches_the_audit_record(tmp_path, monkeypatch):
    import json

    from asa.components import logging_sub

    monkeypatch.setattr(logging_sub, "AUDIT_LOG", tmp_path / "audit.jsonl")
    monkeypatch.setattr(logging_sub, "LOG_DIR", tmp_path)
    usage = {"calls": 5, "input_tokens": 5000, "output_tokens": 300}
    logging_sub.end_run("run1", {"llm_usage": usage})
    record = json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert record["llm_usage"] == usage


# --- the spend cap: an estimated-dollars ledger that stops every call at a limit ---


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    path = tmp_path / "spend.json"
    monkeypatch.setenv("OPENAI_SPEND_LEDGER", str(path))
    monkeypatch.setenv("OPENAI_SPEND_CAP_USD", "")  # empty, so load_dotenv() cannot bring back a cap from .env
    return path


def test_the_cost_of_a_call_is_tokens_times_the_price_of_the_model():
    assert llm.estimate_cost("gpt-4o-mini", 1_000_000, 1_000_000) == pytest.approx(0.75)
    assert llm.estimate_cost("gpt-4.1", 1_000_000, 0) == pytest.approx(2.0)
    assert llm.estimate_cost("gpt-4o-mini", None, None) == 0.0


def test_a_dated_model_name_is_priced_like_its_family_and_an_unknown_one_conservatively():
    assert llm.estimate_cost("gpt-4o-mini-2024-07-18", 1_000_000, 0) == pytest.approx(0.15)
    assert llm.estimate_cost("some-new-model", 1_000_000, 0) == pytest.approx(llm.UNKNOWN_MODEL_PRICE[0])
    assert llm.estimate_cost(None, 1_000_000, 0) == pytest.approx(llm.UNKNOWN_MODEL_PRICE[0])


def test_without_a_cap_nothing_is_recorded_or_limited(ledger):
    for _ in range(3):
        ask()
    assert not ledger.exists()
    assert llm.spend_cap() is None


def test_each_call_adds_its_estimated_cost_to_the_ledger(ledger, monkeypatch):
    monkeypatch.setenv("OPENAI_SPEND_CAP_USD", "5")
    FakeOpenAI.tokens_per_call = 1_000_010  # 1,000,000 input and 10 output tokens
    llm.client.chat.completions.create(model="gpt-4o-mini", messages=[])
    llm.client.chat.completions.create(model="gpt-4o-mini", messages=[])
    state = llm.read_ledger()
    assert state["calls"] == 2 and state["estimated_usd"] == pytest.approx(0.3, abs=0.001)


def test_once_the_cap_is_reached_the_next_call_is_refused_before_it_is_sent(ledger, monkeypatch):
    monkeypatch.setenv("OPENAI_SPEND_CAP_USD", "0.20")
    FakeOpenAI.tokens_per_call = 1_000_010
    llm.client.chat.completions.create(model="gpt-4o-mini", messages=[])  # about $0.15
    llm.client.chat.completions.create(model="gpt-4o-mini", messages=[])  # about $0.30: now over the cap
    sent = len(FakeOpenAI.instances[0].calls)
    with pytest.raises(llm.SpendCapReached, match="reached the cap"):
        ask()
    assert len(FakeOpenAI.instances[0].calls) == sent  # nothing was sent


def test_the_ledger_is_shared_by_a_new_process_and_can_be_reset_or_topped_up(ledger, monkeypatch):
    monkeypatch.setenv("OPENAI_SPEND_CAP_USD", "1.00")
    llm.add_spend(0.60, calls=0)  # spend made before the ledger existed
    assert llm.read_ledger()["estimated_usd"] == pytest.approx(0.6)
    llm.reset_client()  # a new process would start here, with the file as the only memory
    assert llm.read_ledger()["estimated_usd"] == pytest.approx(0.6)
    llm.add_spend(0.5, calls=0)
    with pytest.raises(llm.SpendCapReached):
        ask()
    llm.reset_spend()
    ask()  # allowed again


def test_a_damaged_ledger_counts_as_empty_and_does_not_crash(ledger, monkeypatch):
    monkeypatch.setenv("OPENAI_SPEND_CAP_USD", "1")
    ledger.write_text("{not json", encoding="utf-8")
    assert llm.read_ledger() == {"estimated_usd": 0.0, "calls": 0}
    ask()  # still works, and the ledger is rewritten
    assert llm.read_ledger()["calls"] == 1


@pytest.mark.parametrize("value", ["", "0", "-1", "abc"])
def test_a_cap_that_is_not_a_positive_number_means_no_cap(ledger, monkeypatch, value):
    monkeypatch.setenv("OPENAI_SPEND_CAP_USD", value)
    assert llm.spend_cap() is None


def test_concurrent_calls_all_reach_the_ledger(ledger, monkeypatch):
    import contextvars
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setenv("OPENAI_SPEND_CAP_USD", "100")
    FakeOpenAI.tokens_per_call = 1_000_010
    context = contextvars.copy_context()
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(
            pool.map(
                lambda _: context.copy().run(llm.client.chat.completions.create, model="gpt-4o-mini", messages=[]),
                range(24),
            )
        )
    assert llm.read_ledger()["calls"] == 24


def test_the_spend_script_shows_adds_and_resets(ledger, monkeypatch, capsys):
    import importlib.util

    monkeypatch.setenv("OPENAI_SPEND_CAP_USD", "0.55")
    spec = importlib.util.spec_from_file_location("openai_spend", REPO_ROOT / "scripts" / "openai_spend.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    assert script.main(["--add", "0.30"]) == 0
    out = capsys.readouterr().out
    assert "$0.3000" in out and "room left: $0.2500" in out
    script.main(["--add", "0.30"])
    assert "CAP REACHED" in capsys.readouterr().out
    script.main(["--reset"])
    assert "$0.0000" in capsys.readouterr().out


# --- run_parallel and the diagnostic validation that uses it ---


def test_run_parallel_keeps_the_order_and_runs_at_the_same_time():
    import time

    started = time.monotonic()
    results = llm.run_parallel([lambda i=i: (time.sleep(0.2), i)[1] for i in range(6)])
    assert results == list(range(6))
    assert time.monotonic() - started < 0.8  # one after another would take 1.2 s


def test_run_parallel_raises_the_first_error_and_handles_no_tasks():
    def boom():
        raise ValueError("first")

    assert llm.run_parallel([]) == []
    with pytest.raises(ValueError, match="first"):
        llm.run_parallel([lambda: 1, boom, lambda: 3])


def test_run_parallel_carries_the_request_budget_into_the_workers():
    with llm.request_budget(limit=0) as counter:
        llm.run_parallel([ask for _ in range(5)])
    assert counter["calls"] == 5


def test_the_diagnostic_validation_runs_all_claims_at_once_and_keeps_their_order(monkeypatch):
    import threading
    import time
    from types import SimpleNamespace as NS

    from asa.agents import diagnostic
    from asa.graph.state import Chunk, RankedCause, Step

    chunk = Chunk(
        chunk_id="c1",
        doc_id="d",
        section_id="s",
        revision="0",
        equipment_model="thermal_station",
        text="evidence",
        score=1.0,
        source_file="f.pdf",
        page=1,
        section_title="t",
    )
    causes = [RankedCause(cause=f"cause {i}", likelihood=0.5, supporting_chunk_ids=["c1"]) for i in range(4)]
    steps = [Step(order=i, action=f"check {i}", supporting_chunk_ids=["c1"]) for i in range(4)]
    unsupported = [RankedCause(cause="no evidence", likelihood=0.5, supporting_chunk_ids=["missing"])]
    inside, peak, lock = [], [], threading.Lock()

    def create(**kwargs):
        text = kwargs["messages"][1]["content"]
        with lock:
            inside.append(1)
            peak.append(len(inside))
        time.sleep(0.15)
        with lock:
            inside.pop()
        odd = any(f"{word} {n}" in text for word in ("cause", "check") for n in (1, 3))
        return NS(choices=[NS(message=NS(content="UNSUPPORTED" if odd else "SUPPORTED"))])

    monkeypatch.setattr(diagnostic.client.chat.completions, "create", create)
    started = time.monotonic()
    valid_causes, valid_steps = diagnostic._validate_diagnostic_claims("symptom", [chunk], causes + unsupported, steps)
    assert time.monotonic() - started < 0.7  # 8 calls one after another would take 1.2 s
    assert max(peak) > 1
    assert [c.cause for c in valid_causes] == ["cause 0", "cause 2"]
    assert [s.action for s in valid_steps] == ["check 0", "check 2"]
