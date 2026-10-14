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
