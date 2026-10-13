"""Unit tests for the Agentic RAG agent (junior/senior retrieval loop) and its graph node.

No network, no OpenAI key, no Chroma index: the retriever and the OpenAI client are replaced
by deterministic fakes.
"""

import json
from types import SimpleNamespace

import pytest

import asa.agents.rag as rag
from asa.graph.state import Chunk


EQUIPMENT = "thermal_station"
QUESTION = "How do I replace the coolant pump?"


def _chunk(chunk_id="c1", page=10, title="4.1 Pump", equipment=EQUIPMENT, text="Step 1: power off.", score=0.5):
    return Chunk(
        chunk_id=chunk_id,
        doc_id="aem_thermal_station",
        section_id="4.1",
        revision="0",
        equipment_model=equipment,
        text=text,
        score=score,
        source_file="aem_thermal_station.pdf",
        page=page,
        section_title=title,
    )


class FakeRetriever:
    """Stands in for AEMHybridRetriever; returns canned chunks per query and records calls."""

    def __init__(self, by_query=None, default=None):
        self.by_query = by_query or {}
        self.default = default if default is not None else []
        self.calls = []

    def hybrid_search(self, query, equipment_model, k=5):
        self.calls.append({"query": query, "equipment_model": equipment_model, "k": k})
        return list(self.by_query.get(query, self.default))


def _msg(content=None, tool_calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))])


def _tool_call(arguments, name="search_service_documents"):
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return [SimpleNamespace(function=SimpleNamespace(name=name, arguments=raw))]


class FakeClient:
    """Fake OpenAI client. Replies are queued per stage, stage detected from the system prompt."""

    STAGES = (
        ("Junior Retrieval Agent for an", "junior"),
        ("evaluate whether retrieved", "assess"),
        ("reformulate AEM", "reformulate"),
        ("You are the Senior Retrieval Agent", "senior"),
        ("generate grounded", "answer"),
    )

    def __init__(self, **queues):
        self.queues = {stage: list(replies) for stage, replies in queues.items()}
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        system = kwargs["messages"][0]["content"]
        stage = next(name for marker, name in self.STAGES if marker in system)
        self.calls.append((stage, kwargs))
        queue = self.queues.get(stage)
        if not queue:
            raise AssertionError(f"unexpected extra model call at stage {stage!r}")
        return queue.pop(0)

    def stage_calls(self, stage):
        return [kwargs for name, kwargs in self.calls if name == stage]


def _make_agent(monkeypatch, retriever=None, **queues):
    monkeypatch.setattr(rag, "AEMHybridRetriever", lambda *a, **k: retriever or FakeRetriever())
    agent = rag.AgenticRAGAgent()
    agent.client = FakeClient(**queues)
    return agent


def _search(query):
    return _msg(tool_calls=_tool_call({"query": query}))


def _good():
    return _msg("GOOD")


def _poor():
    return _msg("POOR")


# ----------------------------------------------------------------------
# Tool definition / execution
# ----------------------------------------------------------------------


def test_tool_definition_requires_query_and_equipment_and_forbids_extras(monkeypatch):
    """The search tool schema makes query and equipment_model mandatory and rejects extra fields."""
    agent = _make_agent(monkeypatch)
    function = agent.get_tool_definitions()[0]["function"]

    assert function["name"] == "search_service_documents"
    assert function["parameters"]["required"] == ["query", "equipment_model"]
    assert function["parameters"]["additionalProperties"] is False
    assert set(function["parameters"]["properties"]["equipment_model"]["enum"]) == {
        "thermal_station",
        "thermal_retrofit_1kw",
    }


def test_execute_tool_returns_chunk_dicts_and_passes_arguments_to_retriever(monkeypatch):
    """execute_tool turns Chunks into the dict shape the loop expects and forwards query/equipment/k."""
    retriever = FakeRetriever(default=[_chunk(score=0.42)])
    agent = _make_agent(monkeypatch, retriever)

    results = agent.execute_tool(
        "search_service_documents",
        {"query": "coolant pump", "equipment_model": EQUIPMENT, "k": "3"},
    )

    assert retriever.calls == [{"query": "coolant pump", "equipment_model": EQUIPMENT, "k": 3}]
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["rrf_score"] == 0.42
    assert results[0]["source_file"] == "aem_thermal_station.pdf"
    assert results[0]["page"] == 10


@pytest.mark.parametrize(
    "tool_name, arguments, message",
    [
        ("search_service_documents", {"equipment_model": EQUIPMENT}, "requires query"),
        ("search_service_documents", {"query": "x"}, "requires equipment_model"),
        ("delete_everything", {"query": "x", "equipment_model": EQUIPMENT}, "Unknown tool"),
    ],
)
def test_execute_tool_rejects_missing_arguments_and_unknown_tools(monkeypatch, tool_name, arguments, message):
    """Missing query/equipment and unknown tool names raise instead of searching."""
    retriever = FakeRetriever()
    agent = _make_agent(monkeypatch, retriever)

    with pytest.raises(ValueError, match=message):
        agent.execute_tool(tool_name, arguments)

    assert retriever.calls == []


# ----------------------------------------------------------------------
# Junior search: equipment enforced by the application
# ----------------------------------------------------------------------


def test_junior_search_forces_tool_choice_and_uses_model_query(monkeypatch):
    """The junior model must call the search tool (tool_choice forced) and its query is what is searched."""
    retriever = FakeRetriever(default=[_chunk()])
    agent = _make_agent(monkeypatch, retriever, junior=[_search("coolant pump replacement")])

    out = agent.junior_search(QUESTION, EQUIPMENT, k=4)

    call = agent.client.stage_calls("junior")[0]
    assert call["tool_choice"] == {"type": "function", "function": {"name": "search_service_documents"}}
    assert call["tools"] == agent.get_tool_definitions()
    assert call["temperature"] == 0
    assert out["search_query"] == "coolant pump replacement"
    assert retriever.calls == [{"query": "coolant pump replacement", "equipment_model": EQUIPMENT, "k": 4}]


def test_junior_search_overrides_equipment_and_k_chosen_by_model(monkeypatch):
    """The model cannot switch equipment or inflate k: the application's values always win."""
    retriever = FakeRetriever(default=[_chunk()])
    hostile = {"query": "pump", "equipment_model": "thermal_retrofit_1kw", "k": 99}
    agent = _make_agent(monkeypatch, retriever, junior=[_msg(tool_calls=_tool_call(hostile))])

    out = agent.junior_search(QUESTION, EQUIPMENT, k=5)

    assert retriever.calls == [{"query": "pump", "equipment_model": EQUIPMENT, "k": 5}]
    assert out["equipment_model"] == EQUIPMENT


def test_junior_search_requires_equipment_model(monkeypatch):
    """An empty equipment model is refused before any model call is made."""
    agent = _make_agent(monkeypatch, junior=[_search("x")])

    with pytest.raises(ValueError, match="equipment_model"):
        agent.junior_search(QUESTION, "")

    assert agent.client.calls == []


def test_junior_search_without_a_tool_call_searches_the_engineers_question(monkeypatch):
    """A reply with no tool call must not fail the request: the mandatory search runs on the question itself."""
    retriever = FakeRetriever(default=[_chunk()])
    agent = _make_agent(monkeypatch, retriever, junior=[_msg("I think the answer is...", tool_calls=None)])

    out = agent.junior_search(QUESTION, EQUIPMENT, k=4)

    assert out["search_query"] == QUESTION
    assert out["tool_name"] == "search_service_documents"
    assert retriever.calls == [{"query": QUESTION, "equipment_model": EQUIPMENT, "k": 4}]


@pytest.mark.parametrize(
    "tool_calls",
    [
        _tool_call("{not json"),
        _tool_call("[]"),
        _tool_call("null"),
        _tool_call("\"just a string\""),
        _tool_call({}),
        _tool_call({"query": ""}),
        _tool_call({"query": "   "}),
        _tool_call({"query": None}),
        _tool_call({"query": 42}),
        _tool_call({"query": "pump removal"}, name="some_other_tool"),
        [],
    ],
    ids=["bad-json", "list", "null", "string", "no-query", "empty-query", "blank-query", "null-query", "number-query", "wrong-tool", "empty-calls"],
)
def test_junior_search_with_an_unusable_tool_call_falls_back_to_the_question(monkeypatch, tool_calls):
    """Unreadable arguments, a missing or blank query, or the wrong tool all fall back to searching the question."""
    retriever = FakeRetriever(default=[_chunk()])
    agent = _make_agent(monkeypatch, retriever, junior=[_msg(tool_calls=tool_calls)])

    out = agent.junior_search(QUESTION, EQUIPMENT)

    assert out["search_query"] == QUESTION
    assert out["tool_name"] == "search_service_documents"
    assert len(retriever.calls) == 1
    assert retriever.calls[0]["query"] == QUESTION


def test_the_fallback_search_is_still_pinned_to_the_selected_equipment(monkeypatch):
    """The fallback keeps the isolation boundary: equipment and result count come from the application."""
    retriever = FakeRetriever(default=[_chunk()])
    agent = _make_agent(monkeypatch, retriever, junior=[_msg(tool_calls=None)])

    agent.junior_search(QUESTION, EQUIPMENT, k=3)

    assert retriever.calls[0]["equipment_model"] == EQUIPMENT
    assert retriever.calls[0]["k"] == 3


def test_a_usable_tool_call_is_not_replaced_by_the_fallback(monkeypatch):
    """The model's own query is used whenever it is a real one."""
    retriever = FakeRetriever(default=[_chunk()])
    agent = _make_agent(monkeypatch, retriever, junior=[_search("coolant pump removal")])

    out = agent.junior_search(QUESTION, EQUIPMENT)

    assert out["search_query"] == "coolant pump removal"
    assert retriever.calls[0]["query"] == "coolant pump removal"


def test_the_retrieval_loop_survives_a_junior_agent_that_makes_no_tool_call(monkeypatch):
    """The whole loop completes when the junior reply is unusable: the question is searched and assessed."""
    retriever = FakeRetriever(default=[_chunk()])
    agent = _make_agent(monkeypatch, retriever, junior=[_msg("no tool", tool_calls=None)], assess=[_good()])

    out = agent.run_retrieval_loop(QUESTION, EQUIPMENT)

    assert out["retrieval_status"] == "GOOD"
    assert out["final_query"] == QUESTION
    assert out["retrieval_attempts"] == 1


# ----------------------------------------------------------------------
# Evidence assessment and reformulation
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "reply, expected",
    [("GOOD", True), ("  good\n", True), ("POOR", False), ("GOOD enough, I suppose", False), ("", False)],
)
def test_assess_retrieval_only_accepts_the_single_word_good(monkeypatch, reply, expected):
    """Assessment is True only for exactly GOOD (case/whitespace-insensitive); anything else is POOR."""
    agent = _make_agent(monkeypatch, assess=[_msg(reply)])
    evidence = [_chunk_dict()]

    assert agent.assess_retrieval(QUESTION, EQUIPMENT, evidence) is expected


def _chunk_dict(chunk_id="c1", **kwargs):
    chunk = _chunk(chunk_id, **kwargs)
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "source_file": chunk.source_file,
        "page": chunk.page,
        "section_id": chunk.section_id,
        "section_title": chunk.section_title,
        "revision": chunk.revision,
        "equipment_model": chunk.equipment_model,
        "text": chunk.text,
        "rrf_score": chunk.score,
    }


def test_assess_retrieval_with_no_results_is_poor_without_calling_the_model(monkeypatch):
    """Empty retrieval is POOR immediately and costs no model call."""
    agent = _make_agent(monkeypatch)

    assert agent.assess_retrieval(QUESTION, EQUIPMENT, []) is False
    assert agent.client.calls == []


def test_assess_retrieval_prompt_carries_question_equipment_and_evidence(monkeypatch):
    """The assessor sees the engineer question, the selected equipment and the evidence text."""
    agent = _make_agent(monkeypatch, assess=[_good()])

    agent.assess_retrieval(QUESTION, EQUIPMENT, [_chunk_dict(text="UNIQUE-EVIDENCE-TEXT")])

    prompt = agent.client.stage_calls("assess")[0]["messages"][1]["content"]
    assert QUESTION in prompt
    assert EQUIPMENT in prompt
    assert "UNIQUE-EVIDENCE-TEXT" in prompt
    assert "[EVIDENCE 1]" in prompt


def test_reformulate_query_returns_stripped_query_and_keeps_equipment_in_prompt(monkeypatch):
    """Reformulation returns the trimmed new query and the prompt pins the selected equipment."""
    agent = _make_agent(monkeypatch, reformulate=[_msg("  coolant pump removal procedure \n")])

    new_query = agent.reformulate_query(QUESTION, EQUIPMENT, [_chunk_dict()])

    assert new_query == "coolant pump removal procedure"
    prompt = agent.client.stage_calls("reformulate")[0]["messages"][1]["content"]
    assert EQUIPMENT in prompt and QUESTION in prompt


@pytest.mark.parametrize("reply", ["", "   ", None])
def test_reformulate_query_with_an_empty_reply_falls_back_to_the_original_question(monkeypatch, reply):
    """An empty or missing reformulation must not abort the request: the original question is searched again."""
    agent = _make_agent(monkeypatch, reformulate=[_msg(reply)])

    assert agent.reformulate_query(QUESTION, EQUIPMENT, [_chunk_dict()]) == QUESTION


def test_reformulate_query_with_no_choices_falls_back_to_the_original_question(monkeypatch):
    """A response with no choices is treated like an empty reformulation."""
    agent = _make_agent(monkeypatch, reformulate=[SimpleNamespace(choices=[])])

    assert agent.reformulate_query(QUESTION, EQUIPMENT, [_chunk_dict()]) == QUESTION


def test_loop_with_an_empty_reformulation_escalates_instead_of_failing(monkeypatch):
    """An empty reformulation leads to a second look at the same question, then the senior review escalates."""
    retriever = FakeRetriever(default=[_chunk()])
    agent = _make_agent(
        monkeypatch,
        retriever,
        junior=[_search("first")],
        assess=[_poor(), _poor()],
        reformulate=[_msg("")],
        senior=[_msg("DECISION: ESCALATE\nREASON: not documented\nQUERY: NONE")],
    )

    out = agent.run_retrieval_loop(QUESTION, EQUIPMENT)

    assert retriever.calls[1]["query"] == QUESTION
    assert out["retrieval_status"] == "INSUFFICIENT"
    assert out["escalation_required"] is True


# ----------------------------------------------------------------------
# Senior review parsing
# ----------------------------------------------------------------------


def _review(agent_text, monkeypatch):
    agent = _make_agent(monkeypatch, senior=[_msg(agent_text)])
    return agent.senior_review(
        QUESTION, EQUIPMENT, "q1", [_chunk_dict()], "q2", [_chunk_dict(chunk_id="c2")]
    )


def test_senior_review_parses_retry_with_new_query(monkeypatch):
    """A well-formed RETRY reply yields decision, reason and the new query."""
    review = _review("DECISION: RETRY\nREASON: try fault codes\nQUERY: pump fault code table", monkeypatch)

    assert review == {"decision": "RETRY", "reason": "try fault codes", "query": "pump fault code table"}


def test_senior_review_parses_escalate_and_ignores_none_query(monkeypatch):
    """ESCALATE with QUERY: NONE gives no query."""
    review = _review("DECISION: ESCALATE\nREASON: procedure absent\nQUERY: NONE", monkeypatch)

    assert review == {"decision": "ESCALATE", "reason": "procedure absent", "query": None}


def test_senior_review_retry_without_usable_query_is_downgraded_to_escalate(monkeypatch):
    """RETRY with QUERY: NONE is unsafe, so it becomes ESCALATE with an explanatory reason."""
    review = _review("DECISION: RETRY\nREASON: maybe\nQUERY: NONE", monkeypatch)

    assert review["decision"] == "ESCALATE"
    assert review["query"] is None
    assert "without providing a valid query" in review["reason"]


@pytest.mark.parametrize("garbage", ["", "I would retry this one.", "DECISION: MAYBE\nQUERY: foo"])
def test_senior_review_malformed_output_defaults_to_escalate(monkeypatch, garbage):
    """Unparseable or invalid senior output fails safe to ESCALATE."""
    review = _review(garbage, monkeypatch)

    assert review["decision"] == "ESCALATE"


def test_senior_review_decision_is_case_insensitive(monkeypatch):
    """A lower-case decision word is still understood."""
    review = _review("DECISION: retry\nREASON: r\nQUERY: new terms", monkeypatch)

    assert review["decision"] == "RETRY" and review["query"] == "new terms"


# ----------------------------------------------------------------------
# Retrieval loop: attempts and limits
# ----------------------------------------------------------------------


def test_loop_good_first_attempt_stops_after_one_retrieval(monkeypatch):
    """GOOD evidence on the first search ends the loop: 1 attempt, no reformulation, no senior."""
    retriever = FakeRetriever(default=[_chunk()])
    agent = _make_agent(monkeypatch, retriever, junior=[_search("pump")], assess=[_good()])

    out = agent.run_retrieval_loop(QUESTION, EQUIPMENT)

    assert out["retrieval_status"] == "GOOD"
    assert out["retrieval_attempts"] == 1
    assert out["final_query"] == "pump"
    assert out["senior_used"] is False and out["escalation_required"] is False
    assert len(retriever.calls) == 1
    assert agent.client.stage_calls("reformulate") == []


def test_loop_reformulates_after_poor_and_succeeds_on_second_attempt(monkeypatch):
    """POOR then reformulated query then GOOD gives 2 attempts with the new query, senior unused."""
    retriever = FakeRetriever(
        by_query={"pump": [_chunk("weak")], "pump removal steps": [_chunk("strong")]},
    )
    agent = _make_agent(
        monkeypatch,
        retriever,
        junior=[_search("pump")],
        assess=[_poor(), _good()],
        reformulate=[_msg("pump removal steps")],
    )

    out = agent.run_retrieval_loop(QUESTION, EQUIPMENT)

    assert [c["query"] for c in retriever.calls] == ["pump", "pump removal steps"]
    assert out["retrieval_attempts"] == 2
    assert out["retrieval_status"] == "GOOD"
    assert out["final_query"] == "pump removal steps"
    assert out["results"][0]["chunk_id"] == "strong"
    assert out["senior_used"] is False
    assert agent.client.stage_calls("senior") == []


def test_loop_senior_retry_gives_third_good_retrieval(monkeypatch):
    """Two POOR attempts then senior RETRY then GOOD gives 3 attempts, senior_decision RETRY."""
    retriever = FakeRetriever(by_query={"third": [_chunk("found")]}, default=[_chunk("weak")])
    agent = _make_agent(
        monkeypatch,
        retriever,
        junior=[_search("first")],
        assess=[_poor(), _poor(), _good()],
        reformulate=[_msg("second")],
        senior=[_msg("DECISION: RETRY\nREASON: other terms\nQUERY: third")],
    )

    out = agent.run_retrieval_loop(QUESTION, EQUIPMENT)

    assert [c["query"] for c in retriever.calls] == ["first", "second", "third"]
    assert out["retrieval_attempts"] == 3
    assert out["retrieval_status"] == "GOOD"
    assert out["senior_used"] is True and out["senior_decision"] == "RETRY"
    assert out["senior_reason"] == "other terms"
    assert out["escalation_required"] is False
    assert out["results"][0]["chunk_id"] == "found"


def test_loop_never_exceeds_three_retrievals_and_escalates_when_third_is_poor(monkeypatch):
    """Even if every assessment is POOR, there are exactly three searches (2 junior + 1 senior) then escalation."""
    retriever = FakeRetriever(default=[_chunk()])
    agent = _make_agent(
        monkeypatch,
        retriever,
        junior=[_search("first")],
        assess=[_poor(), _poor(), _poor()],
        reformulate=[_msg("second")],
        senior=[_msg("DECISION: RETRY\nREASON: once more\nQUERY: third")],
    )

    out = agent.run_retrieval_loop(QUESTION, EQUIPMENT)

    assert len(retriever.calls) == 3
    assert out["retrieval_attempts"] == 3
    assert out["retrieval_status"] == "INSUFFICIENT"
    assert out["escalation_required"] is True
    assert len(agent.client.stage_calls("senior")) == 1
    assert agent.client.queues["assess"] == []


def test_loop_senior_escalate_stops_after_two_retrievals(monkeypatch):
    """Senior ESCALATE means no third search: 2 attempts, INSUFFICIENT, escalation required with the reason."""
    retriever = FakeRetriever(default=[_chunk()])
    agent = _make_agent(
        monkeypatch,
        retriever,
        junior=[_search("first")],
        assess=[_poor(), _poor()],
        reformulate=[_msg("second")],
        senior=[_msg("DECISION: ESCALATE\nREASON: not documented\nQUERY: NONE")],
    )

    out = agent.run_retrieval_loop(QUESTION, EQUIPMENT)

    assert len(retriever.calls) == 2
    assert out["retrieval_attempts"] == 2
    assert out["retrieval_status"] == "INSUFFICIENT"
    assert out["senior_decision"] == "ESCALATE"
    assert out["senior_reason"] == "not documented"
    assert out["escalation_required"] is True


def test_loop_pins_equipment_on_every_retrieval_including_reformulated_and_senior(monkeypatch):
    """All three searches carry the application's equipment model and k."""
    retriever = FakeRetriever(default=[_chunk()])
    hostile = {"query": "first", "equipment_model": "thermal_retrofit_1kw", "k": 50}
    agent = _make_agent(
        monkeypatch,
        retriever,
        junior=[_msg(tool_calls=_tool_call(hostile))],
        assess=[_poor(), _poor(), _good()],
        reformulate=[_msg("second")],
        senior=[_msg("DECISION: RETRY\nREASON: r\nQUERY: third")],
    )

    agent.run_retrieval_loop(QUESTION, EQUIPMENT, k=7)

    assert len(retriever.calls) == 3
    assert {c["equipment_model"] for c in retriever.calls} == {EQUIPMENT}
    assert {c["k"] for c in retriever.calls} == {7}


def test_loop_with_empty_retrieval_goes_through_retries_without_assessor_calls(monkeypatch):
    """Empty results are POOR without model assessment; the loop still reaches senior review and escalates."""
    retriever = FakeRetriever(default=[])
    agent = _make_agent(
        monkeypatch,
        retriever,
        junior=[_search("first")],
        reformulate=[_msg("second")],
        senior=[_msg("DECISION: ESCALATE\nREASON: nothing found\nQUERY: NONE")],
    )

    out = agent.run_retrieval_loop(QUESTION, EQUIPMENT)

    assert out["retrieval_status"] == "INSUFFICIENT"
    assert out["results"] == []
    assert out["escalation_required"] is True
    assert agent.client.stage_calls("assess") == []


# ----------------------------------------------------------------------
# Grounded answer generation and run()
# ----------------------------------------------------------------------

CITED = "Power off first. [Source: aem_thermal_station.pdf, page 10, section 4.1 Pump]"


def test_generate_grounded_answer_returns_cited_text_and_demands_inline_citations(monkeypatch):
    """The answer keeps the model's inline [Source: ...] markers and the prompt requires that format."""
    agent = _make_agent(monkeypatch, answer=[_msg(f"  {CITED}  ")])

    answer = agent.generate_grounded_answer(QUESTION, EQUIPMENT, [_chunk_dict(text="Power off first.")])

    assert answer == CITED
    prompt = agent.client.stage_calls("answer")[0]["messages"][1]["content"]
    assert "[Source: <source_file>, page <page>" in prompt
    assert "Power off first." in prompt
    assert "aem_thermal_station.pdf" in prompt
    assert agent.client.stage_calls("answer")[0]["temperature"] == 0


def test_run_good_evidence_returns_loop_result_plus_grounded_answer(monkeypatch):
    """With GOOD evidence run() adds the generated cited answer to the retrieval result."""
    retriever = FakeRetriever(default=[_chunk()])
    agent = _make_agent(monkeypatch, retriever, junior=[_search("pump")], assess=[_good()], answer=[_msg(CITED)])

    out = agent.run(QUESTION, EQUIPMENT)

    assert out["retrieval_status"] == "GOOD"
    assert out["answer"] == CITED
    assert "[Source:" in out["answer"]
    assert out["results"][0]["chunk_id"] == "c1"


def test_run_insufficient_evidence_escalates_without_generating_an_answer(monkeypatch):
    """When evidence stays insufficient run() returns the escalation message and never calls the answer model."""
    retriever = FakeRetriever(default=[_chunk()])
    agent = _make_agent(
        monkeypatch,
        retriever,
        junior=[_search("first")],
        assess=[_poor(), _poor()],
        reformulate=[_msg("second")],
        senior=[_msg("DECISION: ESCALATE\nREASON: absent\nQUERY: NONE")],
    )

    out = agent.run(QUESTION, EQUIPMENT)

    assert out["escalation_required"] is True
    assert out["answer"].startswith("No grounded answer generated.")
    assert "Human escalation is required" in out["answer"]
    assert agent.client.stage_calls("answer") == []


def test_format_retrieval_context_numbers_evidence_with_citation_fields(monkeypatch):
    """The evidence block numbers each chunk and exposes source, page, section and equipment for citing."""
    agent = _make_agent(monkeypatch)

    text = agent.format_retrieval_context([_chunk_dict("a", page=3, title="2 Safety"), _chunk_dict("b", page=7)])

    assert "[EVIDENCE 1]" in text and "[EVIDENCE 2]" in text
    assert "Source: aem_thermal_station.pdf" in text
    assert "Page: 3" in text and "Section: 2 Safety" in text
    assert f"Equipment: {EQUIPMENT}" in text


def test_get_client_is_lazy_and_cached(monkeypatch):
    """The OpenAI client is only built on first use (so import/construct needs no key) and then reused."""
    created = []
    monkeypatch.setattr(rag, "OpenAI", lambda api_key=None: created.append(api_key) or object())
    monkeypatch.setattr(rag, "AEMHybridRetriever", lambda *a, **k: FakeRetriever())
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    agent = rag.AgenticRAGAgent()

    assert created == []
    first = agent._get_client()
    assert agent._get_client() is first
    assert created == ["sk-fake"]


# ----------------------------------------------------------------------
# Orchestrator node translation
# ----------------------------------------------------------------------


@pytest.fixture
def orchestrator(monkeypatch):
    """Import the orchestrator with the retriever faked so no Chroma index is opened."""
    monkeypatch.setattr(rag, "AEMHybridRetriever", lambda *a, **k: FakeRetriever())
    import asa.graph.orchestrator as module

    return module


def _run_result(status="GOOD", attempts=1, escalation=False, reason=None, results=None, answer="ANSWER"):
    return {
        "retrieval_status": status,
        "retrieval_attempts": attempts,
        "escalation_required": escalation,
        "senior_reason": reason,
        "results": [_chunk_dict()] if results is None else results,
        "answer": answer,
    }


def test_node_translates_good_result_into_state(monkeypatch, orchestrator):
    """A GOOD run becomes sufficient, no retries, no escalation, chunks rebuilt and the answer passed through."""
    seen = {}

    def fake_run(**kwargs):
        seen.update(kwargs)
        return _run_result(results=[_chunk_dict("a", page=3, title="2 Safety")], answer="the answer")

    monkeypatch.setattr(orchestrator.rag_agent, "run", fake_run)

    update = orchestrator.agentic_rag_node(
        {"resolved_query": "  pump?  ", "raw_query": "ignored", "equipment_model": f" {EQUIPMENT} "}
    )

    assert seen == {"question": "pump?", "equipment_model": EQUIPMENT, "k": 5}
    assert update["sufficiency"] is True
    assert update["retry_count"] == 0
    assert update["escalated"] is False
    assert update["escalation_reason"] == ""
    assert update["final_answer"] == "the answer"
    assert update["retrieval_strategy"] == "hybrid"
    chunk = update["retrieved_chunks"][0]
    assert isinstance(chunk, Chunk)
    assert (chunk.chunk_id, chunk.page, chunk.section_title, chunk.score) == ("a", 3, "2 Safety", 0.5)


def test_node_translates_escalated_result_into_state(monkeypatch, orchestrator):
    """An INSUFFICIENT run maps to sufficiency False, retry_count attempts-1, escalated and the senior reason."""
    monkeypatch.setattr(
        orchestrator.rag_agent,
        "run",
        lambda **kw: _run_result("INSUFFICIENT", attempts=3, escalation=True, reason="not documented", answer="No grounded answer"),
    )

    update = orchestrator.agentic_rag_node({"raw_query": "pump?", "equipment_model": EQUIPMENT})

    assert update["sufficiency"] is False
    assert update["retry_count"] == 2
    assert update["escalated"] is True
    assert update["escalation_reason"] == "not documented"


def test_node_handles_empty_results(monkeypatch, orchestrator):
    """An escalated run with no chunks yields an empty chunk list rather than failing."""
    monkeypatch.setattr(
        orchestrator.rag_agent, "run", lambda **kw: _run_result("INSUFFICIENT", 2, True, "none", results=[])
    )

    update = orchestrator.agentic_rag_node({"raw_query": "pump?", "equipment_model": EQUIPMENT})

    assert update["retrieved_chunks"] == []


@pytest.mark.parametrize(
    "state, message",
    [({"equipment_model": EQUIPMENT}, "requires a query"), ({"raw_query": "pump?"}, "requires equipment_model")],
)
def test_node_rejects_missing_query_or_equipment_without_running_the_agent(monkeypatch, orchestrator, state, message):
    """The node refuses blank query/equipment before invoking the RAG agent."""
    monkeypatch.setattr(orchestrator.rag_agent, "run", lambda **kw: pytest.fail("agent must not run"))

    with pytest.raises(ValueError, match=message):
        orchestrator.agentic_rag_node(state)


# ----------------------------------------------------------------------
# S9: the GOOD verdict ignores case, punctuation and markdown, nothing else
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "reply",
    ["GOOD", "good", "Good.", "GOOD.", "**GOOD**", "`GOOD`", "\"GOOD\"", "Good!", "  GOOD \n", "- GOOD", "GOOD\n"],
)
def test_assess_retrieval_accepts_good_with_harmless_formatting(monkeypatch, reply):
    """S9: 'GOOD.' and '**GOOD**' used to count as POOR and cost an extra retrieval; they are GOOD now."""
    agent = _make_agent(monkeypatch, assess=[_msg(reply)])

    assert agent.assess_retrieval(QUESTION, EQUIPMENT, [_chunk_dict()]) is True


@pytest.mark.parametrize(
    "reply",
    [
        "POOR",
        "**POOR**",
        "NOT GOOD",
        "not good.",
        "GOOD enough",
        "GOOD, but incomplete",
        "GOODNESS",
        "GOOD OR POOR",
        "G.O.O.D",
        "BAD",
        "",
        "   ",
        "...",
        "**",
        "Evidence is good",
    ],
)
def test_assess_retrieval_still_rejects_anything_that_is_not_just_good(monkeypatch, reply):
    """S9: loosening the format must not loosen the verdict: any other word keeps the evidence POOR (fail-safe)."""
    agent = _make_agent(monkeypatch, assess=[_msg(reply)])

    assert agent.assess_retrieval(QUESTION, EQUIPMENT, [_chunk_dict()]) is False


def test_the_loop_stops_after_one_search_when_the_assessment_is_good_with_punctuation(monkeypatch):
    """S9 end to end: a 'GOOD.' assessment ends the loop on the first attempt instead of reformulating."""
    retriever = FakeRetriever(default=[_chunk()])
    agent = _make_agent(monkeypatch, retriever, junior=[_search("first")], assess=[_msg("**GOOD**")])

    out = agent.run_retrieval_loop(QUESTION, EQUIPMENT)

    assert out["retrieval_status"] == "GOOD"
    assert out["retrieval_attempts"] == 1
    assert len(retriever.calls) == 1
    assert agent.client.stage_calls("reformulate") == []


# ----------------------------------------------------------------------
# S8: an escalation carries no query
# ----------------------------------------------------------------------


def test_senior_escalate_with_a_query_returns_no_query(monkeypatch):
    """S8: a query the model adds to an ESCALATE is not an instruction to search and is not passed on."""
    review = _review("DECISION: ESCALATE\nREASON: not documented\nQUERY: pump fault code table", monkeypatch)

    assert review["decision"] == "ESCALATE"
    assert review["query"] is None
    assert review["reason"] == "not documented"


@pytest.mark.parametrize("decision", ["MAYBE", "", "retry later", "SEARCH", "proceed"])
def test_senior_invalid_decision_with_a_query_escalates_with_no_query(monkeypatch, decision):
    """S8: an invalid decision fails safe to ESCALATE and drops the query that came with it."""
    review = _review(f"DECISION: {decision}\nREASON: unclear\nQUERY: foo bar", monkeypatch)

    assert review["decision"] == "ESCALATE"
    assert review["query"] is None


def test_senior_reply_with_only_a_query_escalates_with_no_query(monkeypatch):
    """S8: a reply with a query but no decision at all is an escalation without a query."""
    review = _review("QUERY: foo bar", monkeypatch)

    assert review["decision"] == "ESCALATE"
    assert review["query"] is None


def test_senior_retry_keeps_its_query(monkeypatch):
    """S8 must not remove the query a genuine RETRY needs."""
    review = _review("DECISION: RETRY\nREASON: try fault codes\nQUERY: pump fault code table", monkeypatch)

    assert review["decision"] == "RETRY"
    assert review["query"] == "pump fault code table"


def test_senior_retry_without_a_query_escalates_with_no_query(monkeypatch):
    """A RETRY with no usable query is downgraded to ESCALATE and has no query."""
    review = _review("DECISION: RETRY\nREASON: try again\nQUERY: NONE", monkeypatch)

    assert review["decision"] == "ESCALATE"
    assert review["query"] is None


def test_the_loop_does_not_search_again_when_the_senior_escalates_with_a_query(monkeypatch):
    """S8 end to end: an ESCALATE that carries a query still ends after two searches and never runs that query."""
    retriever = FakeRetriever(default=[_chunk()])
    agent = _make_agent(
        monkeypatch,
        retriever,
        junior=[_search("first")],
        assess=[_poor(), _poor()],
        reformulate=[_msg("second")],
        senior=[_msg("DECISION: ESCALATE\nREASON: not documented\nQUERY: sneaky third query")],
    )

    out = agent.run_retrieval_loop(QUESTION, EQUIPMENT)

    assert [call["query"] for call in retriever.calls] == ["first", "second"]
    assert out["escalation_required"] is True
    assert out["retrieval_attempts"] == 2
