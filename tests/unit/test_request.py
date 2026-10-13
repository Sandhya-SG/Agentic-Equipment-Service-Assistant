"""Unit tests for the Request Agent: READY/CLARIFY decisions, deterministic gates and fail-safe parsing.

The model is replaced by a fake, so these tests need no key and no network. They pin what the agent does with
whatever the model says, including malformed output: an unparseable answer must become a clarification, never
a silent READY.
"""

from types import SimpleNamespace

import pytest

import asa.agents.request as request_agent
from asa.agents.clarification import resolve_clarification
from asa.graph.state import MAX_CLARIFICATIONS

EQUIPMENT = "thermal_station"
DEFAULT_QUESTION = "Please provide more detail about the equipment-service request."


@pytest.fixture
def model(monkeypatch):
    """Replace the model call. Set model['reply'] (or model['error']); every call is recorded in model['calls']."""
    box = {"reply": "STATUS: READY\nQUESTION: NONE", "error": None, "calls": []}

    def fake_create(**kwargs):
        box["calls"].append(kwargs)
        if box["error"]:
            raise box["error"]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=box["reply"]))])

    monkeypatch.setattr(request_agent.client.chat.completions, "create", fake_create)
    return box


def run(query="What is the power off procedure?", equipment=EQUIPMENT, **extra):
    state = {"raw_query": query, "equipment_model": equipment, **extra}
    return request_agent.request_node(state)


# --------------------------------------------------------------------------- #
# Deterministic gates: no model call                                          #
# --------------------------------------------------------------------------- #


def test_missing_equipment_asks_to_select_it_without_calling_the_model(model):
    """The equipment gate is deterministic: a model decision must never replace selecting the equipment."""
    result = run(equipment=None)
    assert result["request_status"] == "CLARIFY"
    assert result["clarification_needed"] is True
    assert "select the equipment model" in result["clarification_question"]
    assert model["calls"] == []


def test_blank_equipment_is_treated_as_missing(model):
    """Whitespace is not an equipment selection."""
    assert run(equipment="   ")["request_status"] == "CLARIFY"
    assert model["calls"] == []


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_empty_request_asks_for_a_description_without_calling_the_model(model, query):
    """An empty or whitespace-only question cannot be assessed, so the model is not asked."""
    result = run(query=query)
    assert result["request_status"] == "CLARIFY"
    assert "describe" in result["clarification_question"]
    assert model["calls"] == []


def test_missing_query_keys_are_handled_like_an_empty_request(model):
    """A state with neither raw_query nor resolved_query must not crash."""
    result = request_agent.request_node({"equipment_model": EQUIPMENT})
    assert result["request_status"] == "CLARIFY"
    assert model["calls"] == []


def test_equipment_gate_runs_before_the_empty_request_gate(model):
    """With nothing selected and nothing typed, the engineer is asked for the equipment first."""
    result = run(query="", equipment=None)
    assert "equipment model" in result["clarification_question"]


# --------------------------------------------------------------------------- #
# Model decisions                                                             #
# --------------------------------------------------------------------------- #


def test_ready_decision_proceeds_with_no_question(model):
    """READY clears the clarification fields so the graph routes to planning."""
    model["reply"] = "STATUS: READY\nQUESTION: NONE"
    result = run()
    assert result == {"request_status": "READY", "clarification_needed": False, "clarification_question": None}


def test_ready_decision_drops_a_stray_question(model):
    """A model that says READY but still adds a question must not leave a question behind."""
    model["reply"] = "STATUS: READY\nQUESTION: Which filter do you mean?"
    result = run()
    assert result["request_status"] == "READY"
    assert result["clarification_question"] is None
    assert result["clarification_needed"] is False


def test_clarify_decision_carries_the_models_question(model):
    """CLARIFY returns the one question the model asked."""
    model["reply"] = "STATUS: CLARIFY\nQUESTION: What exactly is not working?"
    result = run(query="It's not working.")
    assert result["request_status"] == "CLARIFY"
    assert result["clarification_needed"] is True
    assert result["clarification_question"] == "What exactly is not working?"


def test_clarify_without_a_question_falls_back_to_the_default_question(model):
    """CLARIFY must always give the engineer something to answer."""
    model["reply"] = "STATUS: CLARIFY"
    assert run()["clarification_question"] == DEFAULT_QUESTION


def test_clarify_with_question_none_keeps_the_default_question(model):
    """'QUESTION: NONE' is not a question to show the engineer."""
    model["reply"] = "STATUS: CLARIFY\nQUESTION: NONE"
    assert run()["clarification_question"] == DEFAULT_QUESTION


def test_status_value_is_case_insensitive(model):
    """'ready' and 'Ready' are accepted as READY."""
    model["reply"] = "STATUS: ready\nQUESTION: none"
    assert run()["request_status"] == "READY"


def test_surrounding_whitespace_and_extra_lines_are_tolerated(model):
    """Padding and commentary around the two expected lines do not change the decision."""
    model["reply"] = "\n  Assessment follows.\n  STATUS: CLARIFY  \n  QUESTION:   Which component?  \n\nThanks."
    result = run()
    assert result["request_status"] == "CLARIFY"
    assert result["clarification_question"] == "Which component?"


# --------------------------------------------------------------------------- #
# Malformed model output fails safe                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "reply",
    [
        "",
        "I think this is fine.",
        "READY",
        "status ready",
        "STATUS READY",
        "{\"status\": \"READY\"}",
        "STATUS: MAYBE\nQUESTION: NONE",
        "STATUS: \nQUESTION: NONE",
    ],
)
def test_unparseable_decision_becomes_a_clarification_never_a_ready(model, reply):
    """If the decision cannot be read, the safe default is to ask, not to proceed."""
    model["reply"] = reply
    result = run()
    assert result["request_status"] == "CLARIFY"
    assert result["clarification_needed"] is True
    assert result["clarification_question"]


def test_a_failing_model_call_propagates_so_the_chat_service_can_report_unavailable(model):
    """The agent does not swallow model errors; the chat layer turns them into 'unavailable'."""
    model["error"] = RuntimeError("model down")
    with pytest.raises(RuntimeError, match="model down"):
        run()


def test_unexpected_reply_object_without_choices_raises(monkeypatch):
    """A response with no choices is an error, not a silent READY."""
    monkeypatch.setattr(request_agent.client.chat.completions, "create", lambda **kw: SimpleNamespace(choices=[]))
    with pytest.raises(IndexError):
        run()


# --------------------------------------------------------------------------- #
# What the model is asked                                                     #
# --------------------------------------------------------------------------- #


def test_model_is_called_once_with_temperature_zero_and_the_request_model(model):
    """Decisions must be repeatable: temperature 0, the configured model, a single call."""
    run()
    assert len(model["calls"]) == 1
    call = model["calls"][0]
    assert call["temperature"] == 0
    assert call["model"] == request_agent.REQUEST_MODEL


def test_prompt_contains_the_question_and_the_selected_equipment(model):
    """The assessment must be about this question and this equipment."""
    run(query="What does error E101 mean?", equipment="thermal_retrofit_1kw")
    prompt = model["calls"][0]["messages"][1]["content"]
    assert "What does error E101 mean?" in prompt
    assert "thermal_retrofit_1kw" in prompt


def test_system_message_forbids_answering_or_diagnosing(model):
    """The Request Agent only assesses completeness."""
    run()
    system = model["calls"][0]["messages"][0]["content"]
    assert "do not answer or diagnose" in system.lower()


def test_prompt_keeps_the_safety_priority_and_the_default_to_ready_rules(model):
    """Guards the prompt rules that stop over-asking and let safety questions through to the Safety Agent."""
    run()
    prompt = model["calls"][0]["messages"][1]["content"]
    assert "The default answer is READY" in prompt
    assert "SAFETY-CRITICAL INTENT HAS PRIORITY OVER" in prompt
    assert "Can I bypass the safety interlock?" in prompt


def test_a_plain_request_is_not_marked_as_a_clarification_continuation(model):
    """The continuation flag is NO unless a resolved query exists."""
    run()
    assert 'Clarification continuation:\nNO' in model["calls"][0]["messages"][1]["content"]


# --------------------------------------------------------------------------- #
# Clarification follow-ups                                                    #
# --------------------------------------------------------------------------- #


def test_resolved_query_replaces_the_raw_query_and_marks_a_continuation(model):
    """After a clarification the combined request is assessed, and the model is told it is a follow-up."""
    resolved = resolve_clarification("It's not working.", "The temperature does not reach the setpoint.")
    run(query="The temperature does not reach the setpoint.", resolved_query=resolved)
    prompt = model["calls"][0]["messages"][1]["content"]
    assert "Original engineer request:\nIt's not working." in prompt
    assert "Engineer clarification:\nThe temperature does not reach the setpoint." in prompt
    assert 'Clarification continuation:\nYES' in prompt


def test_follow_up_can_complete_a_clarified_request(model):
    """The original vague request plus the answer can become READY."""
    model["reply"] = "STATUS: READY\nQUESTION: NONE"
    resolved = resolve_clarification("It's not working.", "The heater does not warm up.")
    assert run(resolved_query=resolved)["request_status"] == "READY"


def test_follow_up_with_a_blank_resolved_query_uses_the_raw_query(model):
    """An empty resolved_query is ignored."""
    run(query="What PPE is required?", resolved_query="")
    prompt = model["calls"][0]["messages"][1]["content"]
    assert "What PPE is required?" in prompt
    assert 'Clarification continuation:\nNO' in prompt


def test_resolve_clarification_combines_both_texts_without_adding_information():
    """The combined request contains the original and the clarification, and nothing else of substance."""
    combined = resolve_clarification("It's not working.", "The pump is noisy.")
    assert combined == "Original engineer request:\nIt's not working.\n\nEngineer clarification:\nThe pump is noisy."


def test_resolve_clarification_with_no_original_returns_the_clarification():
    """Nothing to combine with."""
    assert resolve_clarification("", "  The pump is noisy.  ") == "The pump is noisy."


def test_resolve_clarification_with_no_answer_returns_the_original():
    """A blank answer leaves the original request unchanged."""
    assert resolve_clarification("  It's not working. ", "   ") == "It's not working."


def test_resolve_clarification_handles_none_inputs():
    """None is treated as empty text."""
    assert resolve_clarification(None, None) == ""


def test_the_clarification_limit_is_two_attempts():
    """After this many follow-ups the chat layer stops asking and escalates (see test_chat for the flow)."""
    assert MAX_CLARIFICATIONS == 2
