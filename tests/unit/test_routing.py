"""Unit tests for routing: the Planning Agent's specialist choice and the graph's conditional edges.

The planner's model is replaced by a fake. The tests pin the contract around it: which specialist each reply
selects, that anything unparseable or unknown falls back to Safety (the cautious specialist), that the planner
refuses requests that are not ready, and that the graph never continues to an arbitrary specialist.
"""

from types import SimpleNamespace

import pytest

import asa.agents.planning as planning
from asa.graph.routing import route_after_planning, route_after_request

EQUIPMENT = "thermal_station"
UNPARSED_REASON = "Planner output could not be parsed; routing conservatively to Safety."


@pytest.fixture
def model(monkeypatch):
    """Replace the planner's model call. Set model['reply'] or model['error']; calls are recorded in model['calls']."""
    box = {"reply": "PLAN: AGENTIC_RAG\nREASON: documented information", "error": None, "calls": []}

    def fake_create(**kwargs):
        box["calls"].append(kwargs)
        if box["error"]:
            raise box["error"]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=box["reply"]))])

    monkeypatch.setattr(planning.client.chat.completions, "create", fake_create)
    return box


def plan(query="What is the power off procedure?", **extra):
    state = {"request_status": "READY", "raw_query": query, "equipment_model": EQUIPMENT, **extra}
    return planning.planning_node(state)


# --------------------------------------------------------------------------- #
# Planning Agent: the three specialists                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "reply, expected",
    [
        ("PLAN: AGENTIC_RAG\nREASON: documented procedure", "agentic_rag"),
        ("PLAN: DIAGNOSTIC\nREASON: observed symptom", "diagnostic"),
        ("PLAN: SAFETY\nREASON: unsafe action requested", "safety"),
    ],
)
def test_each_plan_selects_its_specialist(model, reply, expected):
    """RAG, Diagnostic and Safety replies each select the matching graph node."""
    model["reply"] = reply
    result = plan()
    assert result["current_step"] == expected
    assert result["plan"] == [expected]


def test_the_reason_is_returned(model):
    """The routing reason is kept so it can be shown and audited."""
    model["reply"] = "PLAN: DIAGNOSTIC\nREASON: the heater does not warm up"
    assert plan()["planning_reason"] == "the heater does not warm up"


def test_plan_value_is_case_insensitive(model):
    """'plan: diagnostic' style output is accepted."""
    model["reply"] = "PLAN: diagnostic\nREASON: symptom"
    assert plan()["current_step"] == "diagnostic"


def test_padding_and_extra_lines_are_tolerated(model):
    """Commentary around the two expected lines does not change the routing."""
    model["reply"] = "\n  Routing decision:\n  PLAN:   AGENTIC_RAG  \n  REASON:  manual lookup \n"
    result = plan()
    assert result["current_step"] == "agentic_rag"
    assert result["planning_reason"] == "manual lookup"


def test_each_planning_pass_counts_an_iteration(model):
    """iteration_count increases by one on top of what the state already holds."""
    assert plan()["iteration_count"] == 1
    assert plan(iteration_count=3)["iteration_count"] == 4


def test_the_resolved_query_is_routed_after_a_clarification(model):
    """After clarification the combined request is what gets routed, not the last short reply."""
    plan(
        query="The pump is noisy.",
        resolved_query="Original engineer request:\nIt's broken.\n\nEngineer clarification:\nThe pump is noisy.",
    )
    prompt = model["calls"][0]["messages"][1]["content"]
    assert "Original engineer request:\nIt's broken." in prompt


# --------------------------------------------------------------------------- #
# Planning Agent: ambiguous and malformed output falls back to Safety          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "reply",
    [
        "",
        "I would use the diagnostic agent.",
        "AGENTIC_RAG",
        "PLAN AGENTIC_RAG",
        '{"plan": "DIAGNOSTIC"}',
        "PLAN: SOMETHING_ELSE\nREASON: unknown",
        "PLAN: \nREASON: empty",
        "PLAN: RAG\nREASON: wrong name",
    ],
)
def test_unparseable_or_unknown_plan_routes_to_safety(model, reply):
    """When the choice cannot be read, the cautious specialist takes the request."""
    model["reply"] = reply
    result = plan()
    assert result["current_step"] == "safety"
    assert result["plan"] == ["safety"]


def test_unparseable_plan_explains_the_fallback(model):
    """The audit trail says why Safety was chosen."""
    model["reply"] = "no idea"
    assert plan()["planning_reason"] == UNPARSED_REASON


def test_a_reason_without_a_plan_still_routes_to_safety(model):
    """A reason line alone does not select a specialist, but it is kept."""
    model["reply"] = "REASON: this looks like a hazard"
    result = plan()
    assert result["current_step"] == "safety"
    assert result["planning_reason"] == "this looks like a hazard"


# --------------------------------------------------------------------------- #
# Planning Agent: preconditions and failures                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("status", [None, "CLARIFY", "", "ready"])
def test_planner_refuses_a_request_that_is_not_ready(model, status):
    """Only a READY request may be planned; anything else is a wiring error, not a request to route."""
    with pytest.raises(ValueError, match="not READY"):
        planning.planning_node({"request_status": status, "raw_query": "q", "equipment_model": EQUIPMENT})
    assert model["calls"] == []


def test_planner_requires_a_query(model):
    """No question, nothing to route."""
    with pytest.raises(ValueError, match="requires a query"):
        plan(query="   ")
    assert model["calls"] == []


def test_planner_requires_an_equipment_model(model):
    """The equipment is part of the routing context and must be selected."""
    with pytest.raises(ValueError, match="requires equipment_model"):
        planning.planning_node({"request_status": "READY", "raw_query": "q", "equipment_model": " "})
    assert model["calls"] == []


def test_a_failing_model_call_propagates(model):
    """The planner does not hide model failures; the chat layer reports 'unavailable'."""
    model["error"] = RuntimeError("model down")
    with pytest.raises(RuntimeError, match="model down"):
        plan()


# --------------------------------------------------------------------------- #
# Planning Agent: what the model is asked                                     #
# --------------------------------------------------------------------------- #


def test_planner_calls_the_model_once_at_temperature_zero(model):
    """Routing must be repeatable: one call, temperature 0, the configured model."""
    plan()
    assert len(model["calls"]) == 1
    assert model["calls"][0]["temperature"] == 0
    assert model["calls"][0]["model"] == planning.PLANNER_MODEL


def test_prompt_contains_the_question_and_equipment(model):
    """The routing is about this question on this equipment."""
    plan(query="Can I open the panel while it is powered?")
    prompt = model["calls"][0]["messages"][1]["content"]
    assert "Can I open the panel while it is powered?" in prompt
    assert EQUIPMENT in prompt


def test_prompt_keeps_the_routing_priority_safety_first(model):
    """Guards the rules that let safety-critical requests win over diagnostics and lookups."""
    plan()
    prompt = model["calls"][0]["messages"][1]["content"]
    assert "SAFETY has the highest priority" in prompt
    assert "When it is unclear whether the\nengineer is about to perform a task, choose SAFETY." in prompt
    assert "A safety WORD alone" in prompt


def test_system_message_prioritises_safety(model):
    """The system message repeats that safety-critical requests have priority."""
    plan()
    assert "Safety-critical requests have priority" in model["calls"][0]["messages"][0]["content"]


# --------------------------------------------------------------------------- #
# Graph edges after the Request Agent                                         #
# --------------------------------------------------------------------------- #


def test_ready_request_goes_to_planning():
    """READY continues to the Planning Agent."""
    assert route_after_request({"request_status": "READY"}) == "planning"


def test_clarify_request_ends_the_run_for_the_application_to_ask():
    """CLARIFY stops the graph so the engineer can be asked."""
    assert route_after_request({"request_status": "CLARIFY"}) == "clarify"


@pytest.mark.parametrize(
    "state", [{}, {"request_status": None}, {"request_status": "ready"}, {"request_status": "DONE"}]
)
def test_unexpected_request_status_fails_conservatively_to_clarify(state):
    """An unknown status never proceeds to planning."""
    assert route_after_request(state) == "clarify"


# --------------------------------------------------------------------------- #
# Graph edges after the Planning Agent                                        #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("step", ["agentic_rag", "diagnostic", "safety"])
def test_each_specialist_is_routed_to_itself(step):
    """The three valid steps route to the node of the same name."""
    assert route_after_planning({"current_step": step}) == step


@pytest.mark.parametrize(
    "state",
    [{}, {"current_step": None}, {"current_step": ""}, {"current_step": "rag"}, {"current_step": "AGENTIC_RAG"}],
)
def test_unknown_step_falls_back_to_safety_not_an_arbitrary_specialist(state):
    """Planning output the graph does not recognise goes to the cautious specialist."""
    assert route_after_planning(state) == "safety"


def test_planner_output_always_routes_to_a_known_node(model):
    """Whatever the planner replies, the next node is one of the three specialists."""
    for reply in ["PLAN: DIAGNOSTIC", "PLAN: ???", "", "PLAN: SAFETY", "PLAN: AGENTIC_RAG"]:
        model["reply"] = reply
        assert route_after_planning(plan()) in {"agentic_rag", "diagnostic", "safety"}
