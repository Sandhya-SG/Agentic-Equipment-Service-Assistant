"""Integration tests for the LangGraph workflow: Request -> Planning -> one specialist -> END.

The real graph (nodes, edges, routing functions and state reducers from asa.graph.orchestrator) is built, but
every agent is replaced by a recording fake. So these tests check the wiring and the flow of state, not what the
agents say: who runs, in what order, what is skipped, what is passed on and what happens when a node fails.
No key, no model and no index are used.
"""

import importlib
import sys
from types import SimpleNamespace

import pytest

from app.services import graph_runner
from asa.graph.state import TraceEvent, new_state

ORCHESTRATOR = "asa.graph.orchestrator"
SPECIALISTS = ("agentic_rag", "diagnostic", "safety")


@pytest.fixture
def orchestrator(monkeypatch):
    """The orchestrator module, imported with a stub retriever so no embedding model or index is loaded."""
    import asa.agents.rag as rag

    monkeypatch.setattr(rag, "AEMHybridRetriever", lambda *args, **kwargs: SimpleNamespace())
    original = sys.modules.pop(ORCHESTRATOR, None)
    module = importlib.import_module(ORCHESTRATOR)
    try:
        yield module
    finally:
        sys.modules.pop(ORCHESTRATOR, None)
        if original is not None:
            sys.modules[ORCHESTRATOR] = original


class Flow:
    """A graph made of fake nodes. Each node records that it ran and a copy of the state it received."""

    def __init__(self, orchestrator, monkeypatch):
        self.orchestrator = orchestrator
        self.monkeypatch = monkeypatch
        self.calls: list[str] = []
        self.seen: dict[str, dict] = {}
        self.updates = {
            "request": {"request_status": "READY", "clarification_needed": False},
            "planning": {"current_step": "safety", "plan": ["safety"], "planning_reason": "test"},
            "agentic_rag": {"final_answer": "rag answer"},
            "diagnostic": {"final_answer": "diagnostic answer"},
            "safety": {"final_answer": "safety answer", "safety_verdict": "allow"},
        }
        self.errors: dict[str, Exception] = {}
        self.install()

    def _node(self, name):
        def node(state):
            self.calls.append(name)
            self.seen[name] = dict(state)
            if name in self.errors:
                raise self.errors[name]
            update = dict(self.updates[name])
            update["trace"] = [TraceEvent(agent=name, action="completed", timestamp="t")]
            return update

        return node

    def install(self):
        for attribute, name in (
            ("request_node", "request"),
            ("planning_node", "planning"),
            ("agentic_rag_node", "agentic_rag"),
            ("diagnostic_node", "diagnostic"),
            ("safety_node", "safety"),
        ):
            self.monkeypatch.setattr(self.orchestrator, attribute, self._node(name))
        self.graph = self.orchestrator.build_service_graph()

    def choose(self, specialist):
        self.updates["planning"] = {"current_step": specialist, "plan": [specialist], "planning_reason": "test"}

    def invoke(self, query="What is the power off procedure?", equipment="thermal_station"):
        return self.graph.invoke(new_state(raw_query=query, equipment_model=equipment))


@pytest.fixture
def flow(orchestrator, monkeypatch):
    return Flow(orchestrator, monkeypatch)


# --------------------------------------------------------------------------- #
# The happy path                                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("specialist", SPECIALISTS)
def test_ready_request_runs_request_planning_then_the_chosen_specialist(flow, specialist):
    """READY goes Request -> Planning -> the specialist the planner chose, then the run ends."""
    flow.choose(specialist)
    final = flow.invoke()
    assert flow.calls == ["request", "planning", specialist]
    assert final["final_answer"] == flow.updates[specialist]["final_answer"]


@pytest.mark.parametrize("specialist", SPECIALISTS)
def test_exactly_one_specialist_runs(flow, specialist):
    """The other two specialists are never called."""
    flow.choose(specialist)
    flow.invoke()
    assert [call for call in flow.calls if call in SPECIALISTS] == [specialist]


def test_the_run_ends_after_the_specialist(flow):
    """Nothing runs after the specialist: there is no loop back to planning."""
    flow.choose("agentic_rag")
    flow.invoke()
    assert flow.calls[-1] == "agentic_rag"
    assert flow.calls.count("planning") == 1


# --------------------------------------------------------------------------- #
# Clarification short-circuit                                                 #
# --------------------------------------------------------------------------- #


def test_clarification_stops_the_run_before_planning(flow):
    """CLARIFY ends the graph so the application can ask the engineer; no specialist runs."""
    flow.updates["request"] = {
        "request_status": "CLARIFY",
        "clarification_needed": True,
        "clarification_question": "What exactly is not working?",
    }
    final = flow.invoke(query="It's not working.")
    assert flow.calls == ["request"]
    assert final["request_status"] == "CLARIFY"
    assert final["clarification_question"] == "What exactly is not working?"
    assert not final.get("final_answer")


def test_an_unexpected_request_status_also_stops_the_run(flow):
    """A status the router does not know is treated as a clarification, never as READY."""
    flow.updates["request"] = {"request_status": "MAYBE"}
    flow.invoke()
    assert flow.calls == ["request"]


def test_a_missing_request_status_stops_the_run(flow):
    """If the Request Agent returns nothing useful, the run stops instead of guessing."""
    flow.updates["request"] = {}
    flow.invoke()
    assert flow.calls == ["request"]


# --------------------------------------------------------------------------- #
# Routing fallbacks inside the graph                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("step", ["", "rag", "nonsense", None])
def test_an_unknown_planning_step_goes_to_safety(flow, step):
    """An unrecognised step reaches the cautious specialist, not an arbitrary one."""
    flow.updates["planning"] = {"current_step": step}
    flow.invoke()
    assert flow.calls == ["request", "planning", "safety"]


def test_a_planner_that_returns_no_step_goes_to_safety(flow):
    """Missing planning output is handled like an unknown step."""
    flow.updates["planning"] = {}
    flow.invoke()
    assert flow.calls == ["request", "planning", "safety"]


# --------------------------------------------------------------------------- #
# State propagation                                                           #
# --------------------------------------------------------------------------- #


def test_the_planner_sees_what_the_request_agent_decided(flow):
    """The request decision is part of the state the next node receives."""
    flow.invoke(query="What PPE is required?")
    assert flow.seen["planning"]["request_status"] == "READY"
    assert flow.seen["planning"]["raw_query"] == "What PPE is required?"
    assert flow.seen["planning"]["equipment_model"] == "thermal_station"


def test_the_specialist_sees_the_plan_and_the_original_question(flow):
    """The specialist receives the planner's choice and reason, and the question and equipment."""
    flow.choose("diagnostic")
    flow.invoke(query="The heater does not warm up.")
    seen = flow.seen["diagnostic"]
    assert seen["plan"] == ["diagnostic"]
    assert seen["planning_reason"] == "test"
    assert seen["raw_query"] == "The heater does not warm up."
    assert seen["equipment_model"] == "thermal_station"


def test_the_final_state_holds_every_nodes_updates(flow):
    """Updates from all three nodes are merged into the final state."""
    flow.choose("safety")
    final = flow.invoke()
    assert final["request_status"] == "READY"
    assert final["planning_reason"] == "test"
    assert final["safety_verdict"] == "allow"
    assert final["final_answer"] == "safety answer"


def test_the_trace_is_appended_in_the_order_the_agents_ran(flow):
    """The trace uses an append-only reducer, so earlier events are kept and the order is the run order."""
    flow.choose("agentic_rag")
    final = flow.invoke()
    assert [event.agent for event in final["trace"]] == ["request", "planning", "agentic_rag"]


def test_a_clarification_run_has_only_the_request_event_in_its_trace(flow):
    """A short-circuited run leaves a one-step trace."""
    flow.updates["request"] = {"request_status": "CLARIFY"}
    final = flow.invoke()
    assert [event.agent for event in final["trace"]] == ["request"]


def test_the_state_given_to_the_graph_is_not_modified(flow):
    """Running the graph does not mutate the caller's initial state."""
    initial = new_state(raw_query="q", equipment_model="thermal_station")
    before = dict(initial)
    flow.graph.invoke(initial)
    assert initial == before


# --------------------------------------------------------------------------- #
# Failures                                                                    #
# --------------------------------------------------------------------------- #


def test_a_failing_request_agent_raises_and_nothing_else_runs(flow):
    """A node error propagates to the caller (the chat service reports 'unavailable'); later nodes do not run."""
    flow.errors["request"] = RuntimeError("model down")
    with pytest.raises(RuntimeError, match="model down"):
        flow.invoke()
    assert flow.calls == ["request"]


def test_a_failing_planner_raises_and_no_specialist_runs(flow):
    """If planning fails, no specialist is called as a fallback."""
    flow.errors["planning"] = ValueError("not READY")
    with pytest.raises(ValueError):
        flow.invoke()
    assert flow.calls == ["request", "planning"]


@pytest.mark.parametrize("specialist", SPECIALISTS)
def test_a_failing_specialist_raises_instead_of_returning_a_partial_answer(flow, specialist):
    """A specialist error is never turned into an answer inside the graph."""
    flow.choose(specialist)
    flow.errors[specialist] = RuntimeError("specialist failed")
    with pytest.raises(RuntimeError, match="specialist failed"):
        flow.invoke()
    assert flow.calls == ["request", "planning", specialist]


# --------------------------------------------------------------------------- #
# Structure                                                                   #
# --------------------------------------------------------------------------- #


def test_the_graph_has_exactly_the_expected_nodes(flow):
    """Request, Planning and the three specialists, plus the start and end markers."""
    nodes = set(flow.graph.get_graph().nodes)
    assert nodes == {"__start__", "request", "planning", *SPECIALISTS, "__end__"}


def test_the_graph_edges_match_the_workflow(flow):
    """start -> request; request -> planning or end; planning -> each specialist; each specialist -> end."""
    edges = {(edge.source, edge.target) for edge in flow.graph.get_graph().edges}
    expected = {
        ("__start__", "request"),
        ("request", "planning"),
        ("request", "__end__"),
        *(("planning", name) for name in SPECIALISTS),
        *((name, "__end__") for name in SPECIALISTS),
    }
    assert edges == expected


# --------------------------------------------------------------------------- #
# Through the application's graph runner                                      #
# --------------------------------------------------------------------------- #


@pytest.fixture
def runner(flow, monkeypatch):
    monkeypatch.setattr(graph_runner, "_graph", lambda: flow.graph)
    monkeypatch.setattr(graph_runner, "graph_config", lambda: {})
    return flow


def test_the_graph_runner_returns_the_final_state_with_a_trace_of_the_run(runner):
    """run_graph streams the real graph and reports the agents in order with their categorical details."""
    runner.choose("safety")
    state = graph_runner.run_graph("Can I open the panel?", "thermal_station")
    assert [event.agent for event in state["trace"]] == ["request", "planning", "safety"]
    assert state["trace"][0].detail["request_status"] == "READY"
    assert state["trace"][1].detail["current_step"] == "safety"
    assert state["trace"][2].detail["safety_verdict"] == "allow"
    assert state["final_answer"] == "safety answer"


def test_the_graph_runner_passes_the_clarification_context_to_the_request_agent(runner):
    """A follow-up carries the clarification response, the combined query and the attempt count into the graph."""
    graph_runner.run_graph(
        "The pump is noisy.",
        "thermal_station",
        clarification_response="The pump is noisy.",
        resolved_query="Original engineer request:\nIt's broken.\n\nEngineer clarification:\nThe pump is noisy.",
        clarification_count=1,
    )
    seen = runner.seen["request"]
    assert seen["clarification_response"] == "The pump is noisy."
    assert seen["resolved_query"].startswith("Original engineer request:")
    assert seen["clarification_count"] == 1


def test_the_graph_runner_starts_a_new_run_with_clean_state(runner):
    """A first request carries no clarification context and no previous answer."""
    graph_runner.run_graph("What PPE is required?", "thermal_station")
    seen = runner.seen["request"]
    assert seen["clarification_response"] is None
    assert seen["resolved_query"] is None
    assert seen["clarification_count"] == 0
    assert seen["retrieved_chunks"] == []
    assert not seen.get("final_answer")


def test_the_graph_runner_propagates_a_node_failure(runner):
    """An agent error reaches the chat service unchanged, so it can answer 'unavailable'."""
    runner.errors["planning"] = RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        graph_runner.run_graph("q", "thermal_station")


def test_a_clarification_through_the_runner_has_a_single_step_trace(runner):
    """The runner reports a short-circuited run as one completed step."""
    runner.updates["request"] = {"request_status": "CLARIFY", "clarification_question": "Which one?"}
    state = graph_runner.run_graph("it", "thermal_station")
    assert [event.agent for event in state["trace"]] == ["request"]
    assert state["clarification_question"] == "Which one?"
