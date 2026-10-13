"""Unit tests for the Diagnostic Agent (no network, no key, no index, no embeddings)."""

import importlib.util
import json
import os
import sys
import types
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest

from asa.graph.state import Chunk, RankedCause, Step


# --------------------------------------------------------------------------- #
# Import the module without a key, a Chroma index or the embedding model.      #
# diagnostic.py builds an OpenAI client and an AgenticRAGAgent at import time, #
# so both are stubbed while loading it under a private name.                   #
# --------------------------------------------------------------------------- #

def _load_diagnostic_module():
    """Load src/asa/agents/diagnostic.py with a stub RAG module and a dummy key."""
    src = Path(__file__).resolve().parents[2] / "src" / "asa" / "agents" / "diagnostic.py"

    stub = types.ModuleType("asa.agents.rag")
    stub.AgenticRAGAgent = type("AgenticRAGAgent", (), {})

    saved_rag = sys.modules.get("asa.agents.rag")
    saved_key = os.environ.get("OPENAI_API_KEY")
    sys.modules["asa.agents.rag"] = stub
    os.environ["OPENAI_API_KEY"] = "sk-test-not-real"
    try:
        spec = importlib.util.spec_from_file_location("diagnostic_under_test", src)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if saved_rag is None:
            sys.modules.pop("asa.agents.rag", None)
        else:
            sys.modules["asa.agents.rag"] = saved_rag
        if saved_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = saved_key
    return module


diag = _load_diagnostic_module()


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _chunk_dict(chunk_id, text="Evidence text.", **extra):
    item = {
        "chunk_id": chunk_id,
        "doc_id": "aem_thermal_station",
        "section_id": "5.1",
        "revision": "0",
        "equipment_model": "thermal_station",
        "text": text,
        "rrf_score": 0.5,
        "source_file": "aem_thermal_station.pdf",
        "page": 12,
        "section_title": "5.1 Troubleshooting",
    }
    item.update(extra)
    return item


def _chunk(chunk_id, text="Evidence text."):
    return Chunk(
        chunk_id=chunk_id,
        doc_id="aem_thermal_station",
        section_id="5.1",
        revision="0",
        equipment_model="thermal_station",
        text=text,
        score=0.5,
        source_file="aem_thermal_station.pdf",
        page=12,
        section_title="5.1 Troubleshooting",
    )


def _response(text):
    """Minimal fake OpenAI response shape."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


def _good_rag(chunks=None, attempts=1):
    chunks = chunks if chunks is not None else [
        _chunk_dict("c1", "Heater failure prevents reaching setpoint."),
        _chunk_dict("c2", "Check the heater fuse and the thermocouple."),
    ]
    return {
        "retrieval_status": "GOOD",
        "results": chunks,
        "retrieval_attempts": attempts,
    }


class FakeRag:
    """Stands in for AgenticRAGAgent and records the call."""

    def __init__(self, result):
        self.result = result
        self.calls = []

    def run_retrieval_loop(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class FakeModel:
    """
    Fake for client.chat.completions.create.

    The generation call is recognised by response_format; validation calls are
    answered SUPPORTED unless the prompt contains one of `reject`.
    """

    def __init__(self, diagnosis, reject=(), verdict_text=None):
        self.diagnosis = diagnosis
        self.reject = reject
        self.verdict_text = verdict_text
        self.generation_calls = 0
        self.validation_prompts = []

    def __call__(self, **kwargs):
        if kwargs.get("response_format"):
            self.generation_calls += 1
            body = self.diagnosis
            if not isinstance(body, str):
                body = json.dumps(body)
            return _response(body)
        prompt = kwargs["messages"][-1]["content"]
        self.validation_prompts.append(prompt)
        if self.verdict_text is not None:
            return _response(self.verdict_text)
        if any(token in prompt for token in self.reject):
            return _response("UNSUPPORTED")
        return _response("SUPPORTED")


def _install(monkeypatch, rag_result, model):
    rag = FakeRag(rag_result)
    monkeypatch.setattr(diag, "rag_agent", rag)
    monkeypatch.setattr(diag.client.chat.completions, "create", model)
    return rag


STATE = {
    "raw_query": "Temperature cannot reach setpoint",
    "equipment_model": "thermal_station",
}

DIAGNOSIS = {
    "assessment": "ignored by the node",
    "root_cause_confirmed": False,
    "candidate_causes": [
        {"cause": "Heater failure", "likelihood": 0.8, "supporting_chunk_ids": ["c1"]},
        {"cause": "Thermocouple drift", "likelihood": 0.4, "supporting_chunk_ids": ["c2"]},
    ],
    "recommended_checks": [
        {"action": "Check heater fuse", "supporting_chunk_ids": ["c2"], "hazard_flag": True},
        {"action": "Inspect thermocouple", "supporting_chunk_ids": ["c2"]},
    ],
}


# --------------------------------------------------------------------------- #
# Happy path and state shape                                                   #
# --------------------------------------------------------------------------- #

def test_good_evidence_returns_full_state_update(monkeypatch):
    """The node must return exactly the documented state keys for a successful diagnosis."""
    _install(monkeypatch, _good_rag(attempts=2), FakeModel(DIAGNOSIS))

    update = diag.diagnostic_node(dict(STATE))

    assert set(update) == {
        "retrieval_strategy", "retrieved_chunks", "sufficiency", "retry_count",
        "root_causes", "troubleshooting_steps", "escalated",
        "escalation_reason", "final_answer",
    }
    assert update["retrieval_strategy"] == "hybrid"
    assert update["sufficiency"] is True
    assert update["escalated"] is False
    assert update["escalation_reason"] == ""
    assert update["retry_count"] == 1  # attempts - 1
    assert all(isinstance(c, Chunk) for c in update["retrieved_chunks"])
    assert all(isinstance(c, RankedCause) for c in update["root_causes"])
    assert all(isinstance(s, Step) for s in update["troubleshooting_steps"])


def test_retrieved_chunks_map_rrf_score_and_provenance(monkeypatch):
    """RAG result dicts must be converted to Chunk objects with score and citation metadata."""
    _install(monkeypatch, _good_rag([_chunk_dict("c1", page=7, rrf_score=0.91)]),
             FakeModel({"candidate_causes": [], "recommended_checks": []}))

    update = diag.diagnostic_node(dict(STATE))

    chunk = update["retrieved_chunks"][0]
    assert (chunk.chunk_id, chunk.score, chunk.page) == ("c1", 0.91, 7)
    assert chunk.source_file == "aem_thermal_station.pdf"
    assert chunk.section_title == "5.1 Troubleshooting"


def test_retrieval_is_requested_for_the_selected_equipment(monkeypatch):
    """Retrieval must be scoped to the engineer's equipment model and query."""
    rag = _install(monkeypatch, _good_rag(), FakeModel(DIAGNOSIS))

    diag.diagnostic_node(dict(STATE))

    assert rag.calls == [{
        "question": "Temperature cannot reach setpoint",
        "equipment_model": "thermal_station",
        "k": 5,
    }]


def test_resolved_query_takes_priority_over_raw_query(monkeypatch):
    """A clarified (resolved) query must be used for retrieval instead of the raw one."""
    rag = _install(monkeypatch, _good_rag(), FakeModel(DIAGNOSIS))

    diag.diagnostic_node({**STATE, "resolved_query": "Heater stays cold at 150C"})

    assert rag.calls[0]["question"] == "Heater stays cold at 150C"


# --------------------------------------------------------------------------- #
# Ranking, steps, ordering                                                     #
# --------------------------------------------------------------------------- #

def test_causes_carry_likelihood_and_supporting_chunk_ids(monkeypatch):
    """Each retained cause must keep its likelihood and the chunk ids that support it."""
    _install(monkeypatch, _good_rag(), FakeModel(DIAGNOSIS))

    causes = diag.diagnostic_node(dict(STATE))["root_causes"]

    assert [(c.cause, c.likelihood, c.supporting_chunk_ids) for c in causes] == [
        ("Heater failure", 0.8, ["c1"]),
        ("Thermocouple drift", 0.4, ["c2"]),
    ]


def test_steps_are_numbered_in_model_order_and_keep_hazard_flag(monkeypatch):
    """Troubleshooting steps must be ordered 1..n in model order with hazard_flag preserved."""
    _install(monkeypatch, _good_rag(), FakeModel(DIAGNOSIS))

    steps = diag.diagnostic_node(dict(STATE))["troubleshooting_steps"]

    assert [(s.order, s.action, s.hazard_flag) for s in steps] == [
        (1, "Check heater fuse", True),
        (2, "Inspect thermocouple", False),
    ]


def test_step_order_keeps_original_numbers_after_a_step_is_dropped(monkeypatch):
    """Dropping an unsupported step must not renumber the remaining ones (gaps stay visible)."""
    _install(monkeypatch, _good_rag(), FakeModel(DIAGNOSIS, reject=("Check heater fuse",)))

    steps = diag.diagnostic_node(dict(STATE))["troubleshooting_steps"]

    assert [(s.order, s.action) for s in steps] == [(2, "Inspect thermocouple")]


def test_likelihood_is_clamped_to_unit_interval(monkeypatch):
    """Out-of-range likelihoods from the model must be clamped into [0, 1]."""
    diagnosis = {
        "candidate_causes": [
            {"cause": "High", "likelihood": 7, "supporting_chunk_ids": ["c1"]},
            {"cause": "Low", "likelihood": -3, "supporting_chunk_ids": ["c1"]},
        ],
        "recommended_checks": [],
    }
    _install(monkeypatch, _good_rag(), FakeModel(diagnosis))

    causes = diag.diagnostic_node(dict(STATE))["root_causes"]

    assert [c.likelihood for c in causes] == [1.0, 0.0]


def test_missing_optional_fields_use_safe_defaults(monkeypatch):
    """A cause with no likelihood and a step with no hazard_flag must get 0.0 / False."""
    diagnosis = {
        "candidate_causes": [{"cause": " Heater failure ", "supporting_chunk_ids": ["c1"]}],
        "recommended_checks": [{"action": " Check fuse ", "supporting_chunk_ids": ["c2"]}],
    }
    _install(monkeypatch, _good_rag(), FakeModel(diagnosis))

    update = diag.diagnostic_node(dict(STATE))

    assert update["root_causes"][0].cause == "Heater failure"
    assert update["root_causes"][0].likelihood == 0.0
    assert update["troubleshooting_steps"][0].action == "Check fuse"
    assert update["troubleshooting_steps"][0].hazard_flag is False


# --------------------------------------------------------------------------- #
# Evidence validation                                                          #
# --------------------------------------------------------------------------- #

def test_cause_citing_unknown_chunk_id_is_removed_without_a_model_call(monkeypatch):
    """A cause whose cited chunk ids were not retrieved must be dropped (no invented evidence)."""
    diagnosis = {
        "candidate_causes": [
            {"cause": "Ghost cause", "likelihood": 0.9, "supporting_chunk_ids": ["nope"]},
            {"cause": "Real cause", "likelihood": 0.5, "supporting_chunk_ids": ["c1"]},
        ],
        "recommended_checks": [],
    }
    model = FakeModel(diagnosis)
    _install(monkeypatch, _good_rag(), model)

    causes = diag.diagnostic_node(dict(STATE))["root_causes"]

    assert [c.cause for c in causes] == ["Real cause"]
    assert len(model.validation_prompts) == 1  # the ghost cause never reached the validator


def test_claims_with_no_citations_are_removed(monkeypatch):
    """A cause or step with an empty supporting_chunk_ids list must be dropped."""
    diagnosis = {
        "candidate_causes": [{"cause": "Uncited cause", "likelihood": 0.9}],
        "recommended_checks": [{"action": "Uncited check"}],
    }
    _install(monkeypatch, _good_rag(), FakeModel(diagnosis))

    update = diag.diagnostic_node(dict(STATE))

    assert update["root_causes"] == []
    assert update["troubleshooting_steps"] == []


def test_step_citing_unknown_chunk_id_is_removed(monkeypatch):
    """A recommended check citing a chunk id that does not exist must be dropped."""
    diagnosis = {
        "candidate_causes": [],
        "recommended_checks": [
            {"action": "Ghost check", "supporting_chunk_ids": ["zzz"]},
            {"action": "Real check", "supporting_chunk_ids": ["c2"]},
        ],
    }
    _install(monkeypatch, _good_rag(), FakeModel(diagnosis))

    steps = diag.diagnostic_node(dict(STATE))["troubleshooting_steps"]

    assert [s.action for s in steps] == ["Real check"]


def test_validator_receives_only_the_cited_chunk_text(monkeypatch):
    """Validation must judge a claim against its cited chunks only, not all retrieved text."""
    diagnosis = {
        "candidate_causes": [
            {"cause": "Heater failure", "likelihood": 0.5,
             "supporting_chunk_ids": ["c1", "missing"]},
        ],
        "recommended_checks": [],
    }
    model = FakeModel(diagnosis)
    _install(monkeypatch, _good_rag(), model)

    diag.diagnostic_node(dict(STATE))

    prompt = model.validation_prompts[0]
    assert "Heater failure prevents reaching setpoint." in prompt
    assert "Check the heater fuse and the thermocouple." not in prompt
    assert "Temperature cannot reach setpoint" in prompt  # engineer symptom included


def test_unsupported_cause_is_removed_and_supported_one_kept(monkeypatch):
    """Causes the validator marks UNSUPPORTED must be removed from the result."""
    _install(monkeypatch, _good_rag(), FakeModel(DIAGNOSIS, reject=("Thermocouple drift",)))

    causes = diag.diagnostic_node(dict(STATE))["root_causes"]

    assert [c.cause for c in causes] == ["Heater failure"]


def test_validator_verdict_is_case_and_whitespace_tolerant(monkeypatch):
    """A verdict like ' supported \\n' must still count as SUPPORTED."""
    _install(monkeypatch, _good_rag(), FakeModel(DIAGNOSIS, verdict_text="  supported \n"))

    update = diag.diagnostic_node(dict(STATE))

    assert len(update["root_causes"]) == 2
    assert len(update["troubleshooting_steps"]) == 2


@pytest.mark.parametrize("verdict", ["UNSUPPORTED", "Not SUPPORTED", "maybe", "SUPPORTED."])
def test_any_verdict_other_than_exactly_supported_removes_the_claim(monkeypatch, verdict):
    """Fail closed: ambiguous or malformed validator verdicts must drop the claim."""
    _install(monkeypatch, _good_rag(), FakeModel(DIAGNOSIS, verdict_text=verdict))

    update = diag.diagnostic_node(dict(STATE))

    assert update["root_causes"] == []
    assert update["troubleshooting_steps"] == []


# --------------------------------------------------------------------------- #
# Unsupported claims: answer text                                              #
# --------------------------------------------------------------------------- #

def test_answer_lists_only_validated_causes_and_steps(monkeypatch):
    """The engineer-facing answer must not mention a cause or check that validation removed."""
    _install(monkeypatch, _good_rag(),
             FakeModel(DIAGNOSIS, reject=("Thermocouple drift", "Check heater fuse")))

    answer = diag.diagnostic_node(dict(STATE))["final_answer"]

    assert "1. Heater failure" in answer
    assert "Thermocouple drift" not in answer
    assert "Check heater fuse" not in answer
    assert "2. Inspect thermocouple" in answer
    assert "The root cause is not confirmed." in answer


def test_all_claims_removed_gives_not_confirmed_answer_without_escalation(monkeypatch):
    """With evidence but no supported cause, the node says so and lists no cause."""
    _install(monkeypatch, _good_rag(), FakeModel(DIAGNOSIS, verdict_text="UNSUPPORTED"))

    update = diag.diagnostic_node(dict(STATE))

    assert "does not explicitly support a" in update["final_answer"]
    assert "Documented candidate conditions" not in update["final_answer"]
    assert update["sufficiency"] is True
    assert update["escalated"] is False


def test_model_assessment_text_is_never_shown_to_the_engineer(monkeypatch):
    """The free-text assessment from the model must be replaced by the fixed conservative text."""
    diagnosis = {"assessment": "The heater is definitely broken.",
                 "root_cause_confirmed": True,
                 "candidate_causes": [], "recommended_checks": []}
    _install(monkeypatch, _good_rag(), FakeModel(diagnosis))

    answer = diag.diagnostic_node(dict(STATE))["final_answer"]

    assert "definitely broken" not in answer
    assert "not confirmed" in answer


def test_empty_model_lists_yield_empty_results(monkeypatch):
    """An empty or missing candidate_causes / recommended_checks must produce empty lists."""
    _install(monkeypatch, _good_rag(), FakeModel({}))

    update = diag.diagnostic_node(dict(STATE))

    assert update["root_causes"] == []
    assert update["troubleshooting_steps"] == []


# --------------------------------------------------------------------------- #
# Escalation                                                                   #
# --------------------------------------------------------------------------- #

def test_insufficient_evidence_escalates_with_rag_reason_and_skips_the_model(monkeypatch):
    """Non-GOOD retrieval must escalate with the RAG reason and never call the model."""
    rag_result = {
        "retrieval_status": "INSUFFICIENT",
        "results": [_chunk_dict("c9")],
        "retrieval_attempts": 3,
        "senior_reason": "Manual does not cover this fault.",
    }
    model = FakeModel(DIAGNOSIS)
    _install(monkeypatch, rag_result, model)

    update = diag.diagnostic_node(dict(STATE))

    assert update["escalated"] is True
    assert update["escalation_reason"] == "Manual does not cover this fault."
    assert update["sufficiency"] is False
    assert update["root_causes"] == []
    assert update["troubleshooting_steps"] == []
    assert update["retry_count"] == 2
    assert "Human escalation is required." in update["final_answer"]
    assert update["retrieved_chunks"][0].chunk_id == "c9"
    assert model.generation_calls == 0 and model.validation_prompts == []


def test_escalation_uses_default_reason_when_rag_gives_none(monkeypatch):
    """A missing or empty senior_reason must fall back to the default insufficiency reason."""
    rag_result = {"retrieval_status": "POOR", "results": [],
                  "retrieval_attempts": 0, "senior_reason": ""}
    _install(monkeypatch, rag_result, FakeModel(DIAGNOSIS))

    update = diag.diagnostic_node(dict(STATE))

    assert update["escalated"] is True
    assert "does not provide sufficient evidence" in update["escalation_reason"]
    assert update["retry_count"] == 0  # never negative


# --------------------------------------------------------------------------- #
# Input guards, malformed output, safety                                       #
# --------------------------------------------------------------------------- #

def test_missing_query_raises_before_any_retrieval(monkeypatch):
    """An empty query must be rejected without touching retrieval or the model."""
    rag = _install(monkeypatch, _good_rag(), FakeModel(DIAGNOSIS))

    with pytest.raises(ValueError, match="requires a query"):
        diag.diagnostic_node({"raw_query": "   ", "equipment_model": "thermal_station"})

    assert rag.calls == []


def test_missing_equipment_model_raises_before_any_retrieval(monkeypatch):
    """A missing equipment model must be rejected so retrieval is never unscoped."""
    rag = _install(monkeypatch, _good_rag(), FakeModel(DIAGNOSIS))

    with pytest.raises(ValueError, match="requires equipment_model"):
        diag.diagnostic_node({"raw_query": "x", "equipment_model": None})

    assert rag.calls == []


def test_malformed_json_never_yields_an_ungrounded_diagnosis(monkeypatch):
    """Malformed model JSON must not produce causes or steps (currently it raises)."""
    _install(monkeypatch, _good_rag(), FakeModel("{not json"))

    try:
        update = diag.diagnostic_node(dict(STATE))
    except json.JSONDecodeError:
        return  # fails loudly: no ungrounded output reaches the engineer
    assert update["root_causes"] == [] and update["troubleshooting_steps"] == []


def test_step_has_no_executable_or_actuator_field(monkeypatch):
    """Steps are advice only: the type has no command field and extra model keys are ignored."""
    assert {f.name for f in fields(Step)} == {
        "order", "action", "supporting_chunk_ids", "hazard_flag"}
    diagnosis = {
        "candidate_causes": [],
        "recommended_checks": [{
            "action": "Check fuse", "supporting_chunk_ids": ["c2"],
            "command": "set_heater(0)", "execute": True,
        }],
    }
    _install(monkeypatch, _good_rag(), FakeModel(diagnosis))

    step = diag.diagnostic_node(dict(STATE))["troubleshooting_steps"][0]

    assert not hasattr(step, "command") and not hasattr(step, "execute")


def test_hazard_flag_is_passed_through_not_inferred_by_the_node(monkeypatch):
    """Hazard classification belongs to the Safety agent: the node only relays the model flag."""
    diagnosis = {
        "candidate_causes": [],
        "recommended_checks": [{
            "action": "Open the live high-voltage cabinet and measure",
            "supporting_chunk_ids": ["c2"],
        }],
    }
    _install(monkeypatch, _good_rag(), FakeModel(diagnosis))

    step = diag.diagnostic_node(dict(STATE))["troubleshooting_steps"][0]

    assert step.hazard_flag is False


def test_format_context_includes_ids_and_provenance_for_every_chunk():
    """The model prompt must expose chunk ids and provenance so it can cite them."""
    text = diag._format_diagnostic_context([_chunk("c1", "alpha"), _chunk("c2", "beta")])

    assert "[EVIDENCE 1]" in text and "[EVIDENCE 2]" in text
    assert "Chunk ID: c1" in text and "Chunk ID: c2" in text
    assert "Page: 12" in text and "aem_thermal_station.pdf" in text
    assert "alpha" in text and "beta" in text
