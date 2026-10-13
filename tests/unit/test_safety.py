"""Unit tests for the Safety agent decision flow (safety_node).

Every model call and every search is replaced by a fake: no network, no key, no index.
The grounding validators and the deterministic rules have their own files
(test_safety_grounding.py, test_safety_rules.py); these tests pin the DECISION FLOW and,
above all, that every failure mode ends in a cautious outcome (ESCALATE or HALT), never ALLOW.
"""

import json
from types import SimpleNamespace

import pytest

import asa.agents.safety as safety
from asa.graph.state import Chunk, SafetyClaim

POWER_OFF = Chunk(
    chunk_id="c_power",
    doc_id="doc",
    section_id="2.2.2",
    revision="0",
    equipment_model="thermal_station",
    text="Hazardous voltages may exist. Power off before servicing.",
    score=0.9,
    source_file="manual.pdf",
    page=19,
    section_title="2.2.2 General Safety Guidelines",
)
PPE_PAGE = Chunk(
    chunk_id="c_ppe",
    doc_id="doc",
    section_id="2.2.3",
    revision="0",
    equipment_model="thermal_station",
    text="Safety goggles are required when servicing.",
    score=0.8,
    source_file="manual.pdf",
    page=22,
    section_title="2.2.3 Personal Safety Equipment",
)

HAZARD = "Hazardous voltages may exist"
PPE = "Safety goggles"
CONTROL = "Power off before servicing"

STATE = {"raw_query": "What safety precautions apply before servicing?", "equipment_model": "thermal_station"}
SAFETY_QUERY = "servicing safety precautions power off"


def _response(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def _row(chunk, chunk_id=None):
    return {
        "chunk_id": chunk_id or chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "section_id": chunk.section_id,
        "revision": chunk.revision,
        "equipment_model": chunk.equipment_model,
        "text": chunk.text,
        "rrf_score": chunk.score,
        "source_file": chunk.source_file,
        "page": chunk.page,
        "section_title": chunk.section_title,
    }


def _stub_search(monkeypatch, rows=None, by_query=None, searched=None):
    """Replace query building and retrieval; `by_query` maps a query to its rows."""
    monkeypatch.setattr(safety, "_build_safety_search_query", lambda question, equipment_model: SAFETY_QUERY)

    def search(query, equipment_model, k):
        if searched is not None:
            searched.append({"query": query, "equipment_model": equipment_model, "k": k})
        if by_query is not None:
            return by_query.get(query, [])
        return [_row(POWER_OFF), _row(PPE_PAGE)] if rows is None else rows

    monkeypatch.setattr(safety.rag_agent, "search_service_documents", search)


def _install_model(monkeypatch, assess, supported=None, verdict_reply=None):
    """Fake model.

    assess        dict (JSON-encoded), str (raw reply) or Exception (raised) for the assessment call.
    supported     {claim text: [chunk ids]} the claim validator accepts; any other claim is rejected.
    verdict_reply dict/str/Exception for the ALLOW verdict check (default: supported true).
    Returns the list of (kind, prompt) calls, in order of arrival.
    """
    supported = supported or {}
    calls = []

    def reply(value):
        if isinstance(value, Exception):
            raise value
        return _response(value if isinstance(value, str) else json.dumps(value))

    def fake_create(**kwargs):
        prompt = kwargs["messages"][1]["content"]
        if "validating one structured safety claim" in prompt:
            calls.append(("claim", prompt))
            claim = prompt.split("Candidate claim:\n", 1)[1].split("\n", 1)[0]
            ids = supported.get(claim, [])
            return _response(json.dumps({"supported": bool(ids), "supporting_chunk_ids": ids}))
        if "validating a proposed equipment safety verdict" in prompt:
            calls.append(("verdict", prompt))
            return reply({"supported": True} if verdict_reply is None else verdict_reply)
        if "Return VALID JSON only" in prompt:
            calls.append(("assess", prompt))
            return reply(assess)
        raise AssertionError("Unexpected Safety Agent model call.")

    monkeypatch.setattr(safety.client.chat.completions, "create", fake_create)
    return calls


def _assessment(verdict="allow", hazards=(HAZARD,), ppe=(), controls=(), reason="model words"):
    return {
        "verdict": verdict,
        "reason": reason,
        "hazards": list(hazards),
        "ppe_required": list(ppe),
        "required_controls": list(controls),
    }


ALL_SUPPORTED = {HAZARD: ["c_power"], PPE: ["c_ppe"], CONTROL: ["c_power"]}


def _run(monkeypatch, assess, supported=None, verdict_reply=None, state=None, rows=None):
    _stub_search(monkeypatch, rows=rows)
    calls = _install_model(monkeypatch, assess, supported, verdict_reply)
    return safety.safety_node(dict(state or STATE)), calls


def _kinds(calls):
    return [kind for kind, _ in calls]


def _assert_cautious_escalation(result):
    assert result["safety_verdict"] == "escalate"
    assert result["escalated"] is True
    assert result["sufficiency"] is False
    assert result["final_answer"].startswith("Safety decision: ESCALATE")
    assert "Human review is required" in result["final_answer"]
    assert result["escalation_reason"]


# --------------------------------------------------------------------------- #
# The three verdicts and the state they return                                #
# --------------------------------------------------------------------------- #


def test_grounded_allow_reports_hazards_ppe_and_controls_in_state(monkeypatch):
    """A grounded ALLOW returns its validated items and does not escalate."""
    assess = _assessment("allow", [HAZARD], [PPE], [CONTROL])
    result, _ = _run(monkeypatch, assess, ALL_SUPPORTED)
    assert result["safety_verdict"] == "allow"
    assert result["hazards"] == [HAZARD]
    assert result["ppe_required"] == [PPE]
    assert result["escalated"] is False
    assert result["escalation_reason"] == ""
    assert result["sufficiency"] is True
    assert result["retrieval_strategy"] == "hybrid"
    assert result["retry_count"] == 0
    answer = result["final_answer"]
    assert answer.startswith("Safety decision: ALLOW")
    assert f"Documented hazards:\n- {HAZARD}" in answer
    assert f"Documented PPE:\n- {PPE}" in answer
    assert f"Required documented controls:\n- {CONTROL}" in answer


def test_halt_is_kept_without_escalation_and_without_a_verdict_check(monkeypatch):
    """A grounded HALT is final: not escalated, not re-validated, evidence still listed."""
    assess = _assessment("halt", [HAZARD], [], [CONTROL])
    result, calls = _run(monkeypatch, assess, ALL_SUPPORTED)
    assert result["safety_verdict"] == "halt"
    assert result["escalated"] is False
    assert result["escalation_reason"] == ""
    assert result["sufficiency"] is True
    assert "conflicts with the documented safety requirements" in result["safety_reason"]
    assert result["final_answer"].startswith("Safety decision: HALT")
    assert "verdict" not in _kinds(calls)


def test_halt_without_any_validated_item_stays_halt(monkeypatch):
    """Validation only adds caution: a HALT with nothing validated must not be downgraded to ALLOW."""
    result, _ = _run(monkeypatch, _assessment("halt", ["Invented hazard"]), supported={})
    assert result["safety_verdict"] == "halt"
    assert result["hazards"] == []
    assert result["escalated"] is False


def test_model_escalate_returns_the_escalation_text_and_reason(monkeypatch):
    """A model ESCALATE becomes an escalation with the fixed human-review wording."""
    result, calls = _run(monkeypatch, _assessment("escalate", []), supported={})
    _assert_cautious_escalation(result)
    assert result["escalation_reason"] == result["safety_reason"]
    assert "insufficient to make a grounded safety determination" in result["final_answer"]
    assert "verdict" not in _kinds(calls)


def test_verdict_is_normalised_for_case_and_whitespace(monkeypatch):
    """' HALT ' and 'Allow' are read as the verdicts they spell."""
    result, _ = _run(monkeypatch, _assessment(" HALT "), {HAZARD: ["c_power"]})
    assert result["safety_verdict"] == "halt"
    result, _ = _run(monkeypatch, _assessment("Allow"), {HAZARD: ["c_power"]})
    assert result["safety_verdict"] == "allow"


# --------------------------------------------------------------------------- #
# Unsupported or unclear ALLOW must never stand                               #
# --------------------------------------------------------------------------- #


def test_allow_rejected_by_the_verdict_check_becomes_escalate(monkeypatch):
    """The model says ALLOW with valid items, the evidence check says unsupported: ESCALATE."""
    result, calls = _run(
        monkeypatch, _assessment("allow", [HAZARD]), {HAZARD: ["c_power"]}, verdict_reply={"supported": False}
    )
    _assert_cautious_escalation(result)
    assert "verdict" in _kinds(calls)
    assert "Safety decision: ALLOW" not in result["final_answer"]


@pytest.mark.parametrize(
    "reply",
    [{"supported": "yes"}, {"supported": 1}, {"unrelated": True}, [True], "plain words", "", "{broken"],
)
def test_allow_with_an_unclear_verdict_check_becomes_escalate(monkeypatch, reply):
    """Only an explicit boolean true keeps an ALLOW; anything else escalates."""
    result, _ = _run(monkeypatch, _assessment("allow", [HAZARD]), {HAZARD: ["c_power"]}, verdict_reply=reply)
    assert result["safety_verdict"] == "escalate"
    assert result["escalated"] is True


def test_allow_whose_verdict_check_raises_becomes_escalate(monkeypatch):
    """An API failure while checking an ALLOW escalates instead of crashing or allowing."""
    result, _ = _run(
        monkeypatch, _assessment("allow", [HAZARD]), {HAZARD: ["c_power"]}, verdict_reply=RuntimeError("down")
    )
    _assert_cautious_escalation(result)


@pytest.mark.parametrize("verdict", ["proceed", "approved", "ALLOWED", "", None, 5, ["allow"]])
def test_unknown_verdict_values_escalate(monkeypatch, verdict):
    """An unrecognised verdict is never read as permission."""
    result, _ = _run(monkeypatch, _assessment(verdict, [HAZARD]), {HAZARD: ["c_power"]})
    assert result["safety_verdict"] == "escalate"
    assert result["escalated"] is True


def test_missing_verdict_key_escalates(monkeypatch):
    """A reply with no verdict at all escalates."""
    assess = {"reason": "x", "hazards": [HAZARD], "ppe_required": [], "required_controls": []}
    result, _ = _run(monkeypatch, assess, {HAZARD: ["c_power"]})
    assert result["safety_verdict"] == "escalate"


def test_allow_with_only_invented_chunk_ids_escalates(monkeypatch):
    """A claim 'supported' only by chunk ids that were never retrieved is dropped, so the ALLOW escalates."""
    result, _ = _run(monkeypatch, _assessment("allow", [HAZARD]), {HAZARD: ["made_up_id"]})
    assert result["safety_verdict"] == "escalate"
    assert result["hazards"] == []
    assert result["safety_claims"] == []


# --------------------------------------------------------------------------- #
# Missing or insufficient evidence, model failure: fail towards caution       #
# --------------------------------------------------------------------------- #


def test_no_retrieved_evidence_escalates_without_calling_the_model(monkeypatch):
    """With nothing retrieved there is no grounded decision: ESCALATE and no assessment call."""
    _stub_search(monkeypatch, rows=[])
    calls = _install_model(monkeypatch, _assessment("allow"))
    result = safety.safety_node(dict(STATE))
    assert calls == []
    _assert_cautious_escalation(result)
    assert result["retrieved_chunks"] == []
    assert result["hazards"] == []
    assert result["ppe_required"] == []
    assert result["safety_claims"] == []
    assert "No relevant AEM safety documentation was retrieved" in result["final_answer"]


@pytest.mark.parametrize(
    "reply",
    ["", "not json", "{broken", "null", "42", '"allow"', json.dumps(["allow"])],
)
def test_unparseable_or_non_object_assessment_escalates(monkeypatch, reply):
    """Garbage from the model is escalated, with the retrieved evidence kept for the reviewer."""
    result, calls = _run(monkeypatch, reply)
    _assert_cautious_escalation(result)
    assert "could not be completed" in result["final_answer"]
    assert [c.chunk_id for c in result["retrieved_chunks"]] == ["c_power", "c_ppe"]
    assert _kinds(calls) == ["assess"]


def test_assessment_with_empty_content_or_no_choices_escalates(monkeypatch):
    """A reply with content None or no choices escalates."""
    _stub_search(monkeypatch)
    monkeypatch.setattr(safety.client.chat.completions, "create", lambda **kw: _response(None))
    _assert_cautious_escalation(safety.safety_node(dict(STATE)))
    monkeypatch.setattr(
        safety.client.chat.completions, "create", lambda **kw: SimpleNamespace(choices=[])
    )
    _assert_cautious_escalation(safety.safety_node(dict(STATE)))


@pytest.mark.parametrize("error", [RuntimeError("api down"), TimeoutError("slow"), ConnectionError("net")])
def test_model_errors_escalate_instead_of_crashing(monkeypatch, error):
    """Any exception from the assessment call ends as an ESCALATE decision."""
    result, _ = _run(monkeypatch, error)
    _assert_cautious_escalation(result)
    assert result["hazards"] == []
    assert result["safety_claims"] == []


def test_a_model_allow_with_a_wrongly_typed_item_lists_still_cannot_allow(monkeypatch):
    """Hazards sent as numbers or objects give no validated item, so the ALLOW escalates."""
    assess = {"verdict": "allow", "reason": "x", "hazards": {"a": 1}, "ppe_required": None, "required_controls": 7}
    result, _ = _run(monkeypatch, assess, ALL_SUPPORTED)
    assert result["safety_verdict"] == "escalate"


def test_a_string_instead_of_a_list_is_one_item_end_to_end(monkeypatch):
    """A hazard sent as one string is validated as that one string, not letter by letter."""
    assess = {"verdict": "allow", "reason": "x", "hazards": HAZARD, "ppe_required": [], "required_controls": []}
    result, calls = _run(monkeypatch, assess, {HAZARD: ["c_power"]})
    assert result["hazards"] == [HAZARD]
    assert _kinds(calls).count("claim") == 1


# --------------------------------------------------------------------------- #
# PPE and hazards are validated against evidence                              #
# --------------------------------------------------------------------------- #


def test_unsupported_ppe_is_dropped_from_state_and_answer(monkeypatch):
    """PPE the evidence does not support is neither reported in state nor shown to the engineer."""
    assess = _assessment("allow", [HAZARD], ["Arc-flash suit", PPE])
    result, _ = _run(monkeypatch, assess, {HAZARD: ["c_power"], PPE: ["c_ppe"]})
    assert result["ppe_required"] == [PPE]
    assert "Arc-flash suit" not in result["final_answer"]
    assert [c.text for c in result["safety_claims"] if c.category == "ppe"] == [PPE]


def test_unsupported_hazard_is_dropped_but_a_supported_one_keeps_the_allow(monkeypatch):
    """Only validated hazards are reported; the ALLOW rests on the one that survived."""
    assess = _assessment("allow", ["Radiation hazard", HAZARD])
    result, _ = _run(monkeypatch, assess, {HAZARD: ["c_power"]})
    assert result["safety_verdict"] == "allow"
    assert result["hazards"] == [HAZARD]
    assert "Radiation hazard" not in result["final_answer"]


def test_ppe_alone_is_enough_to_back_an_allow(monkeypatch):
    """A validated PPE item is a documented requirement: the ALLOW stands."""
    result, _ = _run(monkeypatch, _assessment("allow", [], [PPE]), {PPE: ["c_ppe"]})
    assert result["safety_verdict"] == "allow"
    assert result["ppe_required"] == [PPE]


def test_the_verdict_check_only_sees_validated_controls(monkeypatch):
    """The ALLOW check is told the controls that passed validation, not the invented ones."""
    assess = _assessment("allow", [HAZARD], [], [CONTROL, "Wear a lead apron"])
    _, calls = _run(monkeypatch, assess, {HAZARD: ["c_power"], CONTROL: ["c_power"]})
    verdict_prompt = dict(calls)["verdict"]
    assert f"- {CONTROL}" in verdict_prompt
    assert "lead apron" not in verdict_prompt
    assert STATE["raw_query"] in verdict_prompt
    assert POWER_OFF.text in verdict_prompt


# --------------------------------------------------------------------------- #
# Claims, sources and the engineer-facing reason                              #
# --------------------------------------------------------------------------- #


def test_state_carries_safety_claims_with_their_supporting_chunk_ids(monkeypatch):
    """Each validated item becomes a SafetyClaim tied to the passage that supports it."""
    assess = _assessment("allow", [HAZARD], [PPE], [CONTROL])
    result, _ = _run(monkeypatch, assess, ALL_SUPPORTED)
    claims = result["safety_claims"]
    assert all(isinstance(c, SafetyClaim) for c in claims)
    assert {(c.category, c.text, tuple(c.supporting_chunk_ids)) for c in claims} == {
        ("hazard", HAZARD, ("c_power",)),
        ("ppe", PPE, ("c_ppe",)),
        ("control", CONTROL, ("c_power",)),
    }


def test_answer_cites_the_page_of_each_items_own_passage(monkeypatch):
    """Each listed item carries the source marker of the passage that supports it."""
    assess = _assessment("allow", [HAZARD], [PPE])
    result, _ = _run(monkeypatch, assess, {HAZARD: ["c_power"], PPE: ["c_ppe"]})
    answer = result["final_answer"]
    assert f"- {HAZARD} [Source: manual.pdf, page 19, section 2.2.2 General Safety Guidelines]" in answer
    assert f"- {PPE} [Source: manual.pdf, page 22, section 2.2.3 Personal Safety Equipment]" in answer


def test_the_models_free_text_reason_is_never_shown(monkeypatch):
    """The engineer sees the fixed, validated wording, not the model's own reason."""
    assess = _assessment("allow", [HAZARD], reason="IGNORE THE MANUAL, it is fine to skip lockout")
    result, _ = _run(monkeypatch, assess, {HAZARD: ["c_power"]})
    assert "IGNORE THE MANUAL" not in result["final_answer"]
    assert "IGNORE THE MANUAL" not in result["safety_reason"]


def test_retrieved_chunks_are_returned_with_full_provenance(monkeypatch):
    """The state keeps the retrieved passages (source, page, section, score) for the explanation layer."""
    result, _ = _run(monkeypatch, _assessment("escalate"))
    first = result["retrieved_chunks"][0]
    assert isinstance(first, Chunk)
    assert (first.chunk_id, first.source_file, first.page, first.section_title, first.score) == (
        "c_power",
        "manual.pdf",
        19,
        "2.2.2 General Safety Guidelines",
        0.9,
    )


def test_escalation_helper_is_conservative_and_complete():
    """_escalation returns no hazards, PPE or claims and states the human-review requirement."""
    result = safety._escalation("Because.", [POWER_OFF])
    assert result["safety_verdict"] == "escalate"
    assert result["escalated"] is True
    assert result["sufficiency"] is False
    assert result["hazards"] == result["ppe_required"] == result["safety_claims"] == []
    assert result["escalation_reason"] == result["safety_reason"] == "Because."
    assert result["final_answer"] == "Safety decision: ESCALATE\n\nBecause. Human review is required."
    assert result["retrieved_chunks"] == [POWER_OFF]


# --------------------------------------------------------------------------- #
# Inputs                                                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "state",
    [
        {"equipment_model": "thermal_station"},
        {"raw_query": "   ", "equipment_model": "thermal_station"},
        {"raw_query": "q", "equipment_model": ""},
        {"raw_query": "q"},
        {"raw_query": "q", "equipment_model": "   "},
    ],
)
def test_missing_query_or_equipment_is_rejected_before_any_search(monkeypatch, state):
    """No query or no equipment is a caller error: no search and no decision is made."""
    searched = []
    _stub_search(monkeypatch, searched=searched)
    with pytest.raises(ValueError):
        safety.safety_node(state)
    assert searched == []


def test_resolved_query_is_used_in_preference_to_the_raw_query(monkeypatch):
    """The resolved (follow-up aware) question is what is searched and assessed."""
    searched = []
    _stub_search(monkeypatch, searched=searched)
    calls = _install_model(monkeypatch, _assessment("escalate"))
    state = {"raw_query": "and then?", "resolved_query": "Can I open the panel while powered?", "equipment_model": "e1"}
    safety.safety_node(state)
    assert [s["query"] for s in searched] == [SAFETY_QUERY, "Can I open the panel while powered?"]
    assert "Can I open the panel while powered?" in dict(calls)["assess"]
    assert all(s["equipment_model"] == "e1" for s in searched)


# --------------------------------------------------------------------------- #
# Search: safety query AND the engineer's own wording                         #
# --------------------------------------------------------------------------- #


def test_search_runs_for_the_safety_query_then_the_engineers_wording(monkeypatch):
    """Both queries are searched, safety query first, each with SEARCH_K passages."""
    searched = []
    _stub_search(monkeypatch, searched=searched)
    _install_model(monkeypatch, _assessment("escalate"))
    safety.safety_node(dict(STATE))
    assert [s["query"] for s in searched] == [SAFETY_QUERY, STATE["raw_query"]]
    assert {s["k"] for s in searched} == {safety.SEARCH_K}


def test_a_passage_found_only_by_the_engineers_wording_reaches_the_model(monkeypatch):
    """Evidence found only with the engineer's own words is part of the assessed evidence."""
    _stub_search(
        monkeypatch,
        by_query={SAFETY_QUERY: [_row(POWER_OFF)], STATE["raw_query"]: [_row(PPE_PAGE)]},
    )
    calls = _install_model(monkeypatch, _assessment("escalate"))
    result = safety.safety_node(dict(STATE))
    assert [c.chunk_id for c in result["retrieved_chunks"]] == ["c_power", "c_ppe"]
    assert "Chunk ID: c_ppe" in dict(calls)["assess"]


def test_one_search_when_the_safety_query_equals_the_question(monkeypatch):
    """No duplicate search when the generated query is the engineer's own sentence."""
    searched = []
    _stub_search(monkeypatch, searched=searched)
    monkeypatch.setattr(safety, "_build_safety_search_query", lambda question, equipment_model: question)
    _install_model(monkeypatch, _assessment("escalate"))
    safety.safety_node(dict(STATE))
    assert [s["query"] for s in searched] == [STATE["raw_query"]]


def test_an_empty_generated_query_falls_back_to_the_question(monkeypatch):
    """A blank generated query still searches, with the engineer's wording."""
    searched = []
    _stub_search(monkeypatch, searched=searched)
    monkeypatch.setattr(safety, "_build_safety_search_query", lambda question, equipment_model: "")
    _install_model(monkeypatch, _assessment("escalate"))
    safety.safety_node(dict(STATE))
    assert [s["query"] for s in searched] == [STATE["raw_query"]]


def test_the_evidence_is_capped_at_the_maximum_passages(monkeypatch):
    """Duplicates are removed and no more than MAX_SAFETY_PASSAGES reach the model."""
    rows = [_row(POWER_OFF, chunk_id=f"x{i}") for i in range(5)]
    _stub_search(monkeypatch, by_query={SAFETY_QUERY: rows, STATE["raw_query"]: rows[:3] + [_row(PPE_PAGE, "y0")]})
    _install_model(monkeypatch, _assessment("escalate"))
    result = safety.safety_node(dict(STATE))
    ids = [c.chunk_id for c in result["retrieved_chunks"]]
    assert len(ids) == safety.MAX_SAFETY_PASSAGES
    assert len(set(ids)) == len(ids)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def test_format_safety_context_lists_every_passage_with_provenance():
    """The evidence block names rank, chunk id, source, page, section and text of each passage."""
    text = safety._format_safety_context([POWER_OFF, PPE_PAGE])
    assert "[EVIDENCE 1]" in text and "[EVIDENCE 2]" in text
    assert "Chunk ID: c_power" in text and "Chunk ID: c_ppe" in text
    assert "Page: 19" in text and "Source: manual.pdf" in text
    assert "Section: 2.2.2 General Safety Guidelines" in text
    assert POWER_OFF.text in text
    assert safety._format_safety_context([]) == ""


def test_search_query_builder_cleans_the_reply_and_sends_the_request(monkeypatch):
    """The generated query is stripped of quotes and spaces and built from the request and equipment."""
    seen = {}

    def fake_create(**kwargs):
        seen.update(kwargs)
        return _response('  "panel access power off"  ')

    monkeypatch.setattr(safety.client.chat.completions, "create", fake_create)
    query = safety._build_safety_search_query("Can I open the panel?", "thermal_station")
    assert query == "panel access power off"
    prompt = seen["messages"][1]["content"]
    assert "Can I open the panel?" in prompt and "thermal_station" in prompt
    assert seen["temperature"] == 0
