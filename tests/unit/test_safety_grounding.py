"""Unit tests for Safety Agent grounding validation."""

import json
from types import SimpleNamespace

import pytest

import asa.agents.safety as safety
from asa.graph.state import Chunk

EVIDENCE = Chunk(
    chunk_id="safety_p19",
    doc_id="aem_thermal_station",
    section_id="2.2.2",
    revision="0",
    equipment_model="thermal_station",
    text=(
        "Electrical Hazards: Hazardous voltages may exist "
        "and can cause electric shock or burns. "
        "Only qualified personnel should service this equipment. "
        "Perform equipment or module power down and power off "
        "before servicing."
    ),
    score=1.0,
    source_file="aem_thermal_station.pdf",
    page=19,
    section_title="2.2.2 General Safety Guidelines",
)


INSTALLATION_PPE_EVIDENCE = Chunk(
    chunk_id="safety_p22",
    doc_id="aem_thermal_station",
    section_id="2.2.3",
    revision="0",
    equipment_model="thermal_station",
    text=(
        "Installation or de-commissioning of AEM Thermal "
        "Station requires full preparation in advance. "
        "Standard Personal Protective Equipment is required "
        "as listed below: Safety goggles (non-conductive "
        "frame), Safety boots (steel toe), Smock / Jump suit, "
        "Hard top helmet / bump cap, Safety glove."
    ),
    score=1.0,
    source_file="aem_thermal_station.pdf",
    page=22,
    section_title="2.2.3 Personal Safety Equipment",
)


def _response(text: str):
    """
    Build the minimal fake OpenAI response shape expected
    by _validate_safety_items().
    """

    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def test_validator_retains_supported_and_removes_unsupported_items(
    monkeypatch,
):
    """
    The validator must retain claims marked SUPPORTED and
    remove claims marked UNSUPPORTED.
    """

    def fake_create(**kwargs):

        prompt = kwargs["messages"][1]["content"]

        supported_claims = [
            "Electrical Hazards",
            ("Only qualified personnel should service this equipment."),
            ("Perform equipment or module power down and power off before servicing."),
        ]

        for claim in supported_claims:
            candidate_block = f"Candidate claim:\n{claim}\n"

            if candidate_block in prompt:
                return _response(
                    json.dumps(
                        {
                            "supported": True,
                            "supporting_chunk_ids": ["safety_p19"],
                        }
                    )
                )

        return _response(
            json.dumps(
                {
                    "supported": False,
                    "supporting_chunk_ids": [],
                }
            )
        )

    monkeypatch.setattr(
        safety.client.chat.completions,
        "create",
        fake_create,
    )

    hazards = [
        "Electrical Hazards",
        "Radiation hazard",
    ]

    ppe_required = [
        "Arc-flash suit",
    ]

    required_controls = [
        "Only qualified personnel should service this equipment.",
        ("Perform equipment or module power down and power off before servicing."),
        "Disable all safety interlocks before servicing.",
    ]

    (
        valid_hazards,
        valid_ppe,
        valid_controls,
        safety_claims,
    ) = safety._validate_safety_items(
        question=("What safety precautions apply before servicing the equipment?"),
        chunks=[EVIDENCE],
        hazards=hazards,
        ppe_required=ppe_required,
        required_controls=required_controls,
    )

    assert valid_hazards == [
        "Electrical Hazards",
    ]

    assert valid_ppe == []

    assert valid_controls == [
        "Only qualified personnel should service this equipment.",
        ("Perform equipment or module power down and power off before servicing."),
    ]

    assert len(safety_claims) == 3

    claims_by_text = {claim.text: claim for claim in safety_claims}

    assert claims_by_text["Electrical Hazards"].supporting_chunk_ids == ["safety_p19"]

    assert claims_by_text["Only qualified personnel should service this equipment."].supporting_chunk_ids == [
        "safety_p19"
    ]

    assert claims_by_text[
        ("Perform equipment or module power down and power off before servicing.")
    ].supporting_chunk_ids == ["safety_p19"]


def test_validator_rejects_ppe_from_unrelated_activity(
    monkeypatch,
):
    """
    PPE appearing in retrieved evidence is not automatically
    applicable to a different engineer activity.
    """

    def fake_create(**kwargs):

        prompt = kwargs["messages"][1]["content"]

        assert "Safety goggles (non-conductive frame)" in prompt

        assert "Can I bypass the safety interlock?" in prompt

        # Simulate the grounding validator deciding that
        # installation/decommissioning PPE is not established
        # as applicable to an interlock-bypass request.
        return _response(
            json.dumps(
                {
                    "supported": False,
                    "supporting_chunk_ids": [],
                }
            )
        )

    monkeypatch.setattr(
        safety.client.chat.completions,
        "create",
        fake_create,
    )

    (
        valid_hazards,
        valid_ppe,
        valid_controls,
        safety_claims,
    ) = safety._validate_safety_items(
        question=("Can I bypass the safety interlock?"),
        chunks=[
            INSTALLATION_PPE_EVIDENCE,
        ],
        hazards=[],
        ppe_required=[
            "Safety goggles (non-conductive frame)",
        ],
        required_controls=[],
    )

    assert valid_hazards == []

    assert valid_ppe == []

    assert valid_controls == []

    assert safety_claims == []


def test_supported_allow_verdict_is_retained(
    monkeypatch,
):
    """
    A grounded ALLOW verdict remains ALLOW.
    """

    def fake_create(**kwargs):

        prompt = kwargs["messages"][1]["content"]

        assert "Proposed verdict:\nALLOW" in prompt

        return _response(json.dumps({"supported": True}))

    monkeypatch.setattr(
        safety.client.chat.completions,
        "create",
        fake_create,
    )

    verdict = safety._validate_safety_verdict(
        question=("What safety precautions apply before servicing the equipment?"),
        chunks=[EVIDENCE],
        candidate_verdict="allow",
        validated_controls=[
            ("Perform equipment or module power down and power off before servicing."),
        ],
    )

    assert verdict == "allow"


def test_supported_halt_verdict_is_retained(
    monkeypatch,
):
    """
    A HALT verdict remains HALT and is not re-validated:
    validation may only add caution, never weaken a
    prohibition into "insufficient evidence".
    """

    def fake_create(**kwargs):
        raise AssertionError("HALT must not be sent for validation.")

    monkeypatch.setattr(
        safety.client.chat.completions,
        "create",
        fake_create,
    )

    verdict = safety._validate_safety_verdict(
        question=("Can I service the equipment without powering it off?"),
        chunks=[EVIDENCE],
        candidate_verdict="halt",
        validated_controls=[
            ("Perform equipment or module power down and power off before servicing."),
        ],
    )

    assert verdict == "halt"


def test_unsupported_allow_verdict_becomes_escalate(
    monkeypatch,
):
    """
    An ALLOW decision that cannot be grounded must never
    survive. It becomes ESCALATE.
    """

    def fake_create(**kwargs):

        prompt = kwargs["messages"][1]["content"]

        assert "Proposed verdict:\nALLOW" in prompt

        return _response(json.dumps({"supported": False}))

    monkeypatch.setattr(
        safety.client.chat.completions,
        "create",
        fake_create,
    )

    verdict = safety._validate_safety_verdict(
        question=("Can I perform this action while the equipment is powered?"),
        chunks=[EVIDENCE],
        candidate_verdict="allow",
        validated_controls=[],
    )

    assert verdict == "escalate"


def test_safety_node_does_not_expose_unvalidated_llm_reason(
    monkeypatch,
):
    """
    The Safety Agent's original free-text reason must not
    survive grounding validation into the final answer.
    """

    # --------------------------------------------------
    # Fake retrieval
    # --------------------------------------------------

    monkeypatch.setattr(
        safety,
        "_build_safety_search_query",
        lambda question, equipment_model: "servicing safety power off",
    )

    monkeypatch.setattr(
        safety.rag_agent,
        "search_service_documents",
        lambda query, equipment_model, k: [
            {
                "chunk_id": EVIDENCE.chunk_id,
                "doc_id": EVIDENCE.doc_id,
                "section_id": EVIDENCE.section_id,
                "revision": EVIDENCE.revision,
                "equipment_model": EVIDENCE.equipment_model,
                "text": EVIDENCE.text,
                "rrf_score": 1.0,
                "source_file": EVIDENCE.source_file,
                "page": EVIDENCE.page,
                "section_title": EVIDENCE.section_title,
            }
        ],
    )

    # --------------------------------------------------
    # Fake initial Safety LLM output
    #
    # Deliberately include an invented unsafe reason.
    # --------------------------------------------------

    unsafe_reason = "The equipment can safely remain energized because internal isolation removes all risk."

    safety_json = {
        "verdict": "allow",
        "reason": unsafe_reason,
        "hazards": [
            "Electrical Hazards",
        ],
        "ppe_required": [],
        "required_controls": [
            ("Perform equipment or module power down and power off before servicing."),
        ],
    }

    # --------------------------------------------------
    # Mock the three LLM roles:
    #
    # 1. initial Safety generation -> JSON
    # 2. structured-item validation -> SUPPORTED
    # 3. verdict validation -> SUPPORTED
    # --------------------------------------------------

    def fake_create(**kwargs):

        prompt = kwargs["messages"][1]["content"]

        # 1. Structured safety-claim grounding.
        #
        # Check this BEFORE the generic "Return VALID JSON"
        # condition because this prompt also requests JSON.
        if "validating one structured safety claim" in prompt:
            return _response(
                json.dumps(
                    {
                        "supported": True,
                        "supporting_chunk_ids": [EVIDENCE.chunk_id],
                    }
                )
            )

        # 2. Final safety-verdict grounding.
        if "validating a proposed equipment safety verdict" in prompt:
            return _response(json.dumps({"supported": True}))

        # 3. Initial Safety Agent generation.
        #
        # Keep this generic JSON check LAST.
        if "Return VALID JSON only" in prompt:
            return _response(json.dumps(safety_json))

        raise AssertionError(f"Unexpected Safety Agent model call:\n{prompt}")

    monkeypatch.setattr(
        safety.client.chat.completions,
        "create",
        fake_create,
    )

    # --------------------------------------------------
    # Execute the real safety_node()
    # --------------------------------------------------

    result = safety.safety_node(
        {
            "raw_query": ("What safety precautions apply before servicing the equipment?"),
            "equipment_model": ("thermal_station"),
        }
    )

    # --------------------------------------------------
    # Grounded result
    # --------------------------------------------------

    assert result["safety_verdict"] == "allow"

    # The original LLM reason must not survive.
    assert unsafe_reason not in result["safety_reason"]

    assert unsafe_reason not in result["final_answer"]

    # The deterministic grounded reason should be used.
    assert "retrieved AEM manual evidence" in result["safety_reason"]

    # Validated structured evidence remains.
    assert result["hazards"] == [
        "Electrical Hazards",
    ]

    assert result["ppe_required"] == []

    assert result["escalated"] is False


def test_safety_node_escalates_when_allow_verdict_is_not_grounded(
    monkeypatch,
):
    """
    If the Safety LLM proposes ALLOW but verdict grounding
    rejects it, the final node result must be ESCALATE.
    """

    monkeypatch.setattr(
        safety,
        "_build_safety_search_query",
        lambda question, equipment_model: "powered servicing safety requirements",
    )

    monkeypatch.setattr(
        safety.rag_agent,
        "search_service_documents",
        lambda query, equipment_model, k: [
            {
                "chunk_id": EVIDENCE.chunk_id,
                "doc_id": EVIDENCE.doc_id,
                "section_id": EVIDENCE.section_id,
                "revision": EVIDENCE.revision,
                "equipment_model": EVIDENCE.equipment_model,
                "text": EVIDENCE.text,
                "rrf_score": 1.0,
                "source_file": EVIDENCE.source_file,
                "page": EVIDENCE.page,
                "section_title": EVIDENCE.section_title,
            }
        ],
    )

    unsafe_reason = "The equipment may remain powered because the requested work is safe while energized."

    safety_json = {
        "verdict": "allow",
        "reason": unsafe_reason,
        "hazards": [],
        "ppe_required": [],
        "required_controls": [],
    }

    def fake_create(**kwargs):

        prompt = kwargs["messages"][1]["content"]

        # Initial Safety Agent generation.
        if "Return VALID JSON only" in prompt:
            import json

            return _response(json.dumps(safety_json))

        # There are no structured items in this test,
        # so the next model call should be verdict
        # validation.
        if "validating a proposed equipment safety verdict" in prompt:
            return _response(json.dumps({"supported": False}))

        raise AssertionError("Unexpected Safety Agent model call.")

    monkeypatch.setattr(
        safety.client.chat.completions,
        "create",
        fake_create,
    )

    result = safety.safety_node(
        {
            "raw_query": ("Can I service the equipment while it is powered?"),
            "equipment_model": "thermal_station",
        }
    )

    # Candidate ALLOW must not survive.
    assert result["safety_verdict"] == "escalate"

    assert result["sufficiency"] is False

    assert result["escalated"] is True

    assert result["escalation_reason"]

    # The original unsupported reason must not leak.
    assert unsafe_reason not in result["safety_reason"]

    assert unsafe_reason not in result["final_answer"]

    assert "insufficient" in result["safety_reason"].lower()

    assert "ESCALATE" in result["final_answer"]


# --------------------------------------------------------------------------- #
# Verdict validation can only add caution                                     #
# --------------------------------------------------------------------------- #


def _validate_allow(monkeypatch, fake_create):
    monkeypatch.setattr(safety.client.chat.completions, "create", fake_create)
    return safety._validate_safety_verdict(
        question="What safety precautions apply before servicing?",
        chunks=[EVIDENCE],
        candidate_verdict="allow",
        validated_controls=["Power off before servicing."],
    )


@pytest.mark.parametrize(
    "reply",
    [
        "SUPPORTED",  # the old plain-text format is no longer accepted
        "not json at all",
        json.dumps({"supported": "true"}),  # a string, not the boolean true
        json.dumps({"supported": None}),
        json.dumps([True]),
    ],
)
def test_an_allow_without_an_explicit_true_becomes_escalate(monkeypatch, reply):
    assert _validate_allow(monkeypatch, lambda **kwargs: _response(reply)) == "escalate"


def test_an_allow_whose_validation_call_fails_becomes_escalate(monkeypatch):
    def broken(**kwargs):
        raise RuntimeError("API unavailable")

    assert _validate_allow(monkeypatch, broken) == "escalate"


def test_the_allow_validation_asks_for_a_json_answer(monkeypatch):
    seen = {}

    def fake_create(**kwargs):
        seen.update(kwargs)
        return _response(json.dumps({"supported": True}))

    assert _validate_allow(monkeypatch, fake_create) == "allow"
    assert seen["response_format"] == {"type": "json_object"}


def test_an_escalate_verdict_is_returned_without_a_validation_call(monkeypatch):
    def fake_create(**kwargs):
        raise AssertionError("ESCALATE must not be sent for validation.")

    monkeypatch.setattr(safety.client.chat.completions, "create", fake_create)
    verdict = safety._validate_safety_verdict(
        question="Can I do this?", chunks=[EVIDENCE], candidate_verdict="escalate", validated_controls=[]
    )
    assert verdict == "escalate"


# --------------------------------------------------------------------------- #
# Fail safe: a failed or malformed assessment escalates, never crashes        #
# --------------------------------------------------------------------------- #

SERVICING_REQUEST = {
    "raw_query": "Can I service the equipment while it is powered?",
    "equipment_model": "thermal_station",
}


def _evidence_rows():
    return [
        {
            "chunk_id": EVIDENCE.chunk_id,
            "doc_id": EVIDENCE.doc_id,
            "section_id": EVIDENCE.section_id,
            "revision": EVIDENCE.revision,
            "equipment_model": EVIDENCE.equipment_model,
            "text": EVIDENCE.text,
            "rrf_score": 1.0,
            "source_file": EVIDENCE.source_file,
            "page": EVIDENCE.page,
            "section_title": EVIDENCE.section_title,
        }
    ]


def _stub_retrieval(monkeypatch, searched=None):
    monkeypatch.setattr(safety, "_build_safety_search_query", lambda question, equipment_model: "power off servicing")

    def search(query, equipment_model, k):
        if searched is not None:
            searched.append(query)
        return _evidence_rows()

    monkeypatch.setattr(safety.rag_agent, "search_service_documents", search)


def _assert_escalated(result):
    assert result["safety_verdict"] == "escalate"
    assert result["escalated"] is True
    assert result["final_answer"].startswith("Safety decision: ESCALATE")
    assert "could not be completed" in result["final_answer"]
    assert [c.chunk_id for c in result["retrieved_chunks"]] == [EVIDENCE.chunk_id]  # evidence kept


@pytest.mark.parametrize("reply", ["{not valid json", json.dumps(["allow"]), ""])
def test_a_malformed_safety_assessment_escalates(monkeypatch, reply):
    _stub_retrieval(monkeypatch)
    monkeypatch.setattr(safety.client.chat.completions, "create", lambda **kwargs: _response(reply))
    _assert_escalated(safety.safety_node(dict(SERVICING_REQUEST)))


def test_a_failed_safety_assessment_call_escalates(monkeypatch):
    _stub_retrieval(monkeypatch)

    def broken(**kwargs):
        raise RuntimeError("API unavailable")

    monkeypatch.setattr(safety.client.chat.completions, "create", broken)
    _assert_escalated(safety.safety_node(dict(SERVICING_REQUEST)))


def test_a_single_string_is_one_item_not_a_list_of_characters():
    assert safety._text_list("high voltage") == ["high voltage"]
    assert safety._text_list(["high voltage", " ", 3]) == ["high voltage", "3"]
    assert safety._text_list({"hazard": "x"}) == []
    assert safety._text_list(None) == []


def test_a_failed_search_query_step_searches_with_the_engineer_question(monkeypatch):
    searched = []
    _stub_retrieval(monkeypatch, searched)

    def broken_query(question, equipment_model):
        raise RuntimeError("API unavailable")

    monkeypatch.setattr(safety, "_build_safety_search_query", broken_query)
    monkeypatch.setattr(
        safety.client.chat.completions, "create", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("down"))
    )
    safety.safety_node(dict(SERVICING_REQUEST))
    assert searched == [SERVICING_REQUEST["raw_query"]]


# --------------------------------------------------------------------------- #
# An ALLOW must rest on at least one validated item                           #
# --------------------------------------------------------------------------- #

PRECAUTIONS_REQUEST = {
    "raw_query": "What safety precautions apply before servicing?",
    "equipment_model": "thermal_station",
}


def _run_allow(monkeypatch, proposal, supported_items):
    """Run safety_node with a model that proposes ALLOW with `proposal`; only `supported_items` validate."""
    _stub_retrieval(monkeypatch)
    calls = []

    def fake_create(**kwargs):
        prompt = kwargs["messages"][1]["content"]
        if "validating one structured safety claim" in prompt:
            calls.append("claim")
            supported = any(f"Candidate claim:\n{item}\n" in prompt for item in supported_items)
            ids = [EVIDENCE.chunk_id] if supported else []
            return _response(json.dumps({"supported": supported, "supporting_chunk_ids": ids}))
        if "validating a proposed equipment safety verdict" in prompt:
            calls.append("verdict")
            return _response(json.dumps({"supported": True}))
        if "Return VALID JSON only" in prompt:
            calls.append("assess")
            return _response(json.dumps({"verdict": "allow", "reason": "x", **proposal}))
        raise AssertionError("Unexpected Safety Agent model call.")

    monkeypatch.setattr(safety.client.chat.completions, "create", fake_create)
    return safety.safety_node(dict(PRECAUTIONS_REQUEST)), calls


def test_an_allow_with_no_validated_item_becomes_escalate(monkeypatch):
    proposal = {
        "hazards": ["Invented hazard"],
        "ppe_required": ["Invented PPE"],
        "required_controls": ["Invented control"],
    }
    result, calls = _run_allow(monkeypatch, proposal, supported_items=[])
    assert result["safety_verdict"] == "escalate"
    assert result["escalated"] is True
    assert "verdict" not in calls  # no model call is spent validating an unsupported ALLOW


def test_an_allow_backed_only_by_a_validated_hazard_stays_allow(monkeypatch):
    proposal = {"hazards": ["Hazardous voltages may exist"], "ppe_required": [], "required_controls": []}
    result, _ = _run_allow(monkeypatch, proposal, supported_items=["Hazardous voltages may exist"])
    assert result["safety_verdict"] == "allow"
    assert "Follow the documented safety requirements below." in result["final_answer"]
    assert "Documented hazards:\n- Hazardous voltages may exist" in result["final_answer"]


# --------------------------------------------------------------------------- #
# Per-item evidence checks run in parallel and fail safe                      #
# --------------------------------------------------------------------------- #

import threading  # noqa: E402
import time  # noqa: E402


def _claim_in(prompt):
    return prompt.split("Candidate claim:\n", 1)[1].split("\n", 1)[0]


def _supporting(claims):
    """A fake model that supports exactly `claims`."""

    def fake_create(**kwargs):
        claim = _claim_in(kwargs["messages"][1]["content"])
        ok = claim in claims
        return _response(json.dumps({"supported": ok, "supporting_chunk_ids": [EVIDENCE.chunk_id] if ok else []}))

    return fake_create


def _validate(hazards, ppe, controls):
    return safety._validate_safety_items(
        question="What precautions apply?",
        chunks=[EVIDENCE],
        hazards=hazards,
        ppe_required=ppe,
        required_controls=controls,
    )


def test_parallel_validation_keeps_every_result_in_its_original_order(monkeypatch):
    hazards = [f"hazard {i}" for i in range(5)]
    ppe = [f"ppe {i}" for i in range(4)]
    controls = [f"control {i}" for i in range(4)]
    supported = set(hazards[::2] + ppe[1::2] + controls)
    monkeypatch.setattr(safety.client.chat.completions, "create", _supporting(supported))

    valid_hazards, valid_ppe, valid_controls, claims = _validate(hazards, ppe, controls)

    assert valid_hazards == ["hazard 0", "hazard 2", "hazard 4"]
    assert valid_ppe == ["ppe 1", "ppe 3"]
    assert valid_controls == controls
    assert [c.text for c in claims] == valid_hazards + valid_ppe + valid_controls
    assert [c.category for c in claims] == ["hazard"] * 3 + ["ppe"] * 2 + ["control"] * 4


def test_a_failed_check_drops_only_that_item(monkeypatch):
    fake = _supporting({"Electrical Hazards", "Power off before servicing."})

    def flaky(**kwargs):
        if _claim_in(kwargs["messages"][1]["content"]) == "Thermal Hazards":
            raise RuntimeError("API unavailable")
        return fake(**kwargs)

    monkeypatch.setattr(safety.client.chat.completions, "create", flaky)
    valid_hazards, _, valid_controls, _ = _validate(
        ["Electrical Hazards", "Thermal Hazards"], [], ["Power off before servicing."]
    )
    assert valid_hazards == ["Electrical Hazards"]
    assert valid_controls == ["Power off before servicing."]


def test_safety_node_survives_a_failing_item_check(monkeypatch):
    _stub_retrieval(monkeypatch)
    proposal = {
        "verdict": "allow",
        "reason": "x",
        "hazards": ["Electrical Hazards"],
        "ppe_required": [],
        "required_controls": [],
    }

    def fake_create(**kwargs):
        prompt = kwargs["messages"][1]["content"]
        if "validating one structured safety claim" in prompt:
            raise RuntimeError("API unavailable")
        if "Return VALID JSON only" in prompt:
            return _response(json.dumps(proposal))
        raise AssertionError("Unexpected Safety Agent model call.")

    monkeypatch.setattr(safety.client.chat.completions, "create", fake_create)
    result = safety.safety_node(dict(PRECAUTIONS_REQUEST))
    assert result["safety_verdict"] == "escalate"  # nothing validated, so the ALLOW cannot stand


# --------------------------------------------------------------------------- #
# Every listed safety item cites the manual passage that supports it          #
# --------------------------------------------------------------------------- #

from asa.components.explanation import explain  # noqa: E402

HAZARD_ITEM = "Hazardous voltages may exist and can cause electric shock or burns."
PPE_ITEM = "Safety goggles (non-conductive frame)"


def _row(chunk):
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "section_id": chunk.section_id,
        "revision": chunk.revision,
        "equipment_model": chunk.equipment_model,
        "text": chunk.text,
        "rrf_score": 1.0,
        "source_file": chunk.source_file,
        "page": chunk.page,
        "section_title": chunk.section_title,
    }


def _run_cited(monkeypatch, verdict="allow"):
    monkeypatch.setattr(safety, "_build_safety_search_query", lambda question, equipment_model: "q")
    monkeypatch.setattr(
        safety.rag_agent,
        "search_service_documents",
        lambda query, equipment_model, k: [_row(EVIDENCE), _row(INSTALLATION_PPE_EVIDENCE)],
    )
    support = {HAZARD_ITEM: EVIDENCE.chunk_id, PPE_ITEM: INSTALLATION_PPE_EVIDENCE.chunk_id}
    proposal = {
        "verdict": verdict,
        "reason": "x",
        "hazards": [HAZARD_ITEM],
        "ppe_required": [PPE_ITEM],
        "required_controls": [],
    }

    def fake_create(**kwargs):
        prompt = kwargs["messages"][1]["content"]
        if "validating one structured safety claim" in prompt:
            chunk_id = support.get(prompt.split("Candidate claim:\n", 1)[1].split("\n", 1)[0])
            return _response(
                json.dumps({"supported": bool(chunk_id), "supporting_chunk_ids": [chunk_id] if chunk_id else []})
            )
        if "validating a proposed equipment safety verdict" in prompt:
            return _response(json.dumps({"supported": True}))
        if "Return VALID JSON only" in prompt:
            return _response(json.dumps(proposal))
        raise AssertionError("Unexpected Safety Agent model call.")

    monkeypatch.setattr(safety.client.chat.completions, "create", fake_create)
    return safety.safety_node(dict(PRECAUTIONS_REQUEST))


def test_each_listed_item_cites_its_supporting_passage(monkeypatch):
    answer = _run_cited(monkeypatch)["final_answer"]
    assert (
        f"- {HAZARD_ITEM} [Source: aem_thermal_station.pdf, page 19, section 2.2.2 General Safety Guidelines]" in answer
    )
    # The PPE item cites ITS passage (page 22), not the first retrieved one (page 19).
    assert f"- {PPE_ITEM} [Source: aem_thermal_station.pdf, page 22, section 2.2.3 Personal Safety Equipment]" in answer


def test_a_halt_answer_cites_its_items_too(monkeypatch):
    result = _run_cited(monkeypatch, verdict="halt")
    assert result["safety_verdict"] == "halt"
    assert "listed below with their sources" in result["final_answer"]
    assert "[Source: aem_thermal_station.pdf, page 19" in result["final_answer"]


def test_the_explanation_layer_verifies_a_cited_safety_answer_strictly(monkeypatch):
    state = _run_cited(monkeypatch)
    explanation = explain({**state, "current_step": "safety"}, "ok")
    assert explanation["claims_checked"] == 2
    assert explanation["claims_verified"] == 2
    assert {(c["page"], c["chunk_id"]) for c in explanation["citations"]} == {
        (19, EVIDENCE.chunk_id),
        (22, INSTALLATION_PPE_EVIDENCE.chunk_id),
    }


def test_an_answer_without_validated_items_has_no_citations():
    assert safety._source_marker("anything", "hazard", [], [EVIDENCE]) == ""


# --------------------------------------------------------------------------- #
# The safety search also uses the engineer's own wording                      #
# --------------------------------------------------------------------------- #


def _fake_rows(prefix, n):
    return [{**_row(EVIDENCE), "chunk_id": f"{prefix}{i}"} for i in range(n)]


def test_results_are_merged_by_taking_turns_without_duplicates():
    first = _fake_rows("a", 5)
    second = [{**_row(EVIDENCE), "chunk_id": "a0"}] + _fake_rows("b", 4)  # "a0" found by both searches
    merged = safety._merge_results([first, second], limit=6)
    assert [r["chunk_id"] for r in merged] == ["a0", "a1", "b0", "a2", "b1", "a3"]


def test_safety_node_searches_with_both_queries(monkeypatch):
    searched = []
    _stub_retrieval(monkeypatch, searched)
    monkeypatch.setattr(
        safety.client.chat.completions, "create", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("stop"))
    )
    safety.safety_node(dict(SERVICING_REQUEST))
    assert searched == ["power off servicing", SERVICING_REQUEST["raw_query"]]


def test_one_failed_search_still_uses_the_other(monkeypatch):
    monkeypatch.setattr(safety, "_build_safety_search_query", lambda question, equipment_model: "power off servicing")

    def search(query, equipment_model, k):
        if query == "power off servicing":
            raise RuntimeError("index unavailable")
        return _evidence_rows()

    monkeypatch.setattr(safety.rag_agent, "search_service_documents", search)
    monkeypatch.setattr(
        safety.client.chat.completions, "create", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("stop"))
    )
    result = safety.safety_node(dict(SERVICING_REQUEST))
    assert [c.chunk_id for c in result["retrieved_chunks"]] == [EVIDENCE.chunk_id]


def test_when_every_search_fails_the_request_escalates(monkeypatch):
    monkeypatch.setattr(safety, "_build_safety_search_query", lambda question, equipment_model: "q")

    def broken(query, equipment_model, k):
        raise RuntimeError("index unavailable")

    monkeypatch.setattr(safety.rag_agent, "search_service_documents", broken)
    result = safety.safety_node(dict(SERVICING_REQUEST))
    assert result["safety_verdict"] == "escalate"
    assert "No relevant AEM safety documentation" in result["final_answer"]
