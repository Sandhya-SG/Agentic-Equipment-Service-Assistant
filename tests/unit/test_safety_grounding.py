"""Unit tests for Safety Agent grounding validation."""

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

    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=text
                )
            )
        ]
    )


def test_validator_retains_supported_and_removes_unsupported_items(
    monkeypatch,
):
    """
    The validator must retain claims marked SUPPORTED and
    remove claims marked UNSUPPORTED.
    """


    def fake_create(**kwargs):

        prompt = kwargs[
            "messages"
        ][1]["content"]

        supported_claims = [
            "Electrical Hazards",
            (
                "Only qualified personnel should "
                "service this equipment."
            ),
            (
                "Perform equipment or module power down "
                "and power off before servicing."
            ),
        ]

        for claim in supported_claims:

            candidate_block = (
                "Candidate claim:\n"
                f"{claim}\n"
            )

            if candidate_block in prompt:
                return _response(
                    "SUPPORTED"
                )

        return _response(
            "UNSUPPORTED"
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
        (
            "Perform equipment or module power down "
            "and power off before servicing."
        ),
        "Disable all safety interlocks before servicing.",
    ]

    (
        valid_hazards,
        valid_ppe,
        valid_controls,
    ) = safety._validate_safety_items(
        question=(
            "What safety precautions apply "
            "before servicing the equipment?"
        ),
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
        (
            "Perform equipment or module power down "
            "and power off before servicing."
        ),
    ]


def test_validator_rejects_ppe_from_unrelated_activity(
    monkeypatch,
):
    """
    PPE appearing in retrieved evidence is not automatically
    applicable to a different engineer activity.
    """

    def fake_create(**kwargs):

        prompt = kwargs[
            "messages"
        ][1]["content"]

        assert (
            "Safety goggles (non-conductive frame)"
            in prompt
        )

        assert (
            "Can I bypass the safety interlock?"
            in prompt
        )

        # Simulate the grounding validator deciding that
        # installation/decommissioning PPE is not established
        # as applicable to an interlock-bypass request.
        return _response(
            "UNSUPPORTED"
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
    ) = safety._validate_safety_items(
        question=(
            "Can I bypass the safety interlock?"
        ),
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


def test_supported_allow_verdict_is_retained(
    monkeypatch,
):
    """
    A grounded ALLOW verdict remains ALLOW.
    """

    def fake_create(**kwargs):

        prompt = kwargs[
            "messages"
        ][1]["content"]

        assert (
            "Proposed verdict:\nALLOW"
            in prompt
        )

        return _response(
            "SUPPORTED"
        )

    monkeypatch.setattr(
        safety.client.chat.completions,
        "create",
        fake_create,
    )

    verdict = safety._validate_safety_verdict(
        question=(
            "What safety precautions apply "
            "before servicing the equipment?"
        ),
        chunks=[EVIDENCE],
        candidate_verdict="allow",
        validated_controls=[
            (
                "Perform equipment or module power down "
                "and power off before servicing."
            ),
        ],
    )

    assert verdict == "allow"


def test_supported_halt_verdict_is_retained(
    monkeypatch,
):
    """
    A grounded HALT verdict remains HALT.
    """

    def fake_create(**kwargs):

        prompt = kwargs[
            "messages"
        ][1]["content"]

        assert (
            "Proposed verdict:\nHALT"
            in prompt
        )

        return _response(
            "SUPPORTED"
        )

    monkeypatch.setattr(
        safety.client.chat.completions,
        "create",
        fake_create,
    )

    verdict = safety._validate_safety_verdict(
        question=(
            "Can I service the equipment "
            "without powering it off?"
        ),
        chunks=[EVIDENCE],
        candidate_verdict="halt",
        validated_controls=[
            (
                "Perform equipment or module power down "
                "and power off before servicing."
            ),
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

        prompt = kwargs[
            "messages"
        ][1]["content"]

        assert (
            "Proposed verdict:\nALLOW"
            in prompt
        )

        return _response(
            "UNSUPPORTED"
        )

    monkeypatch.setattr(
        safety.client.chat.completions,
        "create",
        fake_create,
    )

    verdict = safety._validate_safety_verdict(
        question=(
            "Can I perform this action "
            "while the equipment is powered?"
        ),
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
        lambda question, equipment_model: (
            "servicing safety power off"
        ),
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

    unsafe_reason = (
        "The equipment can safely remain energized "
        "because internal isolation removes all risk."
    )

    safety_json = {
        "verdict": "allow",
        "reason": unsafe_reason,
        "hazards": [
            "Electrical Hazards",
        ],
        "ppe_required": [],
        "required_controls": [
            (
                "Perform equipment or module power down "
                "and power off before servicing."
            ),
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

        prompt = kwargs[
            "messages"
        ][1]["content"]

        if (
            "Return VALID JSON only"
            in prompt
        ):
            import json

            return _response(
                json.dumps(
                    safety_json
                )
            )

        if (
            "validating one structured safety claim"
            in prompt
        ):
            return _response(
                "SUPPORTED"
            )

        if (
            "validating a proposed equipment safety verdict"
            in prompt
        ):
            return _response(
                "SUPPORTED"
            )

        raise AssertionError(
            "Unexpected Safety Agent model call."
        )

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
            "raw_query": (
                "What safety precautions apply "
                "before servicing the equipment?"
            ),
            "equipment_model": (
                "thermal_station"
            ),
        }
    )

    # --------------------------------------------------
    # Grounded result
    # --------------------------------------------------

    assert (
        result["safety_verdict"]
        == "allow"
    )

    # The original LLM reason must not survive.
    assert (
        unsafe_reason
        not in result["safety_reason"]
    )

    assert (
        unsafe_reason
        not in result["final_answer"]
    )

    # The deterministic grounded reason should be used.
    assert (
        "retrieved AEM manual evidence"
        in result["safety_reason"]
    )

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
        lambda question, equipment_model: (
            "powered servicing safety requirements"
        ),
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

    unsafe_reason = (
        "The equipment may remain powered because "
        "the requested work is safe while energized."
    )

    safety_json = {
        "verdict": "allow",
        "reason": unsafe_reason,
        "hazards": [],
        "ppe_required": [],
        "required_controls": [],
    }

    def fake_create(**kwargs):

        prompt = kwargs[
            "messages"
        ][1]["content"]

        # Initial Safety Agent generation.
        if (
            "Return VALID JSON only"
            in prompt
        ):
            import json

            return _response(
                json.dumps(
                    safety_json
                )
            )

        # There are no structured items in this test,
        # so the next model call should be verdict
        # validation.
        if (
            "validating a proposed equipment safety verdict"
            in prompt
        ):
            return _response(
                "UNSUPPORTED"
            )

        raise AssertionError(
            "Unexpected Safety Agent model call."
        )

    monkeypatch.setattr(
        safety.client.chat.completions,
        "create",
        fake_create,
    )

    result = safety.safety_node(
        {
            "raw_query": (
                "Can I service the equipment "
                "while it is powered?"
            ),
            "equipment_model":
                "thermal_station",
        }
    )

    # Candidate ALLOW must not survive.
    assert (
        result["safety_verdict"]
        == "escalate"
    )

    assert (
        result["sufficiency"]
        is False
    )

    assert (
        result["escalated"]
        is True
    )

    assert result[
        "escalation_reason"
    ]

    # The original unsupported reason must not leak.
    assert (
        unsafe_reason
        not in result["safety_reason"]
    )

    assert (
        unsafe_reason
        not in result["final_answer"]
    )

    assert (
        "insufficient"
        in result[
            "safety_reason"
        ].lower()
    )

    assert (
        "ESCALATE"
        in result[
            "final_answer"
        ]
    )