import copy
import random

import pytest

from asa.components.explanation import explain
from asa.graph.state import Chunk, RankedCause, Step


def chunk(chunk_id, page, section, text, source="aem_thermal_station.pdf"):
    return Chunk(
        chunk_id=chunk_id,
        doc_id="aem_thermal_station",
        section_id=f"{source}#p{page}",
        revision="unknown",
        equipment_model="thermal_station",
        text=text,
        score=0.03,
        source_file=source,
        page=page,
        section_title=section,
    )


MAINTENANCE = chunk(
    "c117",
    117,
    "11.1 AEM Thermal Station Preventive Maintenance",
    "Two persons are required when commencing maintenance and servicing. One person monitors the push button\n"
    "panel to prevent accidental activation while the other services the machine. Power off the Thermal Station\n"
    "before performing any maintenance or replacement of parts.",
)
SEMI_ANNUAL = chunk(
    "c118",
    118,
    "11.1.1 Semi Annual Preventive Maintenance",
    "Check all warning label conditions and adhesion. Replace any label that is not legible or is peeling away from "
    "the panel surface. Spring washers must not be reused after removal.",
)
POWER = chunk(
    "c26",
    26,
    "2.5 Safety Hazard",
    "Hazardous voltages may exist and can cause electric shock or burns. Perform equipment or module power down and "
    "power off before servicing. Ensure that all panel doors, which are equipped with safety interlock, are closed.",
)
CHUNKS = [MAINTENANCE, SEMI_ANNUAL, POWER]

RAG_ANSWER = (
    "1. **Two Persons Required**: Always have two people present when commencing maintenance and servicing "
    "[Source: aem_thermal_station.pdf, page 117, section 11.1 AEM Thermal Station Preventive Maintenance].\n"
    "2. **Power Off**: Before any maintenance or replacement of parts, power off the Thermal Station "
    "[Source: aem_thermal_station.pdf, page 117, section 11.1 AEM Thermal Station Preventive Maintenance; "
    "Source: aem_thermal_station.pdf, page 118, section 11.1.1 Semi Annual Preventive Maintenance].\n"
    "3. **Labels**: Replace any warning label that is not legible or is peeling "
    "[Source: aem_thermal_station.pdf, page 118, section 11.1.1 Semi Annual Preventive Maintenance]."
)


def rag_state(**overrides):
    state = {
        "request_status": "READY",
        "current_step": "agentic_rag",
        "final_answer": RAG_ANSWER,
        "retrieved_chunks": list(CHUNKS),
        "sufficiency": True,
        "retry_count": 0,
        "escalated": False,
    }
    state.update(overrides)
    return state


def norm(text):
    return " ".join(text.split())


def chunk_by_id(chunk_id):
    return next(c for c in CHUNKS if c.chunk_id == chunk_id)


# --------------------------------------------------------------------------- #
# What gets explained                                                         #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("status", ["clarification", "blocked", "unavailable"])
def test_nothing_to_explain_without_evidence(status):
    assert explain(rag_state(), status) is None


def test_the_required_keys_are_present():
    result = explain(rag_state(), "ok")
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["confidence_level"] in ("low", "medium", "high")
    assert {"decision", "citations", "notes", "claims_checked", "claims_verified"} <= set(result)


# --------------------------------------------------------------------------- #
# Citations and quotes                                                        #
# --------------------------------------------------------------------------- #


def test_every_quote_is_an_exact_sentence_of_its_passage():
    result = explain(rag_state(), "ok")
    assert result["citations"]
    for citation in result["citations"]:
        source = chunk_by_id(citation["chunk_id"])
        assert norm(citation["quote"]) in norm(source.text)
        assert citation["source_file"] == source.source_file and citation["page"] == source.page
        assert citation["section"] == source.section_title


def test_a_sentence_that_the_manual_wraps_across_lines_is_quoted_whole():
    answer = "One person monitors the push button panel to prevent accidental activation [Source: aem_thermal_station.pdf, page 117, section 11.1]."
    quote = explain(rag_state(final_answer=answer), "ok")["citations"][0]["quote"]
    assert (
        "push button panel to prevent accidental activation" in quote
    )  # the passage breaks the line after "push button"


def test_claims_are_matched_to_the_passage_they_cite():
    result = explain(rag_state(), "ok")
    by_page = {c["page"]: c["quote"] for c in result["citations"]}
    assert "two persons" in by_page[117].lower() or "power off" in by_page[117].lower()
    assert "label" in by_page[118].lower()


def test_one_bracket_may_hold_several_sources():
    result = explain(rag_state(), "ok")
    pages = {c["page"] for c in result["citations"]}
    assert pages == {117, 118}


def test_a_claim_with_no_matching_manual_sentence_is_not_invented():
    answer = (
        "Replace the main boiler gasket with a titanium one [Source: aem_thermal_station.pdf, page 117, section 11.1]."
    )
    result = explain(rag_state(final_answer=answer), "ok")
    assert result["citations"] == []
    assert result["claims_checked"] == 1 and result["claims_verified"] == 0
    assert any("could not be matched" in note for note in result["notes"])


def test_citations_are_not_duplicated():
    answer = (
        "Power off before maintenance [Source: aem_thermal_station.pdf, page 117, section 11.1]. "
        "Power off the Thermal Station before maintenance [Source: aem_thermal_station.pdf, page 117, section 11.1]."
    )
    result = explain(rag_state(final_answer=answer), "ok")
    quotes = [c["quote"] for c in result["citations"]]
    assert len(quotes) == len(set(quotes))


# --------------------------------------------------------------------------- #
# Confidence                                                                  #
# --------------------------------------------------------------------------- #


def test_well_supported_answer_has_high_confidence():
    result = explain(rag_state(), "ok")
    assert result["confidence"] >= 0.85 and result["confidence_level"] == "high"


def test_confidence_adds_up_as_specified():
    # 0.50 base + 0.20 sufficient evidence + 0.15 two sections + 0.15 all claims verified = 1.0
    assert explain(rag_state(), "ok")["confidence"] == 1.0
    # one section only: no section bonus
    one_section = "Power off before maintenance [Source: aem_thermal_station.pdf, page 117, section 11.1]."
    assert explain(rag_state(final_answer=one_section), "ok")["confidence"] == 0.85
    # evidence not judged sufficient: no sufficiency bonus
    assert explain(rag_state(final_answer=one_section, sufficiency=False), "ok")["confidence"] == 0.65


def test_each_retry_lowers_the_confidence_up_to_a_limit():
    one_section = "Power off before maintenance [Source: aem_thermal_station.pdf, page 117, section 11.1]."
    base = explain(rag_state(final_answer=one_section), "ok")["confidence"]
    one = explain(rag_state(final_answer=one_section, retry_count=1), "ok")["confidence"]
    many = explain(rag_state(final_answer=one_section, retry_count=9), "ok")["confidence"]
    assert one == round(base - 0.15, 2)
    assert many == round(base - 0.30, 2)


def test_an_escalated_run_is_capped_and_marked_low():
    result = explain(rag_state(escalated=True, escalation_reason="Insufficient manual evidence."), "escalated")
    assert result["confidence"] <= 0.30 and result["confidence_level"] == "low"
    assert result["decision"]["why"] == "Insufficient manual evidence."
    assert any("capped" in note for note in result["notes"])
    assert any("verify with a qualified engineer" in note for note in result["notes"])


def test_an_answer_with_nothing_quotable_is_low_confidence():
    result = explain(rag_state(retrieved_chunks=[], final_answer="Nothing to cite here at all really."), "ok")
    assert result["citations"] == []
    assert result["confidence"] <= 0.40
    assert any("No manual sentence" in note for note in result["notes"])


@pytest.mark.parametrize(
    "confidence,level", [(0.0, "low"), (0.49, "low"), (0.5, "medium"), (0.69, "medium"), (0.7, "high")]
)
def test_level_boundaries(confidence, level):
    from asa.components.explanation import _confidence_level

    assert _confidence_level(confidence) == level


# --------------------------------------------------------------------------- #
# Other specialists                                                           #
# --------------------------------------------------------------------------- #


def test_safety_halt_is_explained_from_its_own_lines_and_reason():
    answer = (
        "Safety decision: HALT\n\nThe documentation explicitly requires power-off before servicing, including opening panel doors.\n\n"
        "Documented hazards:\n- Electrical Hazards: Hazardous voltages may exist and can cause electric shock or burns.\n\n"
        "Required documented controls:\n- Perform equipment or module power down and power off before servicing.\n"
        "- Ensure that all panel doors, which are equipped with safety interlock, are closed."
    )
    state = rag_state(
        current_step="safety",
        final_answer=answer,
        safety_verdict="halt",
        safety_reason="The manual requires power-off before opening panel doors.",
        retrieved_chunks=[POWER],
    )
    result = explain(state, "halted")
    assert result["decision"]["verdict"] == "halt"
    assert result["decision"]["why"] == "The manual requires power-off before opening panel doors."
    assert result["claims_verified"] >= 2
    assert all(c["page"] == 26 for c in result["citations"])
    assert result["confidence"] >= 0.70  # a documented prohibition is strong evidence


def test_diagnostic_claims_use_their_own_supporting_passages():
    state = rag_state(
        current_step="diagnostic",
        final_answer="Diagnostic assessment: The root cause is not confirmed.",
        root_causes=[
            RankedCause(
                cause="Panel doors are not closed so the interlock stays open",
                likelihood=0.6,
                supporting_chunk_ids=["c26"],
            )
        ],
        troubleshooting_steps=[
            Step(
                order=1,
                action="Check that all panel doors with the safety interlock are closed",
                supporting_chunk_ids=["c26"],
            )
        ],
        retrieved_chunks=[POWER, SEMI_ANNUAL],
    )
    result = explain(state, "ok")
    assert result["claims_checked"] == 2
    assert result["citations"] and all(c["chunk_id"] == "c26" for c in result["citations"])
    assert "root cause is not confirmed" in result["decision"]["why"]


# --------------------------------------------------------------------------- #
# Purity and robustness                                                       #
# --------------------------------------------------------------------------- #


def test_it_does_not_change_the_state_and_is_repeatable():
    state = rag_state()
    before = copy.deepcopy(state)
    first = explain(state, "ok")
    second = explain(state, "ok")
    assert state == before
    assert first == second


def test_passages_may_be_plain_dictionaries():
    as_dicts = [
        {
            "chunk_id": c.chunk_id,
            "source_file": c.source_file,
            "page": c.page,
            "section_title": c.section_title,
            "text": c.text,
        }
        for c in CHUNKS
    ]
    result = explain(rag_state(retrieved_chunks=as_dicts), "ok")
    assert result["citations"] and result["confidence"] == 1.0


def test_odd_input_never_raises():
    odd_states = [
        {},
        {"final_answer": None, "retrieved_chunks": None},
        {"final_answer": 123, "retrieved_chunks": [None, 5, "x"]},
        {"final_answer": "[Source: , page x]", "retrieved_chunks": [{"page": "abc", "text": None}]},
        rag_state(retry_count="many", sufficiency=None),
        rag_state(root_causes=[None, {"cause": None}], troubleshooting_steps=["text"]),
    ]
    for state in odd_states:
        for status in ("ok", "halted", "escalated"):
            try:
                result = explain(state, status)
            except (TypeError, ValueError, AttributeError):
                pytest.fail(f"explain raised for {state!r}")
            assert result is None or 0.0 <= result["confidence"] <= 1.0


def test_random_answers_never_break_the_invariants():
    rng = random.Random(7)
    words = "power off panel label spring washer maintenance Source: page section ] [ ; , . 117 118 26 safety interlock".split()
    for _ in range(200):
        answer = " ".join(rng.choice(words) for _ in range(rng.randint(0, 40)))
        result = explain(rag_state(final_answer=answer, retry_count=rng.randint(0, 4)), "ok")
        assert 0.0 <= result["confidence"] <= 1.0
        for citation in result["citations"]:
            assert norm(citation["quote"]) in norm(chunk_by_id(citation["chunk_id"]).text)


def test_a_citation_to_a_page_that_was_never_retrieved_does_not_verify():
    answer = "Power off the Thermal Station before maintenance [Source: aem_thermal_station.pdf, page 999, section 99.9 Invented]."
    result = explain(rag_state(final_answer=answer), "ok")
    assert result["claims_checked"] == 1 and result["claims_verified"] == 0
    assert result["citations"] == []


def test_an_escalated_answer_quotes_nothing_because_nothing_is_shown():
    result = explain(rag_state(escalated=True), "escalated")
    assert result["citations"] == [] and result["claims_checked"] == 0


def test_page_headers_are_stripped_from_the_front_of_a_quote():
    noisy = chunk(
        "c200",
        118,
        "11.1.1 Semi Annual",
        "11-3 Rev 0 AEM Thermal Station Replace any label that is not legible or is peeling away from the panel.",
    )
    answer = (
        "Replace any label that is not legible or peeling [Source: aem_thermal_station.pdf, page 118, section 11.1.1]."
    )
    quote = explain(rag_state(final_answer=answer, retrieved_chunks=[noisy]), "ok")["citations"][0]["quote"]
    assert quote.startswith("Replace any label") and "Rev 0" not in quote


def test_table_of_contents_lines_are_not_quoted():
    toc = chunk(
        "c6",
        6,
        "Contents",
        "Maintenance 11-1 11.1 Preventive Maintenance 11-2 11.1.1 Daily Check 11-3 11.1.2 Semi Annual 11-4 11.2 Parts",
    )
    answer = "Daily preventive maintenance check [Source: aem_thermal_station.pdf, page 6, section Contents]."
    assert explain(rag_state(final_answer=answer, retrieved_chunks=[toc]), "ok")["citations"] == []
