import copy
import random

import pytest

from asa.components.explanation import explain
from asa.graph.state import (
    Chunk,
    RankedCause,
    SafetyClaim,
    Step,
)


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
    # The three quoted manual lines verify. The narrative opening sentence is the model's own summary, not a manual
    # sentence, so it stays unverified (it says "explicitly" and "opening", which the manual does not) and the
    # confidence is capped just below high.
    assert result["claims_verified"] == 3 and result["claims_checked"] == 4
    assert all(c["page"] == 26 for c in result["citations"])
    assert 0.5 <= result["confidence"] < 0.70


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


def test_safety_claims_use_only_their_supporting_chunks():
    supported = Chunk(
        chunk_id="s19",
        doc_id="manual",
        section_id="2.2.2",
        revision="0",
        equipment_model="thermal_station",
        text=("Electrical Hazards: Hazardous voltages may exist and can cause electric shock or burns."),
        score=1.0,
        source_file="manual.pdf",
        page=19,
        section_title="General Safety Guidelines",
    )

    unrelated = Chunk(
        chunk_id="s22",
        doc_id="manual",
        section_id="2.2.3",
        revision="0",
        equipment_model="thermal_station",
        text=("Standard Personal Protective Equipment is required for installation."),
        score=1.0,
        source_file="manual.pdf",
        page=22,
        section_title="Personal Safety Equipment",
    )

    state = {
        "current_step": "safety",
        "final_answer": (
            "Safety decision: HALT\n\n"
            "Electrical Hazards: Hazardous voltages "
            "may exist and can cause electric shock "
            "or burns."
        ),
        "retrieved_chunks": [
            supported,
            unrelated,
        ],
        "safety_claims": [
            SafetyClaim(
                category="hazard",
                text=("Electrical Hazards: Hazardous voltages may exist and can cause electric shock or burns."),
                supporting_chunk_ids=[
                    "s19",
                ],
            )
        ],
        "sufficiency": True,
        "retry_count": 0,
        "escalated": False,
    }

    result = explain(
        state,
        "halted",
    )

    assert result["citations"]

    assert all(citation["chunk_id"] == "s19" for citation in result["citations"])

    assert all(citation["page"] == 19 for citation in result["citations"])


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


SEMI_ANNUAL_SOURCE = "[Source: aem_thermal_station.pdf, page 118, section 11.1.1 Semi Annual Preventive Maintenance]"


def test_a_marker_after_the_full_stop_cites_the_sentence_before_it():
    # How the RAG agent actually writes citations: the marker follows the full stop.
    explanation = explain(
        rag_state(final_answer=f"Spring washers must not be reused after removal. {SEMI_ANNUAL_SOURCE}"), "ok"
    )
    assert explanation["claims_checked"] == 1
    assert explanation["claims_verified"] == 1
    assert [(c["page"], c["chunk_id"]) for c in explanation["citations"]] == [(118, "c118")]


def test_a_marker_before_the_full_stop_still_cites_its_sentence():
    explanation = explain(
        rag_state(final_answer=f"Spring washers must not be reused after removal {SEMI_ANNUAL_SOURCE}."), "ok"
    )
    assert explanation["claims_verified"] == 1
    assert [c["page"] for c in explanation["citations"]] == [118]


def test_a_marker_on_its_own_line_cites_the_claim_above_it():
    answer = f"1. **Labels**: Replace any warning label that is not legible or is peeling.\n{SEMI_ANNUAL_SOURCE}"
    explanation = explain(rag_state(final_answer=answer), "ok")
    assert explanation["claims_checked"] == 1
    assert [c["page"] for c in explanation["citations"]] == [118]


def test_a_trailing_marker_is_only_checked_against_the_page_it_cites():
    # The sentence exists on page 117, but the answer cites page 118: the claim is not verified,
    # rather than being silently matched to a different page than the one the user is shown.
    answer = f"Power off the Thermal Station before performing any maintenance. {SEMI_ANNUAL_SOURCE}"
    explanation = explain(rag_state(final_answer=answer), "ok")
    assert explanation["claims_checked"] == 1
    assert explanation["claims_verified"] == 0
    assert explanation["citations"] == []


def _contradiction_notes(explanation):
    return [note for note in explanation["notes"] if "contradict" in note]


def test_a_claim_that_drops_the_negation_is_not_verified():
    # The manual says spring washers must NOT be reused; the claim says the opposite with the same words.
    explanation = explain(
        rag_state(final_answer=f"Spring washers must be reused after removal. {SEMI_ANNUAL_SOURCE}"), "ok"
    )
    assert explanation["claims_verified"] == 0
    assert explanation["citations"] == []
    assert _contradiction_notes(explanation) == [
        "1 claim(s) appear to contradict the cited manual sentence (negation, numbers or named terms differ): verify them."
    ]


def test_a_faithful_negative_claim_is_still_verified():
    explanation = explain(
        rag_state(final_answer=f"Spring washers must not be reused after removal. {SEMI_ANNUAL_SOURCE}"), "ok"
    )
    assert explanation["claims_verified"] == 1
    assert _contradiction_notes(explanation) == []


INTERVAL = chunk(
    "c118b",
    118,
    "11.1.1 Semi Annual Preventive Maintenance",
    "Semi Annual Preventive Maintenance to be performed every 6 months by qualified personnel.",
)


def test_a_claim_with_a_different_number_is_not_verified():
    answer = (
        f"Semi annual preventive maintenance is performed every 3 months by qualified personnel. {SEMI_ANNUAL_SOURCE}"
    )
    explanation = explain(rag_state(final_answer=answer, retrieved_chunks=[INTERVAL]), "ok")
    assert explanation["claims_verified"] == 0
    assert len(_contradiction_notes(explanation)) == 1


def test_a_claim_with_the_same_number_is_verified():
    answer = (
        f"Semi annual preventive maintenance is performed every 6 months by qualified personnel. {SEMI_ANNUAL_SOURCE}"
    )
    explanation = explain(rag_state(final_answer=answer, retrieved_chunks=[INTERVAL]), "ok")
    assert explanation["claims_verified"] == 1
    assert [c["chunk_id"] for c in explanation["citations"]] == ["c118b"]
    assert _contradiction_notes(explanation) == []


INVENTED = "Lubricate the compressor bearings with grade 5 grease every week"
PEELING_LABELS = "Replace any warning label that is not legible or is peeling"


def test_an_unverified_claim_caps_the_confidence_below_the_floor():
    answer = f"{PEELING_LABELS}. {SEMI_ANNUAL_SOURCE}\n{INVENTED}. {SEMI_ANNUAL_SOURCE}"
    result = explain(rag_state(final_answer=answer), "ok")
    assert (result["claims_verified"], result["claims_checked"]) == (1, 2)
    assert result["confidence"] <= 0.65 and result["confidence_level"] == "medium"
    assert any("capped below 0.70" in note for note in result["notes"])


def test_a_contradicted_claim_makes_the_confidence_low():
    answer = (
        f"{PEELING_LABELS}. {SEMI_ANNUAL_SOURCE}\nSpring washers must be reused after removal. {SEMI_ANNUAL_SOURCE}"
    )
    result = explain(rag_state(final_answer=answer), "ok")
    assert result["confidence"] <= 0.45 and result["confidence_level"] == "low"


def test_a_fully_verified_answer_is_not_capped():
    result = explain(rag_state(), "ok")
    assert result["confidence"] == 1.0
    assert not any("capped" in note for note in result["notes"])


def test_the_why_states_the_real_claim_counts():
    all_matched = explain(rag_state(), "ok")["decision"]["why"]
    assert all_matched == "Answered from 2 cited manual passage(s); 3 of 3 claim(s) were matched to a manual sentence."

    half = f"{PEELING_LABELS}. {SEMI_ANNUAL_SOURCE}\n{INVENTED}. {SEMI_ANNUAL_SOURCE}"
    assert explain(rag_state(final_answer=half), "ok")["decision"]["why"] == (
        "Answered from 1 cited manual passage(s); 1 of 2 claim(s) were matched to a manual sentence."
    )

    none = f"{INVENTED}. {SEMI_ANNUAL_SOURCE}"
    assert explain(rag_state(final_answer=none), "ok")["decision"]["why"] == (
        "None of the 1 claim(s) could be matched to the 3 retrieved manual passage(s)."
    )


def test_citations_carry_the_fields_of_the_state_citation_contract():
    result = explain(rag_state(), "ok")
    assert result["citations"]
    for citation in result["citations"]:
        source = next(c for c in CHUNKS if c.chunk_id == citation["chunk_id"])
        assert citation["doc_id"] == source.doc_id
        assert citation["section_id"] == source.section_id
        assert citation["revision"] == source.revision
        assert citation["quote_span"] == citation["quote"]


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


FACILITY_TABLE = chunk(
    "c40",
    40,
    "3.4 Facility Specification",
    "3-9\nRev 0\nAEM Thermal Station\nFacility\nWater\nFacility Water Supply\nTemperature Range\n15~25℃\n"
    "Facility Water Return Pressure\n＜1.0bar\nElectrical\nPower Requirement\n380 ~ 420V/50Hz, 3-phase\n"
    "Max Current\n50A\n",
)
TABLE_SOURCE = "[Source: aem_thermal_station.pdf, page 40, section 3.4 Facility Specification]"


def test_a_claim_taken_from_a_specification_table_row_is_verified():
    answer = f"The facility water supply temperature range is 15~25℃. {TABLE_SOURCE}"
    result = explain(rag_state(final_answer=answer, retrieved_chunks=[FACILITY_TABLE]), "ok")
    assert result["claims_verified"] == 1
    quote = result["citations"][0]["quote"]
    assert "Facility Water Supply Temperature Range 15~25" in quote
    assert norm(quote) in norm(FACILITY_TABLE.text)  # still an exact stretch of the passage


def test_a_wrong_value_from_a_specification_table_is_not_verified():
    answer = f"The facility water supply temperature range is 15~35℃. {TABLE_SOURCE}"
    result = explain(rag_state(final_answer=answer, retrieved_chunks=[FACILITY_TABLE]), "ok")
    assert result["claims_verified"] == 0
    assert any("contradict" in note for note in result["notes"])


def test_a_table_row_is_quoted_from_its_own_label_not_from_the_row_before():
    electrical = chunk(
        "c33",
        33,
        "3.1 Facility Requirements",
        "Item\nPhase Type\nVoltage\nWiring Configuration\nFull Load (Amps)\n"
        "PDU\n3 phase, 5 wire\n415VAC\nL1,L2,L3,N,E\n63A\n"
        "Chiller\n3 phase, 5 wire\n415VAC\nL1,L2,L3,N,E\n63A\n",
    )
    answer = "PDU: 3 phase, 5 wire, 415VAC, 63A full load [Source: aem_thermal_station.pdf, page 33, section 3.1]."
    quote = explain(rag_state(final_answer=answer, retrieved_chunks=[electrical]), "ok")["citations"][0]["quote"]
    assert quote.startswith("PDU") and "Chiller" not in quote


def test_a_multi_line_table_of_contents_is_still_not_quoted():
    toc = chunk(
        "c7",
        7,
        "Contents",
        "11 Maintenance 11-1\n11.1 Preventive Maintenance 11-2\n11.1.1 Daily Check 11-3\n11.1.2 Semi Annual Check 11-4\n",
    )
    answer = "Preventive maintenance includes the daily check and the semi annual check [Source: aem_thermal_station.pdf, page 7, section Contents]."
    assert explain(rag_state(final_answer=answer, retrieved_chunks=[toc]), "ok")["citations"] == []


def test_prose_sentences_are_still_preferred_over_table_rows():
    result = explain(rag_state(), "ok")
    for citation in result["citations"]:
        assert citation["quote"].endswith((".", "!", "?"))  # whole sentences, not line fragments


def test_table_of_contents_lines_are_not_quoted():
    toc = chunk(
        "c6",
        6,
        "Contents",
        "Maintenance 11-1 11.1 Preventive Maintenance 11-2 11.1.1 Daily Check 11-3 11.1.2 Semi Annual 11-4 11.2 Parts",
    )
    answer = "Daily preventive maintenance check [Source: aem_thermal_station.pdf, page 6, section Contents]."
    assert explain(rag_state(final_answer=answer, retrieved_chunks=[toc]), "ok")["citations"] == []


def test_a_line_full_of_section_numbers_is_not_quotable_even_when_it_is_mostly_words():
    from asa.components.explanation import _SECTION_NUMBER, _is_furniture

    assert _SECTION_NUMBER.findall("see 11.1 and 11.1.1 and 2.5.6 for details") == ["11.1", "11.1.1", "2.5.6"]
    line = "Preventive maintenance overview see 11.1 daily checks in 11.1.1 and semi annual checks in 11.1.2 for the thermal station"
    assert _is_furniture(line)
    assert not _is_furniture("Replace any label that is not legible or is peeling away from the panel surface.")


# --------------------------------------------------------------------------- #
# Short claims (hazard and PPE labels)                                        #
# --------------------------------------------------------------------------- #

GUIDELINES = chunk(
    "c19",
    19,
    "2.2.2 General Safety Guidelines",
    "Electrical Hazards: Hazardous voltages may exist and can cause electric shock or burns. "
    "Thermal Hazards: Surfaces may be hot during operation.",
)
GUIDELINES_SOURCE = "[Source: aem_thermal_station.pdf, page 19, section 2.2.2 General Safety Guidelines]"


def test_a_two_word_label_is_verified_when_both_words_are_in_the_cited_sentence():
    result = explain(
        rag_state(final_answer=f"- Electrical Hazards {GUIDELINES_SOURCE}", retrieved_chunks=[GUIDELINES]), "ok"
    )
    assert result["claims_verified"] == 1
    assert result["citations"][0]["quote"].startswith("Electrical Hazards")


def test_a_one_word_claim_is_never_verified():
    result = explain(rag_state(final_answer=f"- Electrical {GUIDELINES_SOURCE}", retrieved_chunks=[GUIDELINES]), "ok")
    assert result["claims_verified"] == 0


def test_a_two_word_label_missing_from_its_cited_page_is_not_verified():
    result = explain(
        rag_state(final_answer=f"- Radiation Hazards {GUIDELINES_SOURCE}", retrieved_chunks=[GUIDELINES]), "ok"
    )
    assert result["claims_verified"] == 0
