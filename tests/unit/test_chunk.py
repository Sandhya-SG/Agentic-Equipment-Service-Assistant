import pytest

from asa.ingestion.chunk import (
    CONTENTS_SECTION,
    FRONT_MATTER_SECTION,
    UNKNOWN_SECTION,
    detect_section,
    page_sections,
)

HEADER = "3-2\nRev 0\nAEM 1kW Thermal Retrofit Manual\n"


@pytest.mark.parametrize(
    "heading",
    [
        "3.1 Facility Requirements",
        "2.5.3 Hazardous Energies Listing",
        "6.2.4 Lock Out / Tag Out (LOTO)",
        "10.2 Electrical / Schematic Circuits Content",
        "2 Safety",
    ],
)
def test_real_section_headings_are_detected(heading):
    assert detect_section(f"{HEADER}{heading}\nSome text follows here.") == heading


@pytest.mark.parametrize(
    "line",
    [
        "83.33 L/min @ 36.2psi @EGW -30C",  # a table value
        "2. To the fullest extent permitted by law, AEM shall not be responsible or liable for any",  # legal clause
        "5. The tag must be filled with the information of the personnel who is",  # procedure step
        "6. Remove 4x screws on kit and remove the substrate pusher plate.",  # procedure step
        "3 phase, 5 wire",  # a table cell
    ],
)
def test_table_values_steps_and_clauses_are_not_headings(line):
    assert detect_section(f"{HEADER}{line}\nMore text.") == UNKNOWN_SECTION


def test_a_table_of_contents_page_is_labelled_as_such():
    toc = "Contents\n" + "\n".join(f"{n}.{m} Section {n}{m} {n}-{m}" for n in range(1, 4) for m in range(1, 4))
    assert detect_section(toc) == CONTENTS_SECTION


def test_a_contents_page_with_page_numbers_on_their_own_lines_is_recognised():
    # As in the real manuals: "10.2 Electrical ..." then "10-4" on the next line.
    toc = (
        "vi\nRev 0\nAEM 1kW Thermal Retrofit Manual\n10.2 Electrical / Schematic Circuits Content\n10-4\n"
        "11. Maintenance\n11-1\n11.1\nPreventive Maintenance\n11-2\n11.1.1\nDaily Preventive Check\n11-3\n"
        "11.1.2 Semi Annual Preventive Maintenance\n11-4\n"
    )
    assert detect_section(toc) == CONTENTS_SECTION


def test_a_body_page_with_one_page_number_is_not_a_contents_page():
    assert detect_section(f"{HEADER}3.1 Facility Requirements\n3.1.1 Electrical Requirements\n") == (
        "3.1 Facility Requirements"
    )


def test_a_roman_numbered_page_is_front_matter_even_if_it_lists_a_heading():
    # The end of the contents plus a revision record, as on the real page vi.
    page = "vi\nRev 0\nAEM Thermal Station\n12. Appendix\n12-1\n12.1.1 Isopropyl Alcohol\n12-2\nRevision Record\n"
    assert detect_section(page) == FRONT_MATTER_SECTION


def test_front_matter_is_not_carried_forward():
    pages = [f"{HEADER}1.1 Overview\ntext", "vii\nRev 0\nTerm\nDefinition\nRTD", "no heading here"]
    assert page_sections(pages) == ["1.1 Overview", FRONT_MATTER_SECTION, "1.1 Overview"]


def test_a_page_without_a_heading_continues_the_previous_section():
    pages = [
        "AEM 1kW Thermal Retrofit Manual\nCover page",
        f"{HEADER}2.5 Safety Hazard\nHazardous voltages may exist.",
        f"{HEADER}continued hazard text without a heading",
        f"{HEADER}2.5.3 Hazardous Energies Listing\nTable 2-4",
    ]
    assert page_sections(pages) == [
        UNKNOWN_SECTION,
        "2.5 Safety Hazard",
        "2.5 Safety Hazard",
        "2.5.3 Hazardous Energies Listing",
    ]


def test_the_table_of_contents_is_not_carried_forward():
    toc = "Contents\n" + "\n".join(f"{n}.{m} Section {n}{m} {n}-{m}" for n in range(1, 4) for m in range(1, 4))
    pages = [f"{HEADER}1.1 Overview\ntext", toc, "page after the contents, no heading"]
    assert page_sections(pages) == ["1.1 Overview", CONTENTS_SECTION, "1.1 Overview"]
